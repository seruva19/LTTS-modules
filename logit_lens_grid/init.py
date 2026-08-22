"""
Logit Lens Grid Module

The canonical logit-lens picture: a grid where rows are transformer depth and
columns are token positions, each cell showing the top-1 token the model would
predict *at that position, from that layer*, coloured by confidence. It makes
visible WHERE and WHEN a prediction forms across depth - the single most
instructive view for learning how a transformer refines its output.

`logit_lens` gives the per-layer top-k table + a probability-vs-depth line;
this gives the full layers x positions grid in one image. Cross-layer
accumulator: each block event appends a row and the (growing) grid is re-emitted
under a stable id, so the final image shows every layer.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.core.module_data_sender import get_data_sender  # noqa: E402

logger = logging.getLogger(__name__)

METADATA = {
    "name": "logit_lens_grid",
    "version": "0.1.0",
    "description": "Layers x positions logit-lens grid - the top-1 predicted token at each position decoded from every layer, coloured by confidence",
    "author": "LTTS",
    "event_types": ["layer_after"],
    "dependencies": ["requirements.txt"],
    "methods": {
        "ctor": "initialize_module",
        "dtor": "cleanup_module",
        "ntor": "process_event",
        "utor": "get_ui_schema",
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "max_tokens": 16,
    "apply_final_norm": True,
    "annotate": True,
    "colormap": "viridis",
    "emit_mode": "all",
    "emit_every_n": 1,
}

_SENDER = None

# Each layer event gets a fresh context, so the grid is accumulated in module
# globals keyed by forward pass.
_GRID: Dict[str, Any] = {"forward_pass": None, "rows": []}


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    logger.info("LOGIT_LENS_GRID: initialized")
    return {"status": "initialized"}


def cleanup_module():
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {"name": "max_tokens", "type": "number", "label": "Max token positions",
             "default": 16, "min": 2, "max": 64},
            {"name": "apply_final_norm", "type": "boolean",
             "label": "Apply final LayerNorm before unembedding", "default": True},
            {"name": "annotate", "type": "boolean",
             "label": "Annotate cells with the token", "default": True},
            {"name": "colormap", "type": "select", "label": "Colormap",
             "default": "viridis",
             "options": ["viridis", "magma", "plasma", "cividis"]},
        ],
        "layout": {"type": "vertical"},
        "real_time": {"enabled": True, "interval": 200},
    }


def _resolve_config(ltts_event) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if isinstance(ltts_event.module_state, dict):
        cfg.update(ltts_event.module_state)
    env = getattr(ltts_event, "environment", None)
    if isinstance(env, dict) and isinstance(env.get("module_config"), dict):
        cfg.update(env["module_config"])
    return cfg


def _extract_hidden(outputs: Any) -> Optional[torch.Tensor]:
    candidate = outputs
    if isinstance(candidate, (list, tuple)) and candidate:
        candidate = candidate[0]
    if isinstance(candidate, torch.Tensor) and candidate.dim() == 3:
        return candidate
    return None


def _get_hf_model(ltts_model) -> Any:
    seen = set()
    candidate = ltts_model
    for _ in range(8):
        if candidate is None or id(candidate) in seen:
            break
        seen.add(id(candidate))
        if hasattr(candidate, "get_output_embeddings") or hasattr(candidate, "lm_head"):
            return candidate
        nxt = getattr(candidate, "original_model", None) or getattr(candidate, "model", None)
        candidate = nxt
    return candidate


def _get_unembedding(hf_model):
    try:
        head = hf_model.get_output_embeddings()
        if head is not None:
            return head
    except Exception:
        pass
    return getattr(hf_model, "lm_head", None)


def _get_final_norm(hf_model):
    candidates = [
        ("model", "norm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
        ("model", "final_layernorm"),
    ]
    for parent_name, norm_name in candidates:
        parent = getattr(hf_model, parent_name, None)
        if parent is not None:
            norm = getattr(parent, norm_name, None)
            if isinstance(norm, torch.nn.Module):
                return norm
    return None


def _decode(ltts_model, token_id: int) -> str:
    tokenizer = getattr(ltts_model, "_tokenizer", None)
    if tokenizer is not None:
        try:
            return tokenizer.decode([token_id]).strip() or " "
        except Exception:
            pass
    return f"#{token_id}"


def _input_labels(ltts_model, seq_len: int) -> Optional[List[str]]:
    toks = getattr(ltts_model, "_last_input_tokens", None)
    if toks:
        return [str(t) for t in toks][:seq_len]
    return None


def process_event(ltts_event):
    try:
        evt = getattr(ltts_event, "event_type", "") or ""
        if not (evt == "layer_after" or evt.startswith("layer_after_")):
            return {"status": "skipped", "reason": "event_phase"}

        context = ltts_event.context
        module_class = (context.module_class or "").lower()
        is_block = "decoderlayer" in module_class or module_class.endswith(("block", "layer"))
        if getattr(context, "module_role", None) != "block" and (
            context.layer_type != "unknown" or not is_block
        ):
            return {"status": "skipped", "reason": "not_a_block_boundary"}

        hidden = _extract_hidden(context.outputs)
        if hidden is None:
            return {"status": "skipped", "reason": "no_hidden_state"}

        ltts_model = context.ltts_model
        hf_model = _get_hf_model(ltts_model) if ltts_model is not None else None
        unembed = _get_unembedding(hf_model) if hf_model is not None else None
        if unembed is None:
            return {"status": "skipped", "reason": "no_unembedding"}

        cfg = _resolve_config(ltts_event)
        layer_path = getattr(context, "module_path", None)
        emission = ltts_event.should_emit(METADATA["name"], layer_path, cfg)
        if not emission.get("emit"):
            return {"status": "skipped", "reason": "emission_controller"}

        max_tokens = int(cfg.get("max_tokens", 16) or 16)
        with torch.no_grad():
            states = hidden[0].float()  # [seq, hid]
            seq = min(states.shape[0], max_tokens)
            states = states[:seq]
            if cfg.get("apply_final_norm", True):
                norm = _get_final_norm(hf_model)
                if norm is not None:
                    nd = next(norm.parameters(), states).dtype
                    states = norm(states.to(nd)).float()
            wd = next(unembed.parameters(), states).dtype
            logits = unembed(states.to(wd)).float()  # [seq, vocab]
            probs = torch.softmax(logits, dim=-1)
            top_prob, top_id = probs.max(dim=-1)

        tokens = [_decode(ltts_model, int(i)) for i in top_id.tolist()]
        row_probs = [float(p) for p in top_prob.tolist()]

        # accumulate
        current_pass = emission.get("forward_pass")
        if _GRID["forward_pass"] != current_pass:
            _GRID["forward_pass"] = current_pass
            _GRID["rows"] = []
        _GRID["rows"].append({"tokens": tokens, "probs": row_probs})

        rows = _GRID["rows"]
        n_layers = len(rows)
        seq = min(seq, min(len(r["tokens"]) for r in rows))
        prob_mat = np.array([[r["probs"][j] for j in range(seq)] for r in rows], dtype=np.float32)
        tok_mat = [[rows[i]["tokens"][j] for j in range(seq)] for i in range(n_layers)]

        input_labels = _input_labels(ltts_model, seq)
        annotate = bool(cfg.get("annotate", True)) and (n_layers * seq <= 400)
        cmap = cfg.get("colormap", "viridis")

        fig, ax = plt.subplots(figsize=(max(6, seq * 1.05), max(3.5, n_layers * 0.34)))
        im = ax.imshow(prob_mat, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="top-1 probability")
        ax.set_xlabel("Input token position -> predicted next token")
        ax.set_ylabel("Layer (depth)")
        ax.set_title("Logit lens grid - prediction by layer x position")
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([str(i) for i in range(n_layers)], fontsize=7)
        if input_labels:
            ax.set_xticks(range(seq))
            ax.set_xticklabels(input_labels, rotation=90, fontsize=7)
        if annotate:
            thresh = 0.5
            for i in range(n_layers):
                for j in range(seq):
                    color = "white" if prob_mat[i, j] < thresh else "black"
                    ax.text(j, i, tok_mat[i][j], ha="center", va="center",
                            fontsize=6, color=color)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        if _SENDER:
            _SENDER.send_image(
                uri,
                emit_id="logit_lens_grid:grid",
                layer_path=layer_path,
                emit_mode=cfg.get("emit_mode", "all"),
            )
        return {"status": "ok", "layers": n_layers, "tokens": int(seq)}

    except Exception as exc:
        logger.error("LOGIT_LENS_GRID error: %s", exc, exc_info=True)
        try:
            ltts_event.reactor.signal(
                "error",
                {"message": f"logit_lens_grid error: {exc}", "module": METADATA["name"]},
            )
        except Exception:
            pass
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
