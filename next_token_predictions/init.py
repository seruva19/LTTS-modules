"""
Next-Token Predictions Module

The most direct "what is the model thinking?" view, decoded from the layer it is
attached to (attach it to the last block to see the model's actual next-token
distribution; attach it earlier to see the logit-lens intermediate guess).

Deliberately emits THREE different output types, not a heatmap:
- a scalar: the next-token distribution entropy in bits (low = confident);
- a table: the top-k candidate tokens with probabilities;
- a bar chart: those probabilities, so the shape of the distribution
  (peaked vs. flat) is obvious at a glance.

A good first module for learning: it connects a hidden state to a concrete,
human-readable prediction.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import torch

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "next_token_predictions",
    "version": "0.1.0",
    "description": "Top-k next-token distribution at the attached layer - entropy scalar + token table + probability bar chart",
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
    "top_k": 10,
    "position": -1,
    "apply_final_norm": True,
    "emit_mode": "final",
    "emit_every_n": 1,
}

_SENDER = None


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    logger.info("NEXT_TOKEN_PREDICTIONS: initialized")
    return {"status": "initialized"}


def cleanup_module():
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {"name": "top_k", "type": "number", "label": "Top K tokens",
             "default": 10, "min": 1, "max": 30},
            {"name": "position", "type": "number", "label": "Sequence position (-1 = last)",
             "default": -1, "min": -1},
            {"name": "apply_final_norm", "type": "boolean",
             "label": "Apply final LayerNorm before unembedding", "default": True},
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
    try:
        cfg["top_k"] = max(1, min(30, int(cfg.get("top_k", 10))))
    except (TypeError, ValueError):
        cfg["top_k"] = 10
    try:
        cfg["position"] = int(cfg.get("position", -1))
    except (TypeError, ValueError):
        cfg["position"] = -1
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
        candidate = getattr(candidate, "original_model", None) or getattr(candidate, "model", None)
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
    for parent_name, norm_name in [
        ("model", "norm"), ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"), ("model", "final_layernorm"),
    ]:
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
            return repr(tokenizer.decode([token_id]))
        except Exception:
            pass
    return f"#{token_id}"


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

        with torch.no_grad():
            state = hidden[0, cfg["position"], :].float()
            if cfg.get("apply_final_norm", True):
                norm = _get_final_norm(hf_model)
                if norm is not None:
                    nd = next(norm.parameters(), state).dtype
                    state = norm(state.to(nd)).float()
            wd = next(unembed.parameters(), state).dtype
            logits = unembed(state.to(wd)).float()
            probs = torch.softmax(logits, dim=-1)
            k = min(cfg["top_k"], probs.shape[-1])
            top = torch.topk(probs, k=k)
            # entropy in bits over the full distribution
            p = probs.clamp_min(1e-12)
            entropy_bits = float(-(p * p.log2()).sum().item())

        ids = [int(i) for i in top.indices.tolist()]
        vals = [float(v) for v in top.values.tolist()]
        labels = [_decode(ltts_model, i) for i in ids]

        base = {"layer_path": layer_path, "emit_mode": cfg["emit_mode"]}
        emitted = []

        if _SENDER:
            _SENDER.send_scalar(
                round(entropy_bits, 4),
                label="Next-token entropy (bits)",
                emit_id=f"next_token_predictions:{layer_path}:entropy",
                **base,
            )
            emitted.append("scalar")

            headers = ["Rank", "Token", "Probability"]
            rows = [[r + 1, labels[r], round(vals[r], 5)] for r in range(k)]
            _SENDER.send_table(
                headers, rows,
                emit_id=f"next_token_predictions:{layer_path}:table",
                **base,
            )
            emitted.append("table")

            points = [{"x": r + 1, "y": round(vals[r], 5)} for r in range(k)]
            _SENDER.send_chart(
                {
                    "series": [{"name": "Top-1 = " + labels[0] if labels else "probability",
                                "points": points}],
                    "x_label": "Rank",
                    "y_label": "Probability",
                },
                "bar",
                emit_id=f"next_token_predictions:{layer_path}:chart",
                **base,
            )
            emitted.append("chart")

        if cfg["emit_mode"] == "final":
            try:
                ltts_event.mark_emission_finalized(METADATA["name"], layer_path)
            except Exception:
                pass

        return {"status": "ok", "emitted": emitted, "top_token": labels[0] if labels else None}

    except Exception as exc:
        logger.error("NEXT_TOKEN_PREDICTIONS error: %s", exc, exc_info=True)
        try:
            ltts_event.reactor.signal(
                "error",
                {"message": f"next_token_predictions error: {exc}", "module": METADATA["name"]},
            )
        except Exception:
            pass
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
