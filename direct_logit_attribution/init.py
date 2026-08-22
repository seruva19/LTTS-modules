"""
Direct Logit Attribution Module

Measures how much each decoder block's residual-stream update pushes the
logit of the currently predicted token. At every decoder-block boundary the
hidden state at the last position is diffed against the previous block's
hidden state (delta = h_l - h_{l-1}); the first block uses h_l itself and is
labeled "embedding+block0". Each delta is projected through the final norm
(optional) and dotted with the unembedding row of the top-1 token of the
latest hidden state — a single weight-row dot product per layer, never the
full vocabulary logits per delta.

Because the final predicted token is unknown until the last layer, deltas are
accumulated per forward pass and the full attribution is re-emitted at each
layer against the CURRENT top-1 token, using a stable emit_id so the UI
updates in place.

Emits:
- a bar chart: per-layer logit contribution to the current top-1 token
- optionally a table of the same numbers with layer paths
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "direct_logit_attribution",
    "version": "0.1.0",
    "description": "Direct logit attribution: per-decoder-block residual update contribution to the current top-1 token's logit",
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
    "apply_final_norm": True,
    "emit_chart": True,
    "emit_table": False,
    "emit_mode": "final",
}

_SENDER = None

# Cross-layer accumulator: each layer event gets a fresh LTTSContext, so the
# per-block deltas and the previous block's hidden state must live in module
# state, keyed by forward pass.
_DLA: Dict[str, Any] = {"forward_pass": None, "prev_hidden": None, "entries": []}


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    logger.info("DIRECT_LOGIT_ATTRIBUTION: initialized with config: %s", config)
    return {"status": "initialized", "config": config}


def cleanup_module():
    _DLA["forward_pass"] = None
    _DLA["prev_hidden"] = None
    _DLA["entries"] = []
    logger.info("DIRECT_LOGIT_ATTRIBUTION: cleaned up")
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "apply_final_norm",
                "type": "boolean",
                "label": "Apply final LayerNorm before unembedding",
                "default": DEFAULT_CONFIG["apply_final_norm"],
            },
            {
                "name": "emit_chart",
                "type": "boolean",
                "label": "Emit per-layer attribution bar chart",
                "default": DEFAULT_CONFIG["emit_chart"],
            },
            {
                "name": "emit_table",
                "type": "boolean",
                "label": "Emit per-layer attribution table",
                "default": DEFAULT_CONFIG["emit_table"],
            },
        ],
        "layout": {"type": "vertical"},
        "real_time": {"enabled": True, "interval": 200},
    }


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def _resolve_config(ltts_event) -> Dict[str, Any]:
    config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    if isinstance(ltts_event.module_state, dict):
        config.update(ltts_event.module_state)
    env = getattr(ltts_event, "environment", None)
    env_config = env.get("module_config") if isinstance(env, dict) else None
    if isinstance(env_config, dict):
        config.update(env_config)

    return {
        "apply_final_norm": _coerce_bool(
            config.get("apply_final_norm"), DEFAULT_CONFIG["apply_final_norm"]
        ),
        "emit_chart": _coerce_bool(config.get("emit_chart"), DEFAULT_CONFIG["emit_chart"]),
        "emit_table": _coerce_bool(config.get("emit_table"), DEFAULT_CONFIG["emit_table"]),
        "emit_mode": str(config.get("emit_mode", DEFAULT_CONFIG["emit_mode"])),
        "emit_every_n": config.get("emit_every_n", 1),
    }


def _extract_hidden_state(outputs: Any) -> Optional[torch.Tensor]:
    """Layer outputs may be a tensor or a tuple whose first element is the hidden state."""
    candidate = outputs
    if isinstance(candidate, (list, tuple)) and len(candidate) > 0:
        candidate = candidate[0]
    if isinstance(candidate, torch.Tensor) and candidate.dim() == 3:
        return candidate
    return None


def _get_hf_model(ltts_model) -> Any:
    """Resolve the underlying HuggingFace causal-LM, unwrapping nested LTTS
    wrappers (a model can be wrapped more than once)."""
    seen = set()
    candidate = ltts_model
    for _ in range(8):
        if candidate is None or id(candidate) in seen:
            break
        seen.add(id(candidate))
        if hasattr(candidate, "get_output_embeddings") or hasattr(candidate, "lm_head"):
            return candidate
        nxt = getattr(candidate, "original_model", None)
        if nxt is None:
            nxt = getattr(candidate, "model", None)
        candidate = nxt
    return candidate


def _get_unembedding(hf_model) -> Optional[torch.nn.Module]:
    try:
        head = hf_model.get_output_embeddings()
        if head is not None:
            return head
    except Exception:
        pass
    return getattr(hf_model, "lm_head", None)


def _get_final_norm(hf_model) -> Optional[torch.nn.Module]:
    """Find the final normalization applied before the unembedding, across common architectures."""
    candidates = [
        ("model", "norm"),          # llama, qwen, gemma, smollm
        ("transformer", "ln_f"),    # gpt2
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


def _decode_token(ltts_model, token_id: int) -> str:
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
        # Attribute residual-stream updates only at decoder-block boundaries:
        # attention or MLP sublayer outputs are partial residual updates.
        # Blocks are typed "unknown" by the layer classifier, so confirm via
        # the class name.
        module_class = (context.module_class or "").lower()
        is_block = "decoderlayer" in module_class or module_class.endswith(
            ("block", "layer")
        )
        if getattr(context, "module_role", None) != "block" and (
            context.layer_type != "unknown" or not is_block
        ):
            return {"status": "skipped", "reason": "not_a_block_boundary"}

        hidden = _extract_hidden_state(context.outputs)
        if hidden is None:
            return {"status": "skipped", "reason": "no_hidden_state"}

        ltts_model = context.ltts_model
        if ltts_model is None:
            return {"status": "skipped", "reason": "no_model_reference"}
        hf_model = _get_hf_model(ltts_model)
        if hf_model is None:
            return {"status": "skipped", "reason": "no_hf_model"}
        unembed = _get_unembedding(hf_model)
        if unembed is None:
            return {"status": "skipped", "reason": "no_unembedding"}
        weight = getattr(unembed, "weight", None)
        if not isinstance(weight, torch.Tensor):
            return {"status": "skipped", "reason": "no_unembedding_weight"}

        if not isinstance(ltts_event.module_state, dict):
            ltts_event.module_state = {}
        config = _resolve_config(ltts_event)
        layer_path = getattr(context, "module_path", None)

        emission = ltts_event.should_emit(METADATA["name"], layer_path, config)

        # Accumulate before the emission gate so the delta chain stays
        # complete even when individual layer emissions are suppressed.
        current_pass = emission.get("forward_pass")
        if _DLA["forward_pass"] != current_pass:
            _DLA["forward_pass"] = current_pass
            _DLA["prev_hidden"] = None
            _DLA["entries"] = []
        entries: List[Dict[str, Any]] = _DLA["entries"]

        with torch.no_grad():
            current_hidden = hidden[0, -1, :].detach().float().cpu()

            # Generic and layer-specific event variants may both fire for the
            # same layer in one pass; record the delta only once.
            already_recorded = any(
                entry["layer"] == (layer_path or "unknown") for entry in entries
            )
            if not already_recorded:
                prev_hidden = _DLA["prev_hidden"]
                if prev_hidden is None:
                    delta = current_hidden
                    label = "embedding+block0"
                else:
                    delta = current_hidden - prev_hidden
                    label = f"block{len(entries)}"
                entries.append(
                    {
                        "layer": layer_path or "unknown",
                        "label": label,
                        "delta": delta,
                    }
                )
                _DLA["prev_hidden"] = current_hidden

            if not emission.get("emit"):
                return {"status": "skipped", "reason": "emission_controller"}

            norm = _get_final_norm(hf_model) if config["apply_final_norm"] else None
            device = weight.device

            def _project(vec: torch.Tensor) -> torch.Tensor:
                v = vec.to(device)
                if norm is not None:
                    norm_dtype = next(norm.parameters(), v).dtype
                    v = norm(v.to(norm_dtype))
                return v

            # Current top-1 token of the latest hidden state: a single matvec
            # against the unembedding weight (no per-delta full logits).
            projected_state = _project(current_hidden)
            logits = torch.matmul(weight, projected_state.to(weight.dtype)).float()
            top_id = int(torch.argmax(logits).item())
            token_row = weight[top_id].detach().float()

            contributions = []
            for entry in entries:
                projected_delta = _project(entry["delta"]).float()
                contributions.append(
                    float(torch.dot(token_row, projected_delta).item())
                )

        top_label = _decode_token(ltts_model, top_id)

        emitted = []
        base_kwargs = {"layer_path": layer_path, "emit_mode": config["emit_mode"]}

        if _SENDER and config["emit_chart"]:
            points = [
                {"x": idx, "y": round(value, 5)}
                for idx, value in enumerate(contributions)
            ]
            _SENDER.send_chart(
                {
                    "series": [
                        {
                            "name": f"Logit contribution to {top_label}",
                            "points": points,
                        }
                    ],
                    "x_label": "Layer index",
                    "y_label": "Delta logit",
                },
                "bar",
                # Stable emit_id so the UI updates the same chart in place as
                # later layers re-emit the growing attribution.
                emit_id="direct_logit_attribution:attribution_chart",
                **base_kwargs,
            )
            emitted.append("chart")

        if _SENDER and config["emit_table"]:
            headers = ["Layer", "Path", "Label", "Delta logit"]
            rows = [
                [idx, entry["layer"], entry["label"], round(value, 5)]
                for idx, (entry, value) in enumerate(zip(entries, contributions))
            ]
            _SENDER.send_table(
                headers,
                rows,
                emit_id="direct_logit_attribution:attribution_table",
                **base_kwargs,
            )
            emitted.append("table")

        if config["emit_mode"] == "final":
            try:
                ltts_event.mark_emission_finalized(METADATA["name"], layer_path)
            except Exception:
                logger.debug(
                    "DIRECT_LOGIT_ATTRIBUTION: mark_emission_finalized failed",
                    exc_info=True,
                )

        return {
            "status": "ok",
            "emitted": emitted,
            "top_token": top_label,
            "layers_accumulated": len(entries),
        }

    except Exception as exc:
        logger.error("DIRECT_LOGIT_ATTRIBUTION: error: %s", exc, exc_info=True)
        try:
            ltts_event.reactor.signal(
                "error",
                {
                    "message": f"direct_logit_attribution error: {exc}",
                    "module": METADATA["name"],
                },
            )
        except Exception:
            pass
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
