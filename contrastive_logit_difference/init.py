"""Track a positive-vs-negative token logit difference through depth."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "contrastive_logit_difference",
    "version": "0.1.0",
    "description": "Track the logit difference between two candidate tokens through decoder depth",
    "author": "LTTS",
    "event_types": ["layer_after"],
    "dependencies": ["requirements.txt"],
    "requires": ["hidden_states", "tokenizer_decoding", "unembedding"],
    "methods": {
        "ctor": "initialize_module",
        "dtor": "cleanup_module",
        "ntor": "process_event",
        "utor": "get_ui_schema",
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "positive_token": "",
    "negative_token": "",
    "position": -1,
    "apply_final_norm": True,
    "emit_chart": True,
    "emit_mode": "final",
}

_SENDER = None
_TRAJECTORY: Dict[str, Any] = {"forward_pass": None, "entries": []}


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    return {"status": "initialized", "config": config}


def cleanup_module():
    _TRAJECTORY["entries"] = []
    _TRAJECTORY["forward_pass"] = None
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "positive_token",
                "type": "text",
                "label": "Positive candidate token (blank = auto)",
                "default": "",
            },
            {
                "name": "negative_token",
                "type": "text",
                "label": "Negative candidate token (blank = auto)",
                "default": "",
            },
            {
                "name": "position",
                "type": "number",
                "label": "Sequence position (-1 = last)",
                "default": -1,
                "min": -1,
            },
            {
                "name": "apply_final_norm",
                "type": "boolean",
                "label": "Apply final LayerNorm",
                "default": True,
            },
            {
                "name": "emit_chart",
                "type": "boolean",
                "label": "Emit difference-vs-depth chart",
                "default": True,
            },
        ],
        "layout": {"type": "vertical"},
        "real_time": {"enabled": True, "interval": 200},
    }


def _extract_hidden(outputs: Any) -> Optional[torch.Tensor]:
    candidate = outputs[0] if isinstance(outputs, (list, tuple)) and outputs else outputs
    if isinstance(candidate, torch.Tensor) and candidate.dim() == 3:
        return candidate
    return None


def _unwrap(ltts_model: Any) -> Any:
    candidate = ltts_model
    seen = set()
    for _ in range(8):
        if candidate is None or id(candidate) in seen:
            break
        seen.add(id(candidate))
        if hasattr(candidate, "get_output_embeddings") or hasattr(candidate, "lm_head"):
            return candidate
        candidate = getattr(candidate, "original_model", None) or getattr(candidate, "model", None)
    return candidate


def _unembedding(model: Any):
    try:
        getter = getattr(model, "get_output_embeddings", None)
        if callable(getter):
            result = getter()
            if result is not None:
                return result
    except Exception:
        pass
    return getattr(model, "lm_head", None)


def _final_norm(model: Any):
    for parent_name, norm_name in (
        ("model", "norm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
        ("model", "final_layernorm"),
    ):
        parent = getattr(model, parent_name, None)
        norm = getattr(parent, norm_name, None) if parent is not None else None
        if isinstance(norm, torch.nn.Module):
            return norm
    return None


def _token_id(tokenizer: Any, text: str) -> Optional[int]:
    if not text:
        return None
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return int(ids[-1]) if ids else None
    except Exception:
        return None


def _decode(tokenizer: Any, token_id: int) -> str:
    try:
        return repr(tokenizer.decode([token_id]))
    except Exception:
        return f"#{token_id}"


def _is_decoder_block(context: Any) -> bool:
    if getattr(context, "module_role", None) == "block":
        return True
    module_class = str(getattr(context, "module_class", "") or "").lower()
    return getattr(context, "layer_type", None) == "unknown" and (
        "decoderlayer" in module_class or module_class.endswith(("block", "layer"))
    )


def process_event(ltts_event):
    try:
        if not str(getattr(ltts_event, "event_type", "")).startswith("layer_after"):
            return {"status": "skipped", "reason": "event_phase"}
        context = ltts_event.context
        if not _is_decoder_block(context):
            return {"status": "skipped", "reason": "not_a_block_boundary"}
        hidden = _extract_hidden(context.outputs)
        if hidden is None:
            return {"status": "skipped", "reason": "no_hidden_state"}

        config = dict(DEFAULT_CONFIG)
        if isinstance(ltts_event.module_state, dict):
            config.update(ltts_event.module_state)
        try:
            position = int(config.get("position", -1))
        except (TypeError, ValueError):
            position = -1

        ltts_model = context.ltts_model
        tokenizer = getattr(ltts_model, "_tokenizer", None)
        model = _unwrap(ltts_model)
        head = _unembedding(model)
        if tokenizer is None or head is None:
            return {"status": "skipped", "reason": "tokenizer_or_unembedding_missing"}

        layer_path = getattr(context, "module_path", None)
        emission = ltts_event.should_emit(METADATA["name"], layer_path, config)
        if not emission.get("emit"):
            return {"status": "skipped", "reason": "emission_controller"}

        with torch.no_grad():
            state = hidden[0, position, :].float()
            if config.get("apply_final_norm", True):
                norm = _final_norm(model)
                if norm is not None:
                    dtype = next(norm.parameters(), state).dtype
                    state = norm(state.to(dtype)).float()
            dtype = next(head.parameters(), state).dtype
            logits = head(state.to(dtype)).float()

        positive_id = _token_id(tokenizer, str(config.get("positive_token", "")))
        negative_id = _token_id(tokenizer, str(config.get("negative_token", "")))
        if positive_id is None or negative_id is None:
            top = torch.topk(logits, k=min(2, logits.shape[-1])).indices.tolist()
            if len(top) < 2:
                return {"status": "skipped", "reason": "vocabulary_too_small"}
            positive_id = positive_id if positive_id is not None else int(top[0])
            negative_id = negative_id if negative_id is not None else int(top[1])

        positive_logit = float(logits[positive_id].item())
        negative_logit = float(logits[negative_id].item())
        difference = positive_logit - negative_logit
        positive_label = _decode(tokenizer, positive_id)
        negative_label = _decode(tokenizer, negative_id)

        current_pass = emission.get("forward_pass")
        if _TRAJECTORY["forward_pass"] != current_pass:
            _TRAJECTORY["forward_pass"] = current_pass
            _TRAJECTORY["entries"] = []
        _TRAJECTORY["entries"].append(
            {"layer": layer_path or "unknown", "difference": difference}
        )

        base = {"layer_path": layer_path, "emit_mode": config.get("emit_mode", "final")}
        emitted = []
        if _SENDER:
            _SENDER.send_scalar(
                round(difference, 6),
                label=f"Logit difference {positive_label} − {negative_label}",
                emit_id=f"contrastive_logit_difference:{layer_path}:scalar",
                **base,
            )
            emitted.append("scalar")
            _SENDER.send_table(
                ["Candidate", "Token ID", "Logit"],
                [
                    [positive_label, positive_id, round(positive_logit, 6)],
                    [negative_label, negative_id, round(negative_logit, 6)],
                    ["Difference", None, round(difference, 6)],
                ],
                emit_id=f"contrastive_logit_difference:{layer_path}:table",
                **base,
            )
            emitted.append("table")
            if config.get("emit_chart", True) and len(_TRAJECTORY["entries"]) > 1:
                _SENDER.send_chart(
                    {
                        "series": [
                            {
                                "name": f"{positive_label} − {negative_label}",
                                "points": [
                                    {"x": index, "y": round(item["difference"], 6)}
                                    for index, item in enumerate(_TRAJECTORY["entries"])
                                ],
                            }
                        ],
                        "x_label": "Layer index",
                        "y_label": "Logit difference",
                    },
                    "line",
                    emit_id=f"contrastive_logit_difference:{layer_path}:chart",
                    **base,
                )
                emitted.append("chart")

        return {
            "status": "ok",
            "emitted": emitted,
            "positive_token": positive_label,
            "negative_token": negative_label,
            "difference": difference,
        }
    except Exception as exc:
        logger.error("CONTRASTIVE_LOGIT_DIFFERENCE error: %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
