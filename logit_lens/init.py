"""
Logit Lens Module

Decodes the residual stream at every transformer layer through the model's
unembedding matrix (lm_head), showing which tokens each layer "believes in"
before the final layer. The canonical first tool of mechanistic
interpretability (nostalgebraist, 2020).

Emits per layer:
- a table of top-k predicted tokens with probabilities for the last position
- optionally a chart of the final-prediction token's probability so far
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "logit_lens",
    "version": "0.1.0",
    "description": "Decode each layer's hidden state through the unembedding (logit lens): top-k token predictions per layer",
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
    "top_k": 5,
    "position": -1,  # sequence position to decode; -1 = last token
    "apply_final_norm": True,
    "emit_table": True,
    "emit_probability_chart": True,
    "emit_mode": "final",
}

_SENDER = None

# Cross-layer accumulator: each layer event gets a fresh LTTSContext, so the
# per-depth trajectory must live in module state, keyed by forward pass.
_TRAJECTORY: Dict[str, Any] = {"forward_pass": None, "entries": []}


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    logger.info("LOGIT_LENS: initialized with config: %s", config)
    return {"status": "initialized", "config": config}


def cleanup_module():
    logger.info("LOGIT_LENS: cleaned up")
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "top_k",
                "type": "number",
                "label": "Top K tokens",
                "default": DEFAULT_CONFIG["top_k"],
                "min": 1,
                "max": 20,
            },
            {
                "name": "position",
                "type": "number",
                "label": "Sequence position (-1 = last)",
                "default": DEFAULT_CONFIG["position"],
                "min": -1,
            },
            {
                "name": "apply_final_norm",
                "type": "boolean",
                "label": "Apply final LayerNorm before unembedding",
                "default": DEFAULT_CONFIG["apply_final_norm"],
            },
            {
                "name": "emit_table",
                "type": "boolean",
                "label": "Emit top-k token table",
                "default": DEFAULT_CONFIG["emit_table"],
            },
            {
                "name": "emit_probability_chart",
                "type": "boolean",
                "label": "Emit probability-vs-depth chart",
                "default": DEFAULT_CONFIG["emit_probability_chart"],
            },
        ],
        "layout": {"type": "vertical"},
        "real_time": {"enabled": True, "interval": 200},
    }


def _coerce_int(value: Any, default: int, minimum: int, maximum: Optional[int] = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    if result < minimum:
        result = minimum
    if maximum is not None and result > maximum:
        result = maximum
    return result


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
        "top_k": _coerce_int(config.get("top_k"), DEFAULT_CONFIG["top_k"], 1, 20),
        "position": _coerce_int(config.get("position"), -1, -1),
        "apply_final_norm": _coerce_bool(
            config.get("apply_final_norm"), DEFAULT_CONFIG["apply_final_norm"]
        ),
        "emit_table": _coerce_bool(config.get("emit_table"), True),
        "emit_probability_chart": _coerce_bool(
            config.get("emit_probability_chart"), True
        ),
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


def _decode_tokens(ltts_model, token_ids: List[int]) -> List[str]:
    tokenizer = getattr(ltts_model, "_tokenizer", None)
    labels = []
    for tid in token_ids:
        if tokenizer is not None:
            try:
                labels.append(repr(tokenizer.decode([tid])))
                continue
            except Exception:
                pass
        labels.append(f"#{tid}")
    return labels


def _layer_depth_key(layer_path: str) -> str:
    return layer_path or "unknown"


def process_event(ltts_event):
    try:
        evt = getattr(ltts_event, "event_type", "") or ""
        if not (evt == "layer_after" or evt.startswith("layer_after_")):
            return {"status": "skipped", "reason": "event_phase"}

        context = ltts_event.context
        # Decode the residual stream only at decoder-block boundaries: attention
        # or MLP sublayer outputs are partial residual updates. Blocks are typed
        # "unknown" by the layer classifier, so confirm via the class name.
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

        if not isinstance(ltts_event.module_state, dict):
            ltts_event.module_state = {}
        config = _resolve_config(ltts_event)
        layer_path = getattr(context, "module_path", None)

        emission = ltts_event.should_emit(METADATA["name"], layer_path, config)
        # Throttle emissions only: the trajectory accumulator below must see
        # every layer even when the controller defers the emission.
        should_emit_now = bool(emission.get("emit"))

        position = config["position"]
        with torch.no_grad():
            state = hidden[0, position, :].float()
            if config["apply_final_norm"]:
                norm = _get_final_norm(hf_model)
                if norm is not None:
                    norm_dtype = next(norm.parameters(), state).dtype
                    state = norm(state.to(norm_dtype)).float()
            weight_dtype = next(unembed.parameters(), state).dtype
            logits = unembed(state.to(weight_dtype)).float()
            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, k=min(config["top_k"], probs.shape[-1]))

        top_ids = [int(i) for i in top.indices.tolist()]
        top_probs = [float(p) for p in top.values.tolist()]
        labels = _decode_tokens(ltts_model, top_ids)

        emitted = []
        base_kwargs = {"layer_path": layer_path, "emit_mode": config["emit_mode"]}

        if _SENDER and should_emit_now and config["emit_table"]:
            headers = ["Rank", "Token", "Probability"]
            rows = [
                [rank + 1, label, round(prob, 5)]
                for rank, (label, prob) in enumerate(zip(labels, top_probs))
            ]
            _SENDER.send_table(
                headers,
                rows,
                emit_id=f"logit_lens:{layer_path}:table",
                **base_kwargs,
            )
            emitted.append("table")

        current_pass = emission.get("forward_pass")
        if _TRAJECTORY["forward_pass"] != current_pass:
            _TRAJECTORY["forward_pass"] = current_pass
            _TRAJECTORY["entries"] = []
        trajectory = _TRAJECTORY["entries"]
        trajectory.append(
            {
                "layer": _layer_depth_key(layer_path),
                "top_token": labels[0] if labels else "",
                "top_prob": top_probs[0] if top_probs else 0.0,
            }
        )

        if _SENDER and should_emit_now and config["emit_probability_chart"] and len(trajectory) > 1:
            points = [
                {"x": idx, "y": round(entry["top_prob"], 5)}
                for idx, entry in enumerate(trajectory)
            ]
            _SENDER.send_chart(
                {
                    "series": [{"name": "Top-1 probability", "points": points}],
                    "x_label": "Layer index",
                    "y_label": "Probability",
                },
                "line",
                emit_id=f"logit_lens:{layer_path}:chart",
                **base_kwargs,
            )
            emitted.append("chart")

        if should_emit_now and config["emit_mode"] == "final":
            try:
                ltts_event.mark_emission_finalized(METADATA["name"], layer_path)
            except Exception:
                logger.debug("LOGIT_LENS: mark_emission_finalized failed", exc_info=True)

        return {"status": "ok", "emitted": emitted, "top_token": labels[0] if labels else None}

    except Exception as exc:
        logger.error("LOGIT_LENS: error: %s", exc, exc_info=True)
        try:
            ltts_event.reactor.signal(
                "error",
                {"message": f"logit_lens error: {exc}", "module": METADATA["name"]},
            )
        except Exception:
            pass
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
