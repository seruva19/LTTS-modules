"""
Activation Trajectory Module

Tracks how the residual-stream representation moves through depth. At every
decoder-block boundary it extracts the hidden state at a chosen sequence
position, then measures, layer over layer:

- cosine similarity between consecutive layers (how much direction changes)
- L2 norm of the hidden state (how much magnitude grows)

Emits per layer:
- optionally a scalar: cosine similarity with the previous layer
- optionally (once >= 2 layers are accumulated) a line chart with two series:
  cosine similarity to the previous layer and hidden-state L2 norm vs depth
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "activation_trajectory",
    "version": "0.1.0",
    "description": "Track residual-stream movement through depth: per-layer cosine similarity to the previous layer and hidden-state L2 norm",
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
    "position": -1,  # sequence position to track; -1 = last token
    "emit_scalar": True,
    "emit_chart": True,
    "emit_mode": "final",
}

_SENDER = None

# Cross-layer accumulator: each layer event gets a fresh LTTSContext, so the
# per-depth trajectory must live in module state, keyed by forward pass.
_TRAJECTORY: Dict[str, Any] = {"forward_pass": None, "entries": [], "last_vector": None}


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    logger.info("ACTIVATION_TRAJECTORY: initialized with config: %s", config)
    return {"status": "initialized", "config": config}


def cleanup_module():
    logger.info("ACTIVATION_TRAJECTORY: cleaned up")
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "position",
                "type": "number",
                "label": "Sequence position (-1 = last)",
                "default": DEFAULT_CONFIG["position"],
                "min": -1,
            },
            {
                "name": "emit_scalar",
                "type": "boolean",
                "label": "Emit per-layer cosine-similarity scalar",
                "default": DEFAULT_CONFIG["emit_scalar"],
            },
            {
                "name": "emit_chart",
                "type": "boolean",
                "label": "Emit trajectory chart (cosine sim + L2 norm vs depth)",
                "default": DEFAULT_CONFIG["emit_chart"],
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
        "position": _coerce_int(config.get("position"), -1, -1),
        "emit_scalar": _coerce_bool(
            config.get("emit_scalar"), DEFAULT_CONFIG["emit_scalar"]
        ),
        "emit_chart": _coerce_bool(
            config.get("emit_chart"), DEFAULT_CONFIG["emit_chart"]
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


def _layer_depth_key(layer_path: str) -> str:
    return layer_path or "unknown"


def process_event(ltts_event):
    try:
        evt = getattr(ltts_event, "event_type", "") or ""
        if not (evt == "layer_after" or evt.startswith("layer_after_")):
            return {"status": "skipped", "reason": "event_phase"}

        context = ltts_event.context
        # Track the residual stream only at decoder-block boundaries: attention
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

        if not isinstance(ltts_event.module_state, dict):
            ltts_event.module_state = {}
        config = _resolve_config(ltts_event)
        layer_path = getattr(context, "module_path", None)

        emission = ltts_event.should_emit(METADATA["name"], layer_path, config)
        if not emission.get("emit"):
            return {"status": "skipped", "reason": "emission_controller"}

        position = config["position"]
        with torch.no_grad():
            state = hidden[0, position, :].detach().float().cpu()
            l2_norm = float(torch.linalg.vector_norm(state).item())

        current_pass = emission.get("forward_pass")
        if _TRAJECTORY["forward_pass"] != current_pass:
            _TRAJECTORY["forward_pass"] = current_pass
            _TRAJECTORY["entries"] = []
            _TRAJECTORY["last_vector"] = None

        cosine_sim: Optional[float] = None
        previous = _TRAJECTORY["last_vector"]
        if isinstance(previous, torch.Tensor) and previous.shape == state.shape:
            denom = float(
                torch.linalg.vector_norm(previous).item()
            ) * l2_norm
            if denom > 0.0:
                cosine_sim = float(torch.dot(previous, state).item() / denom)

        _TRAJECTORY["entries"].append(
            {
                "layer": _layer_depth_key(layer_path),
                "l2_norm": l2_norm,
                "cosine_sim": cosine_sim,
            }
        )
        _TRAJECTORY["last_vector"] = state
        trajectory = _TRAJECTORY["entries"]

        emitted = []
        base_kwargs = {"layer_path": layer_path, "emit_mode": config["emit_mode"]}

        if _SENDER and config["emit_scalar"] and cosine_sim is not None:
            _SENDER.send_scalar(
                round(cosine_sim, 5),
                label="Cosine similarity to previous layer",
                emit_id=f"activation_trajectory:{layer_path}:scalar",
                **base_kwargs,
            )
            emitted.append("scalar")

        if _SENDER and config["emit_chart"] and len(trajectory) >= 2:
            cosine_points = [
                {"x": idx, "y": round(entry["cosine_sim"], 5)}
                for idx, entry in enumerate(trajectory)
                if entry["cosine_sim"] is not None
            ]
            norm_points = [
                {"x": idx, "y": round(entry["l2_norm"], 5)}
                for idx, entry in enumerate(trajectory)
            ]
            _SENDER.send_chart(
                {
                    "series": [
                        {"name": "Cosine sim to previous layer", "points": cosine_points},
                        {"name": "Hidden state L2 norm", "points": norm_points},
                    ],
                    "x_label": "Layer index",
                    "y_label": "Value",
                },
                "line",
                emit_id=f"activation_trajectory:{layer_path}:chart",
                **base_kwargs,
            )
            emitted.append("chart")

        if config["emit_mode"] == "final":
            try:
                ltts_event.mark_emission_finalized(METADATA["name"], layer_path)
            except Exception:
                logger.debug(
                    "ACTIVATION_TRAJECTORY: mark_emission_finalized failed",
                    exc_info=True,
                )

        return {
            "status": "ok",
            "emitted": emitted,
            "layers_accumulated": len(trajectory),
            "cosine_sim": cosine_sim,
            "l2_norm": l2_norm,
        }

    except Exception as exc:
        logger.error("ACTIVATION_TRAJECTORY: error: %s", exc, exc_info=True)
        try:
            ltts_event.reactor.signal(
                "error",
                {
                    "message": f"activation_trajectory error: {exc}",
                    "module": METADATA["name"],
                },
            )
        except Exception:
            pass
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
