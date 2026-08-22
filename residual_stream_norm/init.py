"""Residual-stream magnitude diagnostics across decoder depth."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import torch

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "residual_stream_norm",
    "version": "0.1.0",
    "description": "Track L2, RMS, maximum magnitude, and norm growth through decoder depth",
    "author": "LTTS",
    "event_types": ["layer_after"],
    "dependencies": ["requirements.txt"],
    "requires": ["hidden_states"],
    "methods": {
        "ctor": "initialize_module",
        "dtor": "cleanup_module",
        "ntor": "process_event",
        "utor": "get_ui_schema",
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "position": -1,
    "emit_scalar": True,
    "emit_table": True,
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
                "name": "position",
                "type": "number",
                "label": "Sequence position (-1 = last)",
                "default": -1,
                "min": -1,
            },
            {
                "name": "emit_scalar",
                "type": "boolean",
                "label": "Emit L2 norm scalar",
                "default": True,
            },
            {
                "name": "emit_table",
                "type": "boolean",
                "label": "Emit metric table",
                "default": True,
            },
            {
                "name": "emit_chart",
                "type": "boolean",
                "label": "Emit norm trajectory",
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


def _is_decoder_block(context: Any) -> bool:
    if getattr(context, "module_role", None) == "block":
        return True
    module_class = str(getattr(context, "module_class", "") or "").lower()
    return getattr(context, "layer_type", None) == "unknown" and (
        "decoderlayer" in module_class or module_class.endswith(("block", "layer"))
    )


def _config(event) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if isinstance(event.module_state, dict):
        config.update(event.module_state)
    try:
        config["position"] = int(config.get("position", -1))
    except (TypeError, ValueError):
        config["position"] = -1
    return config


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

        config = _config(ltts_event)
        layer_path = getattr(context, "module_path", None)
        emission = ltts_event.should_emit(METADATA["name"], layer_path, config)
        if not emission.get("emit"):
            return {"status": "skipped", "reason": "emission_controller"}

        state = hidden[0, config["position"], :].detach().float()
        l2_norm = float(torch.linalg.vector_norm(state).item())
        rms = float(torch.sqrt(torch.mean(state.square())).item())
        max_abs = float(state.abs().max().item())

        current_pass = emission.get("forward_pass")
        if _TRAJECTORY["forward_pass"] != current_pass:
            _TRAJECTORY["forward_pass"] = current_pass
            _TRAJECTORY["entries"] = []
        previous = _TRAJECTORY["entries"][-1]["l2_norm"] if _TRAJECTORY["entries"] else None
        growth = l2_norm / previous if previous and previous > 0 else None
        entry = {
            "layer": layer_path or "unknown",
            "l2_norm": l2_norm,
            "rms": rms,
            "max_abs": max_abs,
            "growth_ratio": growth,
        }
        _TRAJECTORY["entries"].append(entry)

        base = {"layer_path": layer_path, "emit_mode": config.get("emit_mode", "final")}
        emitted = []
        if _SENDER and config.get("emit_scalar", True):
            _SENDER.send_scalar(
                round(l2_norm, 6),
                label="Residual-stream L2 norm",
                emit_id=f"residual_stream_norm:{layer_path}:scalar",
                **base,
            )
            emitted.append("scalar")
        if _SENDER and config.get("emit_table", True):
            _SENDER.send_table(
                ["Metric", "Value"],
                [
                    ["L2 norm", round(l2_norm, 6)],
                    ["RMS", round(rms, 6)],
                    ["Max abs", round(max_abs, 6)],
                    ["Growth ratio", round(growth, 6) if growth is not None else None],
                ],
                emit_id=f"residual_stream_norm:{layer_path}:table",
                **base,
            )
            emitted.append("table")
        if _SENDER and config.get("emit_chart", True) and len(_TRAJECTORY["entries"]) > 1:
            entries = _TRAJECTORY["entries"]
            _SENDER.send_chart(
                {
                    "series": [
                        {
                            "name": "L2 norm",
                            "points": [
                                {"x": index, "y": round(item["l2_norm"], 6)}
                                for index, item in enumerate(entries)
                            ],
                        },
                        {
                            "name": "RMS",
                            "points": [
                                {"x": index, "y": round(item["rms"], 6)}
                                for index, item in enumerate(entries)
                            ],
                        },
                    ],
                    "x_label": "Layer index",
                    "y_label": "Magnitude",
                },
                "line",
                emit_id=f"residual_stream_norm:{layer_path}:chart",
                **base,
            )
            emitted.append("chart")
        return {"status": "ok", "emitted": emitted, **entry}
    except Exception as exc:
        logger.error("RESIDUAL_STREAM_NORM error: %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
