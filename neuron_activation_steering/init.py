"""Causal neuron steering and clamping during inference."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "neuron_activation_steering",
    "version": "0.1.0",
    "description": "Add, scale, set, or clamp selected activation channels during inference",
    "author": "LTTS",
    "event_types": ["layer_after"],
    "dependencies": ["requirements.txt"],
    "requires": ["hidden_states"],
    "target_roles": ["block", "mlp", "activation"],
    "methods": {
        "ctor": "initialize_module",
        "dtor": "cleanup_module",
        "ntor": "process_event",
        "utor": "get_ui_schema",
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "neuron_indices": "",
    "operation": "add",
    "value": 1.0,
    "clamp_min": -1.0,
    "clamp_max": 1.0,
    "positions": "all",
    "emit_summary": True,
    "emit_mode": "all",
}

_SENDER = None


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    return {"status": "initialized", "config": config}


def cleanup_module():
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "enabled",
                "type": "boolean",
                "label": "Enable steering",
                "default": False,
            },
            {
                "name": "neuron_indices",
                "type": "text",
                "label": "Neuron indices (for example 0,5,10-20)",
                "default": "",
            },
            {
                "name": "operation",
                "type": "select",
                "label": "Operation",
                "default": "add",
                "options": [
                    {"value": "add", "label": "Add"},
                    {"value": "multiply", "label": "Multiply"},
                    {"value": "set", "label": "Set"},
                    {"value": "clamp", "label": "Clamp"},
                ],
            },
            {
                "name": "value",
                "type": "number",
                "label": "Value",
                "default": 1.0,
                "step": 0.1,
            },
            {
                "name": "clamp_min",
                "type": "number",
                "label": "Clamp minimum",
                "default": -1.0,
                "step": 0.1,
            },
            {
                "name": "clamp_max",
                "type": "number",
                "label": "Clamp maximum",
                "default": 1.0,
                "step": 0.1,
            },
            {
                "name": "positions",
                "type": "select",
                "label": "Token positions",
                "default": "all",
                "options": [
                    {"value": "all", "label": "All positions"},
                    {"value": "last", "label": "Last position"},
                ],
            },
            {
                "name": "emit_summary",
                "type": "boolean",
                "label": "Emit intervention metrics",
                "default": True,
            },
        ],
        "layout": {"type": "vertical"},
        "real_time": {"enabled": True, "interval": 200},
    }


def _hidden(outputs: Any) -> Optional[torch.Tensor]:
    candidate = (
        outputs[0] if isinstance(outputs, (tuple, list)) and outputs else outputs
    )
    if isinstance(candidate, torch.Tensor) and candidate.dim() == 3:
        return candidate
    return None


def _indices(spec: str, width: int) -> List[int]:
    values = set()
    for raw in str(spec or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            if "-" in raw:
                start, end = (int(item.strip()) for item in raw.split("-", 1))
                start, end = min(start, end), max(start, end)
                values.update(range(start, end + 1))
            else:
                values.add(int(raw))
        except ValueError:
            logger.warning(
                "NEURON_ACTIVATION_STEERING: ignored malformed index %r", raw
            )
    return sorted(item for item in values if 0 <= item < width)


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in {"1", "true", "yes", "on"}:
            return True
        if value.strip().lower() in {"0", "false", "no", "off"}:
            return False
    return default


def _config(event) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if isinstance(event.module_state, dict):
        config.update(event.module_state)
    environment = getattr(event, "environment", None)
    if isinstance(environment, dict) and isinstance(
        environment.get("module_config"), dict
    ):
        config.update(environment["module_config"])
    config["enabled"] = _bool(config.get("enabled"), False)
    config["emit_summary"] = _bool(config.get("emit_summary"), True)
    config["operation"] = str(config.get("operation", "add")).lower()
    config["positions"] = str(config.get("positions", "all")).lower()
    for key, default in (
        ("value", 1.0),
        ("clamp_min", -1.0),
        ("clamp_max", 1.0),
    ):
        try:
            config[key] = float(config.get(key, default))
        except (TypeError, ValueError):
            config[key] = default
    return config


def _replace(outputs: Any, hidden: torch.Tensor) -> Any:
    if isinstance(outputs, tuple):
        return (hidden,) + outputs[1:]
    if isinstance(outputs, list):
        return [hidden, *outputs[1:]]
    return hidden


def process_event(event):
    try:
        if not str(getattr(event, "event_type", "")).startswith("layer_after"):
            return {"status": "skipped", "reason": "event_phase"}
        config = _config(event)
        if not config["enabled"]:
            return {"status": "skipped", "reason": "disabled"}
        hidden = _hidden(event.context.outputs)
        if hidden is None:
            return {"status": "skipped", "reason": "no_hidden_state"}
        indices = _indices(config.get("neuron_indices", ""), hidden.shape[-1])
        if not indices:
            return {"status": "skipped", "reason": "no_valid_indices"}
        operation = config["operation"]
        if operation not in {"add", "multiply", "set", "clamp"}:
            return {
                "status": "error",
                "message": f"unsupported operation: {operation}",
            }

        modified = hidden.detach().clone()
        index = torch.tensor(indices, device=modified.device, dtype=torch.long)
        if config["positions"] == "all":
            selected = modified[:, :, index]
        else:
            selected = modified[:, -1:, index]
        before_mean = float(selected.float().mean().item())
        if operation == "add":
            changed = selected + config["value"]
        elif operation == "multiply":
            changed = selected * config["value"]
        elif operation == "set":
            changed = torch.full_like(selected, config["value"])
        else:
            low, high = sorted((config["clamp_min"], config["clamp_max"]))
            changed = selected.clamp(min=low, max=high)
        if config["positions"] == "all":
            modified[:, :, index] = changed
        else:
            modified[:, -1:, index] = changed
        delta_norm = float((modified - hidden.detach()).float().norm().item())
        after_mean = float(changed.float().mean().item())
        event.context.analysis_results["modified_outputs"] = _replace(
            event.context.outputs, modified
        )

        layer_path = getattr(event.context, "module_path", None) or "unknown"
        emitted = []
        if _SENDER and config["emit_summary"]:
            emission = event.should_emit(METADATA["name"], layer_path, config)
            if emission.get("emit"):
                _SENDER.send_table(
                    ["Metric", "Value"],
                    [
                        ["Operation", operation],
                        ["Neurons", len(indices)],
                        ["Positions", config["positions"]],
                        ["Mean before", round(before_mean, 6)],
                        ["Mean after", round(after_mean, 6)],
                        ["Delta L2 norm", round(delta_norm, 6)],
                    ],
                    layer_path=layer_path,
                    emit_id=f"neuron_activation_steering:{layer_path}:table",
                    emit_mode=config.get("emit_mode", "all"),
                )
                emitted.append("table")
        return {
            "status": "ok",
            "operation": operation,
            "neurons": len(indices),
            "positions": config["positions"],
            "mean_before": before_mean,
            "mean_after": after_mean,
            "delta_norm": delta_norm,
            "emitted": emitted,
        }
    except Exception as exc:
        logger.error("NEURON_ACTIVATION_STEERING error: %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc)}


def interceptor(event):
    return process_event(event)
