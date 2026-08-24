"""Locate token representations that keep changing late in model depth.

This is a diagnostic companion to recirculation, not a recirculation executor.
Context: Mozer et al., "Recirculation" (2026),
https://arxiv.org/abs/2608.17981
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import torch

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "state_tracking_diagnostics",
    "version": "0.1.0",
    "description": "Find tokens whose residual representations update late in model depth, with convergence and delayed-contextualization diagnostics",
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
    "max_positions": 64,
    "top_tokens": 8,
    "emit_heatmap": True,
    "emit_chart": True,
    "emit_table": True,
    "emit_mode": "final",
}

_SENDER = None
_TRACE: Dict[str, Any] = {
    "forward_pass": None,
    "layers": [],
    "previous": None,
    "updates": [],
    "chart": [],
}


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    return {"status": "initialized", "config": config}


def cleanup_module():
    _reset(None)
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "max_positions",
                "type": "number",
                "label": "Maximum token positions",
                "default": 64,
                "min": 2,
                "max": 512,
            },
            {
                "name": "top_tokens",
                "type": "number",
                "label": "Tokens in delayed-update table",
                "default": 8,
                "min": 1,
                "max": 64,
            },
            {
                "name": "emit_heatmap",
                "type": "boolean",
                "label": "Emit token-by-depth update heatmap",
                "default": True,
            },
            {
                "name": "emit_chart",
                "type": "boolean",
                "label": "Emit depth summary chart",
                "default": True,
            },
            {
                "name": "emit_table",
                "type": "boolean",
                "label": "Emit delayed-token table",
                "default": True,
            },
        ],
        "layout": {"type": "vertical"},
        "real_time": {"enabled": True, "interval": 200},
    }


def _reset(forward_pass):
    _TRACE.update(
        forward_pass=forward_pass,
        layers=[],
        previous=None,
        updates=[],
        chart=[],
    )


def _integer(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _boolean(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "1", "yes", "on"}:
            return True
        if value.lower() in {"false", "0", "no", "off"}:
            return False
    return default


def _config(event) -> Dict[str, Any]:
    values = dict(DEFAULT_CONFIG)
    if isinstance(event.module_state, dict):
        values.update(event.module_state)
    environment = getattr(event, "environment", None)
    supplied = environment.get("module_config") if isinstance(environment, dict) else None
    if isinstance(supplied, dict):
        values.update(supplied)
    return {
        "max_positions": _integer(values.get("max_positions"), 64, 2, 512),
        "top_tokens": _integer(values.get("top_tokens"), 8, 1, 64),
        "emit_heatmap": _boolean(values.get("emit_heatmap"), True),
        "emit_chart": _boolean(values.get("emit_chart"), True),
        "emit_table": _boolean(values.get("emit_table"), True),
        "emit_mode": str(values.get("emit_mode", "final")),
        "emit_every_n": values.get("emit_every_n", 1),
    }


def _hidden(outputs: Any) -> Optional[torch.Tensor]:
    if isinstance(outputs, (tuple, list)) and outputs:
        outputs = outputs[0]
    return outputs if isinstance(outputs, torch.Tensor) and outputs.ndim == 3 else None


def _is_block(context: Any) -> bool:
    if getattr(context, "module_role", None) == "block":
        return True
    name = str(getattr(context, "module_class", "")).lower()
    return getattr(context, "layer_type", None) == "unknown" and (
        "decoderlayer" in name or name.endswith(("block", "layer"))
    )


def _labels(event: Any, positions: int, sequence_length: int):
    start = sequence_length - positions
    fallback = [str(index) for index in range(start, sequence_length)]
    environment = getattr(event, "environment", None)
    ids = environment.get("input_ids") if isinstance(environment, dict) else None
    tokenizer = environment.get("tokenizer") if isinstance(environment, dict) else None
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or not callable(convert):
        return fallback
    try:
        labels = [str(item) for item in convert([int(x) for x in ids[0, -positions:]])]
    except Exception:
        return fallback
    return labels if len(labels) == positions else fallback


def _token_summary(updates: torch.Tensor):
    split = max(1, updates.shape[0] // 2)
    energy = updates.square()
    total = energy.sum(dim=0)
    late = energy[split:].sum(dim=0) if split < len(energy) else torch.zeros_like(total)
    late_share = torch.where(total > 1e-12, late / total, torch.zeros_like(total))
    peak_value, peak_transition = updates.max(dim=0)
    return total, late_share, peak_value, peak_transition


def process_event(event):
    try:
        if not str(getattr(event, "event_type", "")).startswith("layer_after"):
            return {"status": "skipped", "reason": "event_phase"}
        context = event.context
        if not _is_block(context):
            return {"status": "skipped", "reason": "not_a_block_boundary"}
        hidden = _hidden(context.outputs)
        if hidden is None:
            return {"status": "skipped", "reason": "no_hidden_state"}
        config = _config(event)
        layer_path = getattr(context, "module_path", None) or "unknown"
        emission = event.should_emit(METADATA["name"], layer_path, config)
        if not emission.get("emit"):
            return {"status": "skipped", "reason": "emission_controller"}

        positions = min(hidden.shape[1], config["max_positions"])
        current = hidden[0, -positions:, :].detach().float().cpu()
        forward_pass = emission.get("forward_pass")
        if _TRACE["forward_pass"] != forward_pass:
            _reset(forward_pass)
        previous = _TRACE["previous"]
        _TRACE["layers"].append(layer_path)
        _TRACE["previous"] = current
        if not isinstance(previous, torch.Tensor) or previous.shape != current.shape:
            return {"status": "ok", "emitted": [], "layers_accumulated": 1}

        previous_norm = torch.linalg.vector_norm(previous, dim=-1).clamp_min(1e-12)
        update = torch.linalg.vector_norm(current - previous, dim=-1) / previous_norm
        cosine = torch.nn.functional.cosine_similarity(previous, current, dim=-1)
        update = torch.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
        cosine = torch.nan_to_num(cosine, nan=0.0, posinf=0.0, neginf=0.0)
        _TRACE["updates"].append(update)
        updates = torch.stack(_TRACE["updates"])
        total, late_share, peak_value, peak_transition = _token_summary(updates)
        labels = _labels(event, positions, hidden.shape[1])
        metrics = {
            "mean_relative_update": float(update.mean()),
            "mean_consecutive_cosine": float(cosine.mean()),
            "mean_late_update_share": float(late_share.mean()),
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("state-tracking metrics are not finite")
        _TRACE["chart"].append(metrics)

        emitted = []
        base = {"layer_path": layer_path, "emit_mode": config["emit_mode"]}
        if _SENDER and config["emit_heatmap"]:
            _SENDER.send_heatmap(
                updates.T.round(decimals=6).tolist(),
                dimensions={"rows": positions, "cols": updates.shape[0]},
                row_labels=labels,
                column_labels=_TRACE["layers"][1:],
                color_scale="sequential",
                metric="relative_residual_update",
                emit_id="state_tracking_diagnostics:heatmap",
                **base,
            )
            emitted.append("heatmap")
        if _SENDER and config["emit_chart"]:
            _SENDER.send_chart(
                {
                    "series": [
                        {
                            "name": "Relative update",
                            "points": [
                                {"x": i + 1, "y": round(item["mean_relative_update"], 6)}
                                for i, item in enumerate(_TRACE["chart"])
                            ],
                        },
                        {
                            "name": "Consecutive cosine",
                            "points": [
                                {"x": i + 1, "y": round(item["mean_consecutive_cosine"], 6)}
                                for i, item in enumerate(_TRACE["chart"])
                            ],
                        },
                    ],
                    "x_label": "Layer transition",
                    "y_label": "Score",
                },
                "line",
                emit_id="state_tracking_diagnostics:chart",
                **base,
            )
            emitted.append("chart")
        if _SENDER and config["emit_table"]:
            ranked = sorted(
                range(positions),
                key=lambda i: (-float(late_share[i]), -float(total[i]), i),
            )[: config["top_tokens"]]
            rows = [
                [
                    i,
                    labels[i],
                    int(peak_transition[i]) + 1,
                    round(float(peak_value[i]), 6),
                    round(float(late_share[i]), 6),
                    round(float(total[i]), 6),
                ]
                for i in ranked
            ]
            _SENDER.send_table(
                ["Position", "Token", "Peak transition", "Peak update", "Late share", "Update energy"],
                rows,
                emit_id="state_tracking_diagnostics:table",
                **base,
            )
            emitted.append("table")
        return {"status": "ok", "emitted": emitted, "layers_accumulated": len(_TRACE["layers"]), **metrics}
    except Exception as exc:
        logger.error("STATE_TRACKING_DIAGNOSTICS error: %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc)}


def interceptor(event):
    return process_event(event)
