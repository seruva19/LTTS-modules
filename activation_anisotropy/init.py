"""Residual-stream anisotropy and isotropy diagnostics."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "activation_anisotropy",
    "version": "0.1.0",
    "description": "Measure residual-stream anisotropy, isotropy, and dominant directions across layers",
    "author": "LTTS",
    "event_types": ["layer_after"],
    "dependencies": ["requirements.txt"],
    "requires": ["hidden_states"],
    "target_roles": ["block"],
    "methods": {
        "ctor": "initialize_module",
        "dtor": "cleanup_module",
        "ntor": "process_event",
        "utor": "get_ui_schema",
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "max_vectors": 256,
    "emit_scalar": True,
    "emit_table": True,
    "emit_chart": True,
    "emit_mode": "all",
}

_SENDER = None
_TRACE: Dict[str, Any] = {"forward_pass": None, "entries": []}


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    return {"status": "initialized", "config": config}


def cleanup_module():
    _TRACE.update(forward_pass=None, entries=[])
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "max_vectors",
                "type": "number",
                "label": "Maximum token vectors",
                "default": 256,
                "min": 2,
                "max": 4096,
            },
            {
                "name": "emit_scalar",
                "type": "boolean",
                "label": "Emit isotropy score",
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
                "label": "Emit depth chart",
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


def _config(event) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if isinstance(event.module_state, dict):
        config.update(event.module_state)
    environment = getattr(event, "environment", None)
    if isinstance(environment, dict) and isinstance(
        environment.get("module_config"), dict
    ):
        config.update(environment["module_config"])
    try:
        config["max_vectors"] = max(2, min(int(config["max_vectors"]), 4096))
    except (TypeError, ValueError):
        config["max_vectors"] = 256
    return config


def _measure(hidden: torch.Tensor, max_vectors: int) -> Dict[str, float]:
    vectors = hidden.detach().float().reshape(-1, hidden.shape[-1])
    if vectors.shape[0] > max_vectors:
        indices = torch.linspace(
            0, vectors.shape[0] - 1, max_vectors, device=vectors.device
        ).long()
        vectors = vectors.index_select(0, indices)
    vectors = vectors[torch.linalg.vector_norm(vectors, dim=-1) > 1e-12]
    if vectors.shape[0] < 2:
        raise ValueError("at least two non-zero token vectors are required")

    normalized = torch.nn.functional.normalize(vectors, dim=-1)
    similarities = normalized @ normalized.T
    count = similarities.shape[0]
    mean_cosine = float(((similarities.sum() - count) / (count * (count - 1))).item())
    isotropy = max(0.0, min(1.0, 1.0 - abs(mean_cosine)))
    mean_direction_norm = float(normalized.mean(dim=0).norm().item())

    centered = vectors - vectors.mean(dim=0, keepdim=True)
    variances = torch.linalg.svdvals(centered).square()
    total_variance = float(variances.sum().item())
    dominant_ratio = (
        float(variances[0].item() / total_variance) if total_variance > 0 else 0.0
    )
    return {
        "isotropy_score": isotropy,
        "mean_pairwise_cosine": mean_cosine,
        "mean_direction_norm": mean_direction_norm,
        "dominant_variance_ratio": dominant_ratio,
        "vectors": int(vectors.shape[0]),
    }


def process_event(event):
    try:
        if not str(getattr(event, "event_type", "")).startswith("layer_after"):
            return {"status": "skipped", "reason": "event_phase"}
        hidden = _hidden(event.context.outputs)
        if hidden is None:
            return {"status": "skipped", "reason": "no_hidden_state"}
        config = _config(event)
        layer_path = getattr(event.context, "module_path", None) or "unknown"
        emission = event.should_emit(METADATA["name"], layer_path, config)
        if not emission.get("emit"):
            return {"status": "skipped", "reason": "emission_controller"}

        metrics = _measure(hidden, config["max_vectors"])
        forward_pass = emission.get("forward_pass")
        if _TRACE["forward_pass"] != forward_pass:
            _TRACE.update(forward_pass=forward_pass, entries=[])
        entry = {"layer": layer_path, **metrics}
        _TRACE["entries"].append(entry)
        base = {
            "layer_path": layer_path,
            "emit_mode": config.get("emit_mode", "all"),
        }
        emitted = []
        if _SENDER and config.get("emit_scalar", True):
            _SENDER.send_scalar(
                round(metrics["isotropy_score"], 6),
                label="Activation isotropy",
                emit_id=f"activation_anisotropy:{layer_path}:scalar",
                **base,
            )
            emitted.append("scalar")
        if _SENDER and config.get("emit_table", True):
            _SENDER.send_table(
                ["Metric", "Value"],
                [
                    ["Isotropy score", round(metrics["isotropy_score"], 6)],
                    ["Mean pairwise cosine", round(metrics["mean_pairwise_cosine"], 6)],
                    ["Mean-direction norm", round(metrics["mean_direction_norm"], 6)],
                    [
                        "Dominant variance ratio",
                        round(metrics["dominant_variance_ratio"], 6),
                    ],
                    ["Token vectors", metrics["vectors"]],
                ],
                emit_id=f"activation_anisotropy:{layer_path}:table",
                **base,
            )
            emitted.append("table")
        if _SENDER and config.get("emit_chart", True):
            entries = _TRACE["entries"]
            _SENDER.send_chart(
                {
                    "series": [
                        {
                            "name": "Isotropy",
                            "points": [
                                {"x": index, "y": round(item["isotropy_score"], 6)}
                                for index, item in enumerate(entries)
                            ],
                        },
                        {
                            "name": "Dominant variance",
                            "points": [
                                {
                                    "x": index,
                                    "y": round(item["dominant_variance_ratio"], 6),
                                }
                                for index, item in enumerate(entries)
                            ],
                        },
                    ],
                    "x_label": "Observed layer",
                    "y_label": "Score",
                },
                "line",
                emit_id=f"activation_anisotropy:{layer_path}:chart",
                **base,
            )
            emitted.append("chart")
        return {"status": "ok", "emitted": emitted, **entry}
    except Exception as exc:
        logger.error("ACTIVATION_ANISOTROPY error: %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc)}


def interceptor(event):
    return process_event(event)
