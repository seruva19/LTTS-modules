"""Intrinsic dimensionality (Participation Ratio and Effective Rank) across layers."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "intrinsic_dimensionality",
    "version": "0.1.0",
    "description": "Measure layer intrinsic dimensionality via Participation Ratio, Effective Rank, and PCA variance threshold",
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
    "variance_threshold": 0.9,
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
                "default": DEFAULT_CONFIG["max_vectors"],
                "min": 2,
                "max": 4096,
            },
            {
                "name": "variance_threshold",
                "type": "number",
                "label": "Cumulative variance threshold",
                "default": DEFAULT_CONFIG["variance_threshold"],
                "min": 0.5,
                "max": 0.99,
            },
            {
                "name": "emit_scalar",
                "type": "boolean",
                "label": "Emit participation ratio (scalar)",
                "default": DEFAULT_CONFIG["emit_scalar"],
            },
            {
                "name": "emit_table",
                "type": "boolean",
                "label": "Emit dimensionality table",
                "default": DEFAULT_CONFIG["emit_table"],
            },
            {
                "name": "emit_chart",
                "type": "boolean",
                "label": "Emit depth chart",
                "default": DEFAULT_CONFIG["emit_chart"],
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


def _hidden(outputs: Any) -> Optional[torch.Tensor]:
    candidate = (
        outputs[0] if isinstance(outputs, (tuple, list)) and outputs else outputs
    )
    if isinstance(candidate, torch.Tensor) and candidate.dim() == 3:
        return candidate
    return None


def _config(event) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if isinstance(getattr(event, "module_state", None), dict):
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
    try:
        config["variance_threshold"] = max(
            0.5, min(float(config["variance_threshold"]), 0.99)
        )
    except (TypeError, ValueError):
        config["variance_threshold"] = 0.9
    config["emit_scalar"] = _coerce_bool(config.get("emit_scalar"), True)
    config["emit_table"] = _coerce_bool(config.get("emit_table"), True)
    config["emit_chart"] = _coerce_bool(config.get("emit_chart"), True)
    return config


def _measure(hidden: torch.Tensor, max_vectors: int, variance_threshold: float) -> Dict[str, Any]:
    vectors = hidden.detach().float().reshape(-1, hidden.shape[-1])
    if vectors.shape[0] > max_vectors:
        indices = torch.linspace(
            0, vectors.shape[0] - 1, max_vectors, device=vectors.device
        ).long()
        vectors = vectors.index_select(0, indices)
    if vectors.shape[0] < 2:
        raise ValueError("at least two token vectors are required")

    centered = vectors - vectors.mean(dim=0, keepdim=True)
    eigvals = torch.linalg.svdvals(centered).square()
    total = eigvals.sum()
    if float(total.item()) <= 1e-12:
        return {
            "participation_ratio": 0.0,
            "effective_rank": 0.0,
            "dim_at_threshold": 0,
            "top1_variance_ratio": 0.0,
            "vectors": int(vectors.shape[0]),
        }

    pr = float((total.square() / eigvals.square().sum()).item())
    probs = eigvals / total
    erank = float(torch.exp(-(probs * torch.log(probs + 1e-12)).sum()).item())
    cum = torch.cumsum(probs, dim=0)
    dim_thresh = min(len(eigvals), int((cum < variance_threshold).sum().item()) + 1)
    top1_ratio = float(probs[0].item())

    return {
        "participation_ratio": pr,
        "effective_rank": erank,
        "dim_at_threshold": dim_thresh,
        "top1_variance_ratio": top1_ratio,
        "vectors": int(vectors.shape[0]),
    }


def process_event(event):
    try:
        evt = str(getattr(event, "event_type", "") or "")
        if not (evt == "layer_after" or evt.startswith("layer_after_")):
            return {"status": "skipped", "reason": "event_phase"}

        hidden = _hidden(event.context.outputs)
        if hidden is None:
            return {"status": "skipped", "reason": "no_hidden_state"}

        config = _config(event)
        layer_path = getattr(event.context, "module_path", None) or "unknown"
        emission = event.should_emit(METADATA["name"], layer_path, config)

        metrics = _measure(hidden, config["max_vectors"], config["variance_threshold"])

        forward_pass = emission.get("forward_pass")
        if _TRACE["forward_pass"] != forward_pass:
            _TRACE.update(forward_pass=forward_pass, entries=[])

        entries = _TRACE["entries"]
        entry = {
            "layer_path": layer_path,
            "participation_ratio": metrics["participation_ratio"],
            "effective_rank": metrics["effective_rank"],
        }
        replaced = False
        for idx, existing in enumerate(entries):
            if existing["layer_path"] == layer_path:
                entries[idx] = entry
                replaced = True
                break
        if not replaced:
            entries.append(entry)

        if not emission.get("emit"):
            return {"status": "skipped", "reason": "emission_controller"}

        name = METADATA["name"]
        emitted: List[str] = []

        if config["emit_scalar"]:
            _SENDER.send_scalar(
                round(metrics["participation_ratio"], 4),
                label=f"Participation Ratio (eRank={metrics['effective_rank']:.2f})",
                unit="dims",
                layer_path=layer_path,
                emit_id=f"{name}:{layer_path}:scalar",
            )
            emitted.append("scalar")

        if config["emit_table"]:
            pct = int(round(config["variance_threshold"] * 100))
            rows = [
                ["Participation Ratio", round(metrics["participation_ratio"], 4)],
                ["Effective Rank (eRank)", round(metrics["effective_rank"], 4)],
                [f"Dims for {pct}% variance", metrics["dim_at_threshold"]],
                ["Top-1 variance ratio", round(metrics["top1_variance_ratio"], 4)],
                ["Token vectors", metrics["vectors"]],
            ]
            _SENDER.send_table(
                ["Metric", "Value"],
                rows,
                layer_path=layer_path,
                emit_id=f"{name}:{layer_path}:table",
            )
            emitted.append("table")

        if len(entries) >= 2 and config["emit_chart"]:
            _SENDER.send_chart(
                {
                    "series": [
                        {
                            "name": "Participation Ratio",
                            "points": [
                                {"x": idx, "y": round(item["participation_ratio"], 4)}
                                for idx, item in enumerate(entries)
                            ],
                        },
                        {
                            "name": "Effective Rank",
                            "points": [
                                {"x": idx, "y": round(item["effective_rank"], 4)}
                                for idx, item in enumerate(entries)
                            ],
                        },
                    ],
                    "x_label": "Layer",
                    "y_label": "Effective Dimensions",
                },
                "line",
                layer_path=layer_path,
                emit_id=f"{name}:chart",
            )
            emitted.append("chart")

        if config.get("emit_mode") == "final":
            event.mark_emission_finalized(METADATA["name"], layer_path)

        return {
            "status": "processed",
            "layer": layer_path,
            "participation_ratio": metrics["participation_ratio"],
            "effective_rank": metrics["effective_rank"],
            "emitted": emitted,
        }
    except Exception as exc:
        logger.exception("INTRINSIC_DIMENSIONALITY: error processing event")
        reactor = getattr(event, "reactor", None)
        if reactor is not None and hasattr(reactor, "signal"):
            reactor.signal("error", {"message": str(exc), "module": METADATA["name"]})
        return {"status": "error", "error": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
