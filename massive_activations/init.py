"""Massive activations, outlier dimensions, and attention sinks.

A few residual-stream coordinates in a trained transformer hold values orders of
magnitude above the layer median (arXiv 2402.17762). They behave as learned bias
terms, and the positions carrying them collect a large share of every head's
attention mass (arXiv 2309.17453, arXiv 2504.02732). Because they dominate any
norm/sparsity/entropy statistic taken over the residual stream, they are isolated
here explicitly rather than left to skew the other modules' numbers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "massive_activations",
    "version": "0.1.0",
    "description": "Isolate outlier residual-stream activations (massive activations / super weights) and the attention sinks they create",
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
    "top_k": 5,
    "massive_ratio_threshold": 100.0,
    "top_sink_heads": 8,
    "emit_outlier_table": True,
    "emit_ratio_scalar": True,
    "emit_chart": True,
    "emit_sink_scalar": True,
    "emit_sink_table": True,
    "emit_mode": "final",
}

# Upper bounds exist because this runs inside a live forward pass: a mistyped
# config value must not turn into a multi-thousand-row payload per layer.
_MAX_TOP_K = 50
_MAX_SINK_HEADS = 64

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
                "name": "top_k",
                "type": "number",
                "label": "Top outlier activations",
                "default": DEFAULT_CONFIG["top_k"],
                "min": 1,
                "max": _MAX_TOP_K,
            },
            {
                "name": "massive_ratio_threshold",
                "type": "number",
                "label": "Massive threshold (x median |activation|)",
                "default": DEFAULT_CONFIG["massive_ratio_threshold"],
                "min": 1,
            },
            {
                "name": "top_sink_heads",
                "type": "number",
                "label": "Top attention-sink heads",
                "default": DEFAULT_CONFIG["top_sink_heads"],
                "min": 1,
                "max": _MAX_SINK_HEADS,
            },
            {
                "name": "emit_outlier_table",
                "type": "boolean",
                "label": "Emit outlier table",
                "default": True,
            },
            {
                "name": "emit_ratio_scalar",
                "type": "boolean",
                "label": "Emit massiveness scalar",
                "default": True,
            },
            {
                "name": "emit_chart",
                "type": "boolean",
                "label": "Emit max-vs-median depth chart",
                "default": True,
            },
            {
                "name": "emit_sink_scalar",
                "type": "boolean",
                "label": "Emit attention-sink scalar",
                "default": True,
            },
            {
                "name": "emit_sink_table",
                "type": "boolean",
                "label": "Emit attention-sink head table",
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


def _extract_attention(attention_weights: Any) -> Optional[torch.Tensor]:
    candidate = attention_weights
    if isinstance(candidate, (list, tuple)) and candidate:
        candidate = candidate[0]
    if isinstance(candidate, torch.Tensor) and candidate.dim() == 4:
        return candidate
    return None


def _token_labels(context: Any) -> List[str]:
    ltts_model = getattr(context, "ltts_model", None)
    tokens = getattr(ltts_model, "_last_input_tokens", None)
    if isinstance(tokens, (list, tuple)) and tokens:
        return [str(token) for token in tokens]
    return []


def _token_at(tokens: List[str], position: int) -> Optional[str]:
    return tokens[position] if 0 <= position < len(tokens) else None


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def _config(event) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if isinstance(event.module_state, dict):
        config.update(event.module_state)
    config["top_k"] = _clamp_int(config.get("top_k"), DEFAULT_CONFIG["top_k"], 1, _MAX_TOP_K)
    config["top_sink_heads"] = _clamp_int(
        config.get("top_sink_heads"), DEFAULT_CONFIG["top_sink_heads"], 1, _MAX_SINK_HEADS
    )
    try:
        config["massive_ratio_threshold"] = max(1.0, float(config.get("massive_ratio_threshold")))
    except (TypeError, ValueError):
        config["massive_ratio_threshold"] = DEFAULT_CONFIG["massive_ratio_threshold"]
    return config


def _attention_sinks(attention: torch.Tensor, top_heads: int) -> Dict[str, Any]:
    """Per-head attention mass on position 0 and on each head's own sink key."""
    weights = attention[0].detach().float()
    # Averaging over queries first collapses the [q, k] grid to one row per head,
    # so everything below is a handful of values instead of a per-query scan.
    per_key = weights.mean(dim=1)
    position_zero = per_key[:, 0]
    peak_mass, peak_index = per_key.max(dim=1)
    order = torch.argsort(position_zero, descending=True)[:top_heads]

    return {
        "mean_position_zero": float(position_zero.mean().item()),
        "heads": [
            {
                "head": int(head),
                "position_zero": float(zero),
                "sink_position": int(sink_pos),
                "sink_mass": float(sink_mass),
            }
            for head, zero, sink_pos, sink_mass in zip(
                order.cpu().tolist(),
                position_zero[order].cpu().tolist(),
                peak_index[order].cpu().tolist(),
                peak_mass[order].cpu().tolist(),
            )
        ],
    }


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

        magnitudes = hidden[0].detach().float().abs()
        seq_len, hidden_dim = magnitudes.shape
        median = float(magnitudes.median().item())
        threshold = config["massive_ratio_threshold"]
        # A zero median (an all-zero layer) would make every ratio infinite;
        # report no massiveness rather than divide by it.
        scale = median if median > 0 else None

        top_k = min(config["top_k"], magnitudes.numel())
        values, flat_indices = torch.topk(magnitudes.reshape(-1), top_k)
        # One host transfer per tensor: never .item() inside the loop below.
        signed = hidden[0].detach().float().reshape(-1)[flat_indices].cpu().tolist()
        indices = flat_indices.cpu().tolist()
        magnitudes_top = values.cpu().tolist()

        massive_dims = 0
        if scale is not None:
            massive_dims = int((magnitudes > threshold * scale).any(dim=0).sum().item())

        tokens = _token_labels(context)
        outliers = []
        for rank, (flat_index, magnitude, value) in enumerate(
            zip(indices, magnitudes_top, signed), start=1
        ):
            position = flat_index // hidden_dim
            outliers.append(
                {
                    "rank": rank,
                    "position": position,
                    "token": _token_at(tokens, position),
                    "dimension": flat_index % hidden_dim,
                    "value": value,
                    "ratio": value / scale if scale is not None else None,
                    "magnitude_ratio": magnitude / scale if scale is not None else None,
                }
            )

        max_abs = magnitudes_top[0] if magnitudes_top else 0.0
        max_ratio = max_abs / scale if scale is not None else None

        current_pass = emission.get("forward_pass")
        if _TRAJECTORY["forward_pass"] != current_pass:
            _TRAJECTORY["forward_pass"] = current_pass
            _TRAJECTORY["entries"] = []
        _TRAJECTORY["entries"].append(
            {"layer": layer_path or "unknown", "max_abs": max_abs, "median": median}
        )

        sinks = None
        attention = _extract_attention(getattr(context, "attention_weights", None))
        if attention is not None and attention.shape[-1] > 0:
            sinks = _attention_sinks(attention, config["top_sink_heads"])

        base = {"layer_path": layer_path, "emit_mode": config.get("emit_mode", "final")}
        emitted = []
        if _SENDER and config.get("emit_outlier_table", True):
            _SENDER.send_table(
                ["Rank", "Position", "Token", "Dimension", "Value", "x median"],
                [
                    [
                        item["rank"],
                        item["position"],
                        item["token"],
                        item["dimension"],
                        round(item["value"], 6),
                        round(item["ratio"], 3) if item["ratio"] is not None else None,
                    ]
                    for item in outliers
                ],
                emit_id=f"massive_activations:{layer_path}:outliers",
                **base,
            )
            emitted.append("table")
        if _SENDER and config.get("emit_ratio_scalar", True):
            _SENDER.send_scalar(
                round(max_ratio, 3) if max_ratio is not None else None,
                label=(
                    f"Max massiveness (x median) - {massive_dims} dims above "
                    f"{round(threshold, 3)}x"
                ),
                emit_id=f"massive_activations:{layer_path}:ratio",
                **base,
            )
            emitted.append("scalar")
        if _SENDER and config.get("emit_chart", True) and len(_TRAJECTORY["entries"]) > 1:
            entries = _TRAJECTORY["entries"]
            _SENDER.send_chart(
                {
                    "series": [
                        {
                            "name": "Max |activation|",
                            "points": [
                                {"x": index, "y": round(item["max_abs"], 6)}
                                for index, item in enumerate(entries)
                            ],
                        },
                        {
                            "name": "Median |activation|",
                            "points": [
                                {"x": index, "y": round(item["median"], 6)}
                                for index, item in enumerate(entries)
                            ],
                        },
                    ],
                    "x_label": "Layer index",
                    "y_label": "|Activation|",
                },
                "line",
                emit_id=f"massive_activations:{layer_path}:chart",
                **base,
            )
            emitted.append("chart")
        if _SENDER and sinks and config.get("emit_sink_scalar", True):
            _SENDER.send_scalar(
                round(sinks["mean_position_zero"], 6),
                label="Attention mass on position 0 (mean over heads)",
                emit_id=f"massive_activations:{layer_path}:sink_scalar",
                **base,
            )
            emitted.append("sink_scalar")
        if _SENDER and sinks and config.get("emit_sink_table", True):
            _SENDER.send_table(
                ["Head", "Mass on pos 0", "Sink position", "Sink token", "Sink mass"],
                [
                    [
                        head["head"],
                        round(head["position_zero"], 6),
                        head["sink_position"],
                        _token_at(tokens, head["sink_position"]),
                        round(head["sink_mass"], 6),
                    ]
                    for head in sinks["heads"]
                ],
                emit_id=f"massive_activations:{layer_path}:sink_heads",
                **base,
            )
            emitted.append("sink_table")
        return {
            "status": "ok",
            "emitted": emitted,
            "layer": layer_path or "unknown",
            "seq_len": int(seq_len),
            "median_abs": median,
            "max_abs": max_abs,
            "max_ratio": max_ratio,
            "massive_dimensions": massive_dims,
            "outliers": outliers,
            "sink_position_zero": sinks["mean_position_zero"] if sinks else None,
        }
    except Exception as exc:
        logger.error("MASSIVE_ACTIVATIONS error: %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
