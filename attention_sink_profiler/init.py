"""
Attention Sink Profiler Module

Profiles attention-sink concentration on initial token positions across heads
and attention layers (StreamingLLM, arXiv:2309.17453; Why the First Token,
arXiv:2504.02732).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import torch

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "attention_sink_profiler",
    "version": "0.1.0",
    "description": "Profile attention-sink mass on initial tokens across heads and layers (StreamingLLM / first-token sink)",
    "author": "LTTS",
    "event_types": ["layer_after"],
    "dependencies": ["requirements.txt"],
    "requires": ["attention_weights"],
    "target_roles": ["self_attention"],
    "methods": {
        "ctor": "initialize_module",
        "dtor": "cleanup_module",
        "ntor": "process_event",
        "utor": "get_ui_schema",
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "sink_positions": 1,
    "sink_threshold": 0.3,
    "exclude_sink_queries": True,
    "emit_scalar": True,
    "emit_table": True,
    "emit_chart": True,
    "emit_heatmap": True,
    "emit_mode": "final",
}

_SENDER = None
_PROFILE: Dict[str, Any] = {"forward_pass": None, "layers": []}


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    return {"status": "initialized", "config": config}


def cleanup_module():
    _PROFILE["forward_pass"] = None
    _PROFILE["layers"] = []
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "sink_positions",
                "type": "number",
                "label": "Initial sink token positions",
                "default": DEFAULT_CONFIG["sink_positions"],
                "min": 1,
                "max": 8,
            },
            {
                "name": "sink_threshold",
                "type": "number",
                "label": "Sink-head threshold",
                "default": DEFAULT_CONFIG["sink_threshold"],
                "min": 0.05,
                "max": 0.95,
            },
            {
                "name": "exclude_sink_queries",
                "type": "boolean",
                "label": "Exclude sink positions from query average",
                "default": DEFAULT_CONFIG["exclude_sink_queries"],
            },
            {
                "name": "emit_scalar",
                "type": "boolean",
                "label": "Emit mean sink ratio (scalar)",
                "default": DEFAULT_CONFIG["emit_scalar"],
            },
            {
                "name": "emit_table",
                "type": "boolean",
                "label": "Emit per-head sink table",
                "default": DEFAULT_CONFIG["emit_table"],
            },
            {
                "name": "emit_chart",
                "type": "boolean",
                "label": "Emit sink-vs-depth chart",
                "default": DEFAULT_CONFIG["emit_chart"],
            },
            {
                "name": "emit_heatmap",
                "type": "boolean",
                "label": "Emit layer x head sink heatmap",
                "default": DEFAULT_CONFIG["emit_heatmap"],
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
    if isinstance(getattr(ltts_event, "module_state", None), dict):
        config.update(ltts_event.module_state)
    env = getattr(ltts_event, "environment", None)
    env_config = env.get("module_config") if isinstance(env, dict) else None
    if isinstance(env_config, dict):
        config.update(env_config)

    try:
        sink_positions = max(1, min(8, int(config.get("sink_positions", 1))))
    except (TypeError, ValueError):
        sink_positions = 1

    try:
        sink_threshold = max(0.05, min(0.95, float(config.get("sink_threshold", 0.3))))
    except (TypeError, ValueError):
        sink_threshold = 0.3

    return {
        "sink_positions": sink_positions,
        "sink_threshold": sink_threshold,
        "exclude_sink_queries": _coerce_bool(
            config.get("exclude_sink_queries"), DEFAULT_CONFIG["exclude_sink_queries"]
        ),
        "emit_scalar": _coerce_bool(config.get("emit_scalar"), DEFAULT_CONFIG["emit_scalar"]),
        "emit_table": _coerce_bool(config.get("emit_table"), DEFAULT_CONFIG["emit_table"]),
        "emit_chart": _coerce_bool(config.get("emit_chart"), DEFAULT_CONFIG["emit_chart"]),
        "emit_heatmap": _coerce_bool(config.get("emit_heatmap"), DEFAULT_CONFIG["emit_heatmap"]),
        "emit_mode": str(config.get("emit_mode", DEFAULT_CONFIG["emit_mode"])),
        "emit_every_n": config.get("emit_every_n", 1),
    }


def _extract_attention(attention_weights: Any) -> Optional[torch.Tensor]:
    candidate = attention_weights
    if isinstance(candidate, (list, tuple)) and len(candidate) > 0:
        candidate = candidate[0]
    if isinstance(candidate, torch.Tensor) and candidate.dim() == 4:
        return candidate
    return None


def _candidate_attention_keys(module_path: str) -> List[str]:
    keys: List[str] = []

    def _append_unique(value: str) -> None:
        if value and value not in keys:
            keys.append(value)

    if module_path:
        _append_unique(module_path)
        base_path = re.sub(
            r"^(original_model\.model\.|original_model\.|model\.)", "", module_path
        )
        for prefix in ("", "model.", "original_model.", "original_model.model."):
            _append_unique(f"{prefix}{base_path}" if prefix else base_path)
        for existing in list(keys):
            if ".self_attn" not in existing:
                _append_unique(f"{existing}.self_attn")
        match = re.search(r"layers\.(\d+)(?:\.|$)", module_path)
        if not match:
            match = re.search(r"transformer\.h\.(\d+)(?:\.|$)", module_path)
        if match:
            idx = match.group(1)
            _append_unique(f"model.layers.{idx}.self_attn")
            _append_unique(f"layers.{idx}.self_attn")
    return keys


def _get_layer_attention_weights(ltts_event) -> Optional[torch.Tensor]:
    try:
        context = ltts_event.context
        ltts_model = getattr(context, "ltts_model", None)
        module_path = (getattr(context, "module_path", "") or "").strip()
        candidate_keys = _candidate_attention_keys(module_path)

        if ltts_model is not None and hasattr(ltts_model, "captured_attn"):
            captured = ltts_model.captured_attn or {}
            for key in candidate_keys:
                if key in captured and captured[key]:
                    latest_step = max(captured[key].keys())
                    return captured[key][latest_step]

        if ltts_model is not None and hasattr(ltts_model, "layer_attention_weights"):
            law = ltts_model.layer_attention_weights or {}
            for key in candidate_keys:
                if key in law:
                    return law[key]

        attn = getattr(context, "attention_weights", None)
        if isinstance(attn, torch.Tensor):
            return attn

        try:
            from miners.utils import extract_attention_weights

            return extract_attention_weights(getattr(context, "outputs", None))
        except Exception:
            return None
    except Exception as exc:
        logger.warning("ATTENTION_SINK_PROFILER: error getting attention: %s", exc)
        return None


def process_event(ltts_event):
    try:
        evt = getattr(ltts_event, "event_type", "") or ""
        if not (evt == "layer_after" or evt.startswith("layer_after_")):
            return {"status": "skipped", "reason": "event_phase"}

        context = ltts_event.context
        if getattr(context, "layer_type", None) != "attention":
            return {"status": "skipped", "reason": "not_attention_layer"}

        attention = _extract_attention(_get_layer_attention_weights(ltts_event))
        if attention is None:
            return {"status": "skipped", "reason": "no_attention_weights"}

        if not isinstance(getattr(ltts_event, "module_state", None), dict):
            ltts_event.module_state = {}
        config = _resolve_config(ltts_event)
        layer_path = getattr(context, "module_path", None) or "unknown"

        emission = ltts_event.should_emit(METADATA["name"], layer_path, config)

        with torch.no_grad():
            attn = attention[0].detach().float().cpu()  # [H, Q, K]
            num_heads, num_queries, key_len = attn.shape
            s = min(config["sink_positions"], key_len)
            q_attn = attn[:, s:, :] if (config["exclude_sink_queries"] and num_queries > s) else attn
            sink_mass = q_attn[:, :, :s].sum(dim=-1)  # [H, Q']
            sink_ratios = [float(v) for v in sink_mass.mean(dim=-1).tolist()]
            max_sinks = [float(v) for v in sink_mass.amax(dim=-1).tolist()]

        sink_threshold = config["sink_threshold"]
        sink_flags = [ratio >= sink_threshold for ratio in sink_ratios]
        sink_count = sum(1 for flag in sink_flags if flag)
        mean_sink = sum(sink_ratios) / len(sink_ratios) if sink_ratios else 0.0
        sink_head_fraction = sink_count / num_heads if num_heads else 0.0

        current_pass = emission.get("forward_pass")
        if _PROFILE["forward_pass"] != current_pass:
            _PROFILE["forward_pass"] = current_pass
            _PROFILE["layers"] = []

        layers = _PROFILE["layers"]
        entry = {
            "layer_path": layer_path,
            "mean_sink": mean_sink,
            "sink_head_fraction": sink_head_fraction,
            "sink_ratios": sink_ratios,
        }
        replaced = False
        for idx, existing in enumerate(layers):
            if existing["layer_path"] == layer_path:
                layers[idx] = entry
                replaced = True
                break
        if not replaced:
            layers.append(entry)

        if not emission.get("emit"):
            return {"status": "skipped", "reason": "emission_controller"}

        name = METADATA["name"]
        emitted: List[str] = []

        if config["emit_scalar"]:
            _SENDER.send_scalar(
                round(mean_sink, 4),
                label=f"Mean sink ratio ({sink_count}/{num_heads} heads >= {sink_threshold:.2f})",
                unit="ratio",
                layer_path=layer_path,
                emit_id=f"{name}:{layer_path}:scalar",
            )
            emitted.append("scalar")

        if config["emit_table"]:
            rows = [
                [
                    head_idx,
                    round(sink_ratios[head_idx], 4),
                    round(max_sinks[head_idx], 4),
                    "yes" if sink_flags[head_idx] else "no",
                ]
                for head_idx in range(num_heads)
            ]
            _SENDER.send_table(
                ["Head", "Sink Ratio", "Max Sink Weight", "Sink Head"],
                rows,
                layer_path=layer_path,
                emit_id=f"{name}:{layer_path}:table",
            )
            emitted.append("table")

        if len(layers) >= 2:
            if config["emit_chart"]:
                _SENDER.send_chart(
                    {
                        "series": [
                            {
                                "name": "Mean Sink Ratio",
                                "points": [
                                    {"x": idx, "y": round(item["mean_sink"], 4)}
                                    for idx, item in enumerate(layers)
                                ],
                            },
                            {
                                "name": "Sink Head Fraction",
                                "points": [
                                    {"x": idx, "y": round(item["sink_head_fraction"], 4)}
                                    for idx, item in enumerate(layers)
                                ],
                            },
                        ],
                        "x_label": "Attention layer",
                        "y_label": "Ratio",
                    },
                    "line",
                    layer_path=layer_path,
                    emit_id=f"{name}:chart",
                )
                emitted.append("chart")

            if config["emit_heatmap"]:
                cols = len(layers[0]["sink_ratios"])
                if cols > 0 and all(len(item["sink_ratios"]) == cols for item in layers):
                    matrix = [
                        [round(val, 4) for val in item["sink_ratios"]]
                        for item in layers
                    ]
                    _SENDER.send_heatmap(
                        matrix,
                        dimensions={"rows": len(layers), "cols": cols},
                        layer_path=layer_path,
                        emit_id=f"{name}:heatmap",
                    )
                    emitted.append("heatmap")

        if config.get("emit_mode") == "final":
            ltts_event.mark_emission_finalized(METADATA["name"], layer_path)

        return {
            "status": "processed",
            "layer": layer_path,
            "mean_sink_ratio": mean_sink,
            "sink_heads": sink_count,
            "emitted": emitted,
        }
    except Exception as exc:
        logger.exception("ATTENTION_SINK_PROFILER: error processing event")
        reactor = getattr(ltts_event, "reactor", None)
        if reactor is not None and hasattr(reactor, "signal"):
            reactor.signal("error", {"message": str(exc), "module": METADATA["name"]})
        return {"status": "error", "error": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
