"""
Sample Scalar Module

Provides a predictable example module that emits at most a single scalar value and
an optional text snippet each time it is invoked. Designed for debugging the LTTS
pipeline with one scalar and optional text payload per invocation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "sample_module",
    "version": "0.3.1",
    "description": "Minimal scalar emitter for LTTS debugging",
    "author": "LTTS",
    "event_types": ["layer_after", "layer_before"],
    "dependencies": ["requirements.txt"],
    "methods": {
        "ctor": "initialize_module",
        "dtor": "cleanup_module",
        "ntor": "process_event",
        "utor": "get_ui_schema",
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "emit_scalar": True,
    "scalar_label": "Sample Scalar",
    "scale_factor": 1.0,
    "scalar_precision": 4,
    "emit_text": False,
    "text_content": "Sample module message",
    "emit_mode": "final",
    "emit_table": False,
    "emit_json": False,
    "emit_image": False,
    "emit_chart": False,
}

_VALID_EMIT_MODES = {"all", "first", "every_n", "final", "manual"}

_SENDER = None
_RNG = np.random.default_rng()

_SAMPLE_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
)


def _round_value(value: float, precision: int) -> float:
    try:
        return round(float(value), precision)
    except Exception:
        return float(value)


def _generate_sample_table(scale_factor: float, precision: int):
    headers = ["Index", "Random Value", "Scaled"]
    random_values = _RNG.random(3)
    rows = [
        [
            idx,
            _round_value(val, precision),
            _round_value(val * scale_factor, precision),
        ]
        for idx, val in enumerate(random_values)
    ]
    return headers, rows


def _generate_sample_json(scale_factor: float):
    random_values = _RNG.random(5) * scale_factor
    return {
        "module": METADATA["name"],
        "scale_factor": scale_factor,
        "statistics": {
            "min": float(np.min(random_values)),
            "max": float(np.max(random_values)),
            "mean": float(np.mean(random_values)),
            "std": float(np.std(random_values)),
        },
        "values": [float(val) for val in random_values],
    }


def _generate_sample_chart(scale_factor: float, precision: int):
    steps = list(range(6))
    base_wave = np.sin(np.linspace(0, np.pi, len(steps)))
    noise = _RNG.normal(0.0, 0.05, len(steps))
    series = [
        {
            "x": step,
            "y": _round_value((value + jitter) * scale_factor, precision),
        }
        for step, value, jitter in zip(steps, base_wave, noise)
    ]
    return {
        "series": [
            {
                "name": "Sample Wave",
                "points": series,
            }
        ],
        "x_label": "Step",
        "y_label": "Scaled Output",
    }


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    logger.info("Sample scalar module initialized with config: %s", config)
    return {"status": "initialized", "config": config}


def cleanup_module():
    logger.info("Sample scalar module cleaned up")
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "emit_scalar",
                "type": "boolean",
                "label": "Emit Scalar",
                "default": DEFAULT_CONFIG["emit_scalar"],
            },
            {
                "name": "scalar_label",
                "type": "text",
                "label": "Scalar Label",
                "default": DEFAULT_CONFIG["scalar_label"],
            },
            {
                "name": "scale_factor",
                "type": "number",
                "label": "Scale Factor",
                "default": DEFAULT_CONFIG["scale_factor"],
                "min": 0.0,
            },
            {
                "name": "scalar_precision",
                "type": "number",
                "label": "Scalar Precision",
                "default": DEFAULT_CONFIG["scalar_precision"],
                "min": 0,
                "max": 8,
            },
            {
                "name": "emit_text",
                "type": "boolean",
                "label": "Emit Text Message",
                "default": DEFAULT_CONFIG["emit_text"],
            },
            {
                "name": "text_content",
                "type": "text",
                "label": "Text Content",
                "default": DEFAULT_CONFIG["text_content"],
            },
            {
                "name": "emit_table",
                "type": "boolean",
                "label": "Emit Table",
                "default": DEFAULT_CONFIG["emit_table"],
            },
            {
                "name": "emit_json",
                "type": "boolean",
                "label": "Emit JSON",
                "default": DEFAULT_CONFIG["emit_json"],
            },
            {
                "name": "emit_image",
                "type": "boolean",
                "label": "Emit Image",
                "default": DEFAULT_CONFIG["emit_image"],
            },
            {
                "name": "emit_chart",
                "type": "boolean",
                "label": "Emit Chart",
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


def _coerce_float(value: Any, default: float, minimum: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if result < minimum:
        result = minimum
    return result


def _coerce_precision(value: Any, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if result < 0:
        result = 0
    if result > 8:
        result = 8
    return result


def _resolve_config(ltts_event) -> Dict[str, Any]:
    config: Dict[str, Any] = dict(DEFAULT_CONFIG)

    state = ltts_event.module_state if isinstance(ltts_event.module_state, dict) else {}
    config.update(state)

    env = getattr(ltts_event, "environment", None)
    env_config = env.get("module_config") if isinstance(env, dict) else None
    if isinstance(env_config, dict):
        config.update(env_config)

    emit_mode = str(config.get("emit_mode", DEFAULT_CONFIG["emit_mode"]))
    try:
        emit_every_n_val = int(config.get("emit_every_n", 1))
        if emit_every_n_val < 1:
            emit_every_n_val = 1
    except Exception:
        emit_every_n_val = 1
    expected_passes_val = config.get("expected_generation_passes")

    sanitized = {
        "emit_scalar": _coerce_bool(
            config.get("emit_scalar"), DEFAULT_CONFIG["emit_scalar"]
        ),
        "scalar_label": str(config.get("scalar_label", DEFAULT_CONFIG["scalar_label"])),
        "scale_factor": _coerce_float(
            config.get("scale_factor"), DEFAULT_CONFIG["scale_factor"], 0.0
        ),
        "scalar_precision": _coerce_precision(
            config.get("scalar_precision"), DEFAULT_CONFIG["scalar_precision"]
        ),
        "emit_text": _coerce_bool(config.get("emit_text"), DEFAULT_CONFIG["emit_text"]),
        "text_content": str(config.get("text_content", DEFAULT_CONFIG["text_content"])),
        "emit_table": _coerce_bool(
            config.get("emit_table"), DEFAULT_CONFIG["emit_table"]
        ),
        "emit_json": _coerce_bool(
            config.get("emit_json"), DEFAULT_CONFIG["emit_json"]
        ),
        "emit_image": _coerce_bool(
            config.get("emit_image"), DEFAULT_CONFIG["emit_image"]
        ),
        "emit_chart": _coerce_bool(
            config.get("emit_chart"), DEFAULT_CONFIG["emit_chart"]
        ),
        "emit_mode": emit_mode,
        "emit_every_n": emit_every_n_val,
        "expected_generation_passes": expected_passes_val,
    }

    if isinstance(ltts_event.module_state, dict):
        ltts_event.module_state.update(sanitized)
    else:
        ltts_event.module_state = dict(sanitized)

    return sanitized


def process_event(ltts_event):
    try:
        # Only respond to layer-after (including layer-specific) events to avoid duplicate emissions
        evt = getattr(ltts_event, "event_type", "") or ""
        if not (evt == "layer_after" or evt.startswith("layer_after_")):
            return {"status": "skipped", "reason": "event_phase"}

        if not isinstance(ltts_event.module_state, dict):
            ltts_event.module_state = {}

        config = _resolve_config(ltts_event)

        # Read the layer path from the event context.
        layer_path = getattr(ltts_event.context, "module_path", None)
        module_name = METADATA["name"]

        # Apply the framework emission policy to this layer.
        emission_info = ltts_event.should_emit(module_name, layer_path, config)

        if bool(ltts_event.module_state.get("emitted_once", False)):
            logger.debug(
                "SAMPLE_MODULE: emission latched for %s (event=%s, pass=%s)",
                layer_path,
                getattr(ltts_event, "event_type", None),
                emission_info.get("forward_pass"),
            )
            return {"status": "skipped", "reason": "latched"}

        # The framework controls emission frequency.
        if not emission_info.get("emit"):
            logger.debug(
                "SAMPLE_MODULE: Skipping emission for %s (mode=%s, pass=%s)",
                layer_path,
                emission_info.get("mode"),
                emission_info.get("forward_pass"),
            )
            return {
                "status": "skipped",
                "mode": emission_info.get("mode"),
                "forward_pass": emission_info.get("forward_pass"),
                "reason": "emission_controller",
            }

        emitted = []

        active_outputs = {
            "scalar": bool(config["emit_scalar"]),
            "text": bool(config["emit_text"]),
            "table": bool(config.get("emit_table")),
            "json": bool(config.get("emit_json")),
            "image": bool(config.get("emit_image")),
            "chart": bool(config.get("emit_chart")),
        }

        if not any(active_outputs.values()):
            return {"status": "noop", "reason": "outputs_disabled"}

        forward_pass = emission_info.get("forward_pass")
        base_kwargs = {
            "layer_name": layer_path,
            "layer_path": layer_path,
            "emit_mode": config.get("emit_mode"),
            "forward_pass": forward_pass,
        }

        if _SENDER and config["emit_scalar"]:
            raw_value = float(_RNG.random())
            scaled_value = raw_value * config["scale_factor"]
            precision = config["scalar_precision"]
            value = round(scaled_value, precision)
            logger.info(
                "SAMPLE_MODULE: emitting scalar=%s label=%s for layer=%s",
                value,
                config["scalar_label"],
                layer_path,
            )
            _SENDER.send_scalar(
                value,
                label=config["scalar_label"],
                emit_id=f"sample_module:{layer_path}:scalar",
                **base_kwargs,
            )
            emitted.append("scalar")

        if _SENDER and config["emit_text"]:
            _SENDER.send_text(
                config["text_content"],
                preformatted=False,
                emit_id=f"sample_module:{layer_path}:text",
                **base_kwargs,
            )
            emitted.append("text")

        if _SENDER and active_outputs["table"]:
            headers, rows = _generate_sample_table(
                config["scale_factor"], config["scalar_precision"]
            )
            _SENDER.send_table(
                headers,
                rows,
                emit_id=f"sample_module:{layer_path}:table",
                **base_kwargs,
            )
            emitted.append("table")

        if _SENDER and active_outputs["json"]:
            json_payload = _generate_sample_json(config["scale_factor"])
            _SENDER.send_visualization(
                {"type": "json", "data": json_payload},
                "json",
                emit_id=f"sample_module:{layer_path}:json",
                **base_kwargs,
            )
            emitted.append("json")

        if _SENDER and active_outputs["image"]:
            _SENDER.send_image(
                _SAMPLE_IMAGE_B64,
                emit_id=f"sample_module:{layer_path}:image",
                **base_kwargs,
            )
            emitted.append("image")

        if _SENDER and active_outputs["chart"]:
            chart_payload = _generate_sample_chart(
                config["scale_factor"], config["scalar_precision"]
            )
            _SENDER.send_chart(
                chart_payload,
                "line",
                emit_id=f"sample_module:{layer_path}:chart",
                **base_kwargs,
            )
            emitted.append("chart")

        if not emitted:
            return {"status": "noop", "reason": "no_output_enabled"}

        # Mark as finalized if the mode was 'final'
        if config.get("emit_mode") == "final":
            try:
                ltts_event.mark_emission_finalized(module_name, layer_path)
            except Exception:
                logger.debug(
                    "SAMPLE_MODULE: mark_emission_finalized failed", exc_info=True
                )

        try:
            ltts_event.module_state["emitted_once"] = True
        except Exception:
            pass

        return {"status": "ok", "emitted": emitted}

    except Exception as exc:
        logger.error("Error in sample module: %s", exc, exc_info=True)
        ltts_event.reactor.signal(
            "error",
            {
                "message": f"Sample module error: {exc}",
                "module": METADATA["name"],
            },
        )
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
