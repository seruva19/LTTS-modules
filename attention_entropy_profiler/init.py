"""
Attention Entropy Profiler Module

Profiles attention concentration across heads, layers, and decoding steps.
For every attention layer it computes the per-head entropy of the attention
distribution over the key dimension (averaged over query positions) and
normalizes it by log(seq_len) so the value lands in [0, 1]:

    H = -sum(p * log(p + 1e-9))      (over keys, per head, per query)
    H_norm = H / log(seq_len)

Low normalized entropy means the head focuses on few tokens (concentrated);
high entropy means diffuse, near-uniform attention.

Emits per attention layer (configurable):
- a scalar: mean normalized entropy across this layer's heads
- a table: head index, raw entropy, normalized entropy, max attention weight
- after >= 2 attention layers accumulated, a line chart of mean normalized
  entropy vs attention-layer index for this forward pass
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional

import torch

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "attention_entropy_profiler",
    "version": "0.1.0",
    "description": "Per-head attention entropy (normalized by log(seq_len)) profiled across attention layers: scalar, head table, and depth chart",
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
    "emit_scalar": True,
    "emit_table": False,
    "emit_chart": True,
    "emit_mode": "final",
}

_SENDER = None

# Cross-layer accumulator: each layer event gets a fresh LTTSContext, so the
# per-depth entropy profile must live in module state, keyed by forward pass.
_PROFILE: Dict[str, Any] = {"forward_pass": None, "layers": []}


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    logger.info("ATTENTION_ENTROPY_PROFILER: initialized with config: %s", config)
    return {"status": "initialized", "config": config}


def cleanup_module():
    _PROFILE["forward_pass"] = None
    _PROFILE["layers"] = []
    logger.info("ATTENTION_ENTROPY_PROFILER: cleaned up")
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "emit_scalar",
                "type": "boolean",
                "label": "Emit mean normalized entropy (scalar)",
                "default": DEFAULT_CONFIG["emit_scalar"],
            },
            {
                "name": "emit_table",
                "type": "boolean",
                "label": "Emit per-head entropy table",
                "default": DEFAULT_CONFIG["emit_table"],
            },
            {
                "name": "emit_chart",
                "type": "boolean",
                "label": "Emit entropy-vs-depth chart",
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


def _resolve_config(ltts_event) -> Dict[str, Any]:
    config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    if isinstance(ltts_event.module_state, dict):
        config.update(ltts_event.module_state)
    env = getattr(ltts_event, "environment", None)
    env_config = env.get("module_config") if isinstance(env, dict) else None
    if isinstance(env_config, dict):
        config.update(env_config)

    return {
        "emit_scalar": _coerce_bool(config.get("emit_scalar"), DEFAULT_CONFIG["emit_scalar"]),
        "emit_table": _coerce_bool(config.get("emit_table"), DEFAULT_CONFIG["emit_table"]),
        "emit_chart": _coerce_bool(config.get("emit_chart"), DEFAULT_CONFIG["emit_chart"]),
        "emit_mode": str(config.get("emit_mode", DEFAULT_CONFIG["emit_mode"])),
        "emit_every_n": config.get("emit_every_n", 1),
    }


def _extract_attention(attention_weights: Any) -> Optional[torch.Tensor]:
    """Attention weights are expected as [batch, heads, seq_q, seq_k]."""
    candidate = attention_weights
    if isinstance(candidate, (list, tuple)) and len(candidate) > 0:
        candidate = candidate[0]
    if isinstance(candidate, torch.Tensor) and candidate.dim() == 4:
        return candidate
    return None


def _candidate_attention_keys(module_path: str) -> List[str]:
    """Build candidate captured_attn keys, mirroring attn_visualizer."""
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
    """Locate attention weights for the current layer from any available source.

    Modern attention backends (sdpa/flash) do not populate
    ``context.attention_weights`` via ``output_attentions``; the QKV-hook path
    records them on the model instead. Mirror attn_visualizer's multi-source
    lookup so the profiler works regardless of attention implementation.
    """
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
        logger.warning("ATTENTION_ENTROPY_PROFILER: error getting attention: %s", exc)
        return None


def process_event(ltts_event):
    try:
        evt = getattr(ltts_event, "event_type", "") or ""
        if not (evt == "layer_after" or evt.startswith("layer_after_")):
            return {"status": "skipped", "reason": "event_phase"}

        context = ltts_event.context
        if context.layer_type != "attention":
            return {"status": "skipped", "reason": "not_attention_layer"}

        attention = _extract_attention(_get_layer_attention_weights(ltts_event))
        if attention is None:
            return {"status": "skipped", "reason": "no_attention_weights"}

        if not isinstance(ltts_event.module_state, dict):
            ltts_event.module_state = {}
        config = _resolve_config(ltts_event)
        layer_path = getattr(context, "module_path", None)

        emission = ltts_event.should_emit(METADATA["name"], layer_path, config)

        with torch.no_grad():
            # [heads, seq_q, seq_k] for the first batch element.
            attn = attention[0].detach().float().cpu()
            key_len = attn.shape[-1]
            # Per-head entropy over the key dimension, averaged over queries.
            entropy_per_query = -(attn * torch.log(attn + 1e-9)).sum(dim=-1)
            raw_entropy = entropy_per_query.mean(dim=-1)  # [heads]
            log_len = math.log(key_len) if key_len > 1 else 1.0
            norm_entropy = raw_entropy / log_len
            max_weight = attn.amax(dim=(-2, -1))  # [heads]

        raw_list = [float(v) for v in raw_entropy.tolist()]
        norm_list = [float(v) for v in norm_entropy.tolist()]
        max_list = [float(v) for v in max_weight.tolist()]
        mean_norm = sum(norm_list) / len(norm_list) if norm_list else 0.0

        # Accumulate before the emission gate so the depth profile stays
        # complete even when individual layer emissions are suppressed.
        current_pass = emission.get("forward_pass")
        if _PROFILE["forward_pass"] != current_pass:
            _PROFILE["forward_pass"] = current_pass
            _PROFILE["layers"] = []
        layers = _PROFILE["layers"]
        # Generic and layer-specific event variants may both fire for the same
        # layer in one pass; record it only once.
        if not any(entry["layer"] == (layer_path or "unknown") for entry in layers):
            layers.append(
                {
                    "layer": layer_path or "unknown",
                    "mean_norm_entropy": mean_norm,
                }
            )

        if not emission.get("emit"):
            return {"status": "skipped", "reason": "emission_controller"}

        emitted = []
        base_kwargs = {"layer_path": layer_path, "emit_mode": config["emit_mode"]}

        if _SENDER and config["emit_scalar"]:
            _SENDER.send_scalar(
                round(mean_norm, 5),
                label="Mean normalized attention entropy",
                emit_id=f"attention_entropy_profiler:{layer_path}:scalar",
                **base_kwargs,
            )
            emitted.append("scalar")

        if _SENDER and config["emit_table"]:
            headers = ["Head", "Entropy", "Normalized entropy", "Max attention weight"]
            rows = [
                [head, round(raw, 5), round(norm, 5), round(peak, 5)]
                for head, (raw, norm, peak) in enumerate(
                    zip(raw_list, norm_list, max_list)
                )
            ]
            _SENDER.send_table(
                headers,
                rows,
                emit_id=f"attention_entropy_profiler:{layer_path}:table",
                **base_kwargs,
            )
            emitted.append("table")

        if _SENDER and config["emit_chart"] and len(layers) >= 2:
            points = [
                {"x": idx, "y": round(entry["mean_norm_entropy"], 5)}
                for idx, entry in enumerate(layers)
            ]
            _SENDER.send_chart(
                {
                    "series": [
                        {"name": "Mean normalized entropy", "points": points}
                    ],
                    "x_label": "Attention layer index",
                    "y_label": "Normalized entropy",
                },
                "line",
                emit_id=f"attention_entropy_profiler:{layer_path}:chart",
                **base_kwargs,
            )
            emitted.append("chart")

        if config["emit_mode"] == "final":
            try:
                ltts_event.mark_emission_finalized(METADATA["name"], layer_path)
            except Exception:
                logger.debug(
                    "ATTENTION_ENTROPY_PROFILER: mark_emission_finalized failed",
                    exc_info=True,
                )

        return {
            "status": "ok",
            "emitted": emitted,
            "mean_normalized_entropy": mean_norm,
            "heads": len(norm_list),
        }

    except Exception as exc:
        logger.error("ATTENTION_ENTROPY_PROFILER: error: %s", exc, exc_info=True)
        try:
            ltts_event.reactor.signal(
                "error",
                {
                    "message": f"attention_entropy_profiler error: {exc}",
                    "module": METADATA["name"],
                },
            )
        except Exception:
            pass
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
