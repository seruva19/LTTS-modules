"""
Lookback Lens Module

Measures the Lookback Ratio (context attention vs. newly generated token
attention) per head and layer (Chuang et al., arXiv:2407.07071).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import torch

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "lookback_lens",
    "version": "0.1.0",
    "description": "Measure Lookback Ratio (context vs. newly generated token attention) across heads and layers",
    "author": "LTTS",
    "event_types": ["layer_after"],
    "dependencies": ["requirements.txt"],
    "requires": ["attention_weights"],
    "target_roles": ["attention"],
    "methods": {
        "ctor": "initialize_module",
        "dtor": "cleanup_module",
        "ntor": "process_event",
        "utor": "get_ui_schema",
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "context_tokens": 0,
    "context_fraction": 0.5,
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
                "name": "context_tokens",
                "type": "number",
                "label": "Context prefix length (0 = auto)",
                "default": DEFAULT_CONFIG["context_tokens"],
                "min": 0,
                "max": 4096,
            },
            {
                "name": "context_fraction",
                "type": "number",
                "label": "Auto context fraction",
                "default": DEFAULT_CONFIG["context_fraction"],
                "min": 0.1,
                "max": 0.9,
            },
            {
                "name": "emit_scalar",
                "type": "boolean",
                "label": "Emit mean lookback ratio (scalar)",
                "default": DEFAULT_CONFIG["emit_scalar"],
            },
            {
                "name": "emit_table",
                "type": "boolean",
                "label": "Emit per-head lookback table",
                "default": DEFAULT_CONFIG["emit_table"],
            },
            {
                "name": "emit_chart",
                "type": "boolean",
                "label": "Emit lookback-vs-depth chart",
                "default": DEFAULT_CONFIG["emit_chart"],
            },
            {
                "name": "emit_heatmap",
                "type": "boolean",
                "label": "Emit layer x head lookback heatmap",
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
        context_tokens = max(0, min(4096, int(config.get("context_tokens", 0))))
    except (TypeError, ValueError):
        context_tokens = 0

    try:
        context_fraction = max(0.1, min(0.9, float(config.get("context_fraction", 0.5))))
    except (TypeError, ValueError):
        context_fraction = 0.5

    return {
        "context_tokens": context_tokens,
        "context_fraction": context_fraction,
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
        logger.warning("LOOKBACK_LENS: error getting attention: %s", exc)
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
            if key_len < 2 or num_queries < 2:
                return {"status": "skipped", "reason": "sequence_too_short"}

            ctx_len = config["context_tokens"]
            if 0 < ctx_len < key_len:
                split_idx = ctx_len
            else:
                split_idx = max(1, min(key_len - 1, int(round(key_len * config["context_fraction"]))))

            q_start = min(split_idx, num_queries - 1)
            lr_steps = []
            ctx_mass_steps = []
            new_mass_steps = []
            for t in range(q_start, num_queries):
                a_ctx = attn[:, t, :split_idx].mean(dim=-1)
                end_new = min(t + 1, key_len)
                if end_new > split_idx:
                    a_new = attn[:, t, split_idx:end_new].mean(dim=-1)
                else:
                    a_new = attn[:, t, split_idx:].mean(dim=-1)
                lr_steps.append(a_ctx / (a_ctx + a_new + 1e-9))
                ctx_mass_steps.append(attn[:, t, :split_idx].sum(dim=-1))
                new_mass_steps.append(attn[:, t, split_idx:].sum(dim=-1))

            lookback_ratios = [float(v) for v in torch.stack(lr_steps, dim=0).mean(dim=0).tolist()]
            ctx_masses = [float(v) for v in torch.stack(ctx_mass_steps, dim=0).mean(dim=0).tolist()]
            new_masses = [float(v) for v in torch.stack(new_mass_steps, dim=0).mean(dim=0).tolist()]

        mean_lr = sum(lookback_ratios) / len(lookback_ratios) if lookback_ratios else 0.0

        current_pass = emission.get("forward_pass")
        if _PROFILE["forward_pass"] != current_pass:
            _PROFILE["forward_pass"] = current_pass
            _PROFILE["layers"] = []

        layers = _PROFILE["layers"]
        entry = {
            "layer_path": layer_path,
            "mean_lookback": mean_lr,
            "lookback_ratios": lookback_ratios,
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
                round(mean_lr, 4),
                label=f"Mean lookback ratio (ctx={split_idx}/{key_len})",
                unit="ratio",
                layer_path=layer_path,
                emit_id=f"{name}:{layer_path}:scalar",
            )
            emitted.append("scalar")

        if config["emit_table"]:
            rows = [
                [
                    head_idx,
                    round(lookback_ratios[head_idx], 4),
                    round(ctx_masses[head_idx], 4),
                    round(new_masses[head_idx], 4),
                ]
                for head_idx in range(num_heads)
            ]
            _SENDER.send_table(
                ["Head", "Lookback Ratio", "Context Mass", "New-Token Mass"],
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
                                "name": "Mean Lookback Ratio",
                                "points": [
                                    {"x": idx, "y": round(item["mean_lookback"], 4)}
                                    for idx, item in enumerate(layers)
                                ],
                            }
                        ],
                        "x_label": "Attention layer",
                        "y_label": "Lookback Ratio",
                    },
                    "line",
                    layer_path=layer_path,
                    emit_id=f"{name}:chart",
                )
                emitted.append("chart")

            if config["emit_heatmap"]:
                cols = len(layers[0]["lookback_ratios"])
                if cols > 0 and all(len(item["lookback_ratios"]) == cols for item in layers):
                    matrix = [
                        [round(val, 4) for val in item["lookback_ratios"]]
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
            "mean_lookback_ratio": mean_lr,
            "context_split": split_idx,
            "emitted": emitted,
        }
    except Exception as exc:
        logger.exception("LOOKBACK_LENS: error processing event")
        reactor = getattr(ltts_event, "reactor", None)
        if reactor is not None and hasattr(reactor, "signal"):
            reactor.signal("error", {"message": str(exc), "module": METADATA["name"]})
        return {"status": "error", "error": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
