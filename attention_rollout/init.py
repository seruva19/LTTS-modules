"""
Attention Rollout Module

Implements attention rollout (Abnar & Zuidema, 2020): per attention layer the
head-fused attention matrix is mixed with the identity (residual connection),
row-normalized, and multiplied into the running rollout product. The rollout
matrix approximates token-to-token influence through the network up to the
current layer.

Emits per attention layer:
- a heatmap image of the current rollout matrix
- optionally a table of the top-k source tokens influencing the last position
"""

from __future__ import annotations

import io
import base64
import logging
from typing import Any, Dict, List, Optional

import torch
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - degrade visibly, not silently
    plt = None

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "attention_rollout",
    "version": "0.1.0",
    "description": "Attention rollout (Abnar & Zuidema 2020): cumulative token-to-token influence matrices across attention layers",
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
    "head_fusion": "mean",
    "add_residual": True,
    "emit_heatmap": True,
    "emit_top_sources_table": True,
    "top_k": 5,
    "emit_mode": "final",
}

_SENDER = None

# Cross-layer accumulator: each layer event gets a fresh LTTSContext, so the
# running rollout product must live in module state, keyed by forward pass
# (same pattern as logit_lens._TRAJECTORY).
_ROLLOUT: Dict[str, Any] = {"forward_pass": None, "matrix": None, "layer_count": 0}


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    logger.info("ATTENTION_ROLLOUT: initialized with config: %s", config)
    return {"status": "initialized", "config": config}


def cleanup_module():
    _ROLLOUT["forward_pass"] = None
    _ROLLOUT["matrix"] = None
    _ROLLOUT["layer_count"] = 0
    logger.info("ATTENTION_ROLLOUT: cleaned up")
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "head_fusion",
                "type": "select",
                "label": "Head fusion",
                "default": DEFAULT_CONFIG["head_fusion"],
                "options": [
                    {"value": "mean", "label": "Mean"},
                    {"value": "max", "label": "Max"},
                    {"value": "min", "label": "Min"},
                ],
            },
            {
                "name": "add_residual",
                "type": "boolean",
                "label": "Add residual (0.5*A + 0.5*I)",
                "default": DEFAULT_CONFIG["add_residual"],
            },
            {
                "name": "emit_heatmap",
                "type": "boolean",
                "label": "Emit rollout heatmap image",
                "default": DEFAULT_CONFIG["emit_heatmap"],
            },
            {
                "name": "emit_top_sources_table",
                "type": "boolean",
                "label": "Emit top-k source tokens for last position",
                "default": DEFAULT_CONFIG["emit_top_sources_table"],
            },
            {
                "name": "top_k",
                "type": "number",
                "label": "Top K source tokens",
                "default": DEFAULT_CONFIG["top_k"],
                "min": 1,
                "max": 20,
            },
        ],
        "layout": {"type": "vertical"},
        "real_time": {"enabled": True, "interval": 200},
    }


def _coerce_int(value: Any, default: int, minimum: int, maximum: Optional[int] = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    if result < minimum:
        result = minimum
    if maximum is not None and result > maximum:
        result = maximum
    return result


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

    head_fusion = str(config.get("head_fusion", "mean")).strip().lower()
    if head_fusion not in {"mean", "max", "min"}:
        head_fusion = "mean"

    return {
        "head_fusion": head_fusion,
        "add_residual": _coerce_bool(
            config.get("add_residual"), DEFAULT_CONFIG["add_residual"]
        ),
        "emit_heatmap": _coerce_bool(
            config.get("emit_heatmap"), DEFAULT_CONFIG["emit_heatmap"]
        ),
        "emit_top_sources_table": _coerce_bool(
            config.get("emit_top_sources_table"),
            DEFAULT_CONFIG["emit_top_sources_table"],
        ),
        "top_k": _coerce_int(config.get("top_k"), DEFAULT_CONFIG["top_k"], 1, 20),
        "emit_mode": str(config.get("emit_mode", DEFAULT_CONFIG["emit_mode"])),
        "emit_every_n": config.get("emit_every_n", 1),
    }


def _candidate_attention_keys(module_path: str) -> List[str]:
    """Build candidate captured_attn keys, mirroring attn_visualizer."""
    import re

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
    """Locate attention weights for the current layer, mirroring attn_visualizer."""
    try:
        context = ltts_event.context
        ltts_model = getattr(context, "ltts_model", None)
        module_path = (getattr(context, "module_path", "") or "").strip()
        candidate_keys = _candidate_attention_keys(module_path)

        # 1) Dynamically captured attention per-layer (preferred)
        if ltts_model is not None and hasattr(ltts_model, "captured_attn"):
            captured = ltts_model.captured_attn or {}
            for key in candidate_keys:
                if key in captured and captured[key]:
                    latest_step = max(captured[key].keys())
                    return captured[key][latest_step]

        # 2) Fallback per-layer weights
        if ltts_model is not None and hasattr(ltts_model, "layer_attention_weights"):
            law = ltts_model.layer_attention_weights or {}
            for key in candidate_keys:
                if key in law:
                    return law[key]

        # 3) Direct context injection
        attn = getattr(context, "attention_weights", None)
        if isinstance(attn, torch.Tensor):
            return attn

        # 4) Fallback to extracting from current outputs
        try:
            from miners.utils import extract_attention_weights

            return extract_attention_weights(getattr(context, "outputs", None))
        except Exception:
            return None
    except Exception as exc:
        logger.warning("ATTENTION_ROLLOUT: error getting attention weights: %s", exc)
        return None


def _fuse_heads(attention: torch.Tensor, head_fusion: str) -> Optional[np.ndarray]:
    """Reduce attention weights to a fused [seq, seq] numpy matrix."""
    tensor = attention.detach().float().cpu()
    if tensor.dim() == 4:  # [batch, heads, seq, seq]
        tensor = tensor[0]
    if tensor.dim() == 3:  # [heads, seq, seq]
        if head_fusion == "max":
            tensor = tensor.max(dim=0).values
        elif head_fusion == "min":
            tensor = tensor.min(dim=0).values
        else:
            tensor = tensor.mean(dim=0)
    if tensor.dim() != 2:
        return None
    matrix = tensor.numpy()
    if matrix.shape[0] != matrix.shape[1]:
        # Rollout requires square (full-sequence) attention matrices; decode-step
        # slices with q_len != k_len cannot be chained.
        return None
    return matrix


def _get_token_labels(ltts_model, seq_len: int) -> Optional[List[str]]:
    try:
        if ltts_model is None:
            return None
        tokens = getattr(ltts_model, "_last_input_tokens", None)
        if isinstance(tokens, (list, tuple)) and tokens:
            labels = [str(t) for t in tokens]
        else:
            tokenizer = getattr(ltts_model, "_tokenizer", None)
            ids = getattr(ltts_model, "_last_input_ids", None)
            if tokenizer is None or not isinstance(ids, (list, tuple)) or not ids:
                return None
            labels = []
            for tid in ids:
                try:
                    decoded = tokenizer.decode([int(tid)], skip_special_tokens=False)
                    labels.append(decoded.strip() or f"<{tid}>")
                except Exception:
                    labels.append(f"<{tid}>")
        if len(labels) < seq_len:
            labels = labels + [f"pos {i}" for i in range(len(labels), seq_len)]
        return labels[:seq_len]
    except Exception:
        return None


def _save_plot_as_base64() -> str:
    """Render the current matplotlib figure as a PNG base64 data URI."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=150, facecolor="white")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{image_base64}"


def _render_rollout_heatmap(
    rollout: np.ndarray, layer_count: int, labels: Optional[List[str]]
) -> Optional[str]:
    if plt is None:
        return None
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(rollout, cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_title(f"Attention rollout after {layer_count} attention layer(s)")
        seq_len = rollout.shape[-1]
        if labels and seq_len <= 48:
            positions = list(range(seq_len))
            ax.set_xticks(positions)
            ax.set_yticks(positions)
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_yticklabels(labels, rotation=0, fontsize=7)
            ax.set_xlabel("Source tokens")
            ax.set_ylabel("Target tokens")
        else:
            ax.set_xlabel("Source position")
            ax.set_ylabel("Target position")
        return _save_plot_as_base64()
    except Exception as exc:
        logger.warning("ATTENTION_ROLLOUT: heatmap render failed: %s", exc)
        try:
            plt.close()
        except Exception:
            pass
        return None


def process_event(ltts_event):
    try:
        evt = getattr(ltts_event, "event_type", "") or ""
        if not (evt == "layer_after" or evt.startswith("layer_after_")):
            return {"status": "skipped", "reason": "event_phase"}

        context = ltts_event.context
        if getattr(context, "layer_type", None) != "attention":
            return {"status": "skipped", "reason": "not_attention_layer"}

        if not isinstance(ltts_event.module_state, dict):
            ltts_event.module_state = {}
        config = _resolve_config(ltts_event)
        layer_path = getattr(context, "module_path", None)

        emission = ltts_event.should_emit(METADATA["name"], layer_path, config)

        attention = _get_layer_attention_weights(ltts_event)
        if attention is None or not isinstance(attention, torch.Tensor):
            return {"status": "skipped", "reason": "no_attention_weights"}

        with torch.no_grad():
            fused = _fuse_heads(attention, config["head_fusion"])
        if fused is None:
            return {"status": "skipped", "reason": "non_square_or_bad_shape"}

        seq_len = fused.shape[-1]

        # Reset the accumulator when a new forward pass begins. Accumulation
        # always runs (even when emission is throttled) so the rollout product
        # never misses a layer.
        current_pass = emission.get("forward_pass")
        if _ROLLOUT["forward_pass"] != current_pass:
            _ROLLOUT["forward_pass"] = current_pass
            _ROLLOUT["matrix"] = None
            _ROLLOUT["layer_count"] = 0

        layer_attn = fused
        if config["add_residual"]:
            layer_attn = 0.5 * layer_attn + 0.5 * np.eye(seq_len, dtype=layer_attn.dtype)
        row_sums = layer_attn.sum(axis=-1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        layer_attn = layer_attn / row_sums

        previous = _ROLLOUT["matrix"]
        if previous is None or previous.shape[-1] != seq_len:
            previous = np.eye(seq_len, dtype=layer_attn.dtype)
            _ROLLOUT["layer_count"] = 0
        rollout = layer_attn @ previous
        _ROLLOUT["matrix"] = rollout
        _ROLLOUT["layer_count"] += 1
        layer_count = _ROLLOUT["layer_count"]

        if not emission.get("emit"):
            return {"status": "skipped", "reason": "emission_controller"}

        ltts_model = getattr(context, "ltts_model", None)
        labels = _get_token_labels(ltts_model, seq_len)

        emitted = []
        base_kwargs = {"layer_path": layer_path, "emit_mode": config["emit_mode"]}

        if _SENDER and config["emit_heatmap"]:
            image = _render_rollout_heatmap(rollout, layer_count, labels)
            if image:
                _SENDER.send_image(
                    image,
                    emit_id=f"attention_rollout:{layer_path}:heatmap",
                    **base_kwargs,
                )
                emitted.append("heatmap")
            elif plt is None:
                _SENDER.send_text(
                    "attention_rollout: matplotlib is not available; install it "
                    "(see ltts_modules/attention_rollout/requirements.txt) to "
                    "render rollout heatmaps.",
                    emit_id=f"attention_rollout:{layer_path}:no_matplotlib",
                    **base_kwargs,
                )
                emitted.append("matplotlib_warning")

        if _SENDER and config["emit_top_sources_table"]:
            last_row = rollout[-1]
            k = min(config["top_k"], seq_len)
            top_indices = np.argsort(last_row)[::-1][:k]
            headers = ["Rank", "Source position", "Source token", "Rollout influence"]
            rows = []
            for rank, idx in enumerate(top_indices):
                idx = int(idx)
                label = labels[idx] if labels and idx < len(labels) else f"pos {idx}"
                rows.append([rank + 1, idx, label, round(float(last_row[idx]), 5)])
            _SENDER.send_table(
                headers,
                rows,
                emit_id=f"attention_rollout:{layer_path}:top_sources",
                **base_kwargs,
            )
            emitted.append("top_sources_table")

        if config["emit_mode"] == "final":
            try:
                ltts_event.mark_emission_finalized(METADATA["name"], layer_path)
            except Exception:
                logger.debug(
                    "ATTENTION_ROLLOUT: mark_emission_finalized failed", exc_info=True
                )

        return {
            "status": "ok",
            "emitted": emitted,
            "layer_count": layer_count,
            "seq_len": seq_len,
        }

    except Exception as exc:
        logger.error("ATTENTION_ROLLOUT: error: %s", exc, exc_info=True)
        try:
            ltts_event.reactor.signal(
                "error",
                {
                    "message": f"attention_rollout error: {exc}",
                    "module": METADATA["name"],
                },
            )
        except Exception:
            pass
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
