"""
Representation Similarity Module

Measures how similar the representations of different layers are across one
forward pass using linear Centered Kernel Alignment (CKA, Kornblith et al.,
2019). At every decoder-block boundary the hidden states of the last positions
are accumulated; once at least two layers are available, the pairwise linear
CKA matrix between all accumulated layers is computed:

    CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

on column-centered matrices (rows = positions, cols = hidden dims).

Emits:
- optionally a layer-by-layer CKA heatmap image, re-emitted with a stable
  emit_id so the UI replaces it as more layers arrive (the last emission wins)
- optionally a scalar: CKA between the first and last accumulated layer
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "representation_similarity",
    "version": "0.1.0",
    "description": "Layer-by-layer representation similarity (linear CKA) across the forward pass, rendered as a heatmap",
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
    "max_positions": 64,  # last min(seq, max_positions) positions are used
    "emit_heatmap": True,
    "emit_scalar": False,
    "emit_mode": "final",
}

_SENDER = None

# Cross-layer accumulator: each layer event gets a fresh LTTSContext, so the
# per-depth trajectory must live in module state, keyed by forward pass.
_TRAJECTORY: Dict[str, Any] = {"forward_pass": None, "entries": []}


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    logger.info("REPRESENTATION_SIMILARITY: initialized with config: %s", config)
    return {"status": "initialized", "config": config}


def cleanup_module():
    logger.info("REPRESENTATION_SIMILARITY: cleaned up")
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "max_positions",
                "type": "number",
                "label": "Max sequence positions used per layer",
                "default": DEFAULT_CONFIG["max_positions"],
                "min": 8,
                "max": 512,
            },
            {
                "name": "emit_heatmap",
                "type": "boolean",
                "label": "Emit layer-by-layer CKA heatmap",
                "default": DEFAULT_CONFIG["emit_heatmap"],
            },
            {
                "name": "emit_scalar",
                "type": "boolean",
                "label": "Emit first-vs-last layer CKA scalar",
                "default": DEFAULT_CONFIG["emit_scalar"],
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

    return {
        "max_positions": _coerce_int(
            config.get("max_positions"), DEFAULT_CONFIG["max_positions"], 8, 512
        ),
        "emit_heatmap": _coerce_bool(
            config.get("emit_heatmap"), DEFAULT_CONFIG["emit_heatmap"]
        ),
        "emit_scalar": _coerce_bool(
            config.get("emit_scalar"), DEFAULT_CONFIG["emit_scalar"]
        ),
        "emit_mode": str(config.get("emit_mode", DEFAULT_CONFIG["emit_mode"])),
        "emit_every_n": config.get("emit_every_n", 1),
    }


def _extract_hidden_state(outputs: Any) -> Optional[torch.Tensor]:
    """Layer outputs may be a tensor or a tuple whose first element is the hidden state."""
    candidate = outputs
    if isinstance(candidate, (list, tuple)) and len(candidate) > 0:
        candidate = candidate[0]
    if isinstance(candidate, torch.Tensor) and candidate.dim() == 3:
        return candidate
    return None


def _layer_depth_key(layer_path: str) -> str:
    return layer_path or "unknown"


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two [positions, hidden] matrices (column-centered)."""
    rows = min(x.shape[0], y.shape[0])
    x = x[:rows].astype(np.float64)
    y = y[:rows].astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = float(np.linalg.norm(x.T @ y, ord="fro") ** 2)
    self_x = float(np.linalg.norm(x.T @ x, ord="fro"))
    self_y = float(np.linalg.norm(y.T @ y, ord="fro"))
    denom = self_x * self_y
    if denom <= 0.0:
        return 0.0
    return cross / denom


def _compute_cka_matrix(entries: List[Dict[str, Any]]) -> np.ndarray:
    n = len(entries)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            value = _linear_cka(entries[i]["states"], entries[j]["states"])
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def _render_cka_heatmap(matrix: np.ndarray) -> str:
    """Render the CKA matrix to a base64 PNG data URI (Agg-safe, always closed)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
        fig.colorbar(image, ax=ax, label="Linear CKA")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        ax.set_title("Layer-by-layer representation similarity (linear CKA)")
        buffer = io.BytesIO()
        fig.savefig(
            buffer, format="png", bbox_inches="tight", dpi=150, facecolor="white"
        )
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{image_base64}"
    finally:
        plt.close(fig)


def process_event(ltts_event):
    try:
        evt = getattr(ltts_event, "event_type", "") or ""
        if not (evt == "layer_after" or evt.startswith("layer_after_")):
            return {"status": "skipped", "reason": "event_phase"}

        context = ltts_event.context
        # Compare representations only at decoder-block boundaries: attention
        # or MLP sublayer outputs are partial residual updates. Blocks are typed
        # "unknown" by the layer classifier, so confirm via the class name.
        module_class = (context.module_class or "").lower()
        is_block = "decoderlayer" in module_class or module_class.endswith(
            ("block", "layer")
        )
        if getattr(context, "module_role", None) != "block" and (
            context.layer_type != "unknown" or not is_block
        ):
            return {"status": "skipped", "reason": "not_a_block_boundary"}

        hidden = _extract_hidden_state(context.outputs)
        if hidden is None:
            return {"status": "skipped", "reason": "no_hidden_state"}

        if not isinstance(ltts_event.module_state, dict):
            ltts_event.module_state = {}
        config = _resolve_config(ltts_event)
        layer_path = getattr(context, "module_path", None)

        emission = ltts_event.should_emit(METADATA["name"], layer_path, config)
        if not emission.get("emit"):
            return {"status": "skipped", "reason": "emission_controller"}

        with torch.no_grad():
            seq_len = hidden.shape[1]
            positions = min(seq_len, config["max_positions"])
            states = (
                hidden[0, seq_len - positions :, :]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        current_pass = emission.get("forward_pass")
        if _TRAJECTORY["forward_pass"] != current_pass:
            _TRAJECTORY["forward_pass"] = current_pass
            _TRAJECTORY["entries"] = []
        _TRAJECTORY["entries"].append(
            {"layer": _layer_depth_key(layer_path), "states": states}
        )
        trajectory = _TRAJECTORY["entries"]

        emitted = []
        first_last_cka: Optional[float] = None
        base_kwargs = {"layer_path": layer_path, "emit_mode": config["emit_mode"]}

        if len(trajectory) >= 2 and (config["emit_heatmap"] or config["emit_scalar"]):
            matrix = _compute_cka_matrix(trajectory)
            first_last_cka = float(matrix[0, -1])

            if _SENDER and config["emit_heatmap"]:
                # Stable emit_id: each decoder block re-emits the (growing)
                # matrix under the same id, so the UI replaces rather than
                # stacks and the last layer's emission ends up displayed.
                _SENDER.send_image(
                    _render_cka_heatmap(matrix),
                    emit_id="representation_similarity:cka",
                    **base_kwargs,
                )
                emitted.append("heatmap")

            if _SENDER and config["emit_scalar"]:
                _SENDER.send_scalar(
                    round(first_last_cka, 5),
                    label="CKA: first vs last accumulated layer",
                    emit_id="representation_similarity:cka_scalar",
                    **base_kwargs,
                )
                emitted.append("scalar")

        if config["emit_mode"] == "final":
            try:
                ltts_event.mark_emission_finalized(METADATA["name"], layer_path)
            except Exception:
                logger.debug(
                    "REPRESENTATION_SIMILARITY: mark_emission_finalized failed",
                    exc_info=True,
                )

        return {
            "status": "ok",
            "emitted": emitted,
            "layers_accumulated": len(trajectory),
            "first_last_cka": first_last_cka,
        }

    except Exception as exc:
        logger.error("REPRESENTATION_SIMILARITY: error: %s", exc, exc_info=True)
        try:
            ltts_event.reactor.signal(
                "error",
                {
                    "message": f"representation_similarity error: {exc}",
                    "module": METADATA["name"],
                },
            )
        except Exception:
            pass
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
