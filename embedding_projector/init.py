"""
Embedding Projector Module

Projects per-token hidden states of a chosen decoder layer down to 2D
(PCA or t-SNE) and renders a scatter plot annotated with token labels,
showing how the model arranges the sequence in representation space at
that depth.

Hidden states are collected at decoder-block boundaries (same detection
as logit_lens) into a module-global accumulator keyed by forward pass,
so a specific layer index can be targeted, or -1 to follow the deepest
layer seen so far.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "embedding_projector",
    "version": "0.1.0",
    "description": "Project per-token hidden states of a chosen layer to 2D (PCA / t-SNE) and plot the token cloud",
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
    "target_layer_index": -1,  # -1 = deepest decoder block seen so far
    "method": "pca",
    "normalize_vectors": True,
    "emit_image": True,
    "emit_mode": "all",
}

_SENDER = None

# Cross-layer accumulator: each layer event gets a fresh LTTSContext, so the
# per-depth hidden states must live in module state, keyed by forward pass.
_HIDDEN_STATES: Dict[str, Any] = {"forward_pass": None, "layers": []}


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    logger.info("EMBEDDING_PROJECTOR: initialized with config: %s", config)
    return {"status": "initialized", "config": config}


def cleanup_module():
    _HIDDEN_STATES["forward_pass"] = None
    _HIDDEN_STATES["layers"] = []
    logger.info("EMBEDDING_PROJECTOR: cleaned up")
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "target_layer_index",
                "type": "number",
                "label": "Target layer index (-1 = deepest seen so far)",
                "default": DEFAULT_CONFIG["target_layer_index"],
                "min": -1,
            },
            {
                "name": "method",
                "type": "select",
                "label": "Projection method",
                "default": DEFAULT_CONFIG["method"],
                "options": [
                    {"value": "pca", "label": "PCA"},
                    {"value": "tsne", "label": "t-SNE (needs scikit-learn)"},
                ],
            },
            {
                "name": "normalize_vectors",
                "type": "boolean",
                "label": "L2-normalize token vectors",
                "default": DEFAULT_CONFIG["normalize_vectors"],
                "description": "Prevents token-norm outliers from collapsing the projection scale.",
            },
            {
                "name": "emit_image",
                "type": "boolean",
                "label": "Emit scatter plot image",
                "default": DEFAULT_CONFIG["emit_image"],
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

    method = str(config.get("method", "pca")).strip().lower()
    if method not in {"pca", "tsne"}:
        method = "pca"

    return {
        "target_layer_index": _coerce_int(config.get("target_layer_index"), -1, -1),
        "method": method,
        "normalize_vectors": _coerce_bool(
            config.get("normalize_vectors"), DEFAULT_CONFIG["normalize_vectors"]
        ),
        "emit_image": _coerce_bool(config.get("emit_image"), True),
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


def _get_token_labels(ltts_event, seq_len: int) -> List[str]:
    """Token labels from the live model; fall back to position indices."""
    try:
        lm = getattr(ltts_event.context, "ltts_model", None)
        if lm is not None:
            tokens = getattr(lm, "_last_input_tokens", None)
            if isinstance(tokens, (list, tuple)) and tokens:
                labels = [str(t) for t in list(tokens)[:seq_len]]
                if len(labels) == seq_len:
                    return labels
            tokenizer = getattr(lm, "_tokenizer", None)
            inp = getattr(ltts_event.context, "inputs", None)
            if tokenizer is not None and isinstance(inp, dict) and "input_ids" in inp:
                ids = inp["input_ids"][0].detach().cpu().tolist()
                labels = [str(t) for t in tokenizer.convert_ids_to_tokens(ids)[:seq_len]]
                if len(labels) == seq_len:
                    return labels
    except Exception:
        logger.debug("EMBEDDING_PROJECTOR: token label extraction failed", exc_info=True)
    return [str(i) for i in range(seq_len)]


def _project_pca(states: np.ndarray) -> np.ndarray:
    """Center + top-2 SVD components. Pure numpy, no sklearn."""
    centered = states - states.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(2, vt.shape[0])
    coords = centered @ vt[:n_components].T
    if coords.shape[1] < 2:
        coords = np.concatenate(
            [coords, np.zeros((coords.shape[0], 2 - coords.shape[1]), dtype=coords.dtype)],
            axis=1,
        )
    return coords.astype(np.float32)


def _normalize_token_vectors(states: np.ndarray) -> np.ndarray:
    """Project token directions instead of letting vector norms dominate PCA."""
    norms = np.linalg.norm(states, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, np.finfo(np.float32).eps)
    return (states / safe_norms).astype(np.float32)


def _project(states: np.ndarray, method: str) -> tuple:
    """Project [seq, hidden] -> [seq, 2]. Returns (coords, method_actually_used)."""
    if method == "tsne":
        n = states.shape[0]
        if n < 4:
            logger.warning(
                "EMBEDDING_PROJECTOR: t-SNE needs at least 4 points (got %d); falling back to PCA",
                n,
            )
            return _project_pca(states), "pca"
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            logger.warning(
                "EMBEDDING_PROJECTOR: scikit-learn not available; falling back to PCA"
            )
            return _project_pca(states), "pca"
        try:
            perplexity = min(30.0, max(2.0, (n - 1) / 3.0))
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                random_state=0,
            )
            return tsne.fit_transform(states).astype(np.float32), "tsne"
        except Exception as exc:
            logger.warning(
                "EMBEDDING_PROJECTOR: t-SNE failed (%s); falling back to PCA", exc
            )
            return _project_pca(states), "pca"
    return _project_pca(states), "pca"


def _render_scatter(coords: np.ndarray, labels: List[str], title: str) -> Optional[str]:
    """Render scatter plot to a base64 PNG data URI. Always closes the figure."""
    if not MATPLOTLIB_AVAILABLE:
        return None
    import base64
    import io

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(coords[:, 0], coords[:, 1], s=28, c="#3b6fd4", alpha=0.8)
        for idx in range(coords.shape[0]):
            label = labels[idx] if idx < len(labels) else str(idx)
            ax.annotate(
                label,
                (coords[idx, 0], coords[idx, 1]),
                fontsize=7,
                alpha=0.85,
                xytext=(3, 3),
                textcoords="offset points",
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150, facecolor="white")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{image_base64}"
    finally:
        plt.close("all")


def process_event(ltts_event):
    try:
        evt = getattr(ltts_event, "event_type", "") or ""
        if not (evt == "layer_after" or evt.startswith("layer_after_")):
            return {"status": "skipped", "reason": "event_phase"}

        context = ltts_event.context
        # Project the residual stream only at decoder-block boundaries:
        # attention or MLP sublayer outputs are partial residual updates.
        # Blocks are typed "unknown" by the layer classifier, so confirm via
        # the class name (same detection as logit_lens).
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
            states = hidden[0].detach().float().cpu().numpy().astype(np.float32)
        if states.ndim != 2 or states.shape[0] < 2:
            return {"status": "skipped", "reason": "sequence_too_short"}

        # Accumulate per-layer hidden states for this forward pass.
        current_pass = emission.get("forward_pass")
        if _HIDDEN_STATES["forward_pass"] != current_pass:
            _HIDDEN_STATES["forward_pass"] = current_pass
            _HIDDEN_STATES["layers"] = []
        _HIDDEN_STATES["layers"].append({"path": layer_path, "states": states})
        layer_index = len(_HIDDEN_STATES["layers"]) - 1

        target = config["target_layer_index"]
        if target >= 0 and layer_index != target:
            return {"status": "skipped", "reason": "not_target_layer"}

        method = config["method"]
        projection_states = (
            _normalize_token_vectors(states)
            if config["normalize_vectors"]
            else states
        )
        coords, method_used = _project(projection_states, method)
        labels = _get_token_labels(ltts_event, states.shape[0])
        geometry = "cosine" if config["normalize_vectors"] else "raw"
        title = (
            f"{layer_path or 'unknown'} - {method_used.upper()} projection "
            f"({geometry} geometry)"
        )

        # Stable emit_id: with target -1 each deeper layer replaces the previous
        # projection instead of stacking new cards.
        if target >= 0:
            emit_id = f"embedding_projector:layer_{target}:{method_used}"
        else:
            emit_id = f"embedding_projector:deepest:{method_used}"

        emitted = []
        base_kwargs = {"layer_path": layer_path, "emit_mode": config["emit_mode"]}

        if _SENDER:
            if config["emit_image"] and MATPLOTLIB_AVAILABLE:
                image_uri = _render_scatter(coords, labels, title)
                if image_uri:
                    _SENDER.send_image(image_uri, emit_id=emit_id, **base_kwargs)
                    emitted.append("image")
            else:
                if config["emit_image"] and not MATPLOTLIB_AVAILABLE:
                    logger.warning(
                        "EMBEDDING_PROJECTOR: matplotlib not available; emitting raw coordinates"
                    )
                _SENDER.send_visualization(
                    {
                        "type": "json",
                        "data": {
                            "title": title,
                            "method": method_used,
                            "points": [
                                {
                                    "token": labels[i] if i < len(labels) else str(i),
                                    "x": round(float(coords[i, 0]), 5),
                                    "y": round(float(coords[i, 1]), 5),
                                }
                                for i in range(coords.shape[0])
                            ],
                        },
                    },
                    "json",
                    emit_id=emit_id,
                    **base_kwargs,
                )
                emitted.append("json")

        if config["emit_mode"] == "final":
            try:
                ltts_event.mark_emission_finalized(METADATA["name"], layer_path)
            except Exception:
                logger.debug(
                    "EMBEDDING_PROJECTOR: mark_emission_finalized failed", exc_info=True
                )

        return {
            "status": "ok",
            "emitted": emitted,
            "layer_index": layer_index,
            "method": method_used,
            "normalized": config["normalize_vectors"],
            "num_tokens": int(states.shape[0]),
        }

    except Exception as exc:
        logger.error("EMBEDDING_PROJECTOR: error: %s", exc, exc_info=True)
        try:
            ltts_event.reactor.signal(
                "error",
                {
                    "message": f"embedding_projector error: {exc}",
                    "module": METADATA["name"],
                },
            )
        except Exception:
            pass
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
