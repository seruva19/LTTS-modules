"""
Attention Head Grid Module

Renders every attention head of a layer as a small heatmap in a grid, so head
specialization is visible at a glance — diagonal (self), sub-diagonal
(previous-token), first-column (BOS / attention-sink), and the off-diagonal
stripes of induction heads. `attn_visualizer` shows one head (or the average);
this shows the whole "zoo" of a layer at once, which is the single most useful
view for *learning* what attention heads do.

Attaches to any layer card: it resolves the block's captured attention the same
way `attn_visualizer` does, so attaching to the attention or post-attention-norm
card both work. On layers with no attention it renders nothing.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.core.module_data_sender import get_data_sender  # noqa: E402

logger = logging.getLogger(__name__)

METADATA = {
    "name": "attention_head_grid",
    "version": "0.1.0",
    "description": "Grid of per-head attention heatmaps for a layer - see head specialization (diagonal / previous-token / sink / induction) at a glance",
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
    "max_heads": 16,
    "max_tokens": 40,
    "show_token_labels": True,
    "colormap": "viridis",
    "emit_mode": "final",
    "emit_every_n": 1,
}

_SENDER = None


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    logger.info("ATTENTION_HEAD_GRID: initialized")
    return {"status": "initialized"}


def cleanup_module():
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {"name": "max_heads", "type": "number", "label": "Max heads to show",
             "default": 16, "min": 1, "max": 64},
            {"name": "max_tokens", "type": "number", "label": "Max tokens (truncate)",
             "default": 40, "min": 4, "max": 200},
            {"name": "show_token_labels", "type": "boolean",
             "label": "Token labels (short sequences)", "default": True},
            {"name": "colormap", "type": "select", "label": "Colormap",
             "default": "viridis",
             "options": ["viridis", "plasma", "inferno", "magma", "cividis"]},
        ],
        "layout": {"type": "vertical"},
        "real_time": {"enabled": True, "interval": 200},
    }


def _resolve_config(ltts_event) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if isinstance(ltts_event.module_state, dict):
        cfg.update(ltts_event.module_state)
    env = getattr(ltts_event, "environment", None)
    if isinstance(env, dict) and isinstance(env.get("module_config"), dict):
        cfg.update(env["module_config"])
    return cfg


def _resolve_attention(ltts_event) -> Optional[torch.Tensor]:
    """Find this layer's attention weights — mirrors attn_visualizer's lookup."""
    ctx = ltts_event.context
    ltts_model = getattr(ctx, "ltts_model", None)
    module_path = (getattr(ctx, "module_path", "") or "").strip()

    keys: List[str] = []

    def _add(k):
        if k and k not in keys:
            keys.append(k)

    if module_path:
        _add(module_path)
        base = re.sub(r"^(original_model\.model\.|original_model\.|model\.)", "", module_path)
        for prefix in ("", "model.", "original_model.", "original_model.model."):
            _add(f"{prefix}{base}" if prefix else base)
        for existing in list(keys):
            if ".self_attn" not in existing:
                _add(f"{existing}.self_attn")
        m = re.search(r"layers\.(\d+)(?:\.|$)", module_path) or re.search(
            r"transformer\.h\.(\d+)(?:\.|$)", module_path
        )
        if m:
            idx = m.group(1)
            _add(f"model.layers.{idx}.self_attn")
            _add(f"layers.{idx}.self_attn")

    if ltts_model is not None and hasattr(ltts_model, "captured_attn"):
        captured = ltts_model.captured_attn or {}
        for key in keys:
            if key in captured and captured[key]:
                return captured[key][max(captured[key].keys())]
    if ltts_model is not None and hasattr(ltts_model, "layer_attention_weights"):
        law = ltts_model.layer_attention_weights or {}
        for key in keys:
            if key in law:
                return law[key]
    try:
        from miners.utils import extract_attention_weights

        return extract_attention_weights(getattr(ctx, "outputs", None))
    except Exception:
        return None


def _to_heads_seq_seq(attn) -> Optional[np.ndarray]:
    try:
        a = (
            attn.detach().to(torch.float32).cpu().numpy()
            if isinstance(attn, torch.Tensor)
            else np.array(attn, dtype=np.float32)
        )
    except Exception:
        return None
    if a.ndim == 4:  # [batch, heads, seq, seq]
        a = a[0]
    if a.ndim == 3:  # [heads, seq, seq]
        return a
    if a.ndim == 2:  # single map
        return a[None, :, :]
    return None


def _token_labels(ltts_event, seq_len: int) -> Optional[List[str]]:
    try:
        lm = getattr(ltts_event.context, "ltts_model", None)
        toks = getattr(lm, "_last_input_tokens", None)
        if toks:
            return [str(t) for t in toks][:seq_len]
    except Exception:
        pass
    return None


def process_event(ltts_event):
    try:
        evt = getattr(ltts_event, "event_type", "") or ""
        if not (evt == "layer_after" or evt.startswith("layer_after_")):
            return {"status": "skipped", "reason": "event_phase"}

        attn = _resolve_attention(ltts_event)
        grid = _to_heads_seq_seq(attn) if attn is not None else None
        if grid is None:
            return {"status": "skipped", "reason": "no_attention"}

        cfg = _resolve_config(ltts_event)
        layer_path = getattr(ltts_event.context, "module_path", None)
        emission = ltts_event.should_emit(METADATA["name"], layer_path, cfg)
        if not emission.get("emit"):
            return {"status": "skipped", "reason": "emission_controller"}

        n_heads = int(grid.shape[0])
        show = min(n_heads, int(cfg.get("max_heads", 16) or 16))
        t = min(int(grid.shape[-1]), int(cfg.get("max_tokens", 40) or 40))
        cmap = cfg.get("colormap", "viridis")

        cols = int(np.ceil(np.sqrt(show)))
        rows = int(np.ceil(show / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.1), squeeze=False)
        labels = _token_labels(ltts_event, t) if cfg.get("show_token_labels", True) else None

        for i in range(rows * cols):
            ax = axes[i // cols][i % cols]
            if i < show:
                ax.imshow(grid[i][:t, :t], cmap=cmap, aspect="auto", vmin=0.0)
                ax.set_title(f"H{i}", fontsize=8)
                if labels and t <= 24:
                    ax.set_xticks(range(t))
                    ax.set_yticks(range(t))
                    ax.set_xticklabels(labels, rotation=90, fontsize=5)
                    ax.set_yticklabels(labels, fontsize=5)
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                ax.axis("off")

        fig.suptitle(
            f"Attention heads - {layer_path or 'layer'} ({show}/{n_heads} heads)",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        if _SENDER:
            _SENDER.send_image(
                uri,
                emit_id=f"attention_head_grid:{layer_path}",
                layer_path=layer_path,
                emit_mode=cfg.get("emit_mode", "final"),
            )
        if cfg.get("emit_mode") == "final":
            try:
                ltts_event.mark_emission_finalized(METADATA["name"], layer_path)
            except Exception:
                pass
        return {"status": "ok", "heads": show}

    except Exception as exc:
        logger.error("ATTENTION_HEAD_GRID error: %s", exc, exc_info=True)
        try:
            ltts_event.reactor.signal(
                "error",
                {"message": f"attention_head_grid error: {exc}", "module": METADATA["name"]},
            )
        except Exception:
            pass
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
