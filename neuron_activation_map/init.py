"""
Neuron Activation Map Module

Heatmap of a layer's hidden state: rows = input tokens, columns = the most
informative neurons (residual-stream channels) selected by activation variance.
Cells are signed activations on a diverging colormap, so you can see *which
neurons fire on which tokens*, how sparse the representation is, and which
channels carry token-specific vs. constant signal.

Complements `neuron_tracker` (which reports per-neuron statistics / histograms)
by giving the full token x neuron picture for one forward pass.
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
import matplotlib.pyplot as plt  # noqa: E402

from scripts.core.module_data_sender import get_data_sender  # noqa: E402

logger = logging.getLogger(__name__)

METADATA = {
    "name": "neuron_activation_map",
    "version": "0.1.0",
    "description": "Token x neuron activation heatmap for a layer - see which neurons fire on which tokens and how sparse the representation is",
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
    "top_neurons": 48,
    "max_tokens": 50,
    "selection": "variance",
    "colormap": "coolwarm",
    "emit_mode": "final",
    "emit_every_n": 1,
}

_SENDER = None


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    logger.info("NEURON_ACTIVATION_MAP: initialized")
    return {"status": "initialized"}


def cleanup_module():
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {"name": "top_neurons", "type": "number", "label": "Neurons to show",
             "default": 48, "min": 4, "max": 256},
            {"name": "max_tokens", "type": "number", "label": "Max tokens (truncate)",
             "default": 50, "min": 4, "max": 200},
            {"name": "selection", "type": "select", "label": "Neuron selection",
             "default": "variance",
             "options": [
                 {"value": "variance", "label": "Highest variance"},
                 {"value": "magnitude", "label": "Highest mean |activation|"},
                 {"value": "first", "label": "First N channels"},
             ]},
            {"name": "colormap", "type": "select", "label": "Colormap",
             "default": "coolwarm",
             "options": ["coolwarm", "viridis", "magma", "RdBu_r", "PiYG"]},
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


def _extract_hidden(outputs: Any) -> Optional[torch.Tensor]:
    candidate = outputs
    if isinstance(candidate, (list, tuple)) and candidate:
        candidate = candidate[0]
    if isinstance(candidate, torch.Tensor) and candidate.dim() == 3:
        return candidate
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

        hidden = _extract_hidden(getattr(ltts_event.context, "outputs", None))
        if hidden is None:
            return {"status": "skipped", "reason": "no_hidden_state"}

        cfg = _resolve_config(ltts_event)
        layer_path = getattr(ltts_event.context, "module_path", None)
        emission = ltts_event.should_emit(METADATA["name"], layer_path, cfg)
        if not emission.get("emit"):
            return {"status": "skipped", "reason": "emission_controller"}

        h = hidden[0].detach().to(torch.float32).cpu().numpy()  # [seq, hid]
        seq = min(h.shape[0], int(cfg.get("max_tokens", 50) or 50))
        h = h[:seq]
        hid = h.shape[1]
        n = min(hid, int(cfg.get("top_neurons", 48) or 48))

        selection = cfg.get("selection", "variance")
        if selection == "magnitude":
            score = np.abs(h).mean(axis=0)
            idx = np.argsort(score)[::-1][:n]
            idx = np.sort(idx)
        elif selection == "first":
            idx = np.arange(n)
        else:  # variance
            score = h.var(axis=0)
            idx = np.argsort(score)[::-1][:n]
            idx = np.sort(idx)

        sub = h[:, idx]  # [seq, n]
        vmax = float(np.abs(sub).max()) or 1.0
        cmap = cfg.get("colormap", "coolwarm")

        fig, ax = plt.subplots(figsize=(max(6, n * 0.16), max(4, seq * 0.28)))
        im = ax.imshow(sub, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="activation")
        ax.set_xlabel(f"Neuron (top {n} by {selection})")
        ax.set_ylabel("Tokens (decoded)")
        ax.set_title(f"Neuron activations - {layer_path or 'layer'}")

        labels = _token_labels(ltts_event, seq)
        if labels:
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7)
        step = max(1, n // 16)
        ax.set_xticks(range(0, n, step))
        ax.set_xticklabels([str(int(idx[i])) for i in range(0, n, step)], rotation=90, fontsize=6)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        if _SENDER:
            _SENDER.send_image(
                uri,
                emit_id=f"neuron_activation_map:{layer_path}",
                layer_path=layer_path,
                emit_mode=cfg.get("emit_mode", "final"),
            )
        if cfg.get("emit_mode") == "final":
            try:
                ltts_event.mark_emission_finalized(METADATA["name"], layer_path)
            except Exception:
                pass
        return {"status": "ok", "neurons": int(n), "tokens": int(seq)}

    except Exception as exc:
        logger.error("NEURON_ACTIVATION_MAP error: %s", exc, exc_info=True)
        try:
            ltts_event.reactor.signal(
                "error",
                {"message": f"neuron_activation_map error: {exc}", "module": METADATA["name"]},
            )
        except Exception:
            pass
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
