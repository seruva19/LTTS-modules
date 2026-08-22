"""
Induction Head Detector Module

Scores every attention head for induction behavior (Olsson et al., 2022): an
induction head at query position i attends to the token immediately AFTER a
previous occurrence of the current token (token[j-1] == token[i]).

When input token ids are unavailable, falls back to pattern-only proxies:
a "previous-token head" score (attention mass on j == i-1) and a
"first-token head" score (attention mass on j == 0), with the table labeled
accordingly.

Emits per attention layer:
- a table of per-head scores (induction / prev-token / first-token)
- optionally a bar chart of induction scores per head
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import numpy as np

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "induction_head_detector",
    "version": "0.1.0",
    "description": "Score every attention head for induction behavior (attends to the token after a previous occurrence of the current token)",
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
    "emit_table": True,
    "emit_chart": False,
    "min_seq_len": 4,
    "emit_mode": "final",
}

_SENDER = None

# Cross-layer accumulator: each layer event gets a fresh LTTSContext, so the
# per-layer per-head scores must live in module state, keyed by forward pass
# (same pattern as logit_lens._TRAJECTORY).
_SCORES: Dict[str, Any] = {"forward_pass": None, "layers": {}}


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    logger.info("INDUCTION_HEAD_DETECTOR: initialized with config: %s", config)
    return {"status": "initialized", "config": config}


def cleanup_module():
    _SCORES["forward_pass"] = None
    _SCORES["layers"] = {}
    logger.info("INDUCTION_HEAD_DETECTOR: cleaned up")
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "emit_table",
                "type": "boolean",
                "label": "Emit per-head score table",
                "default": DEFAULT_CONFIG["emit_table"],
            },
            {
                "name": "emit_chart",
                "type": "boolean",
                "label": "Emit induction score bar chart",
                "default": DEFAULT_CONFIG["emit_chart"],
            },
            {
                "name": "min_seq_len",
                "type": "number",
                "label": "Minimum sequence length",
                "default": DEFAULT_CONFIG["min_seq_len"],
                "min": 2,
                "max": 4096,
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
        "emit_table": _coerce_bool(config.get("emit_table"), DEFAULT_CONFIG["emit_table"]),
        "emit_chart": _coerce_bool(config.get("emit_chart"), DEFAULT_CONFIG["emit_chart"]),
        "min_seq_len": _coerce_int(
            config.get("min_seq_len"), DEFAULT_CONFIG["min_seq_len"], 2, 4096
        ),
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
        logger.warning(
            "INDUCTION_HEAD_DETECTOR: error getting attention weights: %s", exc
        )
        return None


def _to_heads_matrix(attention: torch.Tensor) -> Optional[np.ndarray]:
    """Normalize attention weights to a [heads, seq, seq] numpy array."""
    tensor = attention.detach().float().cpu()
    if tensor.dim() == 4:  # [batch, heads, seq, seq]
        tensor = tensor[0]
    if tensor.dim() == 2:  # [seq, seq] -> single pseudo-head
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3:
        return None
    matrix = tensor.numpy()
    if matrix.shape[-2] != matrix.shape[-1]:
        # Scores require square (full-sequence) attention matrices.
        return None
    return matrix


def _get_token_ids(ltts_model, seq_len: int) -> Optional[List[Any]]:
    """Return per-position token identities (ids preferred, strings as backup)."""
    try:
        if ltts_model is None:
            return None
        ids = getattr(ltts_model, "_last_input_ids", None)
        if isinstance(ids, (list, tuple)) and len(ids) >= seq_len:
            return list(ids)[:seq_len]
        tokens = getattr(ltts_model, "_last_input_tokens", None)
        if isinstance(tokens, (list, tuple)) and len(tokens) >= seq_len:
            return list(tokens)[:seq_len]
    except Exception:
        pass
    return None


def _compute_head_scores(
    heads: np.ndarray, tokens: Optional[List[Any]]
) -> Dict[str, Any]:
    """Compute induction / prev-token / first-token scores per head.

    heads: [num_heads, seq, seq]; tokens: per-position identities or None.
    Induction score per head = mean over query positions i of the attention
    mass on key positions j >= 1 (j <= i) where token[j-1] == token[i].
    """
    num_heads, seq_len, _ = heads.shape
    positions = np.arange(seq_len)

    has_tokens = tokens is not None and len(tokens) >= seq_len
    induction_scores: Optional[List[float]] = None

    if has_tokens:
        token_arr = np.array(tokens[:seq_len], dtype=object)
        # mask[i, j] = True when j >= 1, j <= i, and token[j-1] == token[i]
        match = token_arr[:, None] == token_arr[None, :]  # match[i, m] = tok[i]==tok[m]
        mask = np.zeros((seq_len, seq_len), dtype=bool)
        mask[:, 1:] = match[:, :-1]  # token[j-1] == token[i]
        causal = positions[None, :] <= positions[:, None]  # j <= i
        mask &= causal
        mask[:, 0] = False  # j >= 1
        induction_scores = [
            float((heads[h] * mask).sum(axis=-1).mean()) for h in range(num_heads)
        ]

    prev_token_scores = [
        float(np.diagonal(heads[h], offset=-1).mean()) if seq_len > 1 else 0.0
        for h in range(num_heads)
    ]
    first_token_scores = [float(heads[h][:, 0].mean()) for h in range(num_heads)]

    return {
        "has_token_ids": has_tokens,
        "induction": induction_scores,
        "prev_token": prev_token_scores,
        "first_token": first_token_scores,
        "num_heads": num_heads,
        "seq_len": seq_len,
    }


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
            heads = _to_heads_matrix(attention)
        if heads is None:
            return {"status": "skipped", "reason": "bad_attention_shape"}

        seq_len = heads.shape[-1]
        if seq_len < config["min_seq_len"]:
            return {"status": "skipped", "reason": "sequence_too_short"}

        ltts_model = getattr(context, "ltts_model", None)
        tokens = _get_token_ids(ltts_model, seq_len)
        scores = _compute_head_scores(heads, tokens)

        # Accumulate per-layer per-head scores, keyed by forward pass. Runs
        # before emission gating so the accumulator never misses a layer.
        current_pass = emission.get("forward_pass")
        if _SCORES["forward_pass"] != current_pass:
            _SCORES["forward_pass"] = current_pass
            _SCORES["layers"] = {}
        _SCORES["layers"][layer_path or "unknown"] = scores

        if not emission.get("emit"):
            return {"status": "skipped", "reason": "emission_controller"}

        has_induction = scores["has_token_ids"] and scores["induction"] is not None
        num_heads = scores["num_heads"]

        if has_induction:
            per_head = [
                {
                    "head": h,
                    "induction": scores["induction"][h],
                    "prev_token": scores["prev_token"][h],
                    "first_token": scores["first_token"][h],
                }
                for h in range(num_heads)
            ]
            per_head.sort(key=lambda entry: entry["induction"], reverse=True)
        else:
            per_head = [
                {
                    "head": h,
                    "prev_token": scores["prev_token"][h],
                    "first_token": scores["first_token"][h],
                }
                for h in range(num_heads)
            ]
            per_head.sort(key=lambda entry: entry["prev_token"], reverse=True)

        emitted = []
        base_kwargs = {"layer_path": layer_path, "emit_mode": config["emit_mode"]}

        if _SENDER and config["emit_table"]:
            if has_induction:
                headers = [
                    "Head",
                    "Induction score",
                    "Prev-token score",
                    "First-token score",
                ]
                rows = [
                    [
                        entry["head"],
                        round(entry["induction"], 4),
                        round(entry["prev_token"], 4),
                        round(entry["first_token"], 4),
                    ]
                    for entry in per_head
                ]
            else:
                # Token ids unavailable: pattern-only proxy scores, labeled as such.
                headers = [
                    "Head",
                    "Prev-token score (proxy, no token ids)",
                    "First-token score (proxy, no token ids)",
                ]
                rows = [
                    [
                        entry["head"],
                        round(entry["prev_token"], 4),
                        round(entry["first_token"], 4),
                    ]
                    for entry in per_head
                ]
            _SENDER.send_table(
                headers,
                rows,
                emit_id=f"induction_head_detector:{layer_path}:table",
                **base_kwargs,
            )
            emitted.append("table")

        if _SENDER and config["emit_chart"]:
            if has_induction:
                series_name = "Induction score"
                values = scores["induction"]
            else:
                series_name = "Prev-token score (proxy)"
                values = scores["prev_token"]
            points = [
                {"x": h, "y": round(float(values[h]), 4)} for h in range(num_heads)
            ]
            _SENDER.send_chart(
                {
                    "series": [{"name": series_name, "points": points}],
                    "x_label": "Head index",
                    "y_label": "Score",
                },
                "bar",
                emit_id=f"induction_head_detector:{layer_path}:chart",
                **base_kwargs,
            )
            emitted.append("chart")

        if config["emit_mode"] == "final":
            try:
                ltts_event.mark_emission_finalized(METADATA["name"], layer_path)
            except Exception:
                logger.debug(
                    "INDUCTION_HEAD_DETECTOR: mark_emission_finalized failed",
                    exc_info=True,
                )

        top_head = per_head[0]["head"] if per_head else None
        return {
            "status": "ok",
            "emitted": emitted,
            "mode": "induction" if has_induction else "pattern_proxy",
            "num_heads": num_heads,
            "top_head": top_head,
        }

    except Exception as exc:
        logger.error("INDUCTION_HEAD_DETECTOR: error: %s", exc, exc_info=True)
        try:
            ltts_event.reactor.signal(
                "error",
                {
                    "message": f"induction_head_detector error: {exc}",
                    "module": METADATA["name"],
                },
            )
        except Exception:
            pass
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
