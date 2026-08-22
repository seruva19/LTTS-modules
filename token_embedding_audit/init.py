"""Under-trained / glitch-token audit from input-embedding geometry."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "token_embedding_audit",
    "version": "0.1.0",
    "description": "Flag under-trained 'glitch' tokens from input-embedding norm and closeness to the mean embedding",
    "author": "LTTS",
    "event_types": ["model_after"],
    "dependencies": ["requirements.txt"],
    "tags": ["tokenizer", "embeddings", "glitch-tokens"],
    "methods": {
        "ctor": "initialize_module",
        "dtor": "cleanup_module",
        "ntor": "process_event",
        "utor": "get_ui_schema",
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "bottom_k": 25,
    "norm_std_threshold": 2.0,
    "histogram_bins": 50,
    "emit_scalar": True,
    "emit_table": True,
    "emit_chart": True,
    "emit_text": True,
    "emit_mode": "first",
}

_SENDER = None
# Embedding geometry is a property of the checkpoint, not of the prompt, so it is
# computed once per weight matrix and reused across forward passes.
_CACHE: Dict[str, Any] = {"signature": None, "norms": None, "cosine": None}


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    return {"status": "initialized", "config": config}


def cleanup_module():
    _CACHE["signature"] = None
    _CACHE["norms"] = None
    _CACHE["cosine"] = None
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "bottom_k",
                "type": "number",
                "label": "Under-trained candidates to list",
                "default": 25,
                "min": 1,
                "max": 200,
            },
            {
                "name": "norm_std_threshold",
                "type": "number",
                "label": "Low-norm cutoff (mean - n*std)",
                "default": 2,
                "min": 0,
                "max": 10,
            },
            {
                "name": "histogram_bins",
                "type": "number",
                "label": "Norm histogram bins",
                "default": 50,
                "min": 2,
                "max": 512,
            },
            {
                "name": "emit_scalar",
                "type": "boolean",
                "label": "Emit low-norm token count",
                "default": True,
            },
            {
                "name": "emit_table",
                "type": "boolean",
                "label": "Emit candidate table",
                "default": True,
            },
            {
                "name": "emit_chart",
                "type": "boolean",
                "label": "Emit norm histogram",
                "default": True,
            },
            {
                "name": "emit_text",
                "type": "boolean",
                "label": "Emit summary text",
                "default": True,
            },
        ],
        "layout": {"type": "vertical"},
        "real_time": {"enabled": False, "interval": 1000},
    }


def _config(event) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if isinstance(event.module_state, dict):
        config.update(event.module_state)
    try:
        config["bottom_k"] = max(1, min(200, int(config.get("bottom_k", 25))))
    except (TypeError, ValueError):
        config["bottom_k"] = 25
    try:
        config["histogram_bins"] = max(2, min(512, int(config.get("histogram_bins", 50))))
    except (TypeError, ValueError):
        config["histogram_bins"] = 50
    try:
        config["norm_std_threshold"] = float(config.get("norm_std_threshold", 2.0))
    except (TypeError, ValueError):
        config["norm_std_threshold"] = 2.0
    return config


def _hf_model(context) -> Optional[Any]:
    ltts_model = getattr(context, "ltts_model", None)
    return getattr(ltts_model, "original_model", None) if ltts_model is not None else None


def _input_embedding_weight(hf_model) -> Optional[torch.Tensor]:
    getter = getattr(hf_model, "get_input_embeddings", None)
    if not callable(getter):
        return None
    try:
        embedding = getter()
    except Exception:
        return None
    weight = getattr(embedding, "weight", None)
    # Quantised checkpoints expose an integer-packed weight that carries no usable
    # geometry, so they are treated as absent rather than measured wrongly.
    if isinstance(weight, torch.Tensor) and weight.dim() == 2 and weight.is_floating_point():
        return weight
    return None


def _ties_embeddings(hf_model, weight: torch.Tensor) -> Optional[bool]:
    tied = getattr(getattr(hf_model, "config", None), "tie_word_embeddings", None)
    if isinstance(tied, bool):
        return tied
    getter = getattr(hf_model, "get_output_embeddings", None)
    output = getter() if callable(getter) else None
    out_weight = getattr(output, "weight", None)
    if isinstance(out_weight, torch.Tensor):
        return out_weight.data_ptr() == weight.data_ptr()
    return None


def _geometry(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row L2 norm and cosine of every row to the mean embedding row."""
    with torch.no_grad():
        # `dtype=` accumulates in fp32 without materialising a second copy of a
        # matrix that reaches gigabytes at 250k vocabularies.
        norms = torch.linalg.vector_norm(weight, dim=1, dtype=torch.float32)
        mean_row = weight.mean(dim=0, dtype=torch.float32)
        mean_norm = torch.linalg.vector_norm(mean_row)
        dots = torch.mv(weight, mean_row.to(weight.dtype)).float()
        cosine = dots / (norms * mean_norm).clamp_min(1e-12)
    return norms.detach().cpu(), cosine.detach().cpu()


def _cached_geometry(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    signature = (weight.data_ptr(), tuple(weight.shape), str(weight.dtype))
    if _CACHE["signature"] != signature:
        norms, cosine = _geometry(weight)
        _CACHE["signature"] = signature
        _CACHE["norms"] = norms
        _CACHE["cosine"] = cosine
    return _CACHE["norms"], _CACHE["cosine"]


def _decode(tokenizer, token_id: int) -> str:
    if tokenizer is not None:
        try:
            return repr(tokenizer.decode([token_id]))
        except Exception:
            pass
    return f"#{token_id}"


def _histogram(norms: torch.Tensor, bins: int) -> List[Dict[str, float]]:
    low = float(norms.min())
    high = float(norms.max())
    if not high > low:
        return [{"x": round(low, 6), "y": int(norms.numel())}]
    counts = torch.histc(norms, bins=bins, min=low, max=high).tolist()
    width = (high - low) / bins
    return [
        {"x": round(low + (index + 0.5) * width, 6), "y": int(count)}
        for index, count in enumerate(counts)
    ]


def process_event(ltts_event):
    try:
        if not str(getattr(ltts_event, "event_type", "")).startswith("model_after"):
            return {"status": "skipped", "reason": "event_phase"}
        context = ltts_event.context
        hf_model = _hf_model(context)
        if hf_model is None:
            return {"status": "skipped", "reason": "no_model"}
        weight = _input_embedding_weight(hf_model)
        if weight is None:
            return {"status": "skipped", "reason": "no_input_embeddings"}

        config = _config(ltts_event)
        layer_path = getattr(context, "module_path", None) or "model"
        emission = ltts_event.should_emit(METADATA["name"], layer_path, config)
        if not emission.get("emit"):
            return {"status": "skipped", "reason": "emission_controller"}

        norms, cosine = _cached_geometry(weight)
        vocab_size, hidden_size = int(weight.shape[0]), int(weight.shape[1])
        norm_mean = float(norms.mean())
        norm_std = max(float(norms.std()), 1e-12)
        cos_mean = float(cosine.mean())
        cos_std = max(float(cosine.std()), 1e-12)

        z_norm = (norms - norm_mean) / norm_std
        # An under-trained row is both short and unusually close to the mean
        # embedding, so the two deviations reinforce each other in one score.
        score = z_norm - (cosine - cos_mean) / cos_std

        bottom_k = min(config["bottom_k"], vocab_size)
        worst = torch.topk(score, bottom_k, largest=False)
        worst_ids = worst.indices.tolist()
        worst_scores = worst.values.tolist()
        cutoff = norm_mean - config["norm_std_threshold"] * norm_std
        low_norm_count = int((norms < cutoff).sum())

        tokenizer = getattr(getattr(context, "ltts_model", None), "_tokenizer", None)
        tied = _ties_embeddings(hf_model, weight)

        base = {"layer_path": layer_path, "emit_mode": config.get("emit_mode", "first")}
        emitted = []
        if _SENDER and config.get("emit_scalar", True):
            _SENDER.send_scalar(
                low_norm_count,
                label=f"Tokens below mean - {config['norm_std_threshold']:g}*std of embedding norm",
                emit_id=f"token_embedding_audit:{layer_path}:scalar",
                **base,
            )
            emitted.append("scalar")
        if _SENDER and config.get("emit_table", True):
            rows = [
                [
                    rank + 1,
                    token_id,
                    _decode(tokenizer, token_id),
                    round(float(norms[token_id]), 6),
                    round(float(z_norm[token_id]), 4),
                    round(float(cosine[token_id]), 6),
                    round(candidate_score, 4),
                ]
                for rank, (token_id, candidate_score) in enumerate(
                    zip(worst_ids, worst_scores)
                )
            ]
            _SENDER.send_table(
                ["Rank", "Token id", "Token", "Norm", "Norm z", "Cos to mean", "Score"],
                rows,
                emit_id=f"token_embedding_audit:{layer_path}:table",
                **base,
            )
            emitted.append("table")
        if _SENDER and config.get("emit_chart", True):
            _SENDER.send_chart(
                {
                    "series": [
                        {
                            "name": "Token count",
                            "points": _histogram(norms, config["histogram_bins"]),
                        }
                    ],
                    "x_label": "Embedding L2 norm (bin centre)",
                    "y_label": "Tokens",
                },
                "bar",
                emit_id=f"token_embedding_audit:{layer_path}:chart",
                **base,
            )
            emitted.append("chart")
        if _SENDER and config.get("emit_text", True):
            if tied is True:
                tie_note = (
                    "Input and output embeddings are TIED: every row doubles as an "
                    "unembedding, so a low-norm row also means the model can barely "
                    "predict that token, and untied-model intuitions do not carry over."
                )
            elif tied is False:
                tie_note = (
                    "Input and output embeddings are untied: this audit covers the "
                    "input side only; the unembedding row for the same token may be trained."
                )
            else:
                tie_note = "Embedding tying could not be determined for this model."
            _SENDER.send_text(
                f"Vocabulary {vocab_size} x {hidden_size}. "
                f"Norm mean {norm_mean:.4f}, std {norm_std:.4f}; "
                f"cosine-to-mean mean {cos_mean:.4f}, std {cos_std:.4f}. "
                f"{low_norm_count} tokens fall below {cutoff:.4f}. "
                f"Score = norm z-score minus cosine-to-mean z-score; the lowest scores "
                f"are under-trained candidates. {tie_note}",
                emit_id=f"token_embedding_audit:{layer_path}:text",
                **base,
            )
            emitted.append("text")

        if config.get("emit_mode") == "final":
            ltts_event.mark_emission_finalized(METADATA["name"], layer_path)

        return {
            "status": "ok",
            "emitted": emitted,
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "low_norm_count": low_norm_count,
            "tied_embeddings": tied,
            "candidates": worst_ids,
        }
    except Exception as exc:
        logger.error("TOKEN_EMBEDDING_AUDIT error: %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
