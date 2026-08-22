"""
Neuron Ablation Module

Causal intervention: zeroes out (or mean-replaces) chosen hidden-state
channels ("neurons") in the residual stream at the layers the user attaches
it to. The classic knockout experiment of mechanistic interpretability —
if behaviour changes when a unit is silenced, that unit was causally involved.

Interventions are OFF by default: a freshly attached module never changes
model behaviour until 'enabled' is switched on and neuron indices are given.

Works on any layer whose output carries an extractable 3D hidden state
[batch, seq, hidden]: decoder-block boundaries, MLP or attention sublayers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "neuron_ablation",
    "version": "0.1.0",
    "description": "Ablate chosen hidden-state channels (zero or mean replacement) at attached layers — causal knockout intervention",
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
    "enabled": False,  # interventions must be off by default
    "neuron_indices": "",
    "ablation_mode": "zero",
    "positions": "all",
    "emit_summary": True,
    "layer_filter": "",
    "emit_mode": "all",
}

_SENDER = None

# Per-forward-pass emission dedup: emit the summary once per layer per pass.
_EMITTED: Dict[str, Any] = {"forward_pass": None, "layers": set()}


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    logger.info("NEURON_ABLATION: initialized with config: %s", config)
    return {"status": "initialized", "config": config}


def cleanup_module():
    logger.info("NEURON_ABLATION: cleaned up")
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "enabled",
                "type": "boolean",
                "label": "Enable ablation (intervention)",
                "default": DEFAULT_CONFIG["enabled"],
            },
            {
                "name": "neuron_indices",
                "type": "text",
                "label": "Neuron indices (e.g. 0,5,10-20)",
                "default": DEFAULT_CONFIG["neuron_indices"],
            },
            {
                "name": "ablation_mode",
                "type": "select",
                "label": "Ablation mode",
                "default": DEFAULT_CONFIG["ablation_mode"],
                "options": [
                    {"value": "zero", "label": "Zero"},
                    {"value": "mean", "label": "Mean over sequence"},
                ],
            },
            {
                "name": "positions",
                "type": "select",
                "label": "Positions to ablate",
                "default": DEFAULT_CONFIG["positions"],
                "options": [
                    {"value": "all", "label": "All positions"},
                    {"value": "last", "label": "Last position only"},
                ],
            },
            {
                "name": "emit_summary",
                "type": "boolean",
                "label": "Emit ablation summary",
                "default": DEFAULT_CONFIG["emit_summary"],
            },
            {
                "name": "layer_filter",
                "type": "text",
                "label": "Layer filter (substring of layer path, empty = all)",
                "default": DEFAULT_CONFIG["layer_filter"],
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


def _coerce_choice(value: Any, choices: Tuple[str, ...], default: str) -> str:
    candidate = str(value).strip().lower() if value is not None else ""
    return candidate if candidate in choices else default


def _resolve_config(ltts_event) -> Dict[str, Any]:
    config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    if isinstance(ltts_event.module_state, dict):
        config.update(ltts_event.module_state)
    env = getattr(ltts_event, "environment", None)
    env_config = env.get("module_config") if isinstance(env, dict) else None
    if isinstance(env_config, dict):
        config.update(env_config)

    return {
        "enabled": _coerce_bool(config.get("enabled"), DEFAULT_CONFIG["enabled"]),
        "neuron_indices": str(config.get("neuron_indices") or ""),
        "ablation_mode": _coerce_choice(
            config.get("ablation_mode"), ("zero", "mean"), DEFAULT_CONFIG["ablation_mode"]
        ),
        "positions": _coerce_choice(
            config.get("positions"), ("all", "last"), DEFAULT_CONFIG["positions"]
        ),
        "emit_summary": _coerce_bool(
            config.get("emit_summary"), DEFAULT_CONFIG["emit_summary"]
        ),
        "layer_filter": str(config.get("layer_filter") or ""),
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


def _parse_neuron_indices(spec: str, hidden_dim: int) -> List[int]:
    """Parse '0,5,10-20' into a sorted list of unique in-range indices.

    Ranges are inclusive. Out-of-range indices are clamped away (dropped)
    with a warning; malformed parts are ignored with a warning.
    """
    requested: set = set()
    malformed: List[str] = []
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            pieces = part.split("-", 1)
            try:
                lo = int(pieces[0].strip())
                hi = int(pieces[1].strip())
            except ValueError:
                malformed.append(part)
                continue
            if lo > hi:
                lo, hi = hi, lo
            requested.update(range(lo, hi + 1))
        else:
            try:
                requested.add(int(part))
            except ValueError:
                malformed.append(part)

    in_range = sorted(i for i in requested if 0 <= i < hidden_dim)
    dropped = len(requested) - len(in_range)
    if dropped:
        logger.warning(
            "NEURON_ABLATION: %d requested indices outside [0, %d) were ignored",
            dropped,
            hidden_dim,
        )
    if malformed:
        logger.warning(
            "NEURON_ABLATION: ignoring malformed index parts: %s", malformed
        )
    return in_range


def _mark_emitted_once(forward_pass: Any, layer_path: str) -> bool:
    """True the first time a (forward_pass, layer) pair is seen."""
    if _EMITTED["forward_pass"] != forward_pass:
        _EMITTED["forward_pass"] = forward_pass
        _EMITTED["layers"] = set()
    if layer_path in _EMITTED["layers"]:
        return False
    _EMITTED["layers"].add(layer_path)
    return True


def process_event(ltts_event):
    try:
        evt = getattr(ltts_event, "event_type", "") or ""
        if not (evt == "layer_after" or evt.startswith("layer_after_")):
            return {"status": "skipped", "reason": "event_phase"}

        if not isinstance(ltts_event.module_state, dict):
            ltts_event.module_state = {}
        config = _resolve_config(ltts_event)

        if not config["enabled"]:
            return {"status": "skipped", "reason": "disabled"}

        context = ltts_event.context
        layer_path = getattr(context, "module_path", None) or "unknown"

        layer_filter = config["layer_filter"].strip()
        if layer_filter and layer_filter not in layer_path:
            return {"status": "skipped", "reason": "layer_filter"}

        # Intervene on whatever layer the event carries (block boundary, mlp,
        # attention, ...) as long as a 3D hidden state is extractable.
        hidden = _extract_hidden_state(context.outputs)
        if hidden is None:
            return {"status": "skipped", "reason": "no_hidden_state"}

        hidden_dim = int(hidden.shape[-1])
        indices = _parse_neuron_indices(config["neuron_indices"], hidden_dim)
        if not indices:
            return {"status": "skipped", "reason": "no_valid_indices"}

        mode = config["ablation_mode"]
        positions = config["positions"]

        with torch.no_grad():
            # Never mutate the original tensor in place.
            new_hidden = hidden.detach().clone()
            idx = torch.tensor(indices, device=new_hidden.device, dtype=torch.long)
            if mode == "mean":
                # Per-batch, per-channel mean over the sequence dimension.
                means = hidden.detach()[:, :, idx].mean(dim=1)  # [batch, n]
                if positions == "all":
                    new_hidden[:, :, idx] = means.unsqueeze(1).to(new_hidden.dtype)
                else:
                    new_hidden[:, -1, idx] = means.to(new_hidden.dtype)
            else:  # zero
                if positions == "all":
                    new_hidden[:, :, idx] = 0
                else:
                    new_hidden[:, -1, idx] = 0
            diff_norm = float(
                torch.linalg.vector_norm((new_hidden - hidden.detach()).float()).item()
            )

        outputs = context.outputs
        if isinstance(outputs, (list, tuple)):
            context.analysis_results["modified_outputs"] = (new_hidden,) + tuple(
                outputs[1:]
            )
        else:
            context.analysis_results["modified_outputs"] = new_hidden

        emitted = False
        if config["emit_summary"]:
            try:
                emission = ltts_event.should_emit(METADATA["name"], layer_path, config)
            except Exception:
                emission = {"emit": True, "forward_pass": None}
            forward_pass = emission.get("forward_pass") if isinstance(emission, dict) else None
            if (
                _SENDER
                and isinstance(emission, dict)
                and emission.get("emit")
                and _mark_emitted_once(forward_pass, layer_path)
            ):
                _SENDER.send_text(
                    (
                        f"Ablated {len(indices)} neuron(s) at {layer_path} "
                        f"(mode={mode}, positions={positions}); "
                        f"L2 norm of difference = {diff_norm:.6f}"
                    ),
                    preformatted=True,
                    layer_path=layer_path,
                    emit_id=f"neuron_ablation:{layer_path}:summary",
                    emit_mode=config["emit_mode"],
                )
                emitted = True
            if config["emit_mode"] == "final":
                try:
                    ltts_event.mark_emission_finalized(METADATA["name"], layer_path)
                except Exception:
                    logger.debug(
                        "NEURON_ABLATION: mark_emission_finalized failed", exc_info=True
                    )

        return {
            "status": "ok",
            "ablated_neurons": len(indices),
            "mode": mode,
            "positions": positions,
            "diff_norm": diff_norm,
            "emitted": emitted,
        }

    except Exception as exc:
        logger.error("NEURON_ABLATION: error: %s", exc, exc_info=True)
        try:
            ltts_event.reactor.signal(
                "error",
                {"message": f"neuron_ablation error: {exc}", "module": METADATA["name"]},
            )
        except Exception:
            pass
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
