"""
Activation Patching Module

Causal tracing (Meng et al., ROME): cache hidden states from a "source"
forward pass, then patch them into a later "target" pass at the same layer.
If patching layer L restores (or destroys) a behaviour, the information that
drives it flows through layer L.

Workflow:
1. set mode = "cache", run a source prompt — activations are stored per layer
2. set mode = "patch", run a target prompt — cached activations overwrite the
   target run's hidden states at the chosen positions
3. "Clear cached activations" button resets the store

Interventions are OFF by default (mode = "off"): a freshly attached module
never changes model behaviour.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch

from scripts.core.ltts_module import LTTSModuleBase
from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "activation_patching",
    "version": "0.1.0",
    "description": "Activation patching (causal tracing): cache hidden states from a source run and patch them into a target run at the same layer",
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
    "mode": "off",  # interventions must be off by default
    "positions": "last",
    "emit_summary": True,
    "layer_filter": "",
    "emit_mode": "all",
}

_SENDER = None

# Module-global activation store: layer_path -> detached cloned CPU tensor.
_CACHE: Dict[str, torch.Tensor] = {}

# Tracks the forward_pass that produced each cached entry. Lets us accumulate
# per-decode-step activations along the sequence dim for the same source pass
# instead of overwriting the full-sequence (prefill) activation with a 1-token
# decode-step one.
_CACHE_PASS: Dict[str, Any] = {}

# Per-forward-pass emission dedup: emit the summary once per layer per pass.
_EMITTED: Dict[str, Any] = {"forward_pass": None, "layers": set()}


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    logger.info("ACTIVATION_PATCHING: initialized with config: %s", config)
    return {"status": "initialized", "config": config}


def cleanup_module():
    _CACHE.clear()
    _CACHE_PASS.clear()
    logger.info("ACTIVATION_PATCHING: cleaned up")
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "mode",
                "type": "select",
                "label": "Mode",
                "default": DEFAULT_CONFIG["mode"],
                "options": [
                    {"value": "off", "label": "Off"},
                    {"value": "cache", "label": "Cache (source run)"},
                    {"value": "patch", "label": "Patch (target run)"},
                ],
            },
            {
                "name": "positions",
                "type": "select",
                "label": "Positions to patch",
                "default": DEFAULT_CONFIG["positions"],
                "options": [
                    {"value": "all", "label": "All positions"},
                    {"value": "last", "label": "Last position only"},
                ],
            },
            {
                "name": "emit_summary",
                "type": "boolean",
                "label": "Emit cache/patch summary",
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
        "mode": _coerce_choice(
            config.get("mode"), ("off", "cache", "patch"), DEFAULT_CONFIG["mode"]
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


def _mark_emitted_once(forward_pass: Any, layer_path: str) -> bool:
    """True the first time a (forward_pass, layer) pair is seen."""
    if _EMITTED["forward_pass"] != forward_pass:
        _EMITTED["forward_pass"] = forward_pass
        _EMITTED["layers"] = set()
    if layer_path in _EMITTED["layers"]:
        return False
    _EMITTED["layers"].add(layer_path)
    return True


def _emit_text(ltts_event, config: Dict[str, Any], layer_path: str, message: str) -> bool:
    """Emit a text summary once per layer per forward pass. Never raises."""
    if not config["emit_summary"]:
        return False
    try:
        emission = ltts_event.should_emit(METADATA["name"], layer_path, config)
    except Exception:
        emission = {"emit": True, "forward_pass": None}
    if not isinstance(emission, dict) or not emission.get("emit"):
        return False
    if not _mark_emitted_once(emission.get("forward_pass"), layer_path):
        return False
    if _SENDER is None:
        return False
    try:
        _SENDER.send_text(
            message,
            preformatted=True,
            layer_path=layer_path,
            emit_id=f"activation_patching:{layer_path}:summary",
            emit_mode=config["emit_mode"],
        )
    except Exception:
        logger.warning("ACTIVATION_PATCHING: failed to emit summary", exc_info=True)
        return False
    if config["emit_mode"] == "final":
        try:
            ltts_event.mark_emission_finalized(METADATA["name"], layer_path)
        except Exception:
            logger.debug(
                "ACTIVATION_PATCHING: mark_emission_finalized failed", exc_info=True
            )
    return True


def process_event(ltts_event):
    try:
        evt = getattr(ltts_event, "event_type", "") or ""
        if not (evt == "layer_after" or evt.startswith("layer_after_")):
            return {"status": "skipped", "reason": "event_phase"}

        if not isinstance(ltts_event.module_state, dict):
            ltts_event.module_state = {}
        config = _resolve_config(ltts_event)

        if config["mode"] == "off":
            return {"status": "skipped", "reason": "disabled"}

        context = ltts_event.context
        layer_path = getattr(context, "module_path", None) or "unknown"
        model_id = getattr(getattr(context, "ltts_model", None), "_model_id", None) or "default"
        # Cache is module-global: include model_id so runs against different
        # models never read each other's cached activations.
        cache_key = f"{model_id}:{layer_path}"

        layer_filter = config["layer_filter"].strip()
        if layer_filter and layer_filter not in layer_path:
            return {"status": "skipped", "reason": "layer_filter"}

        hidden = _extract_hidden_state(context.outputs)
        if hidden is None:
            return {"status": "skipped", "reason": "no_hidden_state"}

        if config["mode"] == "cache":
            incoming = hidden.detach().clone().cpu()
            incoming_seq = int(incoming.shape[1])
            existing = _CACHE.get(cache_key)
            action = "cached"

            if existing is None:
                # First activation for this layer in this source run.
                _CACHE[cache_key] = incoming
            elif incoming_seq > 1:
                # A new prefill / full-sequence forward starts a fresh source
                # run for this layer: replace whatever was there.
                _CACHE[cache_key] = incoming
            else:
                # incoming_seq == 1: a decode step. Accumulate it onto the
                # existing cache along the sequence dim so the cache reconstructs
                # the full generated sequence, instead of overwriting a richer
                # (prefill) activation with a single-position one.
                if (
                    int(existing.shape[0]) == int(incoming.shape[0])
                    and int(existing.shape[-1]) == int(incoming.shape[-1])
                ):
                    _CACHE[cache_key] = torch.cat([existing, incoming], dim=1)
                    action = "accumulated"
                else:
                    # Shapes are incompatible for concatenation. Keep the richer
                    # existing entry rather than clobbering it with 1 position.
                    action = "kept_existing"

            _CACHE_PASS[cache_key] = id(getattr(context, "outputs", None))
            cached_now = _CACHE[cache_key]
            _emit_text(
                ltts_event,
                config,
                layer_path,
                f"{action} {layer_path} {list(cached_now.shape)}",
            )
            return {
                "status": "ok",
                "action": action,
                "layer": layer_path,
                "shape": list(cached_now.shape),
            }

        # mode == "patch"
        if cache_key not in _CACHE:
            return {"status": "skipped", "reason": "no_cached_activation"}

        cached = _CACHE[cache_key]
        if int(cached.shape[-1]) != int(hidden.shape[-1]):
            logger.warning(
                "ACTIVATION_PATCHING: hidden dim mismatch at %s (cached %d vs current %d), skipping",
                layer_path,
                int(cached.shape[-1]),
                int(hidden.shape[-1]),
            )
            return {"status": "skipped", "reason": "hidden_dim_mismatch"}

        seq_len = int(hidden.shape[1])
        cached_seq = int(cached.shape[1])
        batch = min(int(hidden.shape[0]), int(cached.shape[0]))
        positions = config["positions"]

        with torch.no_grad():
            source = cached.to(device=hidden.device, dtype=hidden.dtype)
            # Never mutate the original tensor in place.
            new_hidden = hidden.detach().clone()
            prefix_note = ""
            if positions == "all":
                if cached_seq == 1 and seq_len > 1:
                    # The cache only holds a single decode-step activation, so
                    # 'all' cannot meaningfully cover the full current sequence.
                    # Fall back to last-position patching and say so honestly
                    # instead of pretending we patched a 1-token prefix.
                    new_hidden[:batch, -1, :] = source[:batch, -1, :]
                    patched_positions = 1
                    prefix_note = (
                        " WARNING: cached activation has only 1 position "
                        "(decode step); 'all' patching falls back to "
                        "last-position only -- cache during a single prefill "
                        "forward for full-sequence patching"
                    )
                else:
                    overlap = min(seq_len, cached_seq)
                    new_hidden[:batch, :overlap, :] = source[:batch, :overlap, :]
                    patched_positions = overlap
                    if seq_len != cached_seq:
                        prefix_note = (
                            f" (seq lengths differ: cached {cached_seq} vs current "
                            f"{seq_len}, patched overlapping prefix of {overlap})"
                        )
            else:  # last: overwrite only the last position with the cached last
                new_hidden[:batch, -1, :] = source[:batch, -1, :]
                patched_positions = 1
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

        emitted = _emit_text(
            ltts_event,
            config,
            layer_path,
            (
                f"patched {layer_path}: {patched_positions} position(s) "
                f"(positions={positions}); L2 norm of difference = "
                f"{diff_norm:.6f}{prefix_note}"
            ),
        )

        return {
            "status": "ok",
            "action": "patched",
            "layer": layer_path,
            "patched_positions": patched_positions,
            "diff_norm": diff_norm,
            "emitted": emitted,
        }

    except Exception as exc:
        logger.error("ACTIVATION_PATCHING: error: %s", exc, exc_info=True)
        try:
            ltts_event.reactor.signal(
                "error",
                {
                    "message": f"activation_patching error: {exc}",
                    "module": METADATA["name"],
                },
            )
        except Exception:
            pass
        return {"status": "error", "message": str(exc)}


class ActivationPatchingModule(LTTSModuleBase):
    """Stateful controls: clear the activation cache on demand."""

    def _register_module_routes(self):
        self.add_stateful_button(
            {
                "id": "clear_cache",
                "label": "Clear cached activations",
                "variant": "secondary",
            },
            "handle_clear_cache",
        )

    def handle_clear_cache(self, event_data, module_state):
        cleared = len(_CACHE)
        _CACHE.clear()
        _CACHE_PASS.clear()
        logger.info("ACTIVATION_PATCHING: cleared %d cached activation(s)", cleared)
        return {"status": "ok", "cleared": cleared}


def interceptor(ltts_event):
    return process_event(ltts_event)
