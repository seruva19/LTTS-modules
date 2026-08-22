"""
SAE Features Module

Runs a pretrained sparse autoencoder (via the sae_lens library) over the
residual stream at decoder-block boundaries and reports the most active
features. Sparse autoencoders decompose dense activations into a (mostly)
monosemantic, overcomplete feature basis — the workhorse of current
dictionary-learning interpretability.

sae_lens is an optional heavy dependency: it lives in this module's isolated
venv (ltts_modules/sae_features/venv). If it is missing, the module degrades
gracefully with a one-time explanatory message instead of raising.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch

from scripts.core.module_data_sender import get_data_sender

# Try to import sae_lens components (optional heavy dependency, isolated venv)
try:
    from sae_lens import SAE

    SAE_AVAILABLE = True
except ImportError:
    SAE_AVAILABLE = False
    SAE = None

logger = logging.getLogger(__name__)

METADATA = {
    "name": "sae_features",
    "version": "0.1.0",
    "description": "Sparse autoencoder feature inspection: top active SAE features in the residual stream at decoder-block boundaries (sae_lens)",
    "author": "LTTS",
    "event_types": ["layer_after"],
    "dependencies": ["requirements.txt"],
    "target_roles": ["block"],
    "supported_topologies": ["decoder_only"],
    "methods": {
        "ctor": "initialize_module",
        "dtor": "cleanup_module",
        "ntor": "process_event",
        "utor": "get_ui_schema",
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "sae_release": "gpt2-small-res-jb",
    "sae_id": "blocks.5.hook_resid_pre",
    "layer_filter": "",
    "top_k_features": 10,
    "position": "last",  # last | mean
    "emit_table": True,
    "emit_scalar": False,
    "emit_mode": "final",
}

_SENDER = None

# Module-global session state:
# - SAE cache keyed by (release, sae_id)
# - one-shot latches so warnings are emitted once per session, not per event
_STATE: Dict[str, Any] = {
    "sae_cache": {},          # (release, sae_id) -> SAE instance
    "failed_loads": set(),    # (release, sae_id) keys that failed; do not retry
    "missing_dep_notified": False,
    "dim_mismatch_notified": set(),  # (release, sae_id) keys already warned about
}


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    logger.info("SAE_FEATURES: initialized with config: %s", config)
    return {"status": "initialized", "config": config}


def cleanup_module():
    _STATE["sae_cache"].clear()
    _STATE["failed_loads"].clear()
    _STATE["dim_mismatch_notified"].clear()
    _STATE["missing_dep_notified"] = False
    logger.info("SAE_FEATURES: cleaned up")
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "enabled",
                "type": "boolean",
                "label": "Enabled",
                "default": DEFAULT_CONFIG["enabled"],
            },
            {
                "name": "sae_release",
                "type": "text",
                "label": "SAE release (sae_lens)",
                "default": DEFAULT_CONFIG["sae_release"],
            },
            {
                "name": "sae_id",
                "type": "text",
                "label": "SAE id (hook point)",
                "default": DEFAULT_CONFIG["sae_id"],
            },
            {
                "name": "layer_filter",
                "type": "text",
                "label": "Layer filter (substring of layer path)",
                "default": DEFAULT_CONFIG["layer_filter"],
            },
            {
                "name": "top_k_features",
                "type": "number",
                "label": "Top K features",
                "default": DEFAULT_CONFIG["top_k_features"],
                "min": 1,
                "max": 50,
            },
            {
                "name": "position",
                "type": "select",
                "label": "Sequence position",
                "default": DEFAULT_CONFIG["position"],
                "options": [
                    {"value": "last", "label": "Last token"},
                    {"value": "mean", "label": "Mean over positions"},
                ],
            },
            {
                "name": "emit_table",
                "type": "boolean",
                "label": "Emit top-feature table",
                "default": DEFAULT_CONFIG["emit_table"],
            },
            {
                "name": "emit_scalar",
                "type": "boolean",
                "label": "Emit active-feature count (L0)",
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

    position = str(config.get("position", DEFAULT_CONFIG["position"])).strip().lower()
    if position not in {"last", "mean"}:
        position = DEFAULT_CONFIG["position"]

    return {
        "enabled": _coerce_bool(config.get("enabled"), DEFAULT_CONFIG["enabled"]),
        "sae_release": str(
            config.get("sae_release") or DEFAULT_CONFIG["sae_release"]
        ).strip(),
        "sae_id": str(config.get("sae_id") or DEFAULT_CONFIG["sae_id"]).strip(),
        "layer_filter": str(config.get("layer_filter") or "").strip(),
        "top_k_features": _coerce_int(
            config.get("top_k_features"), DEFAULT_CONFIG["top_k_features"], 1, 50
        ),
        "position": position,
        "emit_table": _coerce_bool(config.get("emit_table"), DEFAULT_CONFIG["emit_table"]),
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


def _send_text_once(latch_key: str, content: str, layer_path: Optional[str]) -> None:
    """Emit an explanatory text payload once per session for a given latch."""
    if latch_key == "missing_dep":
        if _STATE["missing_dep_notified"]:
            return
        _STATE["missing_dep_notified"] = True
    if _SENDER is not None:
        try:
            _SENDER.send_text(
                content,
                layer_path=layer_path,
                emit_id=f"sae_features:{latch_key}",
            )
        except Exception:
            logger.debug("SAE_FEATURES: send_text failed", exc_info=True)
    logger.warning("SAE_FEATURES: %s", content)


def _get_sae(release: str, sae_id: str, layer_path: Optional[str]):
    """Load and cache the SAE once per (release, id); latch failures so we never retry per event."""
    key: Tuple[str, str] = (release, sae_id)
    if key in _STATE["sae_cache"]:
        return _STATE["sae_cache"][key]
    if key in _STATE["failed_loads"]:
        return None
    try:
        loaded = SAE.from_pretrained(release, sae_id)
        # Older sae_lens versions return (sae, cfg_dict, sparsity); newer return the SAE.
        sae = loaded[0] if isinstance(loaded, tuple) else loaded
        sae.eval()
        _STATE["sae_cache"][key] = sae
        logger.info("SAE_FEATURES: loaded SAE %s / %s", release, sae_id)
        return sae
    except Exception as exc:
        _STATE["failed_loads"].add(key)
        _send_text_once(
            f"load_failed:{release}:{sae_id}",
            f"Failed to load SAE '{sae_id}' from release '{release}': {exc}. "
            "SAE feature inspection is disabled for this release/id until the "
            "configuration changes.",
            layer_path,
        )
        return None


def process_event(ltts_event):
    try:
        if not isinstance(ltts_event.module_state, dict):
            ltts_event.module_state = {}
        config = _resolve_config(ltts_event)

        if not config["enabled"]:
            return {"status": "skipped", "reason": "disabled"}

        context = ltts_event.context
        if _SENDER is not None:
            _SENDER.set_context(context)
        layer_path = getattr(context, "module_path", None)

        if not SAE_AVAILABLE:
            _send_text_once(
                "missing_dep",
                "sae_lens not installed — install it into this module's isolated "
                "venv (ltts_modules/sae_features/venv) to enable SAE feature "
                "inspection",
                layer_path,
            )
            return {"status": "skipped", "reason": "sae_lens_missing"}

        evt = getattr(ltts_event, "event_type", "") or ""
        if evt and not (evt == "layer_after" or evt.startswith("layer_after_")):
            return {"status": "skipped", "reason": "event_phase"}

        # Only act at decoder-block boundaries: sublayer outputs are partial
        # residual updates. Prefer the architecture adapter's semantic role and
        # Keep the class-name fallback for compatible event producers.
        module_class = (getattr(context, "module_class", "") or "").lower()
        is_block = getattr(context, "module_role", None) == "block" or (
            "decoderlayer" in module_class
            or module_class.endswith(("block", "layer"))
        )
        if not is_block:
            return {"status": "skipped", "reason": "not_a_block_boundary"}

        if config["layer_filter"] and config["layer_filter"] not in (layer_path or ""):
            return {"status": "skipped", "reason": "layer_filter"}

        hidden = _extract_hidden_state(getattr(context, "outputs", None))
        if hidden is None:
            return {"status": "skipped", "reason": "no_hidden_state"}

        emission = ltts_event.should_emit(METADATA["name"], layer_path, config)
        if not emission.get("emit"):
            return {"status": "skipped", "reason": "emission_controller"}

        sae = _get_sae(config["sae_release"], config["sae_id"], layer_path)
        if sae is None:
            return {"status": "skipped", "reason": "sae_load_failed"}

        with torch.no_grad():
            if config["position"] == "mean":
                vector = hidden[0].mean(dim=0)
            else:
                vector = hidden[0, -1, :]

            d_in = getattr(getattr(sae, "cfg", None), "d_in", None)
            if d_in is not None and vector.shape[-1] != d_in:
                key = (config["sae_release"], config["sae_id"])
                if key not in _STATE["dim_mismatch_notified"]:
                    _STATE["dim_mismatch_notified"].add(key)
                    _send_text_once(
                        f"dim_mismatch:{config['sae_release']}:{config['sae_id']}",
                        f"Hidden state dim {vector.shape[-1]} does not match SAE "
                        f"d_in {d_in} (release '{config['sae_release']}', id "
                        f"'{config['sae_id']}') — skipping. Pick an SAE trained "
                        "for this model's residual stream.",
                        layer_path,
                    )
                return {"status": "skipped", "reason": "dim_mismatch"}

            sae_device = getattr(sae, "device", None) or next(
                sae.parameters()
            ).device
            sae_dtype = next(sae.parameters()).dtype
            feature_acts = sae.encode(
                vector.to(device=sae_device, dtype=sae_dtype).unsqueeze(0)
            )
            feature_acts = feature_acts.squeeze(0).detach().float().cpu()

            k = min(config["top_k_features"], feature_acts.shape[-1])
            top = torch.topk(feature_acts, k=k)
            active_count = int((feature_acts > 1e-6).sum().item())

        top_indices = [int(i) for i in top.indices.tolist()]
        top_values = [float(v) for v in top.values.tolist()]

        emitted = []
        base_kwargs = {"layer_path": layer_path, "emit_mode": config["emit_mode"]}

        if _SENDER and config["emit_table"]:
            headers = ["Rank", "Feature", "Activation"]
            rows = [
                [rank + 1, f"feature {idx}", round(value, 5)]
                for rank, (idx, value) in enumerate(zip(top_indices, top_values))
            ]
            _SENDER.send_table(
                headers,
                rows,
                emit_id=f"sae_features:{layer_path}:table",
                **base_kwargs,
            )
            emitted.append("table")

        if _SENDER and config["emit_scalar"]:
            _SENDER.send_scalar(
                active_count,
                label="Active SAE features (L0)",
                emit_id=f"sae_features:{layer_path}:scalar",
                **base_kwargs,
            )
            emitted.append("scalar")

        if config["emit_mode"] == "final":
            try:
                ltts_event.mark_emission_finalized(METADATA["name"], layer_path)
            except Exception:
                logger.debug(
                    "SAE_FEATURES: mark_emission_finalized failed", exc_info=True
                )

        return {
            "status": "ok",
            "emitted": emitted,
            "top_features": top_indices,
            "active_count": active_count,
        }

    except Exception as exc:
        logger.error("SAE_FEATURES: error: %s", exc, exc_info=True)
        try:
            ltts_event.reactor.signal(
                "error",
                {"message": f"sae_features error: {exc}", "module": METADATA["name"]},
            )
        except Exception:
            pass
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
