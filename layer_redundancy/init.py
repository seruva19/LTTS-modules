"""Block Influence per decoder block: which layers actually change the representation.

BI = 1 - E_t[cos(X_t, Y_t)] over sequence positions, for a block mapping input
hidden state X to output hidden state Y (Men et al., arXiv:2403.03853,
"ShortGPT"). A near-zero BI means the block returned its input almost unrotated:
a pruning candidate, and a low-BI stretch of depth is mechanistically dead.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "layer_redundancy",
    "version": "0.1.0",
    "description": "Layer-redundancy map: per-block Block Influence (1 - cosine between block input and output) and relative norm change",
    "author": "LTTS",
    "event_types": ["layer_after"],
    "dependencies": ["requirements.txt"],
    "requires": ["hidden_states"],
    "methods": {
        "ctor": "initialize_module",
        "dtor": "cleanup_module",
        "ntor": "process_event",
        "utor": "get_ui_schema",
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "least_influential_k": 5,
    "emit_scalar": True,
    "emit_chart": True,
    "emit_table": True,
    "emit_mode": "final",
}

_SENDER = None

# Each layer event carries a fresh LTTSContext, so the depth profile has to be
# accumulated in module state, keyed by forward pass.
_TRAJECTORY: Dict[str, Any] = {"forward_pass": None, "entries": []}


def initialize_module(context, **config):
    global _SENDER
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    return {"status": "initialized", "config": config}


def cleanup_module():
    _TRAJECTORY["entries"] = []
    _TRAJECTORY["forward_pass"] = None
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "least_influential_k",
                "type": "number",
                "label": "Least-influential layers listed",
                "default": DEFAULT_CONFIG["least_influential_k"],
                "min": 1,
                "max": 50,
            },
            {
                "name": "emit_scalar",
                "type": "boolean",
                "label": "Emit Block Influence scalar",
                "default": True,
            },
            {
                "name": "emit_chart",
                "type": "boolean",
                "label": "Emit Block Influence by depth",
                "default": True,
            },
            {
                "name": "emit_table",
                "type": "boolean",
                "label": "Emit least-influential layers",
                "default": True,
            },
        ],
        "layout": {"type": "vertical"},
        "real_time": {"enabled": True, "interval": 200},
    }


def _extract_hidden(outputs: Any) -> Optional[torch.Tensor]:
    candidate = outputs[0] if isinstance(outputs, (list, tuple)) and outputs else outputs
    if isinstance(candidate, torch.Tensor) and candidate.dim() == 3:
        return candidate
    return None


def _is_decoder_block(context: Any) -> bool:
    if getattr(context, "module_role", None) == "block":
        return True
    module_class = str(getattr(context, "module_class", "") or "").lower()
    return getattr(context, "layer_type", None) == "unknown" and (
        "decoderlayer" in module_class or module_class.endswith(("block", "layer"))
    )


def _matches(candidate: Any, reference: torch.Tensor) -> bool:
    return (
        isinstance(candidate, torch.Tensor)
        and candidate.dim() == 3
        and candidate.shape == reference.shape
    )


def _extract_block_input(inputs: Any, reference: torch.Tensor) -> Optional[torch.Tensor]:
    """Residual stream entering the block, whatever wrapper the hook recorded.

    Only a [batch, seq, hidden] tensor shaped exactly like the block output can
    be the same residual stream; anything else (masks, position ids, caches) is
    rejected rather than guessed at.
    """
    if isinstance(inputs, dict):
        if _matches(inputs.get("hidden_states"), reference):
            return inputs["hidden_states"]
        for value in inputs.values():
            if _matches(value, reference):
                return value
        return None
    if isinstance(inputs, (list, tuple)):
        for value in inputs:
            if _matches(value, reference):
                return value
        return None
    return inputs if _matches(inputs, reference) else None


def _config(event) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if isinstance(event.module_state, dict):
        config.update(event.module_state)
    environment = getattr(event, "environment", None)
    if isinstance(environment, dict) and isinstance(
        environment.get("module_config"), dict
    ):
        config.update(environment["module_config"])
    try:
        config["least_influential_k"] = max(1, min(50, int(config["least_influential_k"])))
    except (TypeError, ValueError):
        config["least_influential_k"] = DEFAULT_CONFIG["least_influential_k"]
    return config


def _block_influence(block_input: torch.Tensor, block_output: torch.Tensor):
    """BI and mean relative norm change, reduced on-device before any transfer."""
    x = block_input[0].detach().float()
    y = block_output[0].detach().float()
    cosine = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    input_norm = torch.linalg.vector_norm(x, dim=-1)
    output_norm = torch.linalg.vector_norm(y, dim=-1)
    norm_ratio = output_norm / input_norm.clamp_min(1e-8)
    return (
        1.0 - float(cosine.mean().item()),
        float(norm_ratio.mean().item()),
    )


def _least_influential(entries: List[Dict[str, Any]], k: int) -> List[List[Any]]:
    ranked = sorted(enumerate(entries), key=lambda item: item[1]["block_influence"])
    return [
        [index, item["layer"], round(item["block_influence"], 6), round(item["norm_ratio"], 6)]
        for index, item in ranked[:k]
    ]


def process_event(ltts_event):
    try:
        if not str(getattr(ltts_event, "event_type", "")).startswith("layer_after"):
            return {"status": "skipped", "reason": "event_phase"}
        context = ltts_event.context
        if not _is_decoder_block(context):
            return {"status": "skipped", "reason": "not_a_block_boundary"}
        hidden = _extract_hidden(context.outputs)
        if hidden is None:
            return {"status": "skipped", "reason": "no_hidden_state"}
        block_input = _extract_block_input(context.inputs, hidden)
        if block_input is None:
            return {"status": "skipped", "reason": "no_matching_block_input"}

        config = _config(ltts_event)
        layer_path = getattr(context, "module_path", None)
        emission = ltts_event.should_emit(METADATA["name"], layer_path, config)
        if not emission.get("emit"):
            return {"status": "skipped", "reason": "emission_controller"}

        with torch.no_grad():
            block_influence, norm_ratio = _block_influence(block_input, hidden)

        current_pass = emission.get("forward_pass")
        if _TRAJECTORY["forward_pass"] != current_pass:
            _TRAJECTORY["forward_pass"] = current_pass
            _TRAJECTORY["entries"] = []
        entry = {
            "layer": layer_path or "unknown",
            "block_influence": block_influence,
            "norm_ratio": norm_ratio,
        }
        _TRAJECTORY["entries"].append(entry)
        entries = _TRAJECTORY["entries"]

        base = {"layer_path": layer_path, "emit_mode": config.get("emit_mode", "final")}
        emitted = []
        if _SENDER and config.get("emit_scalar", True):
            _SENDER.send_scalar(
                round(block_influence, 6),
                label="Block Influence (1 - cos)",
                emit_id=f"layer_redundancy:{layer_path}:scalar",
                **base,
            )
            emitted.append("scalar")
        # The depth profile is a single growing artefact: stable emit_ids make the
        # UI replace it as layers arrive instead of stacking one card per block.
        if _SENDER and config.get("emit_chart", True) and len(entries) > 1:
            _SENDER.send_chart(
                {
                    "series": [
                        {
                            "name": "Block Influence",
                            "points": [
                                {"x": index, "y": round(item["block_influence"], 6)}
                                for index, item in enumerate(entries)
                            ],
                        }
                    ],
                    "x_label": "Layer index",
                    "y_label": "Block Influence",
                },
                "bar",
                emit_id="layer_redundancy:chart",
                **base,
            )
            emitted.append("chart")
        if _SENDER and config.get("emit_table", True):
            _SENDER.send_table(
                ["Layer index", "Layer", "Block Influence", "||Y||/||X||"],
                _least_influential(entries, config["least_influential_k"]),
                emit_id="layer_redundancy:table",
                **base,
            )
            emitted.append("table")

        if config.get("emit_mode") == "final":
            ltts_event.mark_emission_finalized(METADATA["name"], layer_path)
        return {"status": "ok", "emitted": emitted, **entry}
    except Exception as exc:
        logger.error("LAYER_REDUNDANCY error: %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc)}


def interceptor(ltts_event):
    return process_event(ltts_event)
