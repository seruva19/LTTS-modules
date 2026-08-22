"""
Occlusion Attribution Module

Perturbation-based token attribution: runs a baseline forward pass over a
prompt, then re-runs it once per input position with that token replaced by a
neutral token (pad / unk / mask), and reports how much the probability of the
baseline top-1 next-token prediction drops. Attribution[i] = baseline_prob -
perturbed_prob: large positive values mark tokens the prediction depends on.

The module runs after the main model forward and can also be triggered through
its UI button. Extra forwards temporarily suppress LTTS layer events so other
attached analyzers are not invoked repeatedly by the perturbation loop.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from scripts.core.ltts_module import LTTSModuleBase
from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)

METADATA = {
    "name": "occlusion_attribution",
    "version": "0.1.0",
    "description": "Perturbation-based token attribution: occlude each input token and measure the drop in the baseline top-1 prediction probability",
    "author": "LTTS",
    "event_types": ["model_after"],
    "dependencies": ["requirements.txt"],
    "module_type": "model_level",
    "methods": {
        "ctor": "initialize_module",
        "dtor": "cleanup_module",
        "ntor": "process_event",
        "utor": "get_ui_schema",
    },
}

DEFAULT_MAX_POSITIONS = 64

_SENDER = None
_MODULE_INSTANCE = None


def initialize_module(context, **config):
    global _SENDER, _MODULE_INSTANCE
    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)
    _MODULE_INSTANCE = OcclusionAttributionModule()
    logger.info("OCCLUSION_ATTRIBUTION: initialized with config: %s", config)
    return {"status": "initialized", "config": config}


def cleanup_module():
    global _MODULE_INSTANCE
    _MODULE_INSTANCE = None
    logger.info("OCCLUSION_ATTRIBUTION: cleaned up")
    return {"status": "cleaned_up"}


def get_ui_schema():
    return {
        "parameters": [
            {
                "name": "prompt",
                "type": "multiline_text",
                "label": "Prompt to analyze",
                "default": "",
            },
            {
                "name": "max_positions",
                "type": "number",
                "label": "Max positions to occlude",
                "default": DEFAULT_MAX_POSITIONS,
                "min": 4,
                "max": 256,
            },
        ],
        "layout": {"type": "vertical"},
        "real_time": {"enabled": False, "interval": 0},
    }


def process_event(ltts_event):
    """Run bounded token occlusion automatically after the main forward."""
    try:
        if ltts_event.event_type != "model_after":
            return {"status": "skipped", "reason": "requires_model_after"}
        ltts_model = getattr(ltts_event.context, "ltts_model", None)
        prompt = getattr(ltts_model, "_last_input_text", "") if ltts_model else ""
        config = dict(ltts_event.environment.get("module_config") or {})
        config.update(ltts_event.module_state or {})
        config.setdefault("prompt", prompt)
        result = _get_instance()._run_occlusion(
            config,
            ltts_event.module_state,
            ltts_model=ltts_model,
        )
        if result.get("status") == "ok":
            ltts_event.reactor.signal(
                "visualization",
                {
                    "module": METADATA["name"],
                    "subtype": "token_attribution",
                    "event_type": ltts_event.event_type,
                    "layer_path": "model",
                    "data": {
                        "type": "chart",
                        "chart_type": "bar",
                        "data": {
                            "positions": [
                                item["position"] for item in result["attributions"]
                            ],
                            "tokens": [
                                item["token"] for item in result["attributions"]
                            ],
                            "attribution": [
                                item["attribution"]
                                for item in result["attributions"]
                            ],
                        },
                        "baseline_top_token": result["baseline_top_token"],
                        "baseline_probability": result["baseline_probability"],
                    },
                },
            )
        return result
    except Exception as exc:
        logger.error("OCCLUSION_ATTRIBUTION: passive analysis failed: %s", exc)
        ltts_event.reactor.signal(
            "error",
            {"module": METADATA["name"], "message": str(exc)},
        )
        return {"status": "error", "message": str(exc)}


def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, result))


def _get_loaded_model_and_tokenizer(ltts_model=None) -> Tuple[Any, Any, Optional[str]]:
    """Find the currently loaded HF model + tokenizer via the global LTTS
    integration (scripts.core.integration._global_integration, the same
    registry InferenceService's ModelManager reads via
    context.ltts_integration.wrapped_models).

    Returns (model, tokenizer, error_message); error_message is None on
    success. Never raises.
    """
    if ltts_model is not None:
        base_model = getattr(ltts_model, "original_model", None)
        tokenizer = getattr(ltts_model, "_tokenizer", None)
        if base_model is None:
            return None, None, "wrapped model has no underlying HuggingFace model"
        if tokenizer is None:
            return None, None, "loaded model has no tokenizer"
        return base_model, tokenizer, None

    try:
        import scripts.core.integration as integration_mod
    except Exception as exc:
        return None, None, f"could not import LTTS integration: {exc}"

    # Read the existing global instance directly; calling get_integration()
    # here could construct a fresh, empty integration, which is never useful.
    integration = getattr(integration_mod, "_global_integration", None)
    if integration is None:
        return None, None, "LTTS integration not initialized (no inference has run yet)"

    wrapped = getattr(integration, "wrapped_models", None)
    if not wrapped:
        return None, None, "no model is currently loaded/wrapped"

    # Prefer the most recently wrapped model.
    ltts_model = list(wrapped.values())[-1]
    base_model = getattr(ltts_model, "original_model", None) or getattr(
        ltts_model, "model", None
    )
    tokenizer = getattr(ltts_model, "_tokenizer", None)

    if base_model is None:
        return None, None, "wrapped model has no underlying HuggingFace model"
    if tokenizer is None:
        return (
            None,
            None,
            "no tokenizer available on the loaded model (run one inference first so the model caches its tokenizer)",
        )
    return base_model, tokenizer, None


def _pick_occlusion_token_id(tokenizer) -> Optional[int]:
    """First available of pad / unk / mask (eos as last resort)."""
    for attr in ("pad_token_id", "unk_token_id", "mask_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            return int(token_id)
    return None


def _forward_last_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    kwargs = {}
    if getattr(getattr(model, "config", None), "is_encoder_decoder", False):
        # An encoder-decoder refuses to run without a decoder input. Start it
        # from the decoder's own start token so the logits belong to the first
        # decoding step rather than to a borrowed encoder position.
        start_id = getattr(model.config, "decoder_start_token_id", None)
        if start_id is None:
            start_id = getattr(model.config, "pad_token_id", 0) or 0
        kwargs["decoder_input_ids"] = torch.full(
            (input_ids.shape[0], 1), int(start_id),
            dtype=input_ids.dtype, device=input_ids.device,
        )
    outputs = model(input_ids=input_ids, **kwargs)
    logits = getattr(outputs, "logits", None)
    if logits is None:
        logits = outputs[0]
    return logits[0, -1, :].float()


def _decode_token(tokenizer, token_id: int) -> str:
    try:
        return repr(tokenizer.decode([token_id]))
    except Exception:
        return f"#{token_id}"


class OcclusionAttributionModule(LTTSModuleBase):
    """Stateful run-on-demand occlusion analysis."""

    def _register_module_routes(self):
        self.add_stateful_button(
            {
                "id": "run_occlusion",
                "label": "Run occlusion analysis",
                "variant": "primary",
            },
            "handle_run_occlusion",
        )

    def handle_run_occlusion(self, event_data, module_state):
        try:
            return self._run_occlusion(event_data or {}, module_state or {})
        except Exception as exc:
            logger.error("OCCLUSION_ATTRIBUTION: error: %s", exc, exc_info=True)
            return {"status": "error", "message": str(exc)}

    # ------------------------------------------------------------------

    def _resolve_prompt(self, event_data: Dict[str, Any], module_state: Dict[str, Any]) -> str:
        for source in (event_data, event_data.get("module_config"), module_state):
            if isinstance(source, dict):
                prompt = source.get("prompt")
                if isinstance(prompt, str) and prompt.strip():
                    return prompt
        return ""

    def _resolve_max_positions(
        self, event_data: Dict[str, Any], module_state: Dict[str, Any]
    ) -> int:
        for source in (event_data, event_data.get("module_config"), module_state):
            if isinstance(source, dict) and source.get("max_positions") is not None:
                return _coerce_int(
                    source.get("max_positions"), DEFAULT_MAX_POSITIONS, 4, 256
                )
        return DEFAULT_MAX_POSITIONS

    def _run_occlusion(
        self,
        event_data: Dict[str, Any],
        module_state: Dict[str, Any],
        ltts_model=None,
    ) -> Dict[str, Any]:
        prompt = self._resolve_prompt(event_data, module_state)
        if not prompt:
            return {
                "status": "error",
                "message": "no prompt provided: set the 'prompt' parameter before running",
            }
        max_positions = self._resolve_max_positions(event_data, module_state)

        model, tokenizer, error = _get_loaded_model_and_tokenizer(ltts_model)
        if error:
            return {"status": "error", "message": error}

        occlusion_id = _pick_occlusion_token_id(tokenizer)
        if occlusion_id is None:
            return {
                "status": "error",
                "message": "tokenizer has no pad/unk/mask/eos token to occlude with",
            }

        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cpu")

        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        seq_len = int(input_ids.shape[1])
        if seq_len < 1:
            return {"status": "error", "message": "prompt tokenized to an empty sequence"}

        # Cap the number of occluded positions to bound cost: analyze the
        # last `max_positions` tokens of longer prompts.
        start = max(0, seq_len - max_positions)
        truncated = start > 0

        was_training = getattr(model, "training", False)
        if was_training:
            model.eval()

        try:
            if ltts_model is not None:
                ltts_model._suspend_analysis_events = True
            with torch.no_grad():
                baseline_logits = _forward_last_logits(model, input_ids)
                baseline_probs = torch.softmax(baseline_logits, dim=-1)
                top_id = int(torch.argmax(baseline_probs).item())
                baseline_prob = float(baseline_probs[top_id].item())

                attributions: List[Dict[str, Any]] = []
                token_ids = input_ids[0].detach().cpu().tolist()
                for position in range(start, seq_len):
                    perturbed_ids = input_ids.clone()
                    perturbed_ids[0, position] = occlusion_id
                    perturbed_logits = _forward_last_logits(model, perturbed_ids)
                    perturbed_probs = torch.softmax(perturbed_logits, dim=-1)
                    perturbed_prob = float(perturbed_probs[top_id].item())
                    attributions.append(
                        {
                            "position": position,
                            "token": _decode_token(tokenizer, int(token_ids[position])),
                            "attribution": baseline_prob - perturbed_prob,
                        }
                    )
        finally:
            if ltts_model is not None:
                ltts_model._suspend_analysis_events = False
            if was_training:
                model.train()

        top_token_label = _decode_token(tokenizer, top_id)
        self._emit_results(attributions, top_token_label, baseline_prob)

        result: Dict[str, Any] = {
            "status": "ok",
            "prompt": prompt,
            "baseline_top_token": top_token_label,
            "baseline_top_token_id": top_id,
            "baseline_probability": round(baseline_prob, 6),
            "occlusion_token_id": occlusion_id,
            "sequence_length": seq_len,
            "analyzed_positions": len(attributions),
            "truncated": truncated,
            "attributions": [
                {
                    "position": entry["position"],
                    "token": entry["token"],
                    "attribution": round(entry["attribution"], 6),
                }
                for entry in attributions
            ],
        }
        if truncated:
            result["message"] = (
                f"sequence has {seq_len} tokens; only the last {len(attributions)} "
                f"positions were analyzed (max_positions={max_positions})"
            )
        return result

    def _emit_results(
        self,
        attributions: List[Dict[str, Any]],
        top_token_label: str,
        baseline_prob: float,
    ) -> None:
        """Emit a sorted table and a per-position bar chart. Never raises."""
        if _SENDER is None:
            logger.warning(
                "OCCLUSION_ATTRIBUTION: data sender not initialized; returning numbers only"
            )
            return
        try:
            ranked = sorted(
                attributions, key=lambda entry: entry["attribution"], reverse=True
            )
            _SENDER.send_table(
                ["Position", "Token", "Attribution"],
                [
                    [entry["position"], entry["token"], round(entry["attribution"], 5)]
                    for entry in ranked
                ],
                emit_id="occlusion_attribution:table",
                layer_path="model",
            )
            _SENDER.send_chart(
                {
                    "series": [
                        {
                            "name": f"Drop in P({top_token_label}) (baseline {baseline_prob:.4f})",
                            "points": [
                                {
                                    "x": entry["position"],
                                    "y": round(entry["attribution"], 5),
                                }
                                for entry in attributions
                            ],
                        }
                    ],
                    "x_label": "Input position",
                    "y_label": "Attribution (probability drop)",
                },
                "bar",
                emit_id="occlusion_attribution:chart",
                layer_path="model",
            )
        except Exception:
            logger.warning(
                "OCCLUSION_ATTRIBUTION: failed to emit visualization", exc_info=True
            )


def _get_instance():
    global _MODULE_INSTANCE
    if _MODULE_INSTANCE is None:
        _MODULE_INSTANCE = OcclusionAttributionModule()
    return _MODULE_INSTANCE


def interceptor(ltts_event):
    return process_event(ltts_event)
