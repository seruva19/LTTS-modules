"""
Token Ablation Analyzer Module

Analyzes token contributions using ablation/occlusion methods.
Determines token importance by measuring the impact when tokens are removed or replaced.

NOTE: For gradient-based analysis, use gradient_importance module.
NOTE: For attention-based analysis, use attn_visualizer module.
"""

import threading
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from collections import defaultdict
import sys
import os

# module_data_sender will be resolved at runtime via context

logger = logging.getLogger(__name__)

_MODULE_LOCAL = threading.local()

# Reentrancy guard: set while a full-model occlusion forward is running. The
# internal base_model(...) calls re-fire layer_after for every layer, which
# would re-enter this module's occlusion path and recurse without bound. When
# the flag is set, layer interception returns immediately so the nested events
# triggered by our own forward pass do nothing.
_IN_FULL_FORWARD = False

METADATA = {
    "name": "Token Ablation Analyzer",
    "version": "2.0.0",
    "description": "Analyze token contributions using ablation/occlusion methods",
    "author": "LTTS Team",
    "event_types": ["layer_after"],
    "methods": {
        "ctor": "initialize",
        "dtor": "cleanup",
        "ntor": "intercept_layer",
        "utor": "get_ui_schema",
    },
    "dependencies": ["requirements.txt"],
    "tags": ["tokens", "ablation", "occlusion", "interpretability"],
}


class TokenContributionAnalyzer:
    def __init__(self):
        self.token_contributions = {}
        self.token_embeddings = {}
        self.token_predictions = {}
        self.layer_token_data = {}
        self.enabled = True
        self.config = {
            "contribution_method": "occlusion",  # the only method this module implements
            "top_k_tokens": 10,
            "track_predictions": True,
            "aggregate_by_position": True,
            "min_contribution_threshold": 0.01,
            "occlusion_mode": "full_model_forward",
            "occlusion_token_strategy": "pad",
            "max_full_occlusion_tokens": 64,
            "occlusion_use_attention_mask": True,
            "skip_special_tokens": True,
        }
        self.token_vocab = None
        self.baseline_predictions = {}
        # data_sender will be initialized at runtime via context
        self.data_sender = None
        self._full_occlusion_cache = {}

    def initialize(self, context):
        """Initialize the token contribution analyzer."""
        logger.info("Token Contribution Analyzer initialized")

        # Resolve module_data_sender at runtime via context
        if hasattr(context, "module_data_sender"):
            self.data_sender = context.module_data_sender.get_data_sender(
                "token_contributor"
            )
            self.data_sender.set_context(context)
        else:
            logger.warning("module_data_sender not available in context")

        return True

    def cleanup(self):
        """Clean up resources."""
        self.token_contributions.clear()
        self.token_embeddings.clear()
        self.token_predictions.clear()
        self.layer_token_data.clear()
        self.baseline_predictions.clear()
        self._full_occlusion_cache.clear()
        logger.info("Token Contribution Analyzer cleaned up")

    def intercept_layer(self, layer_output, layer_info, context):
        """Intercept layer output to analyze token contributions."""
        if not self.enabled:
            return layer_output

        # Re-entrancy guard: a full-model occlusion forward (below) re-fires
        # layer_after for every layer; ignore those nested events so we do not
        # recurse into the occlusion path.
        if _IN_FULL_FORWARD:
            return layer_output

        try:
            layer_idx = layer_info.get("layer_index", 0)
            layer_name = layer_info.get("layer_name", f"layer_{layer_idx}")

            # Extract token data from layer output
            token_data = self._extract_token_data(layer_output, layer_info)
            if token_data is not None:
                # Store token embeddings and data
                self.token_embeddings[layer_name] = token_data.get("embeddings", None)
                self.layer_token_data[layer_name] = token_data

                # Compute token contributions
                contributions = self._compute_token_contributions(
                    token_data, layer_name, context
                )

                if contributions is not None:
                    self.token_contributions[layer_name] = contributions

                    # Analyze token predictions if enabled
                    if self.config["track_predictions"]:
                        predictions = self._analyze_token_predictions(
                            token_data, layer_name
                        )
                        self.token_predictions[layer_name] = predictions

                    # Send visualization data via data sender
                    try:
                        if self.data_sender is None:
                            logger.warning(
                                "data_sender not initialized, skipping visualization"
                            )
                            return layer_output

                        # Send method info as text
                        self.data_sender.send_text(
                            f"Method: Ablation/Occlusion",
                            preformatted=True,
                        )

                        # Send token importance as chart
                        if contributions and "importance" in contributions:
                            self.data_sender.send_chart(
                                {
                                    "positions": list(
                                        range(len(contributions["importance"]))
                                    ),
                                    "importance": contributions["importance"],
                                },
                                "line",
                                layer_name=layer_name,
                            )

                        # Send attention matrix as heatmap if available
                        if contributions and "attention_matrix" in contributions:
                            self.data_sender.send_heatmap(
                                values=contributions["attention_matrix"],
                                dimensions={
                                    "rows": len(contributions["attention_matrix"]),
                                    "cols": (
                                        len(contributions["attention_matrix"][0])
                                        if contributions["attention_matrix"]
                                        else 0
                                    ),
                                },
                                layer_name=layer_name,
                            )

                        # Send top contributors as table
                        top_contributors = self._get_top_contributors(contributions)
                        if top_contributors:
                            table_data = {
                                "headers": ["Position", "Contribution"],
                                "rows": [
                                    [str(c["position"]), f"{c['contribution']:.4f}"]
                                    for c in top_contributors
                                ],
                            }
                            self.data_sender.send_table(
                                headers=table_data["headers"],
                                rows=table_data["rows"],
                                layer_name=layer_name,
                            )

                        # Send average hidden state norm if available
                        if layer_name in self.token_predictions:
                            predictions = self.token_predictions[layer_name]
                            if predictions.get("token_norm"):
                                avg_norm = np.mean(predictions["token_norm"])
                                self.data_sender.send_scalar(
                                    avg_norm, f"{layer_name} Avg Hidden Norm"
                                )

                    except Exception as e:
                        logger.error(f"Error sending visualization data: {e}")

        except Exception as e:
            logger.error(f"Error in token contribution analyzer: {e}")

        return layer_output

    def _extract_token_data(self, layer_output, layer_info):
        """Extract token-related data from layer output."""
        try:
            token_data = {
                "layer_index": layer_info.get("layer_index", 0),
                "layer_name": layer_info.get("layer_name", "unknown"),
                "embeddings": None,
                "attentions": None,
                "hidden_states": None,
                "input_ids": None,
            }

            # Extract different types of token representations
            if isinstance(layer_output, torch.Tensor):
                token_data["hidden_states"] = layer_output
            elif hasattr(layer_output, "last_hidden_state"):
                token_data["hidden_states"] = layer_output.last_hidden_state
            elif hasattr(layer_output, "hidden_states") and layer_output.hidden_states:
                token_data["hidden_states"] = layer_output.hidden_states[-1]

            # Extract attention weights if available
            if hasattr(layer_output, "attentions"):
                token_data["attentions"] = layer_output.attentions

            # Extract input tokens if available
            if hasattr(layer_output, "input_ids"):
                token_data["input_ids"] = layer_output.input_ids

            return token_data

        except Exception as e:
            logger.error(f"Error extracting token data: {e}")
            return None

    def _resolve_occlusion_token_id(self, tokenizer, strategy: str) -> Optional[int]:
        if tokenizer is None:
            return None
        strategy = (strategy or "").lower()
        if strategy == "mask" and getattr(tokenizer, "mask_token_id", None) is not None:
            return tokenizer.mask_token_id
        if strategy == "unk" and getattr(tokenizer, "unk_token_id", None) is not None:
            return tokenizer.unk_token_id
        if strategy == "eos" and getattr(tokenizer, "eos_token_id", None) is not None:
            return tokenizer.eos_token_id
        if getattr(tokenizer, "pad_token_id", None) is not None:
            return tokenizer.pad_token_id
        if getattr(tokenizer, "eos_token_id", None) is not None:
            return tokenizer.eos_token_id
        return None

    def _get_model_inputs(self, context):
        ltts_model = getattr(context, "ltts_model", None)
        if ltts_model is None:
            return None, None, None, None
        tokenizer = getattr(ltts_model, "_tokenizer", None)
        input_ids = getattr(ltts_model, "_last_input_ids", None)
        attention_mask = getattr(ltts_model, "_last_attention_mask", None)
        base_model = getattr(ltts_model, "original_model", None)
        return base_model, tokenizer, input_ids, attention_mask

    def _compute_full_model_occlusion(self, context):
        base_model, tokenizer, input_ids, attention_mask = self._get_model_inputs(context)
        if base_model is None or input_ids is None:
            logger.warning("Full occlusion requires a loaded model and input_ids")
            return None

        cache_key = (tuple(input_ids), tuple(attention_mask) if attention_mask else None)
        if cache_key in self._full_occlusion_cache:
            return self._full_occlusion_cache[cache_key]

        device = next(base_model.parameters()).device
        input_ids_tensor = torch.tensor([input_ids], device=device)
        attention_tensor = (
            torch.tensor([attention_mask], device=device)
            if attention_mask is not None
            else None
        )

        occlusion_token_id = self._resolve_occlusion_token_id(
            tokenizer, self.config.get("occlusion_token_strategy", "pad")
        )
        if occlusion_token_id is None:
            logger.warning("No suitable occlusion token found; using 0 as fallback")
            occlusion_token_id = 0

        max_tokens = min(len(input_ids), int(self.config.get("max_full_occlusion_tokens", 64)))
        if max_tokens <= 0:
            return None

        global _IN_FULL_FORWARD
        _IN_FULL_FORWARD = True
        ltts_model = getattr(context, "ltts_model", None)
        if ltts_model is not None:
            ltts_model._suspend_analysis_events = True
        # An encoder-decoder will not run without a decoder input; start it from
        # the decoder's own start token so every forward here is comparable.
        decoder_kwargs = {}
        if getattr(getattr(base_model, "config", None), "is_encoder_decoder", False):
            start_id = getattr(base_model.config, "decoder_start_token_id", None)
            if start_id is None:
                start_id = getattr(base_model.config, "pad_token_id", 0) or 0
            decoder_kwargs["decoder_input_ids"] = torch.full(
                (input_ids_tensor.shape[0], 1), int(start_id),
                dtype=input_ids_tensor.dtype, device=input_ids_tensor.device,
            )

        try:
            with torch.no_grad():
                baseline_outputs = base_model(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_tensor,
                    return_dict=True,
                    **decoder_kwargs,
                )
                baseline_logits = getattr(baseline_outputs, "logits", None)
                if baseline_logits is None:
                    logger.warning("Model outputs do not include logits for occlusion")
                    return None

                contributions = np.zeros(max_tokens, dtype=np.float32)
                for idx in range(max_tokens):
                    if self.config.get("skip_special_tokens", True) and tokenizer is not None:
                        if input_ids[idx] in getattr(tokenizer, "all_special_ids", []):
                            continue

                    occluded_ids = input_ids_tensor.clone()
                    occluded_ids[0, idx] = occlusion_token_id

                    occluded_mask = attention_tensor
                    if self.config.get("occlusion_use_attention_mask", True) and attention_tensor is not None:
                        occluded_mask = attention_tensor.clone()
                        occluded_mask[0, idx] = 0

                    occluded_outputs = base_model(
                        input_ids=occluded_ids,
                        attention_mask=occluded_mask,
                        return_dict=True,
                        **decoder_kwargs,
                    )
                    occluded_logits = getattr(occluded_outputs, "logits", None)
                    if occluded_logits is None:
                        continue

                    delta = torch.mean(torch.abs(occluded_logits - baseline_logits)).item()
                    contributions[idx] = delta
        finally:
            if ltts_model is not None:
                ltts_model._suspend_analysis_events = False
            _IN_FULL_FORWARD = False

        result = {
            "importance": contributions.tolist(),
            "method": "occlusion_full_forward",
            "baseline_type": "model_logits_mean_delta",
        }
        self._full_occlusion_cache[cache_key] = result
        return result

    def _compute_token_contributions(self, token_data, layer_name, context):
        """
        Compute token contributions using ablation/occlusion method.

        For gradient-based attribution see `gradient_importance`; for
        attention-based analysis see `attn_visualizer`.
        """
        method = self.config["contribution_method"]

        if method == "occlusion":
            mode = self.config.get("occlusion_mode", "hidden_state_proxy")
            if mode == "full_model_forward":
                return self._compute_full_model_occlusion(context)
            return self._compute_occlusion_contributions(token_data, layer_name, context)
        else:
            logger.warning(f"Unknown contribution method: {method}. Only 'occlusion' is supported.")
            return None

    def _compute_occlusion_contributions(self, token_data, layer_name, context):
        """
        Compute token contributions using occlusion (ablation) method.

        Measures how much replacing a token with the baseline changes the
        hidden state.
        """
        hidden_states = token_data.get("hidden_states")

        if hidden_states is None:
            logger.warning("No hidden states available for occlusion analysis")
            return None

        try:
            # Get sequence length
            seq_len = hidden_states.shape[1]
            contributions = np.zeros(seq_len)

            # Compute baseline (mean hidden state)
            baseline_hidden = torch.mean(hidden_states, dim=1, keepdim=True)

            for i in range(seq_len):
                # Occlude token by replacing with baseline
                occluded_hidden = hidden_states.clone()
                occluded_hidden[:, i, :] = baseline_hidden

                # Contribution proxy taken in hidden-state space: the distance
                # between the token's representation and the baseline. Larger
                # distance marks a token the layer's output depends on more.
                contribution = torch.norm(
                    hidden_states[:, i, :] - baseline_hidden, dim=-1
                )
                contributions[i] = torch.mean(contribution).item()

            return {
                "importance": contributions.tolist(),
                "method": "occlusion",
                "baseline_type": "mean_hidden_state"
            }

        except Exception as e:
            logger.error(f"Error computing occlusion contributions: {e}")
            return None

    def _analyze_token_predictions(self, token_data, layer_name):
        """Analyze token-level hidden state statistics.

        Computes meaningful per-token metrics from hidden states:
        - L2 norm: magnitude of the hidden representation (higher = more activated)
        - Cosine similarity to mean: how typical this token's representation is
        """
        hidden_states = token_data.get("hidden_states")
        if hidden_states is None:
            return {}

        try:
            predictions = {
                "token_norm": [],
                "token_cosine_to_mean": [],
                "top_predictions": [],
            }

            # hidden_states: [batch, seq_len, hidden_dim]
            seq_len = hidden_states.shape[1]
            first_batch = hidden_states[0]  # [seq_len, hidden_dim]

            # Mean hidden state across sequence
            mean_hidden = torch.mean(first_batch, dim=0, keepdim=True)  # [1, hidden_dim]
            mean_norm = torch.norm(mean_hidden, dim=-1, keepdim=True).clamp(min=1e-8)

            for i in range(seq_len):
                token_hidden = first_batch[i]  # [hidden_dim]

                # L2 norm of hidden state (meaningful: represents activation magnitude)
                norm_val = torch.norm(token_hidden).item()
                predictions["token_norm"].append(norm_val)

                # Cosine similarity to mean hidden state (meaningful: how typical)
                cos_sim = (
                    torch.dot(token_hidden, mean_hidden.squeeze())
                    / (torch.norm(token_hidden).clamp(min=1e-8) * mean_norm.squeeze())
                ).item()
                predictions["token_cosine_to_mean"].append(cos_sim)

            # Rank tokens by norm (most activated)
            if predictions["token_norm"]:
                top_indices = np.argsort(predictions["token_norm"])[
                    -self.config["top_k_tokens"] :
                ][::-1]
                predictions["top_predictions"] = [
                    {
                        "position": int(idx),
                        "norm": predictions["token_norm"][idx],
                        "cosine_to_mean": predictions["token_cosine_to_mean"][idx],
                    }
                    for idx in top_indices
                ]

            return predictions

        except Exception as e:
            logger.error(f"Error analyzing token predictions: {e}")
            return {}

    def _get_top_contributors(self, contributions):
        """Get top contributing tokens."""
        if contributions is None:
            return []

        importance = contributions.get("importance", [])
        if not importance:
            return []

        top_indices = np.argsort(importance)[-self.config["top_k_tokens"] :][::-1]

        return [
            {"position": int(idx), "contribution": importance[idx]}
            for idx in top_indices
        ]

    def get_token_summary(self):
        """Get summary of token contribution analysis."""
        return {
            "method": self.config["contribution_method"],
            "layers_analyzed": len(self.token_contributions),
            "total_tokens_analyzed": sum(
                len(data.get("importance", []))
                for data in self.token_contributions.values()
            ),
            "layer_details": {
                layer_name: {
                    "max_contribution": (
                        float(np.max(data.get("importance", [0])))
                        if data.get("importance")
                        else 0
                    ),
                    "avg_contribution": (
                        float(np.mean(data.get("importance", [0])))
                        if data.get("importance")
                        else 0
                    ),
                    "num_tokens": len(data.get("importance", [])),
                }
                for layer_name, data in self.token_contributions.items()
            },
        }


    def reset_data(self):
        """Reset collected token data."""
        self.token_contributions.clear()
        self.token_embeddings.clear()
        self.token_predictions.clear()
        self.layer_token_data.clear()
        self.baseline_predictions.clear()
        self._full_occlusion_cache.clear()


def _get_instance():
    inst = getattr(_MODULE_LOCAL, "instance", None)
    if inst is None:
        inst = TokenContributionAnalyzer()
        _MODULE_LOCAL.instance = inst
    return inst


def initialize(context):
    """Module constructor."""
    analyzer = TokenContributionAnalyzer()
    analyzer.initialize(context)
    _MODULE_LOCAL.instance = analyzer
    return analyzer


def cleanup(module):
    """Module destructor."""
    if hasattr(module, "cleanup"):
        module.cleanup()
    _MODULE_LOCAL.instance = None


def intercept_layer(module, layer_output, layer_info, context):
    """Layer interception method."""
    return module.intercept_layer(layer_output, layer_info, context)


def _infer_layer_index(layer_path: str) -> int:
    if not layer_path:
        return 0
    parts = str(layer_path).split(".")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0


def interceptor(ltts_event):
    """LTTSEvent interceptor for token contribution analysis."""
    # If we are inside our own full-model occlusion forward, the nested
    # layer_after events must be ignored to avoid unbounded recursion.
    if _IN_FULL_FORWARD:
        return {"status": "skipped", "reason": "reentrant_forward"}

    _MODULE_INSTANCE = _get_instance()
    if _MODULE_INSTANCE.data_sender is not None:
        _MODULE_INSTANCE.data_sender.set_context(ltts_event.context)
    if isinstance(ltts_event.module_state, dict):
        _MODULE_INSTANCE.config.update(ltts_event.module_state)
    layer_output = getattr(ltts_event.context, "outputs", None)
    layer_path = getattr(ltts_event.context, "module_path", "unknown")
    layer_info = {
        "layer_name": layer_path,
        "layer_index": _infer_layer_index(layer_path),
    }
    # Run analysis (populates internal state regardless of data_sender)
    result = _MODULE_INSTANCE.intercept_layer(layer_output, layer_info, ltts_event.context)

    # Emit visualization data via reactor for consistent client delivery
    layer_name = layer_info["layer_name"]
    contributions = _MODULE_INSTANCE.token_contributions.get(layer_name)
    if contributions is not None:
        # Send token importance chart
        if "importance" in contributions:
            ltts_event.reactor.signal("visualization", {
                "type": "visualization",
                "subtype": "chart",
                "module": "token_contributor",
                "event_type": "module_event",
                "module_path": layer_path,
                "layer_path": layer_path,
                "data": {
                    "type": "chart",
                    "chart_type": "line",
                    "data": {
                        "positions": list(range(len(contributions["importance"]))),
                        "importance": contributions["importance"],
                    },
                },
            })

        # Send top contributors table
        top_contributors = _MODULE_INSTANCE._get_top_contributors(contributions)
        if top_contributors:
            ltts_event.reactor.signal("visualization", {
                "type": "visualization",
                "subtype": "table",
                "module": "token_contributor",
                "event_type": "module_event",
                "module_path": layer_path,
                "layer_path": layer_path,
                "data": {
                    "type": "table",
                    "headers": ["Position", "Contribution"],
                    "rows": [
                        [str(c["position"]), f"{c['contribution']:.4f}"]
                        for c in top_contributors
                    ],
                },
            })

        # Send attention matrix heatmap if available
        if "attention_matrix" in contributions:
            ltts_event.reactor.signal("visualization", {
                "type": "visualization",
                "subtype": "heatmap",
                "module": "token_contributor",
                "event_type": "module_event",
                "module_path": layer_path,
                "layer_path": layer_path,
                "data": {
                    "type": "heatmap",
                    "values": contributions["attention_matrix"],
                    "dimensions": {
                        "rows": len(contributions["attention_matrix"]),
                        "cols": (
                            len(contributions["attention_matrix"][0])
                            if contributions["attention_matrix"]
                            else 0
                        ),
                    },
                },
            })

    return result


def get_ui_schema():
    """Get UI configuration schema."""
    return {
        "type": "object",
        "properties": {
            "top_k_tokens": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 10,
                "description": "Number of top contributing tokens to track",
            },
            "track_predictions": {
                "type": "boolean",
                "default": True,
                "description": "Track token-level predictions and confidence",
            },
            "min_contribution_threshold": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.01,
                "description": "Minimum contribution threshold for analysis",
            },
            "occlusion_mode": {
                "type": "string",
                "enum": ["hidden_state_proxy", "full_model_forward"],
                "default": "full_model_forward",
                "description": "Choose fast proxy or full forward occlusion",
            },
            "occlusion_token_strategy": {
                "type": "string",
                "enum": ["pad", "mask", "unk", "eos"],
                "default": "pad",
                "description": "Token used to replace occluded positions",
            },
            "max_full_occlusion_tokens": {
                "type": "integer",
                "minimum": 1,
                "maximum": 256,
                "default": 64,
                "description": "Max tokens to occlude in full forward mode",
            },
            "occlusion_use_attention_mask": {
                "type": "boolean",
                "default": True,
                "description": "Zero attention mask at occluded positions",
            },
            "skip_special_tokens": {
                "type": "boolean",
                "default": True,
                "description": "Skip special tokens during occlusion",
            },
        },
        "required": [],
        "help": "This module uses ABLATION/OCCLUSION analysis. For gradient-based analysis use 'gradient_importance', for attention analysis use 'attn_visualizer'."
    }
