"""
Gradient-based Importance Module - gradient attribution with Captum

Computes gradient-based importance scores for tokens and neurons using
Captum's attribution methods with an explicit backward pass.

This implementation uses:
- Captum's Saliency for gradient magnitude
- Captum's InputXGradient for activation-weighted gradients
- Captum's IntegratedGradients for path integral attribution

REQUIRES: Computational graph must be retained for backward pass
"""

import torch
import torch.nn.functional as F
import numpy as np
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:  # Structured LTTS visualizations do not require plotting libs
    plt = None
    sns = None
import io
import base64
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import sys
import os
from collections import defaultdict

from .noise_methods import attribution_sanity_report, noise_ensemble_attribution

logger = logging.getLogger(__name__)

# Captum backs the gradient attribution methods
try:
    from captum.attr import Saliency, InputXGradient, IntegratedGradients
    CAPTUM_AVAILABLE = True
    logger.info("Captum loaded: gradient attribution available")
except ImportError:
    CAPTUM_AVAILABLE = False
    logger.error("Captum not available! Install with: pip install captum")
    logger.error("Falling back to activation magnitude (NOT real gradients)")

# Import token attribution visualizer
try:
    from .token_attribution import TokenAttributionAnalyzer
    TOKEN_ATTR_AVAILABLE = True
except ImportError:
    TOKEN_ATTR_AVAILABLE = False
    logger.warning("Token attribution visualizations not available")

# Import Captum gradient functions
if CAPTUM_AVAILABLE:
    try:
        from .captum_gradients import (
            compute_integrated_gradients_captum,
            check_gradient_availability,
            enable_gradients_for_layer,
            compute_gradient_with_backward_pass,
            compute_layer_wise_gradients,
            verify_gradient_flow,
            hook_into_backward_pass,
            benchmark_gradient_methods
        )
    except ImportError:
        logger.warning("Could not import captum_gradients module")

METADATA = {
    "name": "gradient_importance",
    "version": "3.0.0",
    "description": "Target-score gradients and activation × gradient attribution with an explicit backward pass",
    "author": "LTTS Team",
    "event_types": ["layer_after", "layer_before"],
    "methods": {
        "ctor": "initialize",
        "dtor": "cleanup",
        "ntor": "process_event",
        "utor": "get_ui_schema",
    },
    "dependencies": ["requirements.txt"],
    "requires": ["hidden_states", "gradients"],
    "tags": ["gradients", "importance", "attribution", "interpretability", "captum"],
}

# Global state
_module_state = {
    "initialized": False,
    "activations": {},  # Store activations for gradient computation
    "gradients": {},  # Store computed gradients
    "pending": {},  # Layer events waiting for the model-level target backward pass
    "hooks": [],  # Store hook handles
    "config": {
        "method": "saliency",
        "capture_gradients": True,
        "normalize_scores": True,
    },
    "stats": {
        "processed_events": 0,
        "gradients_captured": 0,
    },
}


def initialize(context):
    """Module constructor."""
    global _module_state

    _module_state["initialized"] = True
    _module_state["context"] = context

    # Check Captum availability
    if not CAPTUM_AVAILABLE:
        logger.error("=" * 70)
        logger.error("CAPTUM NOT INSTALLED - Real gradients NOT available!")
        logger.error("Module will fall back to activation magnitude (not real gradients)")
        logger.error("Install Captum: pip install captum")
        logger.error("=" * 70)
        return {"status": "warning", "message": "Captum not installed", "version": "3.0.0"}

    # Initialize token attribution analyzer
    if TOKEN_ATTR_AVAILABLE:
        _module_state["token_analyzer"] = TokenAttributionAnalyzer()
        logger.info("=" * 70)
        logger.info("Gradient Importance Analyzer initialized with Captum v3.0.0")
        logger.info("=" * 70)
    else:
        logger.info("Gradient Importance Analyzer initialized with Captum (v3.0.0)")

    return {"status": "initialized", "version": "3.0.0", "captum": CAPTUM_AVAILABLE}


def cleanup():
    """Module destructor - remove all hooks."""
    global _module_state

    # Remove all registered hooks
    for hook in _module_state["hooks"]:
        try:
            hook.remove()
        except Exception:
            pass

    _module_state["hooks"].clear()
    _module_state["activations"].clear()
    _module_state["gradients"].clear()
    _module_state["pending"].clear()

    logger.info("Gradient Importance Analyzer cleaned up")
    return {"status": "cleaned_up"}


def _prune_hooks(max_hooks: int = 64) -> None:
    """Drop accumulated hook handles once their tensors are gone from the run."""
    hooks = _module_state["hooks"]
    excess = len(hooks) - max_hooks
    if excess <= 0:
        return
    for hook in hooks[:excess]:
        try:
            hook.remove()
        except Exception:
            pass
    del hooks[:excess]


def get_ui_schema():
    """Get UI configuration schema."""
    return {
        "parameters": [
            {
                "name": "method",
                "type": "select",
                "label": "Attribution Method",
                "default": "saliency",
                "options": [
                    "saliency",
                    "input_x_gradient",
                    "integrated_gradients",
                    "explicit_backward",
                    "smoothgrad",
                    "noisegrad",
                    "vargrad",
                ],
                "help": "Gradient method; noise variants operate on captured activations",
            },
            {
                "name": "visualization_type",
                "type": "select",
                "label": "Visualization Type",
                "default": "gradient_heatmap",
                "options": [
                    "gradient_heatmap",
                    "token_importance",
                    "neuron_importance",
                    "gradient_flow",
                    "sanity_checks",
                ],
                "help": "Type of visualization to generate",
            },
            {
                "name": "normalize_scores",
                "type": "boolean",
                "label": "Normalize Scores",
                "default": True,
                "help": "Normalize importance scores to [0, 1] range",
            },
            {
                "name": "top_k_tokens",
                "type": "number",
                "label": "Top K Tokens",
                "default": 10,
                "min": 1,
                "max": 50,
                "help": "Number of top important tokens to highlight",
            },
            {
                "name": "colormap",
                "type": "select",
                "label": "Color Scheme",
                "default": "RdYlGn",
                "options": ["viridis", "plasma", "RdYlGn", "coolwarm", "seismic"],
                "help": "Color scheme for visualization",
            },
            {
                "name": "target_position",
                "type": "number",
                "label": "Target sequence position",
                "default": -1,
                "min": -1,
                "help": "-1 selects the last valid position; ignored for sequence classification",
            },
            {
                "name": "target_token_id",
                "type": "number",
                "label": "Target token/class id",
                "default": -1,
                "min": -1,
                "help": "-1 selects the model's top prediction at the target position",
            },
            {
                "name": "noise_samples",
                "type": "number",
                "label": "Noise samples",
                "default": 16,
                "min": 2,
                "max": 128,
                "help": "Samples used by SmoothGrad, activation-space NoiseGrad, VarGrad, and sanity checks",
            },
            {
                "name": "noise_std",
                "type": "range",
                "label": "Noise standard deviation",
                "default": 0.1,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
            },
            {
                "name": "random_seed",
                "type": "number",
                "label": "Random seed",
                "default": 0,
                "min": 0,
            },
        ],
        "layout": {"type": "grid", "columns": 2},
        "real_time": {"enabled": True, "interval": 200},
    }


def get_ltts_model_from_context_or_global(ltts_event):
    """Get LTTS model only from event context."""
    return getattr(ltts_event.context, "ltts_model", None)


def _first_tensor(value):
    """Extract the primary differentiable tensor from common HF layer outputs."""

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for key in ("last_hidden_state", "hidden_states", "logits"):
            tensor = _first_tensor(value.get(key))
            if tensor is not None:
                return tensor
    for key in ("last_hidden_state", "hidden_states", "logits"):
        tensor = _first_tensor(getattr(value, key, None))
        if tensor is not None:
            return tensor
    return None


def process_event(ltts_event):
    """Main event processing method."""
    global _module_state

    try:
        _module_state["stats"]["processed_events"] += 1

        # Get parameters
        params = ltts_event.module_state or {}
        params = validate_and_set_defaults(params)

        viz_type = params.get("visualization_type", "gradient_heatmap")
        method = params.get("method", "saliency")

        module_name = METADATA.get("name", "gradient_importance")
        actual_layer_path = getattr(ltts_event.context, "module_path", None)

        # Get the LTTS model to access activations and gradients
        ltts_model = get_ltts_model_from_context_or_global(ltts_event)

        # Try to get output tensor
        outputs = _first_tensor(getattr(ltts_event.context, "outputs", None))

        if outputs is None or not isinstance(outputs, torch.Tensor):
            logger.debug("GRADIENT_IMP: No tensor outputs available")
            return

        # Store activation for gradient computation
        layer_path = getattr(ltts_event.context, "module_path", "unknown")
        _module_state["activations"][layer_path] = outputs.detach()

        # Gradient methods are finalized only after InferenceService backprops a
        # concrete model target (top logit/class score by default). Emitting here
        # would only measure a self-referential activation-energy surrogate.
        deferred_methods = {
            "saliency",
            "input_x_gradient",
            "integrated_gradients",
            "explicit_backward",
        }
        if method in deferred_methods:
            # Hook registration and pending accumulation must happen even when
            # emission is throttled (emit_mode first/every_n/final), otherwise
            # gradients would never be collected for those modes.
            if not outputs.requires_grad:
                return {
                    "status": "unsupported",
                    "reason": "layer output is detached from autograd",
                }
            register_gradient_tracking_hook(outputs, layer_path)
            _module_state["pending"][layer_path] = {
                "event": ltts_event,
                "activation": outputs.detach(),
                "params": dict(params),
                "visualization_type": viz_type,
                "method": method,
            }
            return {"status": "pending_backward", "layer_path": layer_path}

        # Check if we should emit (only the direct-visualization path is gated)
        emission_info = ltts_event.should_emit(module_name, actual_layer_path, params)

        if not emission_info["emit"]:
            logger.debug(f"GRADIENT_IMP: Skipping emission for {actual_layer_path}")
            return

        # Generate visualization based on type
        result = None

        if viz_type == "gradient_heatmap":
            result = create_gradient_heatmap(ltts_event, params, outputs)
        elif viz_type == "token_importance":
            result = create_token_importance_plot(ltts_event, params, outputs)
        elif viz_type == "neuron_importance":
            result = create_neuron_importance_plot(ltts_event, params, outputs)
        elif viz_type == "gradient_flow":
            result = create_gradient_flow_plot(ltts_event, params, outputs)
        elif viz_type == "sanity_checks":
            result = create_attribution_sanity_table(params, outputs)
        else:
            result = create_gradient_heatmap(ltts_event, params, outputs)

        if result:
            payload = {
                "type": "gradient_importance",
                "subtype": viz_type,
                "module": module_name,
                "data": result,
                "params": params,
                "event_type": ltts_event.event_type,
                "module_path": actual_layer_path,
                "layer_path": actual_layer_path,
                "stats": _module_state["stats"],
                "timestamp": __import__("time").time(),
            }

            ltts_event.reactor.signal("visualization", payload)
            logger.info(f"GRADIENT_IMP: Created {viz_type} for {actual_layer_path}")

    except Exception as e:
        logger.error(f"Error in gradient importance analyzer: {e}", exc_info=True)
        ltts_event.reactor.signal(
            "error",
            {
                "message": f"Gradient importance error: {str(e)}",
                "module": "gradient_importance",
            },
        )


def register_gradient_hook(tensor, layer_path):
    """Register a backward hook to capture gradients."""
    global _module_state

    def gradient_hook(grad):
        """Hook function to capture gradients during backward pass."""
        _module_state["gradients"][layer_path] = grad.detach().clone()
        _module_state["stats"]["gradients_captured"] += 1
        logger.debug(
            f"GRADIENT_IMP: Captured gradient for {layer_path}, shape: {grad.shape}"
        )
        return grad  # Return gradient unchanged

    # Register the hook
    handle = tensor.register_hook(gradient_hook)
    _module_state["hooks"].append(handle)


def compute_saliency_with_captum(outputs: torch.Tensor) -> torch.Tensor:
    """
    Compute saliency using a valid surrogate objective over layer outputs.

    Note: At layer-hook time LTTS does not always have a downstream task loss.
    We therefore use an explicit surrogate scalar objective to keep attribution
    mathematically valid and avoid invalid Captum targets.

    Args:
        outputs: Layer outputs [batch, seq, hidden]

    Returns:
        Real gradient magnitudes [batch, seq]
    """
    if not CAPTUM_AVAILABLE:
        logger.warning("Captum not available! Using activation magnitude as fallback (NOT real gradients)")
        importance = outputs.abs()
        if importance.dim() > 2:
            importance = importance.norm(dim=-1, keepdim=False)
        return importance

    try:
        # Check if gradients can be computed
        if not outputs.requires_grad:
            logger.debug("Outputs don't require grad, enabling...")
            outputs = outputs.requires_grad_(True)

        # Check computational graph
        if outputs.grad_fn is None and not outputs.is_leaf:
            logger.warning("Computational graph disconnected! Using activation magnitude fallback")
            importance = outputs.abs()
            if importance.dim() > 2:
                importance = importance.norm(dim=-1, keepdim=False)
            return importance

        # Surrogate objective: per-sample activation energy.
        def forward_func(x):
            return (x * x).sum(dim=(1, 2))

        # Use Captum's Saliency on the surrogate objective.
        saliency = Saliency(forward_func)

        # `target=None` is valid because `forward_func` already returns one value
        # per batch sample.
        attributions = saliency.attribute(outputs, target=None, abs=False)

        # Take absolute value (standard for saliency)
        importance = torch.abs(attributions)

        # Aggregate across hidden dimension
        if importance.dim() > 2:
            importance = torch.norm(importance, p=2, dim=-1)

        logger.debug("[OK] Computed saliency gradients with Captum surrogate objective")
        return importance

    except Exception as e:
        logger.error(f"Error computing real gradients with Captum: {e}")
        logger.warning("Falling back to activation magnitude")
        importance = outputs.abs()
        if importance.dim() > 2:
            importance = importance.norm(dim=-1, keepdim=False)
        return importance


def compute_saliency_from_outputs(outputs: torch.Tensor) -> torch.Tensor:
    """Compute saliency scores from layer outputs using the selected backend."""
    return compute_saliency_with_captum(outputs)


def compute_input_x_gradient_captum(outputs: torch.Tensor) -> torch.Tensor:
    """
    Compute Input × Gradient using the same valid surrogate objective.

    Args:
        outputs: Layer outputs [batch, seq, hidden]

    Returns:
        Real Input×Gradient attributions [batch, seq]
    """
    if not CAPTUM_AVAILABLE:
        logger.warning("Captum not available! Using activation magnitude fallback")
        importance = outputs.abs()
        if importance.dim() > 2:
            importance = importance.norm(dim=-1, keepdim=False)
        return importance

    try:
        # Ensure gradients enabled
        if not outputs.requires_grad:
            outputs = outputs.requires_grad_(True)

        # Check computational graph
        if outputs.grad_fn is None and not outputs.is_leaf:
            logger.warning("Computational graph disconnected for InputXGradient!")
            importance = outputs.abs()
            if importance.dim() > 2:
                importance = importance.norm(dim=-1, keepdim=False)
            return importance

        # Use the same surrogate objective as saliency for consistency.
        def forward_func(x):
            return (x * x).sum(dim=(1, 2))

        # Use Captum's InputXGradient on the surrogate objective.
        input_x_grad = InputXGradient(forward_func)

        # `target=None` is valid because `forward_func` returns [batch].
        attributions = input_x_grad.attribute(outputs, target=None)

        # Take absolute value and aggregate
        importance = torch.abs(attributions)
        if importance.dim() > 2:
            importance = torch.norm(importance, p=2, dim=-1)

        logger.debug(
            "[OK] Computed Input×Gradient with Captum surrogate objective"
        )
        return importance

    except Exception as e:
        logger.error(f"Error computing InputXGradient with Captum: {e}")
        logger.warning("Falling back to activation magnitude")
        importance = outputs.abs()
        if importance.dim() > 2:
            importance = importance.norm(dim=-1, keepdim=False)
        return importance


def _normalize_importance(values: torch.Tensor, enabled: bool) -> torch.Tensor:
    values = values.detach().float()
    if not enabled or values.numel() == 0:
        return values
    minimum = values.amin()
    maximum = values.amax()
    scale = (maximum - minimum).clamp_min(1e-12)
    return (values - minimum) / scale


def _gradient_visualization(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    params: Dict[str, Any],
    method: str,
    visualization_type: str,
    target_info: Dict[str, Any],
) -> Dict[str, Any]:
    if method == "input_x_gradient":
        attribution = activation.float() * gradient.float()
    else:
        attribution = gradient.float()

    absolute = attribution.abs()
    token_scores = (
        torch.linalg.vector_norm(absolute, dim=-1)
        if absolute.ndim >= 3
        else absolute
    )
    token_scores = _normalize_importance(
        token_scores, bool(params.get("normalize_scores", True))
    )

    if visualization_type == "neuron_importance" and absolute.ndim >= 3:
        neuron_scores = absolute.mean(dim=tuple(range(absolute.ndim - 1)))
        top_k = min(int(params.get("top_k_tokens", 10)), neuron_scores.numel())
        values, indices = torch.topk(neuron_scores, k=max(top_k, 1))
        return {
            "type": "table",
            "headers": ["Neuron", "Absolute attribution"],
            "rows": [
                [int(index), float(value)]
                for value, index in zip(values.cpu(), indices.cpu())
            ],
            "target": target_info,
            "method": method,
        }

    if visualization_type == "gradient_flow":
        return {
            "type": "table",
            "headers": ["Statistic", "Value"],
            "rows": [
                ["mean_abs_gradient", float(gradient.detach().float().abs().mean())],
                ["max_abs_gradient", float(gradient.detach().float().abs().max())],
                ["l2_gradient", float(torch.linalg.vector_norm(gradient.detach().float()))],
            ],
            "target": target_info,
            "method": method,
        }

    values = token_scores
    if values.ndim == 1:
        values = values.unsqueeze(0)
    elif values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    return {
        "type": "heatmap",
        "values": values.cpu().tolist(),
        "dimensions": {"rows": int(values.shape[0]), "cols": int(values.shape[1])},
        "target": target_info,
        "method": method,
        "quantity": (
            "absolute activation × target gradient"
            if method == "input_x_gradient"
            else "absolute target gradient"
        ),
    }


def finalize_gradients(
    layer_path: Optional[str] = None, target_info: Optional[Dict[str, Any]] = None
):
    """Emit attribution after the model target backward pass has completed."""

    target_info = dict(target_info or {})
    paths = [layer_path] if layer_path else list(_module_state["pending"])
    emitted = []
    for path in paths:
        pending = _module_state["pending"].pop(path, None)
        gradient = _module_state["gradients"].pop(path, None)
        if not pending or gradient is None:
            continue

        event = pending["event"]
        data = _gradient_visualization(
            pending["activation"],
            gradient,
            pending["params"],
            pending["method"],
            pending["visualization_type"],
            target_info,
        )
        event.reactor.signal(
            "visualization",
            {
                "type": "gradient_importance",
                "subtype": pending["visualization_type"],
                "module": METADATA["name"],
                "data": data,
                "event_type": event.event_type,
                "module_path": path,
                "layer_path": path,
                "target": target_info,
            },
        )
        emitted.append(path)
        # The activation for this path is no longer needed once finalized.
        _module_state["activations"].pop(path, None)
    _prune_hooks()
    return {"status": "emitted" if emitted else "no_gradient", "emitted": emitted}


def compute_input_x_gradient(outputs: torch.Tensor, layer_path: str) -> torch.Tensor:
    """Compute Input × Gradient scores for a layer output tensor."""
    _ = layer_path  # unused: attribution does not depend on the layer path
    return compute_input_x_gradient_captum(outputs)


def compute_explicit_backward_gradients(outputs: torch.Tensor) -> torch.Tensor:
    """
    Compute gradients using explicit backward pass for maximum control.

    This method bypasses Captum and performs a manual backward pass,
    giving you complete control over gradient computation.

    Args:
        outputs: Layer outputs [batch, seq, hidden]

    Returns:
        Real gradient magnitudes [batch, seq]
    """
    if not CAPTUM_AVAILABLE:
        logger.warning("Explicit backward pass not available! Using activation magnitude fallback")
        importance = outputs.abs()
        if importance.dim() > 2:
            importance = importance.norm(dim=-1, keepdim=False)
        return importance

    try:
        gradient_checks = verify_gradient_flow(outputs, "layer_outputs")
        if not gradient_checks["computational_graph_connected"]:
            logger.warning(
                "Computational graph issues detected for explicit backward pass"
            )
            importance = outputs.abs()
            if importance.dim() > 2:
                importance = importance.norm(dim=-1, keepdim=False)
            return importance

        gradients = compute_gradient_with_backward_pass(
            outputs,
            target=None,
            retain_graph=True,
            create_graph=False,
        )
        if gradients.shape == outputs.shape:
            return torch.norm(gradients, p=2, dim=-1)
        return gradients
    except Exception as exc:
        logger.error("Error computing explicit backward gradients: %s", exc)
        importance = outputs.abs()
        if importance.dim() > 2:
            importance = importance.norm(dim=-1, keepdim=False)
        return importance


def _coerce_noise_config(params: Dict[str, Any]) -> Tuple[int, float, int]:
    try:
        samples = max(2, min(128, int(params.get("noise_samples", 16))))
    except (TypeError, ValueError):
        samples = 16
    try:
        noise_std = max(0.0, min(2.0, float(params.get("noise_std", 0.1))))
    except (TypeError, ValueError):
        noise_std = 0.1
    try:
        seed = max(0, int(params.get("random_seed", 0)))
    except (TypeError, ValueError):
        seed = 0
    return samples, noise_std, seed


def compute_importance_by_method(
    outputs: torch.Tensor, method: str, params: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """Route all supported attribution methods through one tested interface."""
    params = params or {}
    if method == "saliency":
        return compute_saliency_with_captum(outputs)
    if method == "input_x_gradient":
        return compute_input_x_gradient_captum(outputs)
    if method == "integrated_gradients":
        return compute_integrated_gradients_captum(outputs)
    if method == "explicit_backward":
        return compute_explicit_backward_gradients(outputs)
    if method in {"smoothgrad", "noisegrad", "vargrad"}:
        samples, noise_std, seed = _coerce_noise_config(params)
        return noise_ensemble_attribution(
            outputs,
            method,
            samples=samples,
            noise_std=noise_std,
            seed=seed,
        )
    return compute_saliency_with_captum(outputs)


def create_attribution_sanity_table(
    params: Dict[str, Any], outputs: torch.Tensor
) -> Dict[str, Any]:
    """Return baseline and perturbation checks as a structured table."""
    samples, noise_std, seed = _coerce_noise_config(params)
    report = attribution_sanity_report(
        outputs,
        samples=samples,
        noise_std=noise_std,
        seed=seed,
    )
    rows = [
        ["Finite attribution fraction", round(report["finite_fraction"], 6), ">= 0.999"],
        ["SmoothGrad similarity", round(report["smoothgrad_similarity"], 6), "context"],
        ["Randomization similarity", round(report["randomization_similarity"], 6), "< 0.95"],
        ["Randomization sensitivity", report["randomization_passed"], True],
        ["Zero-baseline mean distance", round(report["zero_baseline_mean"], 6), "context"],
        ["Mean-baseline mean distance", round(report["mean_baseline_mean"], 6), "context"],
    ]
    return {
        "type": "table",
        "headers": ["Check", "Observed", "Expected"],
        "rows": rows,
        "scope": report["scope"],
        "samples": samples,
        "noise_std": noise_std,
    }


def register_gradient_tracking_hook(tensor: torch.Tensor, layer_path: str) -> bool:
    """
    Register a comprehensive gradient tracking hook for a tensor.

    This function sets up gradient tracking that will capture detailed
    information during the backward pass.

    Args:
        tensor: Tensor to track gradients for
        layer_path: Identifier for the layer

    Returns:
        True if hook was successfully registered
    """
    global _module_state

    try:
        def gradient_analysis_hook(grad):
            """Comprehensive gradient analysis hook."""
            # Store gradient in module state
            _module_state["gradients"][layer_path] = grad.detach().clone()
            _module_state["stats"]["gradients_captured"] += 1

            # Log gradient statistics
            grad_stats = {
                "shape": grad.shape,
                "mean": float(grad.mean().item()),
                "std": float(grad.std().item()),
                "max": float(grad.max().item()),
                "min": float(grad.min().item()),
                "norm": float(grad.norm().item())
            }

            logger.debug(f"Gradient captured for {layer_path}:")
            for stat, value in grad_stats.items():
                if isinstance(value, float):
                    logger.debug(f"  {stat}: {value:.6f}")
                else:
                    logger.debug(f"  {stat}: {value}")

            # Check for gradient issues
            if grad.isnan().any():
                logger.warning(f"NaN gradients detected for {layer_path}!")
            if grad.isinf().any():
                logger.warning(f"Inf gradients detected for {layer_path}!")
            if (grad == 0).all():
                logger.warning(f"Zero gradients detected for {layer_path}!")

            return grad

        # Register the hook
        if CAPTUM_AVAILABLE:
            handle = hook_into_backward_pass(tensor, gradient_analysis_hook, layer_path)
            _module_state["hooks"].append(handle)
            logger.debug(f"Registered gradient tracking hook for {layer_path}")
            return True
        else:
            # Fallback hook registration
            handle = tensor.register_hook(gradient_analysis_hook)
            _module_state["hooks"].append(handle)
            logger.debug(f"Registered fallback gradient hook for {layer_path}")
            return True

    except Exception as e:
        logger.error(f"Failed to register gradient hook for {layer_path}: {e}")
        return False


def create_gradient_heatmap(
    ltts_event, params: Dict[str, Any], outputs: torch.Tensor
) -> str:
    """Create gradient-based importance heatmap."""
    try:
        method = params.get("method", "saliency")
        layer_path = getattr(ltts_event.context, "module_path", "unknown")

        # Enable gradients for backward pass
        if CAPTUM_AVAILABLE:
            outputs = enable_gradients_for_layer(outputs)
            can_compute, reason = check_gradient_availability(outputs)
            if not can_compute:
                logger.warning(f"Cannot compute real gradients: {reason}")

        # Compute importance with the selected gradient method
        importance = compute_importance_by_method(outputs, method, params)

        # Convert to numpy
        importance_np = importance.detach().cpu().numpy()

        # Handle different tensor shapes
        if importance_np.ndim == 3:
            # [batch, seq, features] -> take first batch
            importance_np = importance_np[0]
        elif importance_np.ndim == 1:
            # [features] -> reshape to 2D
            importance_np = importance_np.reshape(1, -1)

        # Normalize if requested
        if params.get("normalize_scores", True):
            importance_np = normalize_scores(importance_np)

        # Limit size for visualization
        if importance_np.shape[0] > 100:
            importance_np = importance_np[:100]
        if importance_np.shape[1] > 100:
            importance_np = importance_np[:, :100]

        # Create heatmap
        plt.figure(figsize=(12, 8))

        sns.heatmap(
            importance_np,
            cmap=params.get("colormap", "RdYlGn"),
            cbar=True,
            square=False,
            xticklabels=False,
            yticklabels=10 if importance_np.shape[0] <= 50 else False,
            cbar_kws={"label": "Importance Score"},
        )

        plt.title(f"Gradient Importance Heatmap - {method}", fontsize=14, pad=20)
        plt.xlabel("Feature Dimension", fontsize=12)
        plt.ylabel("Sequence Position", fontsize=12)

        # Add statistics
        stats_text = f"Mean: {importance_np.mean():.4f} | Max: {importance_np.max():.4f} | Min: {importance_np.min():.4f}"
        plt.figtext(0.02, 0.02, stats_text, fontsize=9, style="italic")

        return save_plot_as_base64()

    except Exception as e:
        logger.error(f"Error creating gradient heatmap: {e}")
        return create_error_visualization_data("gradient_heatmap", str(e))


def create_token_importance_plot(
    ltts_event, params: Dict[str, Any], outputs: torch.Tensor
) -> str:
    """Create token-level importance bar plot."""
    try:
        method = params.get("method", "saliency")
        layer_path = getattr(ltts_event.context, "module_path", "unknown")

        # Compute importance
        importance = compute_importance_by_method(outputs, method, params)

        importance_np = importance.detach().cpu().numpy()

        # Aggregate to token level (average across features)
        if importance_np.ndim == 3:
            token_scores = importance_np[0].mean(axis=-1)  # [seq]
        elif importance_np.ndim == 2:
            token_scores = importance_np.mean(axis=-1)  # [seq]
        else:
            token_scores = importance_np

        # Limit to reasonable number of tokens
        max_tokens = min(len(token_scores), 50)
        token_scores = token_scores[:max_tokens]

        # Normalize
        if params.get("normalize_scores", True):
            token_scores = (token_scores - token_scores.min()) / (
                token_scores.max() - token_scores.min() + 1e-8
            )

        # Get top K
        top_k = params.get("top_k_tokens", 10)
        top_indices = np.argsort(token_scores)[-top_k:][::-1]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: All tokens
        colors = [
            "red" if i in top_indices else "steelblue" for i in range(len(token_scores))
        ]
        ax1.bar(range(len(token_scores)), token_scores, color=colors, alpha=0.7)
        ax1.set_title(f"Token Importance Scores - {method}", fontsize=12)
        ax1.set_xlabel("Token Position")
        ax1.set_ylabel("Importance Score")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Top K tokens
        top_scores = token_scores[top_indices]
        ax2.barh(range(top_k), top_scores, color="coral", alpha=0.7)
        ax2.set_yticks(range(top_k))
        ax2.set_yticklabels([f"Token {i}" for i in top_indices])
        ax2.set_title(f"Top {top_k} Most Important Tokens", fontsize=12)
        ax2.set_xlabel("Importance Score")
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        # Add statistics
        stats_text = f"Total Tokens: {len(token_scores)} | Mean: {token_scores.mean():.4f} | Std: {token_scores.std():.4f}"
        fig.text(0.02, 0.02, stats_text, fontsize=9, style="italic")

        return save_plot_as_base64()

    except Exception as e:
        logger.error(f"Error creating token importance plot: {e}")
        return create_error_visualization_data("token_importance", str(e))


def create_neuron_importance_plot(
    ltts_event, params: Dict[str, Any], outputs: torch.Tensor
) -> str:
    """Create neuron-level importance analysis."""
    try:
        method = params.get("method", "saliency")
        layer_path = getattr(ltts_event.context, "module_path", "unknown")

        # Compute importance
        importance = compute_importance_by_method(outputs, method, params)

        importance_np = importance.detach().cpu().numpy()

        # Aggregate to neuron level (average across sequence)
        if importance_np.ndim == 3:
            neuron_scores = importance_np[0].mean(axis=0)  # [features]
        elif importance_np.ndim == 2:
            neuron_scores = importance_np.mean(axis=0)  # [features]
        else:
            neuron_scores = importance_np

        # Limit to reasonable number
        max_neurons = min(len(neuron_scores), 1000)
        neuron_scores = neuron_scores[:max_neurons]

        # Normalize
        if params.get("normalize_scores", True):
            neuron_scores = (neuron_scores - neuron_scores.min()) / (
                neuron_scores.max() - neuron_scores.min() + 1e-8
            )

        # Create analysis plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Neuron Importance Analysis - {method}", fontsize=16)

        # Plot 1: Distribution histogram
        ax1.hist(
            neuron_scores, bins=50, alpha=0.7, color="steelblue", edgecolor="black"
        )
        ax1.axvline(
            neuron_scores.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {neuron_scores.mean():.3f}",
        )
        ax1.set_title("Neuron Importance Distribution")
        ax1.set_xlabel("Importance Score")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Top neurons
        top_k = min(20, len(neuron_scores))
        top_indices = np.argsort(neuron_scores)[-top_k:][::-1]
        top_scores = neuron_scores[top_indices]

        ax2.barh(range(top_k), top_scores, color="coral", alpha=0.7)
        ax2.set_yticks(range(top_k))
        ax2.set_yticklabels([f"Neuron {i}" for i in top_indices])
        ax2.set_title(f"Top {top_k} Most Important Neurons")
        ax2.set_xlabel("Importance Score")
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis="x")

        # Plot 3: Cumulative importance
        sorted_scores = np.sort(neuron_scores)[::-1]
        cumulative = np.cumsum(sorted_scores) / np.sum(sorted_scores)
        ax3.plot(range(len(cumulative)), cumulative, linewidth=2, color="green")
        ax3.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="90% threshold")
        ax3.set_title("Cumulative Importance")
        ax3.set_xlabel("Number of Neurons")
        ax3.set_ylabel("Cumulative Importance Fraction")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Statistics table
        ax4.axis("off")

        # Calculate statistics
        threshold = 0.1
        active_neurons = np.sum(neuron_scores > threshold)
        sparsity = 1.0 - (active_neurons / len(neuron_scores))

        stats_text = f"""Neuron Statistics:

Total Neurons: {len(neuron_scores)}
Active Neurons (>{threshold:.1f}): {active_neurons}
Sparsity: {sparsity:.2%}

Mean Importance: {neuron_scores.mean():.4f}
Std Importance: {neuron_scores.std():.4f}
Max Importance: {neuron_scores.max():.4f}
Min Importance: {neuron_scores.min():.4f}

Top 10% Threshold: {np.percentile(neuron_scores, 90):.4f}
Top 25% Threshold: {np.percentile(neuron_scores, 75):.4f}
Median: {np.median(neuron_scores):.4f}"""

        ax4.text(
            0.1,
            0.9,
            stats_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()
        return save_plot_as_base64()

    except Exception as e:
        logger.error(f"Error creating neuron importance plot: {e}")
        return create_error_visualization_data("neuron_importance", str(e))


def create_gradient_flow_plot(
    ltts_event, params: Dict[str, Any], outputs: torch.Tensor
) -> str:
    """Create gradient flow visualization showing how importance flows through the layer."""
    try:
        method = params.get("method", "saliency")
        layer_path = getattr(ltts_event.context, "module_path", "unknown")

        # Compute importance
        importance = compute_importance_by_method(outputs, method, params)

        importance_np = importance.detach().cpu().numpy()

        # Handle different shapes
        if importance_np.ndim == 3:
            importance_np = importance_np[0]  # [seq, features]
        elif importance_np.ndim == 1:
            importance_np = importance_np.reshape(1, -1)

        # Normalize
        if params.get("normalize_scores", True):
            importance_np = normalize_scores(importance_np)

        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Gradient Flow Analysis - {method}", fontsize=16)

        # Plot 1: 2D flow heatmap
        # Downsample if too large
        display_imp = importance_np
        if display_imp.shape[0] > 50:
            display_imp = display_imp[:: display_imp.shape[0] // 50]
        if display_imp.shape[1] > 100:
            display_imp = display_imp[:, :: display_imp.shape[1] // 100]

        im1 = ax1.imshow(
            display_imp, cmap=params.get("colormap", "RdYlGn"), aspect="auto"
        )
        ax1.set_title("Importance Flow Heatmap")
        ax1.set_xlabel("Feature Dimension")
        ax1.set_ylabel("Sequence Position")
        plt.colorbar(im1, ax=ax1)

        # Plot 2: Flow across sequence
        seq_flow = importance_np.mean(axis=1)  # Average across features
        ax2.plot(range(len(seq_flow)), seq_flow, linewidth=2, color="steelblue")
        ax2.fill_between(range(len(seq_flow)), seq_flow, alpha=0.3)
        ax2.set_title("Importance Flow Across Sequence")
        ax2.set_xlabel("Sequence Position")
        ax2.set_ylabel("Mean Importance")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Flow across features
        feat_flow = importance_np.mean(axis=0)  # Average across sequence
        # Show only first N features for clarity
        max_feat = min(len(feat_flow), 100)
        ax3.plot(range(max_feat), feat_flow[:max_feat], linewidth=2, color="coral")
        ax3.fill_between(range(max_feat), feat_flow[:max_feat], alpha=0.3)
        ax3.set_title("Importance Flow Across Features")
        ax3.set_xlabel("Feature Index")
        ax3.set_ylabel("Mean Importance")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Flow statistics
        ax4.axis("off")

        # Calculate flow statistics
        total_importance = importance_np.sum()
        max_flow_pos = np.argmax(seq_flow)
        max_flow_feat = np.argmax(feat_flow)

        flow_stats = f"""Gradient Flow Statistics:

Total Importance: {total_importance:.2f}
Mean Importance: {importance_np.mean():.4f}
Std Importance: {importance_np.std():.4f}

Peak Sequence Position: {max_flow_pos}
Peak Feature Index: {max_flow_feat}

Sequence Flow Range: {seq_flow.min():.4f} - {seq_flow.max():.4f}
Feature Flow Range: {feat_flow.min():.4f} - {feat_flow.max():.4f}

Shape: {importance_np.shape}
Method: {method}"""

        ax4.text(
            0.1,
            0.9,
            flow_stats,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()
        return save_plot_as_base64()

    except Exception as e:
        logger.error(f"Error creating gradient flow plot: {e}")
        return create_error_visualization_data("gradient_flow", str(e))


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize importance scores to [0, 1] range."""
    scores_flat = scores.flatten()
    min_val = scores_flat.min()
    max_val = scores_flat.max()

    if max_val > min_val:
        normalized = (scores - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(scores)

    return normalized


def validate_and_set_defaults(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validates parameters and sets defaults."""
    defaults = {
        "method": "saliency",
        "visualization_type": "gradient_heatmap",
        "normalize_scores": True,
        "top_k_tokens": 10,
        "colormap": "RdYlGn",
        "noise_samples": 16,
        "noise_std": 0.1,
        "random_seed": 0,
    }

    for key, default_value in defaults.items():
        if key not in params:
            params[key] = default_value

    return params


def create_error_visualization_data(kind: str, message: str) -> Dict[str, Any]:
    """Return structured error payload instead of synthetic placeholder imagery."""
    return {
        "kind": kind,
        "status": "error",
        "error": message,
    }


def save_plot_as_base64() -> str:
    """Save current matplotlib plot as base64 encoded string."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=150, facecolor="white")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{image_base64}"


# Module interface functions matching LTTS expectations
def interceptor(ltts_event):
    """Compatibility interface."""
    return process_event(ltts_event)
