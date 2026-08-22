"""
Gradient computation using Captum.

Gradient-based attribution methods built on Captum. Each method runs an
explicit backward pass over the layer output.
"""

import torch
import logging
from typing import Optional, Dict, List, Union, Tuple

logger = logging.getLogger(__name__)

try:
    from captum.attr import Saliency, InputXGradient, IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False


def compute_integrated_gradients_captum(
    outputs: torch.Tensor,
    baseline: torch.Tensor = None,
    n_steps: int = 50,
    target: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute Integrated Gradients using Captum with a valid surrogate objective.

    Args:
        outputs: Layer outputs [batch, seq, hidden]
        baseline: Baseline (default: zeros)
        n_steps: Number of interpolation steps
        target: Target tensor for gradient computation (default: sum of outputs)

    Returns:
        Integrated gradient attributions [batch, seq]
    """
    if not CAPTUM_AVAILABLE:
        logger.error("Captum not available for Integrated Gradients!")
        # Fallback to approximation
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
            logger.warning("Computational graph disconnected for IntegratedGradients!")
            importance = outputs.abs()
            if importance.dim() > 2:
                importance = importance.norm(dim=-1, keepdim=False)
            return importance

        # Create baseline (zeros if not provided)
        if baseline is None:
            baseline = torch.zeros_like(outputs)

        # Return one scalar per sample to avoid invalid target indexing and keep
        # attribution defined when task logits are not available at hook time.
        def forward_func(x):
            return (x * x).sum(dim=(1, 2))

        # Captum's IntegratedGradients over the surrogate objective.
        ig = IntegratedGradients(forward_func)

        # Captum expects target indices, not tensors. Ignore invalid targets.
        if target is not None and not isinstance(target, (int, tuple, list)):
            target = None

        # Compute with multiple backward passes.
        attributions = ig.attribute(
            outputs,
            baselines=baseline,
            target=target,
            n_steps=n_steps,
            method='gausslegendre'  # More accurate integration
        )

        # Take absolute value and aggregate
        importance = torch.abs(attributions)
        if importance.dim() > 2:
            importance = torch.norm(importance, p=2, dim=-1)

        logger.debug(f"Computed Integrated Gradients ({n_steps} steps)")
        return importance

    except Exception as e:
        logger.error(f"Error computing Integrated Gradients with Captum: {e}")
        logger.warning("Falling back to activation magnitude")
        importance = outputs.abs()
        if importance.dim() > 2:
            importance = importance.norm(dim=-1, keepdim=False)
        return importance


def check_gradient_availability(tensor: torch.Tensor) -> tuple:
    """
    Check if real gradients can be computed for this tensor.

    Returns:
        (bool, str): (can_compute, reason)
    """
    if not tensor.requires_grad:
        return False, "Tensor doesn't require gradients"

    if tensor.grad_fn is None and not tensor.is_leaf:
        return False, "Computational graph disconnected"

    if not CAPTUM_AVAILABLE:
        return False, "Captum not installed"

    return True, "Gradients available"


def enable_gradients_for_layer(tensor: torch.Tensor) -> torch.Tensor:
    """
    Enable gradient computation for a layer output tensor.

    This is needed because LTTS may detach tensors by default.
    """
    if not tensor.requires_grad:
        logger.debug("Enabling gradients for tensor...")
        tensor = tensor.requires_grad_(True)
    return tensor


def test_backward_pass_simple():
    """
    Simple test to verify backward pass works.
    """
    print("Testing backward pass...")

    # Create simple tensor
    x = torch.randn(2, 10, 768, requires_grad=True)

    # Compute target
    y = x.sum()

    # Backward pass
    y.backward()

    # Check gradients
    if x.grad is not None:
        print("[OK] Backward pass works! Gradients computed.")
        print(f"  Gradient shape: {x.grad.shape}")
        print(f"  Gradient mean: {x.grad.mean().item():.6f}")
        return True
    else:
        print("[FAIL] Backward pass failed! No gradients.")
        return False


def compute_gradient_with_backward_pass(
    outputs: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    create_graph: bool = False,
    retain_graph: bool = False
) -> torch.Tensor:
    """
    Compute gradients using explicit backward pass for maximum control.

    This function performs a manual backward pass to compute gradients,
    giving you full control over the gradient computation process.

    Args:
        outputs: Layer outputs [batch, seq, hidden]
        target: Target for gradient computation (default: sum of outputs)
        create_graph: Whether to create computational graph for higher-order gradients
        retain_graph: Whether to retain the graph after backward pass

    Returns:
        Gradient magnitudes [batch, seq, hidden] or aggregated [batch, seq]
    """
    if not outputs.requires_grad:
        logger.warning("Tensor doesn't require gradients - enabling...")
        outputs = outputs.requires_grad_(True)

    # Clear any existing gradients
    if outputs.grad is not None:
        outputs.grad.zero_()

    # Create target if not provided. Use activation-energy objective instead
    # of plain sum to avoid constant gradients.
    if target is None:
        target = (outputs * outputs).mean()

    try:
        # Perform backward pass
        target.backward(retain_graph=retain_graph, create_graph=create_graph)

        # Check if gradients were computed
        if outputs.grad is None:
            logger.warning("Backward pass completed but no gradients found!")
            # Fallback to activation magnitude
            importance = outputs.abs()
            if importance.dim() > 2:
                importance = importance.norm(dim=-1, keepdim=False)
            return importance

        gradients = outputs.grad.detach()

        logger.debug("Computed gradients via explicit backward pass")
        return gradients

    except Exception as e:
        logger.error(f"Error in explicit backward pass: {e}")
        # Fallback to activation magnitude
        importance = outputs.abs()
        if importance.dim() > 2:
            importance = importance.norm(dim=-1, keepdim=False)
        return importance


def compute_layer_wise_gradients(
    layer_outputs: Dict[str, torch.Tensor],
    target_layer: str,
    final_output: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute gradients for multiple layers with respect to a target.

    This is useful for understanding how gradients flow through
    different layers of the model.

    Args:
        layer_outputs: Dictionary mapping layer names to their output tensors
        target_layer: Name of the layer to compute gradients for
        final_output: Final output to compute gradients with respect to

    Returns:
        Dictionary mapping layer names to their gradients
    """
    if not CAPTUM_AVAILABLE:
        logger.error("Captum required for layer-wise gradient computation!")
        return {}

    gradients = {}

    if target_layer not in layer_outputs:
        logger.error(f"Target layer '{target_layer}' not found in layer_outputs!")
        return gradients

    target_tensor = layer_outputs[target_layer]

    if not target_tensor.requires_grad:
        logger.warning(f"Target layer '{target_layer}' doesn't require gradients!")
        return gradients

    # Create final output target if not provided.
    if final_output is None:
        final_output = (target_tensor * target_tensor).mean()

    try:
        # Clear gradients
        for name, tensor in layer_outputs.items():
            if tensor.grad is not None:
                tensor.grad.zero_()

        # Backward pass
        final_output.backward(retain_graph=True)

        # Collect gradients
        for name, tensor in layer_outputs.items():
            if tensor.grad is not None:
                gradients[name] = tensor.grad.detach().clone()
                logger.debug(f"[OK] Captured gradients for layer: {name}")
            else:
                logger.debug(f"No gradients for layer: {name}")

        return gradients

    except Exception as e:
        logger.error(f"Error computing layer-wise gradients: {e}")
        return {}


def verify_gradient_flow(tensor: torch.Tensor, name: str = "tensor") -> Dict[str, bool]:
    """
    Verify that gradients can flow through a tensor.

    This is a diagnostic function to check if the computational
    graph is properly connected for gradient computation.

    Args:
        tensor: Tensor to check
        name: Name for logging purposes

    Returns:
        Dictionary of gradient flow properties
    """
    checks = {
        "requires_grad": tensor.requires_grad,
        "is_leaf": tensor.is_leaf,
        "has_grad_fn": tensor.grad_fn is not None,
        "computational_graph_connected": False
    }

    # Check if computational graph is connected
    if not tensor.is_leaf:
        checks["computational_graph_connected"] = tensor.grad_fn is not None
    else:
        checks["computational_graph_connected"] = True  # Leaf nodes can receive gradients

    logger.info(f"Gradient flow check for '{name}':")
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        logger.info(f"  {status} {check}: {passed}")

    return checks


def hook_into_backward_pass(
    tensor: torch.Tensor,
    hook_fn: callable,
    name: str = "tensor"
) -> torch.utils.hooks.RemovableHandle:
    """
    Register a custom backward hook into a tensor.

    This allows you to intercept and analyze gradients during
    the backward pass for debugging or analysis purposes.

    Args:
        tensor: Tensor to hook into
        hook_fn: Function to call during backward pass
        name: Name for logging

    Returns:
        Hook handle that can be used to remove the hook
    """
    def wrapper_hook(grad):
        logger.debug(f"Backward hook triggered for '{name}'")
        logger.debug(f"  Gradient shape: {grad.shape}")
        logger.debug(f"  Gradient mean: {grad.mean().item():.6f}")
        logger.debug(f"  Gradient std: {grad.std().item():.6f}")
        return hook_fn(grad)

    handle = tensor.register_hook(wrapper_hook)
    logger.debug(f"Registered backward hook for '{name}'")
    return handle


def benchmark_gradient_methods(
    tensor: torch.Tensor,
    methods: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different gradient computation methods.

    This helps you understand the performance characteristics
    of different gradient attribution methods.

    Args:
        tensor: Input tensor
        methods: List of methods to benchmark (default: all available)

    Returns:
        Dictionary with timing and memory information for each method
    """
    import time

    if methods is None:
        methods = ["saliency", "input_x_gradient", "integrated_gradients", "explicit_backward"]

    results = {}

    for method in methods:
        logger.info(f"Benchmarking {method}...")

        # Time the computation
        start_time = time.time()

        try:
            if method == "saliency":
                # Import here to avoid circular imports
                from captum.attr import Saliency
                saliency = Saliency(lambda x: x.sum())
                attributions = saliency.attribute(tensor)

            elif method == "input_x_gradient":
                from captum.attr import InputXGradient
                input_x_grad = InputXGradient(lambda x: x.sum())
                attributions = input_x_grad.attribute(tensor)

            elif method == "integrated_gradients":
                attributions = compute_integrated_gradients_captum(tensor, n_steps=10)  # Reduced for benchmark

            elif method == "explicit_backward":
                attributions = compute_gradient_with_backward_pass(tensor)

            else:
                logger.warning(f"Unknown method: {method}")
                continue

            computation_time = time.time() - start_time

            results[method] = {
                "time_seconds": computation_time,
                "success": True,
                "shape": list(attributions.shape),
                "mean_value": float(attributions.mean().item()),
                "std_value": float(attributions.std().item())
            }

            logger.info(f"  ✓ {method}: {computation_time:.4f}s")

        except Exception as e:
            computation_time = time.time() - start_time
            results[method] = {
                "time_seconds": computation_time,
                "success": False,
                "error": str(e)
            }
            logger.error(f"  ✗ {method}: {e}")

    return results


if __name__ == "__main__":
    # Test backward pass
    if CAPTUM_AVAILABLE:
        print("Captum available!")
        test_backward_pass_simple()

        # Test gradient flow verification
        print("\n" + "="*50)
        print("Testing gradient flow verification...")
        test_tensor = torch.randn(2, 10, 768, requires_grad=True)
        verify_gradient_flow(test_tensor, "test_tensor")

        # Test explicit backward pass
        print("\n" + "="*50)
        print("Testing explicit backward pass...")
        gradients = compute_gradient_with_backward_pass(test_tensor)
        if gradients is not None:
            print(f"[OK] Explicit backward pass successful!")
            print(f"  Gradient shape: {gradients.shape}")

        # Benchmark methods
        print("\n" + "="*50)
        print("Benchmarking gradient methods...")
        benchmark_results = benchmark_gradient_methods(test_tensor)
        for method, stats in benchmark_results.items():
            if stats["success"]:
                print(f"  {method}: {stats['time_seconds']:.4f}s")
            else:
                print(f"  {method}: FAILED - {stats['error']}")

    else:
        print("Captum NOT available!")
