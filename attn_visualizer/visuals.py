import io
import base64
import logging
import re
from typing import Dict, Any, Optional
from typing_extensions import List

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from miners.utils import extract_attention_weights

logger = logging.getLogger(__name__)


def get_ltts_model_from_context_or_global(ltts_event):
    return getattr(ltts_event.context, "ltts_model", None)


def validate_and_set_defaults(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validates parameters and sets defaults"""
    defaults = {
        "visualization_type": "attention_heatmap",
        "head_index": -1,
        "similarity_metric": "cosine",
        "normalize": True,
        "threshold": 0.01,
        "colormap": "viridis",
        "show_annotations": False,
        "token_range": "0:50",
        "emit_mode": "final",
        "emit_every_n": 1,
        "expected_generation_passes": None,
    }

    for key, default_value in defaults.items():
        if key not in params:
            params[key] = default_value

    # Validate ranges
    if params["head_index"] < -1:
        params["head_index"] = -1

    if not (0 <= params["threshold"] <= 1):
        params["threshold"] = 0.01

    valid_viz_types = [
        "attention_heatmap",
        "similarity_matrix",
        "head_analysis",
        "pattern_matrix",
        "entropy_plot",
    ]
    if params["visualization_type"] not in valid_viz_types:
        params["visualization_type"] = "attention_heatmap"

    valid_colormaps = [
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "blues",
        "reds",
        "coolwarm",
    ]
    if params["colormap"] not in valid_colormaps:
        params["colormap"] = "viridis"

    valid_metrics = ["cosine", "dot", "euclidean"]
    if params["similarity_metric"] not in valid_metrics:
        params["similarity_metric"] = "cosine"

    return params


def parse_token_range(token_range_str: str, max_len: int) -> Optional[tuple]:
    """Parse token range string like '0:50' into (start, end) tuple"""
    try:
        if ":" in token_range_str:
            start, end = token_range_str.split(":")
            start = max(0, int(start))
            end = min(max_len, int(end))
            if start < end:
                return (start, end)
    except Exception:
        pass
    return None


def calculate_similarity_matrix(
    hidden: torch.Tensor, metric: str = "cosine"
) -> np.ndarray:
    """Calculate similarity matrix from hidden states"""
    if hidden.dim() == 3:
        hidden = hidden[0]  # Use first batch element

    x = hidden
    if metric == "cosine":
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        sim = x @ x.transpose(0, 1)
    elif metric == "dot":
        sim = x @ x.transpose(0, 1)
    elif metric == "euclidean":
        # Negative euclidean distance (higher = more similar)
        dist = torch.cdist(x.unsqueeze(0), x.unsqueeze(0))[0]
        sim = -dist
    else:
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        sim = x @ x.transpose(0, 1)

    sim_np = sim.detach().to(torch.float32).cpu().numpy()

    # Normalize to 0..1 for visualization
    mn, mx = sim_np.min(), sim_np.max()
    if mx > mn:
        sim_np = (sim_np - mn) / (mx - mn)
    else:
        sim_np = np.zeros_like(sim_np)

    return sim_np


def analyze_attention_patterns(weights: np.ndarray) -> Dict[str, float]:
    """Analyze attention patterns in the weights"""
    entropy = -np.sum(weights * np.log(weights + 1e-8))
    sparsity = np.mean(weights < 0.01)

    diagonal_sum = np.sum(np.diag(weights))
    total_sum = np.sum(weights)
    diagonal_dominance = diagonal_sum / (total_sum + 1e-8)

    return {
        "entropy": float(entropy),
        "sparsity": float(sparsity),
        "diagonal_dominance": float(diagonal_dominance),
    }


def get_entropy_interpretation(entropy: float) -> str:
    """Get human-readable interpretation of entropy value"""
    if entropy < 1.0:
        return "Very focused attention (low entropy)"
    elif entropy < 2.0:
        return "Moderately focused attention"
    elif entropy < 3.0:
        return "Distributed attention"
    else:
        return "Very distributed attention (high entropy)"


def create_error_visualization_data(message: str) -> Dict[str, Any]:
    """Create structured error payload when visualization cannot be produced."""
    return {"status": "error", "error": message}


def save_plot_as_base64(fig) -> str:
    """Save the supplied figure without relying on pyplot's global state."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150, facecolor="white")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"


def _get_token_labels(ltts_event, seq_len: int) -> Optional[List[str]]:
    try:
        lm = getattr(ltts_event.context, "ltts_model", None)
        if lm is not None and hasattr(lm, "_last_input_tokens"):
            tokens = getattr(lm, "_last_input_tokens", None) or []
            if isinstance(tokens, (list, tuple)) and tokens:
                return list(tokens)[:seq_len]
        # Fallback: if tokenizer + input_ids available on model-level inputs
        if lm is not None and hasattr(lm, "_tokenizer"):
            tok = getattr(lm, "_tokenizer")
            # Model-level events may carry kwargs; layer events usually don't.
            inp = getattr(ltts_event.context, "inputs", None)
            if isinstance(inp, dict) and "input_ids" in inp:
                ids = inp["input_ids"][0].detach().cpu().tolist()
                return tok.convert_ids_to_tokens(ids)[:seq_len]
    except Exception:
        pass
    return None


def create_attention_heatmap(ltts_event, params: Dict[str, Any]) -> Optional[str]:
    model = get_ltts_model_from_context_or_global(ltts_event)
    if model is None:
        raise ValueError("No model available for attention heatmap")

    inputs = getattr(ltts_event.context, "inputs", None)
    attention_map = _get_layer_attention_weights(ltts_event)
    if attention_map is None:
        raise ValueError("No attention weights found for attention heatmap")

    head_index = int(params.get("head_index", -1))
    am2d = reduce_to_2d(attention_map, head_index)
    if (
        am2d is None
        or am2d.ndim != 2
        or not am2d.size
        or not np.isfinite(am2d).any()
    ):
        raise ValueError("Unexpected attention shape for heatmap")

    logger.info(
        "ATTN_VISUALIZER: Creating attention heatmap for layer %s, input shape %s",
        getattr(ltts_event.context, "module_path", None),
        (inputs.shape if hasattr(inputs, "shape") else type(inputs)),
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(am2d, cmap="viridis", annot=False, ax=ax, cbar=True)
    ax.set_title("Attention Heatmap")

    # Use real token labels if available
    labels = _get_token_labels(ltts_event, am2d.shape[-1])
    if labels:
        # Center the tick labels on the heatmap cells by offsetting by 0.5
        positions = [i + 0.5 for i in range(len(labels))]
        ax.set_xticks(positions)
        ax.set_yticks(positions)
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, rotation=0, fontsize=8)
        ax.set_xlabel("Tokens (decoded)")
        ax.set_ylabel("Tokens (decoded)")
    else:
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Tokens")

    return save_plot_as_base64(fig)


def create_similarity_matrix(ltts_event, params: Dict[str, Any]) -> Optional[str]:
    model = get_ltts_model_from_context_or_global(ltts_event)
    if model is None:
        raise ValueError("No model available for similarity matrix")

    outputs = getattr(ltts_event.context, "outputs", None)
    metric = params.get("metric", params.get("similarity_metric", "cosine"))

    logger.info("ATTN_VISUALIZER: Creating similarity matrix with metric=%s", metric)

    similarity = calculate_similarity_matrix(outputs, metric)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(similarity, cmap="coolwarm", annot=False, ax=ax)
    ax.set_title("Similarity Matrix")
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Tokens")

    return save_plot_as_base64(fig)


def create_head_analysis(ltts_event, params: Dict[str, Any]) -> Optional[str]:
    model = get_ltts_model_from_context_or_global(ltts_event)
    if model is None:
        raise ValueError("No model available for head analysis")

    attn_weights = _get_layer_attention_weights(ltts_event)
    if attn_weights is None:
        raise ValueError("No attention weights available for head analysis")

    aw = (
        attn_weights.detach().to(torch.float32).cpu().numpy()
        if isinstance(attn_weights, torch.Tensor)
        else np.array(attn_weights)
    )
    if aw.ndim == 4:
        aw = aw[0]

    head_stats = []
    if aw.ndim == 3:
        num_heads = aw.shape[0]
        for h in range(num_heads):
            p = aw[h]
            row_entropy = -np.sum(p * np.log(p + 1e-8), axis=1)
            entropy = float(np.mean(row_entropy))
            avg_attention = float(np.mean(p))
            head_stats.append(
                {"head": h, "entropy": entropy, "avg_attention": avg_attention}
            )
    elif aw.ndim == 2:
        p = aw
        row_entropy = -np.sum(p * np.log(p + 1e-8), axis=1)
        entropy = float(np.mean(row_entropy))
        avg_attention = float(np.mean(p))
        head_stats.append(
            {"head": 0, "entropy": entropy, "avg_attention": avg_attention}
        )
    else:
        raise ValueError("Unexpected attention shape for head analysis")

    logger.info("ATTN_VISUALIZER: Head analysis for %d heads", len(head_stats))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Avg Attention per Head
    sns.barplot(
        x=[s["head"] for s in head_stats],
        y=[s["avg_attention"] for s in head_stats],
        ax=axes[0],
    )
    axes[0].set_title("Avg Attention per Head")

    # Entropy per Head
    sns.barplot(
        x=[s["head"] for s in head_stats],
        y=[s["entropy"] for s in head_stats],
        ax=axes[1],
    )
    axes[1].set_title("Entropy per Head")

    return save_plot_as_base64(fig)


def create_pattern_matrix(ltts_event, params: Dict[str, Any]) -> Optional[str]:
    model = get_ltts_model_from_context_or_global(ltts_event)
    if model is None:
        raise ValueError("No model available for pattern matrix")

    attn_weights = _get_layer_attention_weights(ltts_event)
    if attn_weights is None:
        raise ValueError("No attention weights available for pattern matrix")

    logger.info(
        "ATTN_VISUALIZER: Creating pattern matrix for %s",
        getattr(ltts_event.context, "module_path", None),
    )

    head_index = int(params.get("head_index", -1))
    aw2d = reduce_to_2d(attn_weights, head_index)
    if aw2d is None:
        raise ValueError("Unexpected attention shape for pattern matrix")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(aw2d, cmap="magma", ax=ax)
    ax.set_title("Attention Patterns")
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Tokens")

    return save_plot_as_base64(fig)


def create_entropy_plot(ltts_event, params: Dict[str, Any]) -> Optional[str]:
    model = get_ltts_model_from_context_or_global(ltts_event)
    if model is None:
        raise ValueError("No model available for entropy plot")

    attn_weights = _get_layer_attention_weights(ltts_event)
    if attn_weights is None:
        raise ValueError("No attention weights available for entropy plot")

    head_index = int(params.get("head_index", -1))
    aw2d = reduce_to_2d(attn_weights, head_index)
    if aw2d is None:
        raise ValueError("Unexpected attention shape for entropy plot")

    entropy_values = -np.sum(aw2d * np.log(aw2d + 1e-8), axis=1)
    avg_entropy = float(np.mean(entropy_values))
    interpretation = get_entropy_interpretation(avg_entropy)

    logger.info(
        "ATTN_VISUALIZER: Entropy avg=%.3f interpretation=%s",
        avg_entropy,
        interpretation,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(entropy_values)
    ax.set_title(f"Attention Entropy (Avg: {avg_entropy:.3f})")

    return save_plot_as_base64(fig)


def create_demo_attention_heatmap(params: Dict[str, Any]) -> Any:
    seq_len = int(params.get("seq_len", 20))
    num_heads = int(params.get("num_heads", 8))
    basic_stats = {"avg_weight": 0.5, "max_weight": 0.9}

    synthetic_weights = create_synthetic_attention_weights(
        basic_stats, seq_len, num_heads
    )

    logger.info(
        "ATTN_VISUALIZER: Demo heatmap seq_len=%d num_heads=%d", seq_len, num_heads
    )

    # Reduce to 2D: average across batch and heads unless head_index is specified
    head_index = int(params.get("head_index", -1))
    sw = (
        synthetic_weights.detach().cpu().numpy()
        if isinstance(synthetic_weights, torch.Tensor)
        else np.array(synthetic_weights)
    )
    if sw.ndim == 4:
        # shape: [batch, heads, seq, seq]
        if 0 <= head_index < sw.shape[1]:
            sw2d = sw[0, head_index]
        else:
            sw2d = sw.mean(axis=(0, 1))
    elif sw.ndim == 3:
        # shape: [heads, seq, seq]
        if 0 <= head_index < sw.shape[0]:
            sw2d = sw[head_index]
        else:
            sw2d = sw.mean(axis=0)
    elif sw.ndim == 2:
        sw2d = sw
    else:
        return create_error_visualization_data("Unexpected synthetic attention shape")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(sw2d, cmap="viridis", ax=ax)
    ax.set_title("Demo Attention Heatmap")

    return save_plot_as_base64(fig)


def create_test_visualization(message: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)
    ax.set_title("Test Visualization")
    ax.axis("off")

    return save_plot_as_base64(fig)


def create_comprehensive_analysis(ltts_event, params: Dict[str, Any]) -> Optional[str]:
    model = get_ltts_model_from_context_or_global(ltts_event)
    if model is None:
        raise ValueError("No model for comprehensive analysis")

    outputs = getattr(ltts_event.context, "outputs", None)
    attn_weights = _get_layer_attention_weights(ltts_event)
    if attn_weights is None:
        raise ValueError("No attention weights available for comprehensive analysis")

    head_index = int(params.get("head_index", -1))
    aw2d = reduce_to_2d(attn_weights, head_index)
    if aw2d is None:
        raise ValueError("Unexpected attention shape for comprehensive analysis")

    patterns = analyze_attention_patterns(aw2d)
    entropy_values = -np.sum(aw2d * np.log(aw2d + 1e-8), axis=1)
    avg_entropy = float(np.mean(entropy_values))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sns.heatmap(aw2d, cmap="viridis", ax=axes[0][0])
    axes[0][0].set_title("Attention Heatmap")

    axes[0][1].plot(entropy_values)
    axes[0][1].set_title(f"Entropy (Avg: {avg_entropy:.3f})")

    sns.heatmap(aw2d, cmap="magma", ax=axes[1][0])
    axes[1][0].set_title("Pattern Matrix")

    similarity = calculate_similarity_matrix(
        outputs, metric=params.get("metric", "cosine")
    )
    sns.heatmap(similarity, cmap="coolwarm", ax=axes[1][1])
    axes[1][1].set_title("Similarity Matrix")

    return save_plot_as_base64(fig)


def create_multi_layer_comparison(ltts_event, params: Dict[str, Any]) -> Optional[str]:
    import re

    layer_path_str = params.get("layer_path") or ""
    layer_paths = (
        [lp.strip() for lp in re.split(r"[,;\s]+", layer_path_str) if lp.strip()]
        if layer_path_str
        else []
    )

    logger.info(
        "ATTN_VISUALIZER: multi-layer comparison for %d layers", len(layer_paths)
    )

    fig, axes = plt.subplots(
        len(layer_paths) or 1, 1, figsize=(10, 5 * max(len(layer_paths), 1))
    )

    if not layer_paths:
        axes = np.atleast_1d(axes)
        axes[0].text(0.5, 0.5, "No layer paths provided", ha="center", va="center")
        axes[0].set_title("Multi-layer Comparison")
        axes[0].axis("off")
        return save_plot_as_base64(fig)

    for i, lp in enumerate(layer_paths):
        ax = axes[i] if len(layer_paths) > 1 else axes
        ax.text(0.5, 0.5, f"Layer: {lp}", ha="center", va="center")
        ax.set_title(f"Layer {i+1}: {lp}")
        ax.axis("off")

    return save_plot_as_base64(fig)


def create_token_level_analysis(ltts_event, params: Dict[str, Any]) -> Optional[str]:
    model = get_ltts_model_from_context_or_global(ltts_event)
    if model is None:
        raise ValueError("No model for token-level analysis")

    attn_weights = _get_layer_attention_weights(ltts_event)
    if attn_weights is None:
        raise ValueError("No attention weights available for token-level analysis")

    head_index = int(params.get("head_index", -1))
    aw2d = reduce_to_2d(attn_weights, head_index)
    if aw2d is None:
        raise ValueError("Unexpected attention shape for token-level analysis")

    token_range = params.get("token_range")
    max_len = aw2d.shape[-1]
    parsed_range = parse_token_range(token_range, max_len)
    if not parsed_range:
        raise ValueError("Invalid token range")

    start, end = parsed_range

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(aw2d.mean(axis=1)[start:end])
    ax.set_title(f"Token-Level Attention ({start}:{end})")

    return save_plot_as_base64(fig)


def create_synthetic_attention_weights(
    basic_stats: Dict[str, Any], seq_len: int, num_heads: int
) -> torch.Tensor:
    mean = basic_stats.get("mean", 0.1)
    std = basic_stats.get("std", 0.05)
    weights = torch.randn(1, num_heads, seq_len, seq_len) * std + mean
    weights = torch.softmax(weights, dim=-1)
    for h in range(num_heads):
        for i in range(seq_len):
            weights[0, h, i, i] += 0.1
    weights = torch.softmax(weights, dim=-1)
    return weights


def generate_synthetic_head_stats(num_heads: int) -> List[Dict[str, Any]]:
    stats = []
    for i in range(num_heads):
        stats.append(
            {
                "head": i,
                "entropy": float(np.random.uniform(1.0, 4.0)),
                "avg_attention": float(np.random.uniform(0.1, 0.8)),
            }
        )
    return stats


def reduce_to_2d(attention: Any, head_index: int = -1) -> Optional[np.ndarray]:
    try:
        aw = (
            attention.detach().to(torch.float32).cpu().numpy()
            if isinstance(attention, torch.Tensor)
            else np.array(attention)
        )
        if aw.ndim == 4:
            # [batch, heads, seq, seq]
            if 0 <= head_index < aw.shape[1]:
                return aw[0, head_index]
            return aw.mean(axis=(0, 1))
        if aw.ndim == 3:
            # [heads, seq, seq]
            if 0 <= head_index < aw.shape[0]:
                return aw[head_index]
            return aw.mean(axis=0)
        if aw.ndim == 2:
            return aw
        return None
    except Exception as e:
        logger.warning("Error reducing attention to 2D: %s", e)
        return None


# Helper: prefer captured layer attention from LTTS model, fallback to outputs


def _get_layer_attention_weights(ltts_event) -> Optional[torch.Tensor]:
    try:
        ltts_model = getattr(ltts_event.context, "ltts_model", None)
        module_path = (getattr(ltts_event.context, "module_path", "") or "").strip()
        outputs = getattr(ltts_event.context, "outputs", None)

        def _append_unique(seq, value):
            if value and value not in seq:
                seq.append(value)

        candidate_keys: List[str] = []
        if module_path:
            _append_unique(candidate_keys, module_path)

            # Strip common prefixes while keeping ordering preference
            base_path = re.sub(
                r"^(original_model\.model\.|original_model\.|model\.)", "", module_path
            )
            for prefix in ("", "model.", "original_model.", "original_model.model."):
                prefixed = f"{prefix}{base_path}" if prefix else base_path
                _append_unique(candidate_keys, prefixed)

            # Ensure self-attention variants exist
            for existing in list(candidate_keys):
                if ".self_attn" not in existing:
                    _append_unique(candidate_keys, f"{existing}.self_attn")

            # Add canonical templates based on layer index if present
            match = re.search(r"layers\.(\d+)(?:\.|$)", module_path)
            if not match:
                match = re.search(r"transformer\.h\.(\d+)(?:\.|$)", module_path)
            if match:
                idx = match.group(1)
                for template in [
                    f"model.layers.{idx}.self_attn",
                    f"layers.{idx}.self_attn",
                ]:
                    _append_unique(candidate_keys, template)

        # 1) Prefer dynamically captured attention per-layer
        if ltts_model and hasattr(ltts_model, "captured_attn"):
            captured = ltts_model.captured_attn or {}
            for key in candidate_keys:
                if key in captured and captured[key]:
                    latest_step = max(captured[key].keys())
                    logger.debug(
                        "ATTN_VISUALIZER: Using captured attention at key '%s' (step=%s)",
                        key,
                        latest_step,
                    )
                    return captured[key][latest_step]
            if candidate_keys:
                logger.debug(
                    "ATTN_VISUALIZER: No captured attention match for '%s'. Candidates=%s, available=%s",
                    module_path,
                    candidate_keys,
                    list(captured.keys())[:6],
                )

        # 2) Fallback per-layer weights
        if ltts_model and hasattr(ltts_model, "layer_attention_weights"):
            law = ltts_model.layer_attention_weights or {}
            for key in candidate_keys:
                if key in law:
                    return law[key]

        # 3) Fallback to extracting from current outputs
        try:
            from miners.utils import extract_attention_weights

            return extract_attention_weights(outputs)
        except Exception:
            return None
    except Exception as e:
        logger.warning("Error getting layer attention: %s", e)
        return None


def create_token_level_analysis(ltts_event, params: Dict[str, Any]) -> Optional[str]:
    model = get_ltts_model_from_context_or_global(ltts_event)
    if model is None:
        raise ValueError("No model for token-level analysis")

    attn_weights = _get_layer_attention_weights(ltts_event)
    if attn_weights is None:
        raise ValueError("No attention weights available for token-level analysis")

    head_index = int(params.get("head_index", -1))
    aw2d = reduce_to_2d(attn_weights, head_index)
    if aw2d is None:
        raise ValueError("Unexpected attention shape for token-level analysis")

    token_range = params.get("token_range")
    max_len = aw2d.shape[-1]
    parsed_range = parse_token_range(token_range, max_len)
    if not parsed_range:
        raise ValueError("Invalid token range")

    start, end = parsed_range

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(aw2d.mean(axis=1)[start:end])
    ax.set_title(f"Token-Level Attention ({start}:{end})")

    return save_plot_as_base64(fig)


def reduce_to_2d(arr: Any, head_index: int = -1) -> Optional[np.ndarray]:
    a = arr.detach().to(torch.float32).cpu().numpy() if isinstance(arr, torch.Tensor) else np.array(arr)
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3:
        if 0 <= head_index < a.shape[0]:
            return a[head_index]
        return a.mean(axis=0)
    if a.ndim == 2:
        return a
    return None
