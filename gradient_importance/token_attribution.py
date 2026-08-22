"""
Token Attribution for NLP Models

PURE gradient-based attribution methods for transformer and language models.
Identifies which tokens/positions contribute most to model predictions through
gradient analysis.

Implements:
- Gradient Magnitude: Simple gradient importance
- Input × Gradient: Activation-weighted gradient attribution
- Integrated Gradients: Path integral approximation

Note: For attention-based analysis, use the attn_visualizer module instead.

Outputs: Base64-encoded visualizations for sequence/token data.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Any, List, Optional, Tuple
import logging
import io
import base64

logger = logging.getLogger(__name__)


class TokenAttributionAnalyzer:
    """
    Analyzes token-level importance using PURE gradient-based methods.

    This module focuses exclusively on gradient attribution - for attention-based
    analysis, use the attn_visualizer module instead.

    Designed for NLP/sequence models, NOT for image saliency maps.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.gradient_cache = {}
        self.activation_cache = {}

    @staticmethod
    def fig_to_base64(fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64-encoded PNG string for sending to client.

        Args:
            fig: Matplotlib figure

        Returns:
            Base64-encoded data URI string
        """
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"

    # ========== ATTRIBUTION METHODS ==========

    def compute_gradient_magnitude(
        self,
        activations: torch.Tensor,
        gradients: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute importance based on gradient magnitude.

        This is the simplest attribution: |∇L/∇activation|
        Shows which activations have the steepest gradient.

        Args:
            activations: Layer activations [batch, seq_len, hidden_dim]
            gradients: Gradients w.r.t activations [batch, seq_len, hidden_dim]

        Returns:
            Token importance scores [batch, seq_len]
        """
        if gradients is not None:
            importance = torch.abs(gradients)
        else:
            # Fallback: activation magnitude
            logger.debug("No gradients available, using activation magnitude")
            importance = torch.abs(activations)

        # Aggregate across hidden dimension (L2 norm)
        if importance.dim() > 2:
            importance = torch.norm(importance, p=2, dim=-1)

        return importance

    def compute_input_x_gradient(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Input × Gradient attribution.

        This method considers both the activation value AND its gradient.
        More sensitive than pure gradient magnitude.

        Formula: importance = activation ⊙ gradient (element-wise product)

        Args:
            activations: Layer activations [batch, seq_len, hidden_dim]
            gradients: Gradients w.r.t activations [batch, seq_len, hidden_dim]

        Returns:
            Token attribution scores [batch, seq_len]
        """
        # Element-wise multiplication
        attribution = activations * gradients

        # Sum across hidden dimension (not absolute - preserves direction)
        if attribution.dim() > 2:
            attribution = attribution.sum(dim=-1)

        # Return absolute value for importance magnitude
        return torch.abs(attribution)

    def compute_integrated_gradients_approx(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
        baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Approximate Integrated Gradients.

        True IG requires multiple forward passes with interpolated inputs.
        This is a simplified version using only the available gradient.

        Formula: IG ≈ (input - baseline) ⊙ gradient

        Args:
            activations: Layer activations [batch, seq_len, hidden_dim]
            gradients: Gradients at current activations [batch, seq_len, hidden_dim]
            baseline: Baseline activations (default: zeros)

        Returns:
            Attribution scores [batch, seq_len]
        """
        if baseline is None:
            baseline = torch.zeros_like(activations)

        # Path integral approximation
        delta = activations - baseline
        attribution = delta * gradients

        # Aggregate across features
        if attribution.dim() > 2:
            attribution = attribution.sum(dim=-1)

        return torch.abs(attribution)


    # ========== VISUALIZATION METHODS ==========

    def create_token_importance_bars(
        self,
        token_scores: torch.Tensor,
        tokens: Optional[List[str]] = None,
        title: str = "Token Importance",
        top_k: int = 10,
        figsize: Tuple[int, int] = (14, 6)
    ) -> str:
        """
        Create bar chart showing token importance.

        Args:
            token_scores: Importance scores [seq_len]
            tokens: Optional token strings
            title: Plot title
            top_k: Highlight top-k tokens
            figsize: Figure size

        Returns:
            Base64-encoded image string (data URI)
        """
        # Convert to numpy and handle batch dimension
        scores_np = token_scores.detach().cpu().numpy()
        if scores_np.ndim > 1:
            scores_np = scores_np[0]  # Take first batch

        # Normalize to [0, 1]
        scores_norm = (scores_np - scores_np.min()) / (scores_np.max() - scores_np.min() + 1e-8)

        # Get top-k indices
        top_indices = np.argsort(scores_np)[-top_k:]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                        gridspec_kw={'height_ratios': [2, 1]})

        # Plot 1: All tokens with top-k highlighted
        colors = ['#e74c3c' if i in top_indices else '#3498db' for i in range(len(scores_np))]
        bars = ax1.bar(range(len(scores_np)), scores_norm, color=colors, alpha=0.7,
                       edgecolor='black', linewidth=0.5)

        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_xlabel('Token Position', fontsize=11)
        ax1.set_ylabel('Importance Score', fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Add token labels if available
        if tokens and len(tokens) == len(scores_np):
            # Show every nth token to avoid crowding
            step = max(1, len(tokens) // 20)
            ax1.set_xticks(range(0, len(tokens), step))
            ax1.set_xticklabels([tokens[i] for i in range(0, len(tokens), step)],
                               rotation=45, ha='right', fontsize=8)

        # Legend
        red_patch = mpatches.Patch(color='#e74c3c', label=f'Top {top_k} tokens')
        blue_patch = mpatches.Patch(color='#3498db', label='Other tokens')
        ax1.legend(handles=[red_patch, blue_patch], loc='upper right')

        # Plot 2: Top-k tokens ranked
        top_indices_sorted = top_indices[np.argsort(scores_np[top_indices])[::-1]]
        top_scores = scores_norm[top_indices_sorted]

        if tokens:
            labels = [tokens[i] if i < len(tokens) else f'T{i}' for i in top_indices_sorted]
        else:
            labels = [f'Token {i}' for i in top_indices_sorted]

        bars2 = ax2.barh(range(len(top_scores)), top_scores,
                         color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_yticks(range(len(top_scores)))
        ax2.set_yticklabels(labels, fontsize=9)
        ax2.set_xlabel('Importance Score', fontsize=11)
        ax2.set_title(f'Top {top_k} Most Important Tokens', fontsize=12)
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x', linestyle='--')

        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars2, top_scores)):
            ax2.text(score + 0.01, i, f'{score:.3f}',
                    va='center', fontsize=8)

        plt.tight_layout()
        return self.fig_to_base64(fig)

    def create_text_heatmap(
        self,
        token_scores: torch.Tensor,
        tokens: List[str],
        title: str = "Token Attribution",
        max_tokens_per_row: int = 15,
        figsize: Tuple[int, int] = (14, 8)
    ) -> str:
        """
        Create text-based heatmap with color-coded tokens.

        This is the NLP equivalent of image saliency maps - showing
        token importance through background colors.

        Args:
            token_scores: Importance scores [seq_len]
            tokens: List of token strings
            title: Plot title
            max_tokens_per_row: Wrap text after N tokens
            figsize: Figure size

        Returns:
            Base64-encoded image string (data URI)
        """
        scores_np = token_scores.detach().cpu().numpy()
        if scores_np.ndim > 1:
            scores_np = scores_np[0]

        # Normalize scores
        scores_norm = (scores_np - scores_np.min()) / (scores_np.max() - scores_np.min() + 1e-8)

        # Create custom colormap (white -> yellow -> red)
        colors_map = LinearSegmentedColormap.from_list(
            'importance',
            ['#ffffff', '#fff3cd', '#ffc107', '#ff6b6b', '#c0392b']
        )

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')

        # Layout tokens in rows
        y_pos = 0.95
        x_pos = 0.05
        token_count = 0

        for i, (token, score) in enumerate(zip(tokens, scores_norm)):
            # Get color based on importance
            color = colors_map(score)

            # Calculate token width (proportional to text length)
            token_width = len(token) * 0.013 + 0.015

            # Wrap to next line if needed
            if token_count >= max_tokens_per_row or x_pos + token_width > 0.98:
                y_pos -= 0.08
                x_pos = 0.05
                token_count = 0

            # Draw token box
            bbox_props = dict(
                boxstyle='round,pad=0.4',
                facecolor=color,
                edgecolor='#2c3e50',
                linewidth=1.5 if score > 0.7 else 0.8,  # Thick border for important tokens
                alpha=0.9
            )

            ax.text(
                x_pos, y_pos,
                token,
                fontsize=11 if score > 0.7 else 9,
                fontweight='bold' if score > 0.7 else 'normal',
                bbox=bbox_props,
                transform=ax.transAxes,
                verticalalignment='center'
            )

            x_pos += token_width
            token_count += 1

        # Title
        ax.text(0.5, 0.98, title, fontsize=16, fontweight='bold',
                ha='center', transform=ax.transAxes)

        # Add colorbar legend
        sm = plt.cm.ScalarMappable(cmap=colors_map, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                           pad=0.02, shrink=0.6, aspect=30)
        cbar.set_label('Token Importance', fontsize=11)

        # Add statistics
        stats_text = (
            f"Tokens: {len(tokens)} | "
            f"Mean: {scores_np.mean():.3f} | "
            f"Max: {scores_np.max():.3f} | "
            f"Std: {scores_np.std():.3f}"
        )
        ax.text(0.5, 0.02, stats_text, fontsize=9, ha='center',
                style='italic', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.8))

        plt.tight_layout()
        return self.fig_to_base64(fig)

    def create_sequence_heatmap(
        self,
        importance_matrix: torch.Tensor,
        tokens: Optional[List[str]] = None,
        title: str = "Sequence Importance",
        colormap: str = 'YlOrRd',
        figsize: Tuple[int, int] = (12, 6)
    ) -> str:
        """
        Create 2D heatmap for sequence x features importance.

        Useful when you have importance scores across both sequence
        positions and feature dimensions.

        Args:
            importance_matrix: Importance [seq_len, feature_dim] or [seq_len]
            tokens: Optional token labels
            title: Plot title
            colormap: Matplotlib colormap
            figsize: Figure size

        Returns:
            Base64-encoded image string (data URI)
        """
        matrix_np = importance_matrix.detach().cpu().numpy()

        # Handle 1D input
        if matrix_np.ndim == 1:
            matrix_np = matrix_np.reshape(-1, 1)
        elif matrix_np.ndim == 3:
            # [batch, seq, feat] -> take first batch
            matrix_np = matrix_np[0]

        # Normalize
        matrix_np = (matrix_np - matrix_np.min()) / (matrix_np.max() - matrix_np.min() + 1e-8)

        # Limit size for visualization
        max_seq = min(matrix_np.shape[0], 100)
        max_feat = min(matrix_np.shape[1], 100)
        matrix_np = matrix_np[:max_seq, :max_feat]

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(matrix_np, cmap=colormap, aspect='auto', interpolation='nearest')

        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Feature Dimension', fontsize=11)
        ax.set_ylabel('Token Position', fontsize=11)

        # Token labels on y-axis
        if tokens and len(tokens) <= 50:
            ax.set_yticks(range(min(len(tokens), max_seq)))
            ax.set_yticklabels(tokens[:max_seq], fontsize=8)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Importance Score')
        cbar.ax.tick_params(labelsize=9)

        # Add grid for clarity
        ax.set_xticks(np.arange(max_feat) - 0.5, minor=True)
        ax.set_yticks(np.arange(max_seq) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)

        plt.tight_layout()
        return self.fig_to_base64(fig)

    def create_method_comparison(
        self,
        attributions: Dict[str, torch.Tensor],
        tokens: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 10)
    ) -> str:
        """
        Compare multiple attribution methods side-by-side.

        Args:
            attributions: Dict mapping method names to attribution tensors
            tokens: Optional token strings
            figsize: Figure size

        Returns:
            Base64-encoded image string (data URI)
        """
        n_methods = len(attributions)
        fig, axes = plt.subplots(n_methods, 1, figsize=figsize, sharex=True)

        if n_methods == 1:
            axes = [axes]

        # Normalize each method's scores
        normalized = {}
        for method, scores in attributions.items():
            scores_np = scores.detach().cpu().numpy()
            if scores_np.ndim > 1:
                scores_np = scores_np[0]
            scores_np = (scores_np - scores_np.min()) / (scores_np.max() - scores_np.min() + 1e-8)
            normalized[method] = scores_np

        # Plot each method
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        for idx, (method, scores_np) in enumerate(normalized.items()):
            ax = axes[idx]
            color = colors[idx % len(colors)]

            ax.bar(range(len(scores_np)), scores_np, alpha=0.7,
                  color=color, edgecolor='black', linewidth=0.5)
            ax.set_ylabel('Importance', fontsize=10)
            ax.set_title(method, fontsize=11, fontweight='bold', loc='left')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_ylim(0, 1.0)

        # X-axis labels only on bottom plot
        axes[-1].set_xlabel('Token Position', fontsize=11)

        if tokens and len(tokens) <= 30:
            axes[-1].set_xticks(range(len(tokens)))
            axes[-1].set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)

        fig.suptitle('Attribution Method Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        return self.fig_to_base64(fig)

    def create_neuron_analysis(
        self,
        importance_matrix: torch.Tensor,
        title: str = "Neuron Importance Analysis",
        figsize: Tuple[int, int] = (14, 10)
    ) -> str:
        """
        Analyze neuron-level importance (feature dimension).

        Shows which neurons/features in the hidden dimension are most active.

        Args:
            importance_matrix: Importance [seq_len, hidden_dim]
            title: Plot title
            figsize: Figure size

        Returns:
            Base64-encoded image string (data URI)
        """
        matrix_np = importance_matrix.detach().cpu().numpy()

        if matrix_np.ndim == 3:
            matrix_np = matrix_np[0]  # Take first batch

        # Aggregate across sequence to get per-neuron importance
        neuron_scores = matrix_np.mean(axis=0) if matrix_np.ndim == 2 else matrix_np

        # Create 4-panel analysis
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Panel 1: Neuron importance distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(neuron_scores, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
        ax1.axvline(neuron_scores.mean(), color='#e74c3c', linestyle='--',
                   linewidth=2, label=f'Mean: {neuron_scores.mean():.4f}')
        ax1.set_xlabel('Importance Score', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title('Distribution', fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Top neurons
        ax2 = fig.add_subplot(gs[0, 1])
        top_k = min(20, len(neuron_scores))
        top_indices = np.argsort(neuron_scores)[-top_k:][::-1]
        top_scores = neuron_scores[top_indices]

        ax2.barh(range(top_k), top_scores, alpha=0.7, color='#e74c3c', edgecolor='black')
        ax2.set_yticks(range(top_k))
        ax2.set_yticklabels([f'Neuron {i}' for i in top_indices], fontsize=8)
        ax2.set_xlabel('Importance', fontsize=10)
        ax2.set_title(f'Top {top_k} Neurons', fontsize=11, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')

        # Panel 3: Cumulative importance
        ax3 = fig.add_subplot(gs[1, 0])
        sorted_scores = np.sort(neuron_scores)[::-1]
        cumulative = np.cumsum(sorted_scores) / np.sum(sorted_scores)

        ax3.plot(range(len(cumulative)), cumulative, linewidth=2, color='#2ecc71')
        ax3.axhline(0.9, color='#e74c3c', linestyle='--', alpha=0.7,
                   label='90% threshold')
        ax3.fill_between(range(len(cumulative)), cumulative, alpha=0.3, color='#2ecc71')
        ax3.set_xlabel('Number of Neurons', fontsize=10)
        ax3.set_ylabel('Cumulative Importance', fontsize=10)
        ax3.set_title('Cumulative Analysis', fontsize=11, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: Statistics table
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        # Calculate statistics
        threshold = neuron_scores.mean()
        active_neurons = np.sum(neuron_scores > threshold)
        sparsity = 1.0 - (active_neurons / len(neuron_scores))

        stats_text = f"""Neuron Statistics

Total Neurons: {len(neuron_scores):,}
Active (>mean): {active_neurons:,}
Sparsity: {sparsity:.1%}

Mean: {neuron_scores.mean():.4f}
Std Dev: {neuron_scores.std():.4f}
Maximum: {neuron_scores.max():.4f}
Minimum: {neuron_scores.min():.4f}

90th percentile: {np.percentile(neuron_scores, 90):.4f}
75th percentile: {np.percentile(neuron_scores, 75):.4f}
Median: {np.median(neuron_scores):.4f}"""

        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1',
                         edgecolor='#2c3e50', linewidth=1.5))

        fig.suptitle(title, fontsize=16, fontweight='bold')
        return self.fig_to_base64(fig)


def example_usage():
    """
    Example demonstrating token attribution for NLP.

    This shows how to use the analyzer in your LTTS module to send
    visualizations to the client as base64-encoded images.
    """

    # Simulate transformer layer output
    batch_size = 1
    seq_len = 15
    hidden_dim = 768

    activations = torch.randn(batch_size, seq_len, hidden_dim)
    gradients = torch.randn(batch_size, seq_len, hidden_dim)

    # Example tokens
    tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the",
              "lazy", "dog", "in", "the", "sunny", "park", ".", "[SEP]"]

    # Create analyzer
    analyzer = TokenAttributionAnalyzer()

    print("Computing token attributions...")

    # Compute different attribution methods
    grad_mag = analyzer.compute_gradient_magnitude(activations, gradients)
    input_x_grad = analyzer.compute_input_x_gradient(activations, gradients)
    integrated_grad = analyzer.compute_integrated_gradients_approx(activations, gradients)

    print("Creating visualizations as base64 strings...")

    # Visualization 1: Token importance bars
    base64_img1 = analyzer.create_token_importance_bars(
        grad_mag,
        tokens=tokens,
        title="Token Importance (Gradient Magnitude)",
        top_k=5
    )
    print(f"[OK] Token importance bars: {len(base64_img1)} chars")

    # Visualization 2: Text heatmap
    base64_img2 = analyzer.create_text_heatmap(
        input_x_grad,
        tokens=tokens,
        title="Token Attribution Heatmap",
        max_tokens_per_row=10
    )
    print(f"[OK] Text heatmap: {len(base64_img2)} chars")

    # Visualization 3: Method comparison
    attributions = {
        "Gradient Magnitude": grad_mag,
        "Input x Gradient": input_x_grad,
        "Integrated Gradients": integrated_grad
    }
    base64_img3 = analyzer.create_method_comparison(attributions, tokens=tokens)
    print(f"[OK] Method comparison: {len(base64_img3)} chars")

    # Visualization 4: Sequence heatmap
    importance_2d = torch.abs(activations[0])  # [seq_len, hidden_dim]
    base64_img4 = analyzer.create_sequence_heatmap(
        importance_2d[:, :64],  # Show first 64 features
        tokens=tokens,
        title="Sequence x Feature Importance"
    )
    print(f"[OK] Sequence heatmap: {len(base64_img4)} chars")

    # Visualization 5: Neuron analysis
    base64_img5 = analyzer.create_neuron_analysis(
        importance_2d,
        title="Hidden Dimension Neuron Analysis"
    )
    print(f"[OK] Neuron analysis: {len(base64_img5)} chars")

    print("\n[SUCCESS] All visualizations created as base64 strings!")
    print("These can now be sent to the client via reactor.signal()")
    print(f"\nToken scores summary:")
    print(f"  Gradient Magnitude - Max token: '{tokens[grad_mag[0].argmax()]}'")
    print(f"  Input x Gradient - Max token: '{tokens[input_x_grad[0].argmax()]}'")

    # Example of how to send to client in LTTS module:
    print("\nExample payload for client:")
    print("""
    payload = {
        "type": "token_attribution",
        "subtype": "token_importance_bars",
        "data": base64_img1,  # Send the base64 string directly
        "tokens": tokens,
        "scores": grad_mag.tolist()
    }
    ltts_event.reactor.signal("visualization", payload)
    """)


if __name__ == "__main__":
    example_usage()
