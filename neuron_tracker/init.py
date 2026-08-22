"""
Neuron Activation Tracker Module

Tracks and analyzes neuron activations across layers to identify
important neurons, dead neurons, and activation patterns.
"""

import threading
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from collections import defaultdict
import sys
import os

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)
_MODULE_LOCAL = threading.local()

METADATA = {
    "name": "Neuron Activation Tracker",
    "version": "1.0.0",
    "description": "Track and analyze neuron activation patterns across layers",
    "author": "LTTS Team",
    "event_types": ["layer_after"],
    "methods": {
        "ctor": "initialize",
        "dtor": "cleanup",
        "ntor": "intercept_layer",
        "utor": "get_ui_schema",
    },
    "dependencies": ["torch", "numpy"],
    "tags": ["neurons", "activations", "analysis", "interpretability"],
}


class NeuronTracker:
    def __init__(self):
        self.activation_history = defaultdict(list)
        self.neuron_stats = {}
        self.dead_neurons = {}
        self.high_activations = {}
        self.enabled = True
        self.config = {
            "activation_threshold": 0.01,
            "dead_neuron_threshold": 0.001,
            "high_activation_threshold": 0.9,
            "max_history_size": 100,
            "track_percentiles": [10, 25, 50, 75, 90, 95, 99],
        }
        self.data_sender = get_data_sender("neuron_tracker")

    def initialize(self, context):
        """Initialize the neuron tracker."""
        logger.info("Neuron Activation Tracker initialized")
        self.data_sender.set_context(context)
        return True

    def cleanup(self):
        """Clean up resources."""
        self.activation_history.clear()
        self.neuron_stats.clear()
        self.dead_neurons.clear()
        self.high_activations.clear()
        logger.info("Neuron Activation Tracker cleaned up")

    def intercept_layer(self, layer_output, layer_info, context):
        """Intercept layer output to track neuron activations."""
        if not self.enabled:
            return layer_output

        try:
            layer_idx = layer_info.get("layer_index", 0)
            layer_name = layer_info.get("layer_name", f"layer_{layer_idx}")

            # Extract activations from the layer output
            activations = self._extract_activations(layer_output)
            if activations is not None:
                analysis = self._analyze_activations(activations, layer_name, layer_idx)

                # Store analysis results
                self.neuron_stats[layer_name] = analysis
                self._update_activation_history(layer_name, activations)

                # Identify dead and highly active neurons
                self._identify_special_neurons(layer_name, activations)

                # Send visualization data via data sender
                try:
                    # Send scalar metrics
                    stats = analysis["statistics"]
                    self.data_sender.send_scalar(
                        stats["mean"], f"{layer_name} Mean Activation"
                    )
                    self.data_sender.send_scalar(
                        analysis["sparsity"], f"{layer_name} Sparsity"
                    )

                    # Send dead neurons count as scalar
                    dead_count = len(self.dead_neurons.get(layer_name, []))
                    self.data_sender.send_scalar(
                        dead_count, f"{layer_name} Dead Neurons"
                    )

                    # Send activation distribution as chart
                    if analysis["activation_distribution"]:
                        self.data_sender.send_chart(
                            {
                                "histogram": analysis["activation_distribution"][
                                    "histogram"
                                ],
                                "bin_edges": analysis["activation_distribution"][
                                    "bin_edges"
                                ],
                            },
                            "bar",
                            layer_name=layer_name,
                        )

                    # Send summary table
                    table_data = {
                        "headers": ["Metric", "Value"],
                        "rows": [
                            ["Neurons", analysis["num_neurons"]],
                            ["Dead Neurons", dead_count],
                            [
                                "High Activations",
                                len(self.high_activations.get(layer_name, [])),
                            ],
                            ["Mean Activation", f"{stats['mean']:.4f}"],
                            ["Sparsity", f"{analysis['sparsity']:.2%}"],
                            ["Std Dev", f"{stats['std']:.4f}"],
                        ],
                    }
                    self.data_sender.send_table(
                        headers=table_data["headers"],
                        rows=table_data["rows"],
                        layer_name=layer_name,
                    )

                except Exception as e:
                    logger.error(f"Error sending visualization data: {e}")

        except Exception as e:
            logger.error(f"Error in neuron tracker: {e}")

        return layer_output

    def _extract_activations(self, layer_output):
        """Extract activation tensor from layer output."""
        # Handle different output formats
        if isinstance(layer_output, torch.Tensor):
            return layer_output
        elif hasattr(layer_output, "last_hidden_state"):
            return layer_output.last_hidden_state
        elif hasattr(layer_output, "hidden_states") and layer_output.hidden_states:
            return layer_output.hidden_states[-1]
        elif isinstance(layer_output, (list, tuple)) and len(layer_output) > 0:
            return layer_output[0]

        return None

    def _analyze_activations(self, activations, layer_name, layer_idx):
        """Analyze activation patterns for a layer."""
        if activations.dim() > 2:
            # Flatten batch and sequence dimensions
            activations_flat = activations.view(-1, activations.shape[-1])
        else:
            activations_flat = activations

        # Convert to numpy for statistical analysis
        activations_np = activations_flat.detach().to(torch.float32).cpu().numpy()

        analysis = {
            "layer_index": layer_idx,
            "layer_name": layer_name,
            "num_neurons": activations_np.shape[1],
            "statistics": {
                "mean": float(np.mean(activations_np)),
                "std": float(np.std(activations_np)),
                "min": float(np.min(activations_np)),
                "max": float(np.max(activations_np)),
                "median": float(np.median(activations_np)),
            },
            "percentiles": {},
            "sparsity": float(
                np.mean(activations_np < self.config["activation_threshold"])
            ),
            "activation_distribution": self._compute_distribution(activations_np),
        }

        # Compute percentiles
        for percentile in self.config["track_percentiles"]:
            analysis["percentiles"][f"p{percentile}"] = float(
                np.percentile(activations_np, percentile)
            )

        return analysis

    def _compute_distribution(self, activations_np):
        """Compute activation distribution histogram."""
        hist, bin_edges = np.histogram(activations_np, bins=20, density=True)
        return {"histogram": hist.tolist(), "bin_edges": bin_edges.tolist()}

    def _update_activation_history(self, layer_name, activations):
        """Update activation history for tracking over time."""
        activations_flat = activations.view(-1, activations.shape[-1])
        mean_activations = torch.mean(activations_flat, dim=0).detach().to(torch.float32).cpu().numpy()

        self.activation_history[layer_name].append(mean_activations)

        # Limit history size
        if len(self.activation_history[layer_name]) > self.config["max_history_size"]:
            self.activation_history[layer_name].pop(0)

    def _identify_special_neurons(self, layer_name, activations):
        """Identify dead and highly active neurons."""
        activations_flat = activations.view(-1, activations.shape[-1])
        mean_activations = torch.mean(activations_flat, dim=0).detach().to(torch.float32).cpu().numpy()

        # Dead neurons (consistently low activation)
        dead_mask = mean_activations < self.config["dead_neuron_threshold"]
        self.dead_neurons[layer_name] = np.where(dead_mask)[0].tolist()

        # Highly active neurons
        high_mask = mean_activations > self.config["high_activation_threshold"]
        self.high_activations[layer_name] = np.where(high_mask)[0].tolist()

    def get_neuron_summary(self):
        """Get summary of neuron tracking data."""
        summary = {
            "layers_tracked": len(self.neuron_stats),
            "total_dead_neurons": sum(
                len(neurons) for neurons in self.dead_neurons.values()
            ),
            "total_high_activations": sum(
                len(neurons) for neurons in self.high_activations.values()
            ),
            "layer_details": {},
        }

        for layer_name in self.neuron_stats:
            summary["layer_details"][layer_name] = {
                "dead_neurons": len(self.dead_neurons.get(layer_name, [])),
                "high_activations": len(self.high_activations.get(layer_name, [])),
                "sparsity": self.neuron_stats[layer_name]["sparsity"],
                "avg_activation": self.neuron_stats[layer_name]["statistics"]["mean"],
            }

        return summary

    def get_activation_trends(self, layer_name, neuron_idx=None):
        """Get activation trends over time for a layer or specific neuron."""
        if layer_name not in self.activation_history:
            return None

        history = np.array(self.activation_history[layer_name])

        if neuron_idx is not None:
            return history[:, neuron_idx].tolist()
        else:
            return {
                "mean_trend": np.mean(history, axis=1).tolist(),
                "std_trend": np.std(history, axis=1).tolist(),
            }

    def reset_data(self):
        """Reset collected neuron data."""
        self.activation_history.clear()
        self.neuron_stats.clear()
        self.dead_neurons.clear()
        self.high_activations.clear()


def _get_instance():
    inst = getattr(_MODULE_LOCAL, "instance", None)
    if inst is None:
        inst = NeuronTracker()
        _MODULE_LOCAL.instance = inst
    return inst


def initialize(context):
    """Module constructor."""
    tracker = NeuronTracker()
    tracker.initialize(context)
    _MODULE_LOCAL.instance = tracker
    return tracker


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
    """LTTSEvent interceptor for neuron tracking."""
    _MODULE_INSTANCE = _get_instance()
    _MODULE_INSTANCE.data_sender.set_context(ltts_event.context)
    if isinstance(ltts_event.module_state, dict):
        _MODULE_INSTANCE.config.update(ltts_event.module_state)
    layer_output = getattr(ltts_event.context, "outputs", None)
    layer_path = getattr(ltts_event.context, "module_path", "unknown")
    layer_info = {
        "layer_name": layer_path,
        "layer_index": _infer_layer_index(layer_path),
    }
    result = _MODULE_INSTANCE.intercept_layer(layer_output, layer_info, ltts_event.context)

    # Emit visualization data via reactor for consistent client delivery
    layer_name = layer_info["layer_name"]
    analysis = _MODULE_INSTANCE.neuron_stats.get(layer_name)
    if analysis:
        stats = analysis.get("statistics", {})
        dead_count = len(_MODULE_INSTANCE.dead_neurons.get(layer_name, []))
        high_count = len(_MODULE_INSTANCE.high_activations.get(layer_name, []))

        # Send mean activation as scalar
        ltts_event.reactor.signal("visualization", {
            "type": "visualization",
            "subtype": "scalar",
            "module": "neuron_tracker",
            "event_type": "module_event",
            "module_path": layer_path,
            "layer_path": layer_path,
            "data": {
                "type": "scalar",
                "value": stats.get("mean", 0),
                "label": f"{layer_name} Mean Activation",
            },
        })

        # Send activation distribution as chart
        dist = analysis.get("activation_distribution")
        if dist and dist.get("histogram"):
            ltts_event.reactor.signal("visualization", {
                "type": "visualization",
                "subtype": "chart",
                "module": "neuron_tracker",
                "event_type": "module_event",
                "module_path": layer_path,
                "layer_path": layer_path,
                "data": {
                    "type": "chart",
                    "chart_type": "bar",
                    "data": {
                        "histogram": dist["histogram"],
                        "bin_edges": dist["bin_edges"],
                    },
                },
            })

        # Send summary table
        ltts_event.reactor.signal("visualization", {
            "type": "visualization",
            "subtype": "table",
            "module": "neuron_tracker",
            "event_type": "module_event",
            "module_path": layer_path,
            "layer_path": layer_path,
            "data": {
                "type": "table",
                "headers": ["Metric", "Value"],
                "rows": [
                    ["Neurons", str(analysis.get("num_neurons", "N/A"))],
                    ["Dead Neurons", str(dead_count)],
                    ["High Activations", str(high_count)],
                    ["Mean Activation", f"{stats.get('mean', 0):.4f}"],
                    ["Sparsity", f"{analysis.get('sparsity', 0):.2%}"],
                    ["Std Dev", f"{stats.get('std', 0):.4f}"],
                ],
            },
        })

    return result


def get_ui_schema():
    """Get UI configuration schema."""
    return {
        "type": "object",
        "properties": {
            "activation_threshold": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.01,
                "description": "Threshold for considering a neuron as active",
            },
            "dead_neuron_threshold": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 0.01,
                "default": 0.001,
                "description": "Threshold for identifying dead neurons",
            },
            "high_activation_threshold": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 1.0,
                "default": 0.9,
                "description": "Threshold for identifying highly active neurons",
            },
            "max_history_size": {
                "type": "integer",
                "minimum": 10,
                "maximum": 1000,
                "default": 100,
                "description": "Maximum number of activation samples to keep in history",
            },
        },
        "required": [],
    }
