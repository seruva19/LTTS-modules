"""
Layer-wise Statistics Module

Comprehensive layer-wise statistics collection and analysis for
monitoring model behavior, detecting anomalies, and understanding
computational patterns across layers.

Scope: per-layer statistical analysis (mean, std, min, max, distributions).
Infrastructure metadata (shapes, dtypes, memory, devices) is reported by the
`layer_inspector` module.
"""

import threading
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from collections import defaultdict, deque
import time
import sys
import os

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)
_MODULE_LOCAL = threading.local()

METADATA = {
    "name": "Layer Statistics Monitor",
    "version": "1.0.0",
    "description": "Comprehensive layer-wise statistics collection and monitoring",
    "author": "LTTS Team",
    "event_types": ["layer_after"],
    "methods": {
        "ctor": "initialize",
        "dtor": "cleanup",
        "ntor": "intercept_layer",
        "utor": "get_ui_schema",
    },
    "dependencies": ["torch", "numpy"],
    "tags": ["statistics", "monitoring", "analysis", "performance"],
}


class LayerStatisticsMonitor:
    def __init__(self):
        self.layer_stats = defaultdict(dict)
        self.computational_stats = defaultdict(dict)
        self.anomaly_detection = defaultdict(dict)
        self.history_buffer = defaultdict(lambda: deque(maxlen=100))
        self.enabled = True
        self.config = {
            "compute_basic_stats": True,
            "compute_distribution_stats": True,
            "compute_similarity_stats": True,
            "anomaly_detection": True,
            "anomaly_threshold": 2.0,  # Standard deviations
            "history_window": 100,
            "compute_performance": True,
            "track_gradients": False,
            "track_memory": True,
            "percentiles": [5, 25, 50, 75, 95],
            "bin_count": 50,
        }
        self.start_time = {}
        self.memory_usage = {}
        self.data_sender = get_data_sender("layer_statistics")

    def initialize(self, context):
        """Initialize the layer statistics monitor."""
        logger.info("Layer Statistics Monitor initialized")
        self.data_sender.set_context(context)
        return True

    def cleanup(self):
        """Clean up resources."""
        self.layer_stats.clear()
        self.computational_stats.clear()
        self.anomaly_detection.clear()
        self.history_buffer.clear()
        self.start_time.clear()
        self.memory_usage.clear()
        logger.info("Layer Statistics Monitor cleaned up")

    def intercept_layer(self, layer_output, layer_info, context):
        """Intercept layer output to compute comprehensive statistics."""
        if not self.enabled:
            return layer_output

        start_time = time.time() if self.config["compute_performance"] else None

        try:
            layer_idx = layer_info.get("layer_index", 0)
            layer_name = layer_info.get("layer_name", f"layer_{layer_idx}")

            # Start timing
            if start_time is not None:
                self.start_time[layer_name] = start_time

            # Extract tensor data for analysis
            tensor_data = self._extract_tensor_data(layer_output)

            if tensor_data is not None:
                # Compute basic statistics
                if self.config["compute_basic_stats"]:
                    basic_stats = self._compute_basic_statistics(
                        tensor_data, layer_name
                    )
                    self.layer_stats[layer_name]["basic"] = basic_stats

                # Compute distribution statistics
                if self.config["compute_distribution_stats"]:
                    dist_stats = self._compute_distribution_statistics(
                        tensor_data, layer_name
                    )
                    self.layer_stats[layer_name]["distribution"] = dist_stats

                # Compute similarity statistics
                if self.config["compute_similarity_stats"]:
                    similarity_stats = self._compute_similarity_statistics(
                        tensor_data, layer_name
                    )
                    self.layer_stats[layer_name]["similarity"] = similarity_stats

                # Track memory usage
                if self.config["track_memory"]:
                    memory_stats = self._compute_memory_statistics(
                        tensor_data, layer_name
                    )
                    self.layer_stats[layer_name]["memory"] = memory_stats

                # Update history and detect anomalies
                self._update_history_and_detect_anomalies(layer_name, tensor_data)

                # Send visualization data via data sender
                try:
                    # Send basic statistics as scalars
                    if "basic" in self.layer_stats[layer_name]:
                        basic_stats = self.layer_stats[layer_name]["basic"]
                        self.data_sender.send_scalar(
                            basic_stats["mean"], f"{layer_name} Mean"
                        )
                        self.data_sender.send_scalar(
                            basic_stats["std"], f"{layer_name} Std Dev"
                        )
                        self.data_sender.send_scalar(
                            basic_stats["zeros_percentage"], f"{layer_name} Zeros %"
                        )

                    # Send distribution histogram as chart
                    if "distribution" in self.layer_stats[layer_name]:
                        dist_stats = self.layer_stats[layer_name]["distribution"]
                        if dist_stats.get("histogram"):
                            self.data_sender.send_chart(
                                {
                                    "histogram": dist_stats["histogram"],
                                    "bin_edges": dist_stats["bin_edges"],
                                },
                                "bar",
                                layer_name=layer_name,
                            )

                        # Send distribution metrics as scalars
                        self.data_sender.send_scalar(
                            dist_stats.get("skewness", 0), f"{layer_name} Skewness"
                        )
                        self.data_sender.send_scalar(
                            dist_stats.get("kurtosis", 0), f"{layer_name} Kurtosis"
                        )

                    # Send similarity matrix as heatmap
                    if "similarity" in self.layer_stats[layer_name]:
                        sim_stats = self.layer_stats[layer_name]["similarity"]
                        if "similarity_matrix" in sim_stats:
                            self.data_sender.send_heatmap(
                                values=sim_stats["similarity_matrix"],
                                dimensions={
                                    "rows": len(sim_stats["similarity_matrix"]),
                                    "cols": (
                                        len(sim_stats["similarity_matrix"][0])
                                        if sim_stats["similarity_matrix"]
                                        else 0
                                    ),
                                },
                                layer_name=layer_name,
                            )

                    # Send memory usage as scalar
                    if "memory" in self.layer_stats[layer_name]:
                        mem_stats = self.layer_stats[layer_name]["memory"]
                        self.data_sender.send_scalar(
                            mem_stats.get("memory_usage_mb", 0),
                            f"{layer_name} Memory (MB)",
                        )

                    # Send anomaly alerts as text
                    if (
                        layer_name in self.anomaly_detection
                        and self.anomaly_detection[layer_name]
                    ):
                        anomalies = self.anomaly_detection[layer_name]
                        anomaly_text = (
                            f"Anomalies detected: {', '.join(anomalies.keys())}"
                        )
                        self.data_sender.send_text(anomaly_text, preformatted=True)

                    # Send summary table
                    basic_stats = self.layer_stats[layer_name].get("basic", {})
                    dist_stats = self.layer_stats[layer_name].get("distribution", {})
                    mem_stats = self.layer_stats[layer_name].get("memory", {})

                    table_data = {
                        "headers": ["Metric", "Value"],
                        "rows": [
                            ["Data Type", basic_stats.get("dtype", "N/A")],
                            ["Shape", str(basic_stats.get("shape", "N/A"))],
                            ["Mean", f"{basic_stats.get('mean', 0):.4f}"],
                            ["Std Dev", f"{basic_stats.get('std', 0):.4f}"],
                            ["Range", f"{basic_stats.get('range', 0):.4f}"],
                            [
                                "Memory (MB)",
                                f"{mem_stats.get('memory_usage_mb', 0):.2f}",
                            ],
                            ["Skewness", f"{dist_stats.get('skewness', 0):.4f}"],
                            ["Kurtosis", f"{dist_stats.get('kurtosis', 0):.4f}"],
                            [
                                "Anomalies",
                                (
                                    "Yes"
                                    if self.anomaly_detection.get(layer_name)
                                    else "No"
                                ),
                            ],
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
            logger.error(f"Error in layer statistics monitor: {e}")

        # End timing and compute performance stats
        if start_time is not None and layer_name in self.start_time:
            elapsed = time.time() - self.start_time[layer_name]
            self.computational_stats[layer_name]["execution_time"] = elapsed
            del self.start_time[layer_name]

        return layer_output

    def _extract_tensor_data(self, layer_output):
        """Extract tensor data from layer output."""
        try:
            if isinstance(layer_output, torch.Tensor):
                return layer_output
            elif hasattr(layer_output, "last_hidden_state"):
                return layer_output.last_hidden_state
            elif hasattr(layer_output, "hidden_states") and layer_output.hidden_states:
                return layer_output.hidden_states[-1]
            elif isinstance(layer_output, (list, tuple)) and len(layer_output) > 0:
                if isinstance(layer_output[0], torch.Tensor):
                    return layer_output[0]

            return None

        except Exception as e:
            logger.error(f"Error extracting tensor data: {e}")
            return None

    def _compute_basic_statistics(self, tensor, layer_name):
        """Compute basic statistical measures."""
        try:
            # Convert to numpy for statistical computation
            if tensor.dim() > 2:
                # Flatten batch and sequence dimensions
                tensor_flat = tensor.view(-1, tensor.shape[-1])
            else:
                tensor_flat = tensor

            tensor_np = tensor_flat.detach().to(torch.float32).cpu().numpy()

            stats = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "count": int(tensor_np.size),
                "mean": float(np.mean(tensor_np)),
                "std": float(np.std(tensor_np)),
                "min": float(np.min(tensor_np)),
                "max": float(np.max(tensor_np)),
                "median": float(np.median(tensor_np)),
                "variance": float(np.var(tensor_np)),
                "range": float(np.max(tensor_np) - np.min(tensor_np)),
                "zeros_count": int(np.sum(tensor_np == 0)),
                "zeros_percentage": float(np.mean(tensor_np == 0)),
                "nan_count": int(np.sum(np.isnan(tensor_np))),
                "inf_count": int(np.sum(np.isinf(tensor_np))),
            }

            # Compute percentiles
            for percentile in self.config["percentiles"]:
                stats[f"percentile_{percentile}"] = float(
                    np.percentile(tensor_np, percentile)
                )

            # Compute additional statistics for non-zero elements
            non_zero = tensor_np[tensor_np != 0]
            if len(non_zero) > 0:
                stats["non_zero_mean"] = float(np.mean(non_zero))
                stats["non_zero_std"] = float(np.std(non_zero))
                stats["non_zero_count"] = int(len(non_zero))

            return stats

        except Exception as e:
            logger.error(f"Error computing basic statistics: {e}")
            return {}

    def _compute_distribution_statistics(self, tensor, layer_name):
        """Compute distribution-based statistics."""
        try:
            tensor_np = tensor.detach().to(torch.float32).cpu().numpy().flatten()

            # Remove NaN and Inf values for histogram
            clean_data = tensor_np[np.isfinite(tensor_np)]

            if len(clean_data) == 0:
                return {"histogram": [], "bin_edges": []}

            # Compute histogram
            hist, bin_edges = np.histogram(
                clean_data, bins=self.config["bin_count"], density=True
            )

            # Compute distribution moments
            stats = {
                "histogram": hist.tolist(),
                "bin_edges": bin_edges.tolist(),
                "skewness": float(self._compute_skewness(clean_data)),
                "kurtosis": float(self._compute_kurtosis(clean_data)),
                "entropy": float(self._compute_entropy(clean_data)),
            }

            # Compute additional distribution properties
            stats.update(
                {
                    "data_range": float(np.max(clean_data) - np.min(clean_data)),
                    "iqr": float(
                        np.percentile(clean_data, 75) - np.percentile(clean_data, 25)
                    ),
                    "mad": float(
                        np.median(np.abs(clean_data - np.median(clean_data)))
                    ),  # Median Absolute Deviation
                }
            )

            return stats

        except Exception as e:
            logger.error(f"Error computing distribution statistics: {e}")
            return {}

    def _compute_similarity_statistics(self, tensor, layer_name):
        """Compute similarity-based statistics."""
        try:
            if tensor.dim() < 2:
                return {}

            # For sequence data, compute token-to-token similarity
            if tensor.dim() == 3:  # [batch, seq_len, hidden_dim]
                # Use first batch item
                token_embeddings = tensor[0]  # [seq_len, hidden_dim]

                # Normalize embeddings
                normalized = torch.nn.functional.normalize(token_embeddings, p=2, dim=1)

                # Compute cosine similarity matrix
                similarity_matrix = torch.mm(normalized, normalized.T)

                # Extract upper triangle (excluding diagonal)
                upper_triangle = torch.triu(similarity_matrix, diagonal=1)
                similarities = upper_triangle[upper_triangle != 0]
                if similarities.numel() == 0:
                    return {
                        "mean_similarity": 0.0,
                        "std_similarity": 0.0,
                        "max_similarity": 0.0,
                        "min_similarity": 0.0,
                        "similarity_matrix": similarity_matrix.detach()
                        .to(torch.float32)
                        .cpu()
                        .numpy()
                        .tolist(),
                    }

                stats = {
                    "mean_similarity": float(torch.mean(similarities).item()),
                    "std_similarity": float(torch.std(similarities).item()),
                    "max_similarity": float(torch.max(similarities).item()),
                    "min_similarity": float(torch.min(similarities).item()),
                    "similarity_matrix": similarity_matrix.detach()
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                    .tolist(),
                }

                return stats

            return {}

        except Exception as e:
            logger.error(f"Error computing similarity statistics: {e}")
            return {}

    def _compute_memory_statistics(self, tensor, layer_name):
        """Compute memory usage statistics."""
        try:
            # Estimate memory usage
            element_size = tensor.element_size()
            total_elements = tensor.numel()
            total_memory_mb = (total_elements * element_size) / (1024 * 1024)

            stats = {
                "memory_usage_mb": float(total_memory_mb),
                "element_size_bytes": int(element_size),
                "total_elements": int(total_elements),
                "tensor_shape": list(tensor.shape),
            }

            # Track memory usage over time
            if layer_name not in self.memory_usage:
                self.memory_usage[layer_name] = []

            self.memory_usage[layer_name].append(total_memory_mb)

            # Keep only recent history
            if len(self.memory_usage[layer_name]) > self.config["history_window"]:
                self.memory_usage[layer_name] = self.memory_usage[layer_name][
                    -self.config["history_window"] :
                ]

            return stats

        except Exception as e:
            logger.error(f"Error computing memory statistics: {e}")
            return {}

    def _update_history_and_detect_anomalies(self, layer_name, tensor):
        """Update history buffer and detect anomalies."""
        try:
            # Compute key metrics for history tracking
            if tensor.dim() > 2:
                tensor_flat = tensor.view(-1, tensor.shape[-1])
            else:
                tensor_flat = tensor

            current_mean = float(torch.mean(tensor_flat).item())
            current_std = float(torch.std(tensor_flat).item())

            # Update history
            self.history_buffer[layer_name].append(
                {
                    "timestamp": time.time(),
                    "mean": current_mean,
                    "std": current_std,
                    "min": float(torch.min(tensor_flat).item()),
                    "max": float(torch.max(tensor_flat).item()),
                }
            )

            # Detect anomalies if enabled and we have enough history
            if (
                self.config["anomaly_detection"]
                and len(self.history_buffer[layer_name]) > 10
            ):

                anomalies = self._detect_anomalies(layer_name)
                self.anomaly_detection[layer_name] = anomalies

        except Exception as e:
            logger.error(f"Error updating history and detecting anomalies: {e}")

    def _detect_anomalies(self, layer_name):
        """Detect statistical anomalies in recent data."""
        try:
            history = list(self.history_buffer[layer_name])
            if len(history) < 10:
                return {}

            # Extract recent values
            recent_means = [h["mean"] for h in history[-10:]]
            recent_stds = [h["std"] for h in history[-10:]]

            # Compute baseline statistics from older data
            if len(history) > 20:
                baseline_means = [h["mean"] for h in history[:-10]]
                baseline_stds = [h["std"] for h in history[:-10]]

                baseline_mean_mean = np.mean(baseline_means)
                baseline_mean_std = np.std(baseline_means)
                baseline_std_mean = np.mean(baseline_stds)
                baseline_std_std = np.std(baseline_stds)

                # Detect anomalies using z-score
                anomalies = {}

                # Check for mean anomalies
                if baseline_mean_std == 0:
                    baseline_mean_std = 1e-8
                recent_mean_z = (
                    abs(np.mean(recent_means) - baseline_mean_mean) / baseline_mean_std
                )
                if recent_mean_z > self.config["anomaly_threshold"]:
                    anomalies["mean_anomaly"] = {
                        "z_score": float(recent_mean_z),
                        "current_mean": float(np.mean(recent_means)),
                        "baseline_mean": float(baseline_mean_mean),
                        "severity": "high" if recent_mean_z > 3 else "medium",
                    }

                # Check for standard deviation anomalies
                if baseline_std_std == 0:
                    baseline_std_std = 1e-8
                recent_std_z = (
                    abs(np.mean(recent_stds) - baseline_std_mean) / baseline_std_std
                )
                if recent_std_z > self.config["anomaly_threshold"]:
                    anomalies["std_anomaly"] = {
                        "z_score": float(recent_std_z),
                        "current_std": float(np.mean(recent_stds)),
                        "baseline_std": float(baseline_std_mean),
                        "severity": "high" if recent_std_z > 3 else "medium",
                    }

                return anomalies

            return {}

        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {}

    def _compute_skewness(self, data):
        """Compute skewness of the data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def _compute_kurtosis(self, data):
        """Compute kurtosis of the data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _compute_entropy(self, data):
        """Compute entropy of the data distribution."""
        # Create histogram
        hist, _ = np.histogram(data, bins=50, density=True)
        hist = hist / np.sum(hist)  # Normalize to probabilities

        # Remove zero probabilities
        hist = hist[hist > 0]

        # Compute entropy
        entropy = -np.sum(hist * np.log(hist))
        return entropy

    def get_statistics_summary(self):
        """Get comprehensive summary of all layer statistics."""
        return {
            "layers_monitored": len(self.layer_stats),
            "total_computations": sum(
                len(stats) for stats in self.computational_stats.values()
            ),
            "anomalies_detected": sum(
                len(anomalies) for anomalies in self.anomaly_detection.values()
            ),
            "memory_usage_total_mb": sum(
                stats.get("memory", {}).get("memory_usage_mb", 0)
                for stats in self.layer_stats.values()
            ),
            "layer_details": {
                layer_name: {
                    "has_anomalies": len(self.anomaly_detection.get(layer_name, {}))
                    > 0,
                    "execution_time_ms": self.computational_stats.get(
                        layer_name, {}
                    ).get("execution_time", 0)
                    * 1000,
                    "memory_usage_mb": stats.get("memory", {}).get(
                        "memory_usage_mb", 0
                    ),
                    "data_range": stats.get("basic", {}).get("range", 0),
                }
                for layer_name, stats in self.layer_stats.items()
            },
        }

    def get_layer_history(self, layer_name, metric="mean"):
        """Get historical data for a specific layer and metric."""
        if layer_name not in self.history_buffer:
            return []

        return [
            {"timestamp": h["timestamp"], "value": h.get(metric, 0)}
            for h in self.history_buffer[layer_name]
        ]

    def reset_data(self):
        """Reset all collected statistics."""
        self.layer_stats.clear()
        self.computational_stats.clear()
        self.anomaly_detection.clear()
        self.history_buffer.clear()
        self.start_time.clear()
        self.memory_usage.clear()


def _get_instance():
    inst = getattr(_MODULE_LOCAL, "instance", None)
    if inst is None:
        inst = LayerStatisticsMonitor()
        _MODULE_LOCAL.instance = inst
    return inst


def initialize(context):
    """Module constructor."""
    monitor = LayerStatisticsMonitor()
    monitor.initialize(context)
    _MODULE_LOCAL.instance = monitor
    return monitor


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
    """LTTSEvent interceptor for layer statistics."""
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
    stats = _MODULE_INSTANCE.layer_stats.get(layer_name, {})
    if stats:
        # Send basic statistics as scalars
        basic = stats.get("basic", {})
        if basic:
            ltts_event.reactor.signal("visualization", {
                "type": "visualization",
                "subtype": "scalar",
                "module": "layer_statistics",
                "event_type": "module_event",
                "module_path": layer_path,
                "layer_path": layer_path,
                "data": {
                    "type": "scalar",
                    "value": basic.get("mean", 0),
                    "label": f"{layer_name} Mean",
                },
            })

        # Send distribution histogram as chart
        dist = stats.get("distribution", {})
        if dist and dist.get("histogram"):
            ltts_event.reactor.signal("visualization", {
                "type": "visualization",
                "subtype": "chart",
                "module": "layer_statistics",
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

        # Send similarity matrix as heatmap
        sim = stats.get("similarity", {})
        if sim and "similarity_matrix" in sim:
            ltts_event.reactor.signal("visualization", {
                "type": "visualization",
                "subtype": "heatmap",
                "module": "layer_statistics",
                "event_type": "module_event",
                "module_path": layer_path,
                "layer_path": layer_path,
                "data": {
                    "type": "heatmap",
                    "values": sim["similarity_matrix"],
                    "dimensions": {
                        "rows": len(sim["similarity_matrix"]),
                        "cols": (
                            len(sim["similarity_matrix"][0])
                            if sim["similarity_matrix"]
                            else 0
                        ),
                    },
                },
            })

        # Send summary table
        mem = stats.get("memory", {})
        ltts_event.reactor.signal("visualization", {
            "type": "visualization",
            "subtype": "table",
            "module": "layer_statistics",
            "event_type": "module_event",
            "module_path": layer_path,
            "layer_path": layer_path,
            "data": {
                "type": "table",
                "headers": ["Metric", "Value"],
                "rows": [
                    ["Data Type", basic.get("dtype", "N/A")],
                    ["Shape", str(basic.get("shape", "N/A"))],
                    ["Mean", f"{basic.get('mean', 0):.4f}"],
                    ["Std Dev", f"{basic.get('std', 0):.4f}"],
                    ["Range", f"{basic.get('range', 0):.4f}"],
                    ["Memory (MB)", f"{mem.get('memory_usage_mb', 0):.2f}"],
                    ["Skewness", f"{dist.get('skewness', 0):.4f}"],
                    ["Kurtosis", f"{dist.get('kurtosis', 0):.4f}"],
                    [
                        "Anomalies",
                        (
                            "Yes"
                            if _MODULE_INSTANCE.anomaly_detection.get(layer_name)
                            else "No"
                        ),
                    ],
                ],
            },
        })

    return result


def get_ui_schema():
    """Get UI configuration schema."""
    return {
        "type": "object",
        "properties": {
            "compute_basic_stats": {
                "type": "boolean",
                "default": True,
                "description": "Compute basic statistical measures (mean, std, etc.)",
            },
            "compute_distribution_stats": {
                "type": "boolean",
                "default": True,
                "description": "Compute distribution statistics (histogram, skewness, kurtosis)",
            },
            "compute_similarity_stats": {
                "type": "boolean",
                "default": True,
                "description": "Compute similarity statistics between tokens/layers",
            },
            "anomaly_detection": {
                "type": "boolean",
                "default": True,
                "description": "Enable statistical anomaly detection",
            },
            "anomaly_threshold": {
                "type": "number",
                "minimum": 1.0,
                "maximum": 5.0,
                "default": 2.0,
                "description": "Z-score threshold for anomaly detection",
            },
            "history_window": {
                "type": "integer",
                "minimum": 10,
                "maximum": 1000,
                "default": 100,
                "description": "Number of recent computations to keep in history",
            },
            "track_memory": {
                "type": "boolean",
                "default": True,
                "description": "Track memory usage statistics",
            },
        },
        "required": [],
    }
