"""
Layer Inspector Module - Comprehensive Layer Information Extraction

This module intercepts layer forward passes to extract infrastructure metadata about:
- Input/Output tensor shapes and dtypes
- Memory usage (CPU/GPU allocation)
- Device placement information
- Gradient flow metadata (requires_grad flags)
- Activation metadata (NaN/Inf detection, element counts)

Scope: infrastructure metadata only. Per-layer statistical analysis (mean,
std, distributions) is reported by the `layer_statistics` module.
"""

# Module metadata required by LTTS
METADATA = {
    "name": "layer_inspector",
    "description": "Comprehensive layer analysis and information extraction module",
    "version": "1.0.0",
    "author": "LTTS Team",
    "event_types": ["layer_after", "layer_before"],
    "methods": {
        "ctor": "initialize",      # Module constructor
        "dtor": "cleanup",        # Module destructor
        "ntor": "interceptor",    # Main event processor - interceptor method
        "utor": "get_ui_schema"   # Returns UI schema for client rendering
    },
    "parameters": {
        "extract_shapes": {
            "type": "boolean",
            "default": True,
            "description": "Extract tensor shape information"
        },
        "extract_dtypes": {
            "type": "boolean",
            "default": True,
            "description": "Extract data type information"
        },
        "extract_memory_usage": {
            "type": "boolean",
            "default": True,
            "description": "Calculate memory usage statistics"
        },
        "extract_activation_metadata": {
            "type": "boolean",
            "default": True,
            "description": "Extract activation metadata (NaN/Inf detection, element counts)"
        },
        "extract_gradients": {
            "type": "boolean",
            "default": False,
            "description": "Extract gradient information (if available)"
        },
        "extract_device_info": {
            "type": "boolean",
            "default": True,
            "description": "Extract device placement information"
        },
        "extract_performance": {
            "type": "boolean",
            "default": True,
            "description": "Measure layer execution time"
        },
        "emit_mode": {
            "type": "select",
            "default": "all",
            "description": "Control when visualization data is emitted",
            "options": ["all", "first", "every_n", "final", "manual"]
        },
        "emit_every_n": {
            "type": "integer",
            "default": 5,
            "description": "Emit data every N passes when emit_mode=every_n"
        }
    }
}

import torch
import time
import logging
from pathlib import Path
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from collections import defaultdict

try:
    from scripts.core.emission_controller import (  # type: ignore
        should_emit as emission_should_emit,
        mark_finalized as emission_mark_finalized,
        update_expected_passes as emission_update_expected,
        reset_entry as emission_reset_entry,
    )
    from scripts.core.ltts_event import LTTSReactor  # type: ignore
except ImportError:
    core_path = Path(__file__).resolve().parents[2] / 'scripts' / 'core'
    if str(core_path) not in sys.path:
        sys.path.append(str(core_path))
    from emission_controller import (  # type: ignore
        should_emit as emission_should_emit,
        mark_finalized as emission_mark_finalized,
        update_expected_passes as emission_update_expected,
        reset_entry as emission_reset_entry,
    )
    from ltts_event import LTTSReactor  # type: ignore

logger = logging.getLogger(__name__)


def append_debug_log(message: str):
    # Route diagnostics through the configured server logger.
    logger.debug("layer_inspector: %s", message)


# Runtime configuration and state shared across interceptor calls
RUNTIME_LAYER_CONFIG: Dict[str, Dict[str, Any]] = defaultdict(dict)
PENDING_LAYER_EMITS: set[str] = set()
LAST_LAYER_CONTEXT: Dict[str, Any] = {}
LAST_LAYER_DATA: Dict[str, Any] = {}

# Singleton instance cache so state persists between calls
_MODULE_SINGLETON: Optional["LayerInspectorModule"] = None

class LayerInspectorModule:
    """
    Extracts layer operations, tensor shapes, precision, and compute statistics.
    """

    def __init__(self):
        self.name = "layer_inspector"
        self.version = "1.0.0"
        self.description = "Layer metadata and runtime statistics"
        self.enabled = True

        # Store context for later use
        self.context = None

        # Configuration
        self.config = {
            "extract_shapes": True,
            "extract_dtypes": True,
            "extract_memory_usage": True,
            "extract_activation_metadata": True,
            "extract_gradient_info": True,
            "extract_device_info": True,
            "compute_flops": False,  # Expensive, disabled by default
            "emit_mode": "all",
            "emit_every_n": 5,
            "expected_generation_passes": None
        }

        # Storage for extracted information
        self.layer_info = defaultdict(dict)
        self.shape_history = defaultdict(list)
        self.memory_tracker = {}

        # Runtime tracking
        self.runtime_layer_config = RUNTIME_LAYER_CONFIG
        self.pending_layers = PENDING_LAYER_EMITS
        self.last_layer_context = LAST_LAYER_CONTEXT
        self.last_layer_data = LAST_LAYER_DATA
        self.finalized_layers: set[str] = set()
        self.layer_aliases: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Emission helpers

    def _emit_signal(
        self,
        signal_type: str,
        payload: Dict[str, Any],
        ltts_event=None,
        context=None,
    ) -> bool:
        """Emit data to the frontend using LTTS reactors/message bus."""

        if ltts_event is not None and hasattr(ltts_event, "reactor"):
            ltts_event.reactor.signal(signal_type, payload)
            return True

        ctx = context or getattr(self, "context", None)
        message_bus = getattr(ctx, "message_bus", None) if ctx else None
        if not message_bus:
            logger.warning("LayerInspector: no message bus available for signal %s", signal_type)
            return False

        reactor = LTTSReactor(message_bus=message_bus)
        reactor.signal(signal_type, payload)
        return True

    def _emit_visualization(self, payload: Dict[str, Any], ltts_event=None, context=None) -> bool:
        return self._emit_signal("visualization", payload, ltts_event=ltts_event, context=context)

    def _emit_error(self, payload: Dict[str, Any], ltts_event=None, context=None) -> bool:
        return self._emit_signal("error", payload, ltts_event=ltts_event, context=context)

    def _emit_layer_inference_event(self, layer_name: str, layer_idx: int, context=None) -> bool:
        """
        Emit a layer inference event to notify the frontend about the current layer being processed.

        Args:
            layer_name: The name of the layer being processed
            layer_idx: The index of the layer in the model
            context: The context object

        Returns:
            bool: True if the event was emitted successfully, False otherwise
        """
        payload = {
            "layer_name": layer_name,
            "layer_index": layer_idx,
            "timestamp": time.time()
        }
        return self._emit_signal("layer_inference", payload, None, context)

    def get_config_schema(self):
        """Return configuration schema for the module."""
        return {
            "extract_shapes": {
                "type": "boolean",
                "default": True,
                "description": "Extract input/output tensor shapes"
            },
            "extract_dtypes": {
                "type": "boolean",
                "default": True,
                "description": "Extract tensor data types and precision"
            },
            "extract_memory_usage": {
                "type": "boolean",
                "default": True,
                "description": "Monitor memory usage per layer"
            },
            "extract_activation_metadata": {
                "type": "boolean",
                "default": True,
                "description": "Extract activation metadata (NaN/Inf detection, element counts)"
            },
            "extract_gradient_info": {
                "type": "boolean",
                "default": True,
                "description": "Extract gradient flow information"
            },
            "extract_device_info": {
                "type": "boolean",
                "default": True,
                "description": "Extract device placement information"
            },
            "compute_flops": {
                "type": "boolean",
                "default": False,
                "description": "Compute FLOPs (computationally expensive)"
            },
            "max_samples_for_stats": {
                "type": "integer",
                "default": 1000,
                "description": "Maximum tensor elements to sample for statistics"
            }
        }

    def update_config(self, new_config: Dict[str, Any]):
        """Update module configuration."""
        normalized = self._normalize_config(new_config)
        self.config.update(normalized)
        logger.info(f"Layer Inspector config updated: {new_config}")

    def _resolve_layer_config(self, layer_name: str) -> Dict[str, Any]:
        """Merge default config with layer-specific overrides."""
        resolved = dict(self.config)
        overrides = self.runtime_layer_config.get(layer_name)
        if not overrides:
            # Try suffix-based matching to handle auto-mapped layer paths
            for candidate, config in self.runtime_layer_config.items():
                if candidate == layer_name:
                    overrides = config
                    break
                if layer_name.endswith(candidate) or candidate.endswith(layer_name):
                    overrides = config
                    break
        if overrides:
            resolved.update(self._normalize_config(overrides))
        return resolved

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize config keys for backward compatibility."""
        if not isinstance(config, dict):
            return {}
        normalized = dict(config)
        if "extract_gradients" in normalized and "extract_gradient_info" not in normalized:
            normalized["extract_gradient_info"] = normalized["extract_gradients"]
        return normalized

    def _should_emit(self, layer_name: str, config: Dict[str, Any]) -> bool:
        """Decide whether to emit visualization for current pass."""
        emit_params = {
            'emit_mode': config.get('emit_mode', 'all'),
            'emit_every_n': config.get('emit_every_n'),
            'expected_generation_passes': config.get('expected_generation_passes')
        }

        info = emission_should_emit(self.name, layer_name, emit_params)
        self.layer_info[layer_name]['forward_passes'] = info['forward_pass']

        mode = info['mode']
        if info['emit']:
            if mode == 'final':
                self.finalized_layers.add(layer_name)
                self.pending_layers.discard(layer_name)
            else:
                self.pending_layers.discard(layer_name)
            return True

        # Not emitting this pass
        if mode in ('final', 'manual') or info.get('deferred'):
            self.pending_layers.add(layer_name)
        else:
            self.pending_layers.discard(layer_name)
        return False

    def set_layer_config(self, layer_name: str, config: Dict[str, Any]):
        """Apply runtime overrides for a layer."""
        if not layer_name:
            return
        # Allow fresh emissions when runtime config is updated
        self.finalized_layers.discard(layer_name)
        overrides = self.runtime_layer_config.get(layer_name, {})
        overrides.update(config)
        self.runtime_layer_config[layer_name] = overrides
        alias_path = overrides.get('layer_path') or config.get('layer_path')
        if alias_path:
            self.layer_aliases[layer_name] = alias_path
        if layer_name in self.layer_info:
            self.layer_info[layer_name]['forward_passes'] = 0
        # Ensure we start counting for this run
        self.layer_info.setdefault(layer_name, {'forward_passes': 0, 'layer_name': layer_name})
        emission_update_expected(self.name, layer_name, overrides.get('expected_generation_passes'))
        emission_reset_entry(self.name, layer_name)
        append_debug_log(f"Runtime config set layer={layer_name} config={config}")

    def clear_layer_config(self, layer_name: str):
        """Drop runtime overrides and cached data."""
        self.runtime_layer_config.pop(layer_name, None)
        self.pending_layers.discard(layer_name)
        self.last_layer_context.pop(layer_name, None)
        self.last_layer_data.pop(layer_name, None)
        self.finalized_layers.discard(layer_name)
        self.layer_aliases.pop(layer_name, None)
        emission_update_expected(self.name, layer_name, None)
        emission_reset_entry(self.name, layer_name)
        append_debug_log(f"Runtime config cleared layer={layer_name}")

    def finalize_layer(self, layer_name: str):
        """Emit cached visualization for deferred layers."""
        if layer_name not in self.pending_layers:
            return
        if layer_name in self.finalized_layers:
            self.pending_layers.discard(layer_name)
            return
        cached = self.last_layer_data.get(layer_name)
        context = self.last_layer_context.get(layer_name)
        if not cached or not context:
            self.pending_layers.discard(layer_name)
            return
        cached_layer_data = cached.get('layer_data', {}) if isinstance(cached, dict) else {}
        display_layer_path = self.layer_aliases.get(layer_name, layer_name)
        payload = {
            "layer_name": layer_name,
            "layer_type": cached.get('layer_type', 'unknown'),
            "layer_data": cached_layer_data,
            "timestamp": time.time(),
            "layer_path": display_layer_path,
            "module_path": display_layer_path,
            "actual_layer_path": layer_name,
            "resolved_layer_path": layer_name,
            "emit_mode": 'final',
            "forward_pass": cached_layer_data.get('forward_passes'),
            "emit_stride": None,
            "deferred": True
        }
        success = self._emit_visualization({
            "type": "layer_inspector",
            "subtype": "layer_data",
            "module": self.name,
            "data": payload,
            "module_path": display_layer_path,
            "layer_path": display_layer_path,
            "actual_layer_path": layer_name,
            "resolved_layer_path": layer_name,
            "emit_mode": 'final',
            "forward_pass": cached_layer_data.get('forward_passes'),
            "deferred": True,
            "timestamp": time.time()
        }, context=context)
        if success:
            self.finalized_layers.add(layer_name)
            emission_mark_finalized(self.name, layer_name)
            append_debug_log(f"Deferred emission finalized layer={layer_name}")
        self.pending_layers.discard(layer_name)

    def intercept_layer(self, layer_output, layer_info, context, target_layer_path=None):
        """
        Intercept layer output to extract comprehensive information.

        Args:
            layer_output: The output from the layer
            layer_info: Information about the layer
            context: Additional context information

        Returns:
            layer_output: Unchanged layer output
        """
        if not self.enabled:
            return layer_output

        try:
            # Use context module_path if available, fallback to layer_info
            layer_name = context.module_path if hasattr(context, 'module_path') and context.module_path else layer_info.get('module_path', layer_info.get('layer_name', 'unknown'))
            layer_idx = layer_info.get('layer_index', 0)

            if target_layer_path:
                self.layer_aliases[layer_name] = target_layer_path
            display_layer_path = self.layer_aliases.get(layer_name, layer_name)

            logger.debug(f"intercept_layer called for: {layer_name}")
            logger.debug(
                "LayerInspector: Layer paths resolved actual=%s target=%s",
                layer_name,
                display_layer_path,
            )

            # Emit a layer inference event to notify the frontend about the current layer being processed
            self._emit_layer_inference_event(layer_name, layer_idx, context)

            resolved_config = self._resolve_layer_config(layer_name)
            logger.debug(f"Resolved config for {layer_name}: {resolved_config}")


            # Initialize layer info if not exists
            if layer_name not in self.layer_info:
                self.layer_info[layer_name] = {
                    'layer_index': layer_idx,
                    'layer_name': layer_name,
                    'forward_passes': 0
                }

            # Extract input information from context if available
            layer_input = getattr(context, 'inputs', None) if context else None

            # Extract comprehensive information
            info_update = {}

            if resolved_config.get("extract_shapes", True):
                shapes_info = self._extract_shapes(layer_input, layer_output)
                info_update.update(shapes_info)

            if resolved_config.get("extract_dtypes", True):
                dtype_info = self._extract_dtypes(layer_input, layer_output)
                info_update.update(dtype_info)

            if resolved_config.get("extract_memory_usage", True):
                memory_info = self._extract_memory_usage(layer_input, layer_output, layer_name)
                info_update.update(memory_info)

            if resolved_config.get("extract_activation_metadata", True):
                metadata_info = self._extract_activation_stats(layer_output)
                info_update.update(metadata_info)

            grad_enabled = resolved_config.get(
                "extract_gradient_info",
                resolved_config.get("extract_gradients", True),
            )
            if grad_enabled:
                grad_info = self._extract_gradient_info(layer_input, layer_output)
                info_update.update(grad_info)

            if resolved_config.get("extract_device_info", True):
                device_info = self._extract_device_info(layer_output)
                info_update.update(device_info)

            # Update layer information
            self.layer_info[layer_name].update(info_update)

            # Cache context and data for deferred emission
            self.last_layer_context[layer_name] = context
            self.last_layer_data[layer_name] = {"layer_type": layer_info.get('layer_type', "unknown"), "layer_data": self.layer_info[layer_name]}
            logger.debug(f"Updated layer_info for {layer_name}")

            # Store in shape history for trend analysis
            if 'input_shape' in info_update:
                self.shape_history[layer_name].append({
                    'timestamp': time.time(),
                    'input_shape': info_update['input_shape'],
                    'output_shape': info_update.get('output_shape')
                })

            # CRITICAL: Update the layer_info in the context so it gets passed to the model structure
            if context and hasattr(context, 'model_state') and context.model_state:
                if 'layer_inspector_data' not in context.model_state:
                    context.model_state['layer_inspector_data'] = {}
                context.model_state['layer_inspector_data'][layer_name] = self.layer_info[layer_name]

            # Note: Visualization data will be sent via reactor.signal() in the interceptor function
            # This method focuses on data extraction and storage

        except Exception as e:
            logger.error(f"Error in layer inspector for {layer_name}: {e}")

        return layer_output

    def _extract_shapes(self, layer_input, layer_output) -> Dict[str, Any]:
        """Extract tensor shape information."""
        shapes_info = {}

        try:
            # Extract input shapes
            if layer_input is not None:
                if isinstance(layer_input, torch.Tensor):
                    shapes_info['input_shape'] = list(layer_input.shape)
                elif isinstance(layer_input, (tuple, list)):
                    shapes_info['input_shapes'] = [list(t.shape) if isinstance(t, torch.Tensor) else str(type(t)) for t in layer_input]
                else:
                    shapes_info['input_type'] = str(type(layer_input))

            # Extract output shapes
            if layer_output is not None:
                if isinstance(layer_output, torch.Tensor):
                    shapes_info['output_shape'] = list(layer_output.shape)
                elif isinstance(layer_output, (tuple, list)):
                    shapes_info['output_shapes'] = [list(t.shape) if isinstance(t, torch.Tensor) else str(type(t)) for t in layer_output]
                else:
                    shapes_info['output_type'] = str(type(layer_output))

        except Exception as e:
            logger.error(f"Error extracting shapes: {e}")
            shapes_info['shape_extraction_error'] = str(e)

        return shapes_info

    def _extract_dtypes(self, layer_input, layer_output) -> Dict[str, Any]:
        """Extract tensor data types and precision information."""
        dtype_info = {}

        try:
            # Input dtype
            if layer_input is not None:
                input_dtype = self._get_tensor_dtype(layer_input)
                if input_dtype:
                    dtype_info['input_dtype'] = input_dtype
                    dtype_info['input_precision'] = self._get_precision_info(input_dtype)

            # Output dtype
            if layer_output is not None:
                output_dtype = self._get_tensor_dtype(layer_output)
                if output_dtype:
                    dtype_info['output_dtype'] = output_dtype
                    dtype_info['output_precision'] = self._get_precision_info(output_dtype)

                    # Check for dtype changes
                    if 'input_dtype' in dtype_info and input_dtype != output_dtype:
                        dtype_info['dtype_changed'] = True
                        dtype_info['dtype_change'] = f"{input_dtype} → {output_dtype}"

        except Exception as e:
            logger.warning(f"Error extracting dtypes: {e}")

        return dtype_info

    def _extract_memory_usage(self, layer_input, layer_output, layer_name: str) -> Dict[str, Any]:
        """Extract memory usage information."""
        memory_info = {}

        try:
            # Calculate tensor memory usage
            if layer_input is not None:
                input_memory = self._calculate_tensor_memory(layer_input)
                memory_info['input_memory_mb'] = input_memory

            if layer_output is not None:
                output_memory = self._calculate_tensor_memory(layer_output)
                memory_info['output_memory_mb'] = output_memory

            # GPU memory if available
            if torch.cuda.is_available():
                memory_info['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
                memory_info['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024

        except Exception as e:
            logger.warning(f"Error extracting memory usage: {e}")

        return memory_info

    def _extract_activation_stats(self, layer_output) -> Dict[str, Any]:
        """
        Extract basic activation metadata (not statistics).

        Note: Detailed statistical analysis (mean/std/min/max/distribution) is handled
        by the layer_statistics module to avoid duplication.
        """
        stats_info = {}

        try:
            tensor = self._extract_primary_tensor(layer_output)
            if tensor is not None:
                if tensor.numel() > 0:
                    with torch.no_grad():
                        flat_tensor = tensor.flatten()

                        # Only extract metadata, not statistical measures
                        stats_info['activation_metadata'] = {
                            'total_elements': int(tensor.numel()),
                            'has_nan': bool(torch.isnan(flat_tensor).any().item()),
                            'has_inf': bool(torch.isinf(flat_tensor).any().item()),
                            'all_zeros': bool((flat_tensor == 0).all().item()),
                            'all_positive': bool((flat_tensor >= 0).all().item()),
                            'all_negative': bool((flat_tensor <= 0).all().item()),
                        }

        except Exception as e:
            logger.warning(f"Error extracting activation metadata: {e}")

        return stats_info

    def _extract_gradient_info(self, layer_input, layer_output) -> Dict[str, Any]:
        """Extract gradient flow information."""
        grad_info = {}

        try:
            # Check input gradient requirements
            if layer_input is not None:
                input_requires_grad = self._check_requires_grad(layer_input)
                grad_info['input_requires_grad'] = input_requires_grad

            # Check output gradient requirements
            if layer_output is not None:
                output_requires_grad = self._check_requires_grad(layer_output)
                grad_info['output_requires_grad'] = output_requires_grad

                # Check if gradients are actually flowing
                grad_info['gradient_enabled'] = torch.is_grad_enabled()

        except Exception as e:
            logger.warning(f"Error extracting gradient info: {e}")

        return grad_info

    def _extract_device_info(self, layer_output) -> Dict[str, Any]:
        """Extract device placement information."""
        device_info = {}

        try:
            tensor = self._extract_primary_tensor(layer_output)
            if tensor is not None:
                device_info['device'] = str(tensor.device)
                device_info['device_type'] = tensor.device.type
                if tensor.device.type == 'cuda':
                    device_info['gpu_index'] = tensor.device.index

        except Exception as e:
            logger.warning(f"Error extracting device info: {e}")

        return device_info

    # Helper methods
    def _extract_primary_tensor(self, layer_output):
        """Extract the primary tensor from various output formats."""
        if isinstance(layer_output, torch.Tensor):
            return layer_output
        elif hasattr(layer_output, 'last_hidden_state'):
            return layer_output.last_hidden_state
        elif hasattr(layer_output, 'hidden_states') and layer_output.hidden_states:
            return layer_output.hidden_states[-1] if isinstance(layer_output.hidden_states, (list, tuple)) else layer_output.hidden_states
        elif isinstance(layer_output, (list, tuple)) and len(layer_output) > 0:
            return layer_output[0] if isinstance(layer_output[0], torch.Tensor) else None
        return None

    def _get_tensor_shape(self, tensor_data):
        """Get tensor shape from various formats."""
        tensor = self._extract_primary_tensor(tensor_data)
        return list(tensor.shape) if tensor is not None else None

    def _get_tensor_numel(self, tensor_data):
        """Get number of elements in tensor."""
        tensor = self._extract_primary_tensor(tensor_data)
        return tensor.numel() if tensor is not None else 0

    def _get_tensor_dtype(self, tensor_data):
        """Get tensor data type."""
        tensor = self._extract_primary_tensor(tensor_data)
        return str(tensor.dtype) if tensor is not None else None

    def _get_precision_info(self, dtype_str: str) -> Dict[str, Any]:
        """Get precision information from dtype string."""
        precision_map = {
            'torch.float32': {'bits': 32, 'type': 'float', 'precision': 'single'},
            'torch.float16': {'bits': 16, 'type': 'float', 'precision': 'half'},
            'torch.bfloat16': {'bits': 16, 'type': 'bfloat', 'precision': 'brain_float'},
            'torch.float64': {'bits': 64, 'type': 'float', 'precision': 'double'},
            'torch.int8': {'bits': 8, 'type': 'int', 'precision': 'int8'},
            'torch.int16': {'bits': 16, 'type': 'int', 'precision': 'int16'},
            'torch.int32': {'bits': 32, 'type': 'int', 'precision': 'int32'},
            'torch.int64': {'bits': 64, 'type': 'int', 'precision': 'int64'},
        }
        return precision_map.get(dtype_str, {'bits': 'unknown', 'type': 'unknown', 'precision': 'unknown'})

    def _calculate_tensor_memory(self, tensor_data) -> float:
        """Calculate memory usage in MB."""
        tensor = self._extract_primary_tensor(tensor_data)
        if tensor is not None:
            return tensor.numel() * tensor.element_size() / 1024 / 1024
        return 0.0

    def _check_requires_grad(self, tensor_data) -> bool:
        """Check if tensor requires gradients."""
        tensor = self._extract_primary_tensor(tensor_data)
        return tensor.requires_grad if tensor is not None else False

    def _analyze_shape_change(self, input_shape: List[int], output_shape: List[int]) -> Dict[str, Any]:
        """Analyze how the tensor shape changed through the layer."""
        return {
            'dimensions_changed': len(input_shape) != len(output_shape),
            'size_changed': input_shape != output_shape,
            'total_elements_ratio': np.prod(output_shape) / np.prod(input_shape) if np.prod(input_shape) > 0 else 0
        }

    def get_layer_info(self, layer_name: str = None) -> Dict[str, Any]:
        """Get extracted information for a specific layer or all layers."""
        if layer_name:
            return self.layer_info.get(layer_name, {})
        return dict(self.layer_info)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all extracted information."""
        return {
            'total_layers_analyzed': len(self.layer_info),
            'layers': list(self.layer_info.keys()),
            'config': self.config,
            'memory_tracking_enabled': self.config.get("extract_memory_usage", True)
        }

    def clear_history(self):
        """Clear stored information and history."""
        self.layer_info.clear()
        self.shape_history.clear()
        self.memory_tracker.clear()
        logger.info("Layer inspector history cleared")

# Module initialization
def get_ui_schema():
    """
    Returns UI schema for client rendering.
    This is the UI schema provider (utor) method referenced in METADATA.

    Returns:
        A dictionary containing the UI schema for the module
    """
    try:
        logger.debug("Generating UI schema for layer inspector module")

        # Create a schema that matches the module's configuration options
        schema = {
            "parameters": [
                {
                    "name": "extract_shapes",
                    "type": "boolean",
                    "label": "Extract Shapes",
                    "default": True,
                    "help": "Extract tensor shape information"
                },
                {
                    "name": "extract_dtypes",
                    "type": "boolean",
                    "label": "Extract Data Types",
                    "default": True,
                    "help": "Extract data type information"
                },
                {
                    "name": "extract_memory_usage",
                    "type": "boolean",
                    "label": "Extract Memory Usage",
                    "default": True,
                    "help": "Calculate memory usage statistics"
                },
                {
                    "name": "extract_activation_metadata",
                    "type": "boolean",
                    "label": "Extract Activation Metadata",
                    "default": True,
                    "help": "Extract activation metadata (NaN/Inf detection, element counts)"
                },
                {
                    "name": "extract_gradients",
                    "type": "boolean",
                    "label": "Extract Gradients",
                    "default": False,
                    "help": "Extract gradient information (if available)"
                },
                {
                    "name": "extract_device_info",
                    "type": "boolean",
                    "label": "Extract Device Info",
                    "default": True,
                    "help": "Extract device placement information"
                },
                {
                    "name": "extract_performance",
                    "type": "boolean",
                    "label": "Extract Performance",
                    "default": True,
                    "help": "Measure layer execution time"
                },
                {
                    "name": "emit_mode",
                    "type": "select",
                    "label": "Emission Strategy",
                    "default": "all",
                    "options": ["all", "first", "every_n", "final", "manual"],
                    "help": "Control when the inspector sends updates"
                },
                {
                    "name": "emit_every_n",
                    "type": "number",
                    "label": "Emit Every N Passes",
                    "default": 5,
                    "min": 1,
                    "help": "Used when emission strategy is every_n"
                }
            ],
            "visualizations": [
                {
                    "name": "layer_data",
                    "type": "custom",
                    "label": "Layer Information",
                    "description": "Detailed information about layer operations"
                }
            ]
        }

        return schema
    except Exception as e:
        logger.error(f"Error generating UI schema: {e}")
        # Return a minimal schema in case of error
        return {
            "parameters": [
                {
                    "name": "extract_shapes",
                    "type": "boolean",
                    "label": "Extract Shapes",
                    "default": True
                }
            ]
        }

def interceptor(ltts_event):
    """
    Main interceptor function for layer inspector module.
    This is the event processor (ntor) method referenced in METADATA.

    Args:
        ltts_event: The event object provided by the LTTS system

    Returns:
        The processed layer output or None
    """
    logger.debug("Layer inspector interceptor starting")
    target_layer_path = None

    try:
        # Get the module instance - use the initialized module from the context
        module = create_module()

        # Get layer information from context
        layer_info = getattr(ltts_event.context, 'layer_info', {})

        # Debug context information
        logger.debug(f"Context module_path: '{ltts_event.context.module_path}'")
        logger.debug(f"Context layer_info: {layer_info}")
        logger.debug(f"Available context attributes: {[attr for attr in dir(ltts_event.context) if not attr.startswith('_')]}")

        # Get layer output from context - check both 'outputs' and 'layer_output'
        layer_output = getattr(ltts_event.context, 'outputs', None)
        if layer_output is None:
            layer_output = getattr(ltts_event.context, 'layer_output', None)

        if layer_output is not None:
            logger.debug(f"Processing layer output for {layer_info.get('module_path', 'unknown')}")

            # Process the layer data
            if hasattr(ltts_event, 'environment'):
                target_layer_path = ltts_event.environment.get('requested_layer_path')
            result = module.intercept_layer(layer_output, layer_info, ltts_event.context, target_layer_path)

            # Get layer name for data retrieval - use context module_path if available
            layer_name = ltts_event.context.module_path or layer_info.get('module_path', layer_info.get('layer_name', 'unknown'))

            # Check if we have collected data for this layer
            if layer_name in module.layer_info:
                resolved_config = module._resolve_layer_config(layer_name)
                if module._should_emit(layer_name, resolved_config):
                    layer_data = module.layer_info[layer_name]

                    current_pass = layer_data.get('forward_passes')
                    emit_mode = resolved_config.get('emit_mode', 'all')

                    visualization_data = {
                        "layer_name": layer_name,
                        "layer_type": layer_info.get('layer_type', 'unknown'),
                        "layer_data": layer_data,
                        "timestamp": time.time(),
                        "layer_path": layer_name,
                        "module_path": layer_name,
                        "emit_mode": emit_mode,
                        "forward_pass": current_pass,
                        "emit_stride": resolved_config.get('emit_every_n') if emit_mode == 'every_n' else None,
                        "deferred": False
                    }

                    logger.debug(f"Sending layer_inspector visualization data for {layer_name}")

                    success = module._emit_visualization({
                        "type": "layer_inspector",
                        "subtype": "layer_data",
                        "module": module.name,
                        "data": visualization_data,
                        "module_path": layer_name,
                        "layer_path": layer_name,
                        "emit_mode": emit_mode,
                        "forward_pass": current_pass,
                        "emit_stride": resolved_config.get('emit_every_n') if emit_mode == 'every_n' else None,
                        "deferred": False,
                        "timestamp": time.time()
                    }, ltts_event=ltts_event)

                    if success:
                        logger.debug(f"Successfully sent layer inspector data for {layer_name}")
                        if resolved_config.get('emit_mode') == 'final':
                            module.finalized_layers.add(layer_name)
                    else:
                        logger.warning(f"Failed to send layer inspector data for {layer_name}")
                else:
                    logger.debug(f"Deferred emission for {layer_name} (mode={resolved_config.get('emit_mode')})")
            else:
                logger.debug(f"No layer data found for '{layer_name}'")
            return result
        else:
            logger.debug("No layer output found in context")

    except Exception as e:
        logger.error(f"ERROR in layer inspector interceptor: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")

        actual_layer = getattr(ltts_event.context, 'module_path', 'unknown')
        error_data = {
            "module": "layer_inspector",
            "error": str(e),
            "layer_path": target_layer_path or actual_layer,
            "actual_layer_path": actual_layer
        }
        module._emit_error(error_data, ltts_event=ltts_event)

    # Return None if we couldn't process the event
    return None

def create_module():
    """Factory function to create (or reuse) the module instance."""
    global _MODULE_SINGLETON
    if _MODULE_SINGLETON is None:
        _MODULE_SINGLETON = LayerInspectorModule()
    return _MODULE_SINGLETON

def initialize(context):
    """
    Initialize the layer inspector module.
    This is the constructor (ctor) method referenced in METADATA.

    Args:
        context: The context object provided by the LTTS system

    Returns:
        The initialized module instance
    """
    try:
        logger.info("Initializing Layer Inspector module")
        module = create_module()

        # Store the context for later use in the interceptor
        module.context = context

        # Extract configuration from context if available
        if hasattr(context, 'config') and context.config:
            module.update_config(context.config)
            logger.info(f"Applied configuration from context: {context.config}")

        # Initialize any resources needed
        logger.info("Layer Inspector module successfully initialized")
        return module
    except Exception as e:
        logger.error(f"Error initializing Layer Inspector module: {e}")
        import traceback
        logger.debug(f"Initialization error traceback: {traceback.format_exc()}")
        # Return module even if there was an error to prevent system crashes
        return create_module()

def set_runtime_config(layer_name: str, config: Dict[str, Any]):
    module = create_module()
    module.set_layer_config(layer_name, config)


def clear_runtime_config(layer_name: str):
    module = create_module()
    module.clear_layer_config(layer_name)


def finalize_layer(layer_name: str):
    module = create_module()
    module.finalize_layer(layer_name)

def cleanup(module):
    """
    Module destructor (dtor) method referenced in METADATA.
    Cleans up resources and performs necessary teardown operations.

    Args:
        module: The module instance to clean up
    """
    try:
        logger.info("Cleaning up Layer Inspector module")

        # Clear any cached data
        if hasattr(module, 'layer_info'):
            module.layer_info.clear()

        if hasattr(module, 'shape_history'):
            module.shape_history.clear()

        if hasattr(module, 'memory_tracker'):
            module.memory_tracker.clear()

        # Clear runtime state
        if hasattr(module, 'runtime_layer_config'):
            module.runtime_layer_config.clear()

        if hasattr(module, 'pending_layers'):
            module.pending_layers.clear()

        if hasattr(module, 'last_layer_context'):
            module.last_layer_context.clear()

        if hasattr(module, 'last_layer_data'):
            module.last_layer_data.clear()

        if hasattr(module, 'layer_info'):
            for layer_name in list(module.layer_info.keys()):
                emission_reset_entry(module.name, layer_name)

        if hasattr(module, 'finalized_layers'):
            for layer_name in list(module.finalized_layers):
                emission_reset_entry(module.name, layer_name)
            module.finalized_layers.clear()

        if hasattr(module, 'layer_aliases'):
            module.layer_aliases.clear()

        # Reset the singleton if needed
        global _MODULE_SINGLETON
        _MODULE_SINGLETON = None

        logger.info("Layer Inspector module cleanup completed")
    except Exception as e:
        logger.error(f"Error during Layer Inspector module cleanup: {e}")
        import traceback
        logger.debug(f"Cleanup error traceback: {traceback.format_exc()}")
