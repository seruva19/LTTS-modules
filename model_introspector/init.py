"""
Model Introspector Module

Extracts and provides model state information including layer structure,
batch info, device info, and parameter counts. Operates at the model level
to provide comprehensive insights into model architecture and state.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional
import logging

from scripts.core.module_data_sender import get_data_sender

logger = logging.getLogger(__name__)
_MODULE_INSTANCE = None

METADATA = {
    "name": "Model Introspector",
    "version": "1.0.0",
    "description": "Extracts comprehensive model state and architecture information",
    "author": "LTTS Team",
    "event_types": ["model_after"],
    "methods": {
        "ctor": "initialize",
        "dtor": "cleanup",
        "ntor": "on_model_after",
        "utor": "get_ui_schema",
    },
    "dependencies": ["torch"],
    "tags": ["introspection", "model-analysis", "architecture", "metadata"],
}


class ModelIntrospector:
    """Extracts comprehensive model state information."""

    def __init__(self):
        self.model = None
        self.layer_wrappers = {}
        self.model_state = {}
        self.batch_info = {}
        self.enabled = True
        self.data_sender = get_data_sender("model_introspector")

    def initialize(self, context):
        """
        Initialize the model introspector.

        Args:
            context: Module context containing model and layer wrapper info
        """
        logger.info("Model Introspector initialized")
        self.data_sender.set_context(context)

        # Get model and layer wrappers from context if available
        if hasattr(context, "model"):
            self.model = context.model
        if hasattr(context, "layer_wrappers"):
            self.layer_wrappers = context.layer_wrappers

        return True

    def cleanup(self):
        """Clean up resources."""
        self.model_state.clear()
        self.batch_info.clear()
        logger.info("Model Introspector cleaned up")

    def on_model_after(self, context):
        """
        Called after model forward pass.

        Args:
            context: Execution context containing model info
        """
        # Extract model and layer wrappers from context on first call
        if not self.model and hasattr(context, "ltts_model"):
            self.model = context.ltts_model.original_model
            self.layer_wrappers = context.ltts_model.layer_wrappers

            # Extract and send model state on first forward
            self.model_state = self.get_model_state()
            self._send_model_info()
            logger.info(
                "Model introspector initialized: %s",
                self.model_state.get("model_class", "Unknown"),
            )

        # Extract and send batch info
        if hasattr(context, "inputs"):
            self.batch_info = self.extract_batch_info(context.inputs)
            if self.batch_info:
                self._send_batch_info()

    def get_model_state(self) -> Dict[str, Any]:
        """
        Extract basic model state information.

        Returns:
            Dictionary containing model state information
        """
        if not self.model:
            return {"error": "No model loaded"}

        try:
            return {
                "model_class": type(self.model).__name__,
                "device": str(next(self.model.parameters()).device),
                "dtype": str(next(self.model.parameters()).dtype),
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                ),
                "layer_structure": self.get_unified_layer_structure(),
                "training": self.model.training,
            }
        except Exception as e:
            logger.warning("Error extracting model state: %s", e)
            return {"error": str(e)}

    def get_unified_layer_structure(self) -> Dict[str, Any]:
        """
        Get comprehensive layer structure using unified traverse_model method.

        Returns:
            Dictionary mapping layer names to layer information
        """
        if not self.model:
            return {}

        from scripts.models.model_struct import traverse_model

        try:
            logger.info("INTROSPECTOR: Starting get_unified_layer_structure")
            # Use unified traverse_model with maximum traversal
            model_struct = traverse_model(self.model, max_traversal=True)
            logger.info(
                "INTROSPECTOR: Generated model_struct with %s layers",
                len(model_struct.layers),
            )

            # Convert to dict format expected by existing code
            structure = {}
            for child in model_struct.layers:
                layer_data = {
                    "class": child.type,
                    "wrapped": child.name in self.layer_wrappers,
                    "category": getattr(child, "blockType", "other"),
                    "parameters": getattr(child, "parameters", 0),
                    "is_transformer_layer": getattr(child, "blockType", "")
                    == "transformer",
                }

                # Add hierarchy information if available
                if hasattr(child, "hierarchy") and child.hierarchy:
                    layer_data["hierarchy"] = child.hierarchy

                structure[child.name] = layer_data

            logger.info(
                "INTROSPECTOR: Completed structure generation with %s entries",
                len(structure),
            )
            return structure
        except Exception as e:
            logger.warning("INTROSPECTOR: ERROR in structure generation: %s", e)
            # Fallback to basic structure
            return {
                name: {
                    "class": type(module).__name__,
                    "wrapped": name in self.layer_wrappers,
                }
                for name, module in self.model.named_modules()
                if name
            }

    def get_layer_structure_for_blueprint(self) -> Dict[str, Any]:
        """
        Get comprehensive layer structure for blueprint serialization.

        Returns:
            Dictionary with detailed layer structure for blueprints
        """
        if not self.model:
            return {
                "error": "No model loaded",
                "model_class": "Unknown",
                "layers": [],
                "total_parameters": 0,
                "trainable_parameters": 0,
            }

        from scripts.models.model_struct import traverse_model

        try:
            # Use unified traverse_model with maximum traversal
            model_struct = traverse_model(self.model, max_traversal=True)

            # Convert to detailed format expected by blueprint serialization
            structure = {
                "model_class": type(self.model).__name__,
                "layers": [],
                "total_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                ),
            }

            # Convert ModelStruct layers to detailed format
            for child in model_struct.layers:
                layer_info = {
                    "name": child.name,
                    "type": child.type,
                    "parameters": getattr(child, "parameters", 0),
                    "trainable": True,  # Assume trainable unless specified
                    "has_children": len(child.layers) > 0,
                    "is_transformer_layer": getattr(child, "blockType", "")
                    == "transformer",
                    "category": getattr(child, "blockType", "other"),
                }

                # Add shape information if available
                if hasattr(child, "inputShape") and child.inputShape:
                    layer_info["weight_shape"] = child.inputShape
                if hasattr(child, "numHeads") and child.numHeads:
                    layer_info["num_heads"] = child.numHeads

                structure["layers"].append(layer_info)

            return structure

        except Exception as e:
            logger.warning(
                "Error extracting unified layer structure for blueprint: %s", e
            )
            # Simple fallback
            return {
                "error": str(e),
                "model_class": type(self.model).__name__,
                "layers": [],
                "total_parameters": 0,
                "trainable_parameters": 0,
            }

    def extract_batch_info(self, inputs: Any) -> Dict[str, Any]:
        """
        Extract batch information from inputs.

        Args:
            inputs: Model inputs (typically tuple or list of tensors)

        Returns:
            Dictionary containing batch information
        """
        try:
            if isinstance(inputs, (tuple, list)) and len(inputs) > 0:
                first_input = inputs[0]
                if isinstance(first_input, torch.Tensor):
                    return {
                        "batch_size": (
                            first_input.shape[0] if first_input.dim() > 0 else 1
                        ),
                        "sequence_length": (
                            first_input.shape[1] if first_input.dim() > 1 else None
                        ),
                        "input_shape": list(first_input.shape),
                    }
            return {}
        except Exception as e:
            logger.warning("Error extracting batch info: %s", e)
            return {"error": str(e)}

    def get_device_info(self) -> Dict[str, Any]:
        """
        Get device information.

        Returns:
            Dictionary containing device information
        """
        if not self.model:
            return {"error": "No model loaded"}

        try:
            device = next(self.model.parameters()).device
            return {
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": (
                    torch.cuda.device_count() if torch.cuda.is_available() else 0
                ),
            }
        except Exception as e:
            logger.warning("Error extracting device info: %s", e)
            return {"error": str(e)}

    def _send_model_info(self):
        """Send model state information to UI."""
        try:
            if not self.model_state:
                return

            # Send model summary as table
            table_rows = [
                ["Model Class", self.model_state.get("model_class", "N/A")],
                ["Device", self.model_state.get("device", "N/A")],
                ["Data Type", self.model_state.get("dtype", "N/A")],
                ["Total Parameters", f"{self.model_state.get('num_parameters', 0):,}"],
                [
                    "Trainable Parameters",
                    f"{self.model_state.get('trainable_parameters', 0):,}",
                ],
                [
                    "Training Mode",
                    "Yes" if self.model_state.get("training", False) else "No",
                ],
                ["Total Layers", len(self.model_state.get("layer_structure", {}))],
            ]

            self.data_sender.send_table(
                headers=["Property", "Value"],
                rows=table_rows,
            )

            # Send layer structure summary
            layer_structure = self.model_state.get("layer_structure", {})
            if layer_structure:
                # Count layer types
                layer_types = {}
                wrapped_count = 0
                for layer_info in layer_structure.values():
                    layer_class = layer_info.get("class", "Unknown")
                    layer_types[layer_class] = layer_types.get(layer_class, 0) + 1
                    if layer_info.get("wrapped", False):
                        wrapped_count += 1

                # Send layer type distribution
                type_rows = [
                    [layer_type, count]
                    for layer_type, count in sorted(
                        layer_types.items(), key=lambda x: x[1], reverse=True
                    )
                ]
                self.data_sender.send_table(
                    headers=["Layer Type", "Count"],
                    rows=type_rows[:10],  # Top 10 layer types
                )

                self.data_sender.send_text(
                    f"Wrapped Layers: {wrapped_count}/{len(layer_structure)}"
                )

        except Exception as e:
            logger.error(f"Error sending model info: {e}")

    def _send_batch_info(self):
        """Send batch information to UI."""
        try:
            if not self.batch_info:
                return

            table_rows = [
                ["Batch Size", self.batch_info.get("batch_size", "N/A")],
                ["Sequence Length", self.batch_info.get("sequence_length", "N/A")],
                ["Input Shape", str(self.batch_info.get("input_shape", "N/A"))],
            ]

            self.data_sender.send_table(
                headers=["Property", "Value"],
                rows=table_rows,
            )

        except Exception as e:
            logger.error(f"Error sending batch info: {e}")


def initialize(context):
    """Module constructor."""
    introspector = ModelIntrospector()
    introspector.initialize(context)
    global _MODULE_INSTANCE
    _MODULE_INSTANCE = introspector
    return introspector


def cleanup(module):
    """Module destructor."""
    if hasattr(module, "cleanup"):
        module.cleanup()
    global _MODULE_INSTANCE
    _MODULE_INSTANCE = None


def on_model_after(module, context):
    """Handle model after event."""
    return module.on_model_after(context)


def interceptor(ltts_event):
    """LTTSEvent interceptor for model introspection."""
    global _MODULE_INSTANCE
    if _MODULE_INSTANCE is None:
        _MODULE_INSTANCE = ModelIntrospector()
        _MODULE_INSTANCE.initialize(ltts_event.context)
    return _MODULE_INSTANCE.on_model_after(ltts_event.context)


def get_ui_schema():
    """Get UI configuration schema."""
    return {
        "type": "object",
        "properties": {
            "enabled": {
                "type": "boolean",
                "default": True,
                "description": "Enable model introspection",
            },
        },
        "required": [],
    }
