"""
Model-Level Control Vector Module with Training Interface

Implements control vector functionality as a model-level extension that can be
attached to entire models rather than specific layers. Uses the repeng library
for control vector extraction and application.
"""

from __future__ import annotations  # Defer type hint evaluation

import asyncio
import inspect
import numpy as np
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
import logging

# repeng 0.4 still references the NumPy 1.x alias removed in NumPy 2.
if not hasattr(np, "float_"):
    np.float_ = np.float64

# Try to import repeng components
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "repeng"))
    from repeng.extract import ControlVector, DatasetEntry
    from repeng.control import ControlModel

    REPNG_AVAILABLE = True
except ImportError:
    REPNG_AVAILABLE = False
    # Define placeholder types when repeng is not available
    if TYPE_CHECKING:
        from repeng.extract import ControlVector, DatasetEntry
        from repeng.control import ControlModel
    else:
        ControlVector = None
        DatasetEntry = None
        ControlModel = None

logger = logging.getLogger(__name__)

METADATA = {
    "name": "model_control_vector",
    "version": "2.1.0",
    "description": "Unified control vector module with training interface, visualization, and repeng integration",
    "author": "LTTS Team",
    "event_types": ["model_after", "model_before", "layer_after"],
    "dependencies": ["requirements.txt"],
    "supported_topologies": ["decoder_only"],
    "methods": {
        "ctor": "initialize_control_vector",
        "ntor": "process_model_event",
        "utor": "get_control_vector_ui",
    },
    "tags": ["control_vector", "training", "model_level", "repeng", "visualization"],
    "module_type": "model_level",  # New field indicating model-level module
}

# Global state
_control_model = None
_control_model_base_id = None
_trained_vector = None
_vector_strength = 1.0
_training_data = []
_training_active = False
_message_bus = None
_model_config = None
_last_inference_time = None
_visualization_count = 0
_current_inference_id = None
_sent_visualizations = set()  # Track which layers already sent visualizations


def initialize_control_vector(context):
    """Initialize the control vector module."""
    global _message_bus, _model_config
    _message_bus = context.message_bus
    _model_config = getattr(context, "model_config", None)

    logger.info(f"MODEL CONTROL VECTOR: Initialized with model config: {_model_config}")
    return {"status": "initialized", "repeng_available": REPNG_AVAILABLE}


def _get_model_and_tokenizer(ltts_event):
    ltts_model = getattr(ltts_event.context, "ltts_model", None)
    base_model = getattr(ltts_model, "original_model", None) if ltts_model else None
    tokenizer = getattr(ltts_model, "_tokenizer", None) if ltts_model else None
    return base_model, tokenizer, ltts_model


def _control_layer_ids(base_model) -> List[int]:
    """Return decoder block ids supported by repeng's ControlModel."""
    if base_model is None:
        return []
    model_body = getattr(base_model, "model", None)
    layers = getattr(model_body, "layers", None)
    if layers is None:
        transformer = getattr(base_model, "transformer", None)
        layers = getattr(transformer, "h", None)
    if layers is None:
        return []
    # repeng 0.4's default training range is ``range(-1, -n_layers, -1)``:
    # after normalization this is layers 1..N-1. Match that set exactly so
    # ControlModel.set_control never indexes a direction that training did not
    # produce (layer 0 would otherwise raise KeyError).
    if len(layers) > 1:
        return list(range(1, len(layers)))
    return [0] if len(layers) == 1 else []


def _ensure_control_model(base_model):
    global _control_model, _control_model_base_id
    if base_model is None or not REPNG_AVAILABLE or ControlModel is None:
        return None
    if _control_model is not None and _control_model_base_id == id(base_model):
        return _control_model
    try:
        layer_ids = _control_layer_ids(base_model)
        if not layer_ids:
            logger.warning(
                "repeng does not expose a supported decoder block list for %s",
                type(base_model).__name__,
            )
            return None
        _control_model = ControlModel(base_model, layer_ids)
        _control_model_base_id = id(base_model)
        return _control_model
    except Exception as e:
        logger.error("Failed to initialize ControlModel: %s", e)
        _control_model = None
        _control_model_base_id = None
        return None


def _train_control_vector(dataset, model, tokenizer, params):
    if not REPNG_AVAILABLE or ControlVector is None:
        return {"error": "repeng library not available"}
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError(
                "Control-vector training requires a tokenizer padding token "
                "or an EOS token that can be used for padding"
            )
    method = params.get("training_method", "pca_diff")
    max_batch_size = params.get("max_batch_size", 32)

    if hasattr(ControlVector, "train"):
        return ControlVector.train(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            method=method,
            batch_size=max_batch_size,
        )
    # Compatibility with older repeng releases.
    if hasattr(ControlVector, "from_dataset"):
        return ControlVector.from_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            method=method,
            batch_size=max_batch_size,
        )
    if hasattr(ControlVector, "from_data"):
        return ControlVector.from_data(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            method=method,
            batch_size=max_batch_size,
        )

    raise RuntimeError("Unsupported repeng ControlVector API")


def get_control_vector_ui():
    """Get UI schema for control vector module."""
    ui_schema = {
        "type": "model_level_control_panel",
        "title": "Model Control Vector",
        "sections": [
            {
                "id": "vector_control",
                "title": "Vector Control",
                "controls": [
                    {
                        "type": "slider",
                        "name": "vector_strength",
                        "label": "Vector Strength",
                        "min": -2.0,
                        "max": 2.0,
                        "default": 1.0,
                        "step": 0.1,
                        "description": "Strength of control vector application",
                    },
                    {
                        "type": "toggle",
                        "name": "vector_enabled",
                        "label": "Enable Control Vector",
                        "default": False,
                        "description": "Toggle control vector on/off",
                    },
                    {
                        "type": "toggle",
                        "name": "enable_visualization",
                        "label": "Enable Visualization",
                        "default": True,
                        "description": "Enable real-time visualization of control vector activations",
                    },
                ],
            },
            {
                "id": "training_interface",
                "title": "Training Interface",
                "controls": [
                    {
                        "type": "textarea",
                        "name": "positive_examples",
                        "label": "Positive Examples",
                        "placeholder": "Enter positive examples (one per line)",
                        "description": "Examples of the behavior you want to enhance",
                    },
                    {
                        "type": "textarea",
                        "name": "negative_examples",
                        "label": "Negative Examples",
                        "placeholder": "Enter negative examples (one per line)",
                        "description": "Examples of the behavior you want to reduce",
                    },
                    {
                        "type": "button",
                        "name": "train_vector",
                        "label": "Train Control Vector",
                        "action": "train_control_vector",
                        "description": "Train a new control vector from examples",
                    },
                    {
                        "type": "button",
                        "name": "reset_vector",
                        "label": "Reset Vector",
                        "action": "reset_control_vector",
                        "description": "Remove trained control vector",
                    },
                ],
            },
            {
                "id": "advanced_options",
                "title": "Advanced Options",
                "controls": [
                    {
                        "type": "select",
                        "name": "training_method",
                        "label": "Training Method",
                        "options": [
                            {"value": "pca_diff", "label": "PCA Difference"},
                            {"value": "pca_center", "label": "PCA Center"},
                        ],
                        "default": "pca_diff",
                    },
                    {
                        "type": "number",
                        "name": "max_batch_size",
                        "label": "Max Batch Size",
                        "min": 1,
                        "max": 64,
                        "default": 32,
                        "description": "Maximum examples processed in one training batch",
                    },
                ],
            },
        ],
    }

    if not REPNG_AVAILABLE:
        ui_schema["warning"] = "repeng library not available - training disabled"

    return ui_schema


def process_model_event(ltts_event):
    """Process model-level events for control vector application."""
    global _control_model, _trained_vector, _vector_strength
    global _last_inference_time, _visualization_count, _current_inference_id, _sent_visualizations

    if ltts_event.event_type not in ["model_before", "model_after"]:
        return

    try:
        current_time = datetime.now()

        # Create a more granular inference ID to properly detect new runs
        inference_id = current_time.strftime("%Y%m%d_%H%M%S_%f")[
            :20
        ]  # Include microseconds

        # Reset for new inference runs (if more than 5 seconds have passed)
        if (
            _last_inference_time is None
            or (current_time - _last_inference_time).total_seconds() > 5
        ):
            _visualization_count = 0
            _current_inference_id = inference_id
            _sent_visualizations.clear()
            _last_inference_time = current_time

        _visualization_count += 1

        logger.debug(f"Event #{_visualization_count} - {ltts_event.event_type}")

        # Get current parameters from event/module state
        params = ltts_event.module_state or {}
        action = params.get("action") or params.get("control_action")
        if params.get("train_vector"):
            action = "train_control_vector"
        if params.get("reset_vector"):
            action = "reset_control_vector"
        vector_enabled = params.get("vector_enabled", False)
        vector_strength = params.get("vector_strength", 1.0)
        enable_visualization = params.get("enable_visualization", True)

        # Extract layer information from context
        layer_path = getattr(ltts_event.context, "module_path", "model")
        layer_type = getattr(ltts_event.context, "layer_type", "model")

        base_model, tokenizer, ltts_wrapper = _get_model_and_tokenizer(ltts_event)
        # Actions are edge-triggered UI commands.  The same attachment state is
        # presented to both model_before and model_after, so running an action
        # in both phases trains or resets twice during one inference.
        if ltts_event.event_type == "model_before" and action == "train_control_vector":
            if base_model is None or tokenizer is None:
                logger.warning("Training requires a loaded model and tokenizer")
            else:
                training_model = _control_model or base_model
                if ltts_wrapper is not None:
                    ltts_wrapper._suspend_analysis_events = True
                try:
                    training_result = handle_training_request(
                        params, training_model, tokenizer
                    )
                finally:
                    if ltts_wrapper is not None:
                        ltts_wrapper._suspend_analysis_events = False
                params["last_training_result"] = training_result
            ltts_event.module_state["action"] = None
            ltts_event.module_state["train_vector"] = False
            ltts_event.module_state["reset_vector"] = False

        if ltts_event.event_type == "model_before" and action == "reset_control_vector":
            _trained_vector = None
            if _control_model is not None:
                _control_model.reset()
            ltts_event.module_state["action"] = None
            ltts_event.module_state["reset_vector"] = False

        if ltts_event.event_type == "model_before":
            _control_model = _ensure_control_model(base_model)
            if (
                vector_enabled
                and _trained_vector is not None
                and _control_model is not None
            ):
                logger.info(f"Applying control vector with strength: {vector_strength}")
                _control_model.set_control(_trained_vector, coeff=vector_strength)
            elif _control_model is not None:
                _control_model.reset()

        # Send visualization if enabled
        if enable_visualization and ltts_event.event_type == "model_after":
            # Create unique key for this exact event
            viz_key = f"{_current_inference_id}_{layer_path}_{_visualization_count}"

            if viz_key not in _sent_visualizations:
                _sent_visualizations.add(viz_key)
                visualization = generate_control_vector_visualization_data(
                    vector_strength, _trained_vector
                )
                ltts_event.reactor.signal(
                    "visualization",
                    {
                        "module": "model_control_vector",
                        "subtype": "control_vector_activation",
                        "event_type": ltts_event.event_type,
                        "layer_path": layer_path,
                        "data": visualization,
                    },
                )
                _send_control_vector_message(
                    vector_strength,
                    ltts_event.context,
                    layer_path,
                    layer_type,
                    "model_control_vector",
                    _trained_vector,
                )

        # Model events can run in an inference worker thread without an asyncio
        # loop.  MessageBus owns the application loop and provides a thread-safe
        # publication method for exactly this case.
        _publish_vector_status(
            {
                "enabled": vector_enabled,
                "strength": vector_strength,
                "trained": _trained_vector is not None,
                "repeng_available": REPNG_AVAILABLE,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Send signal through reactor
        if hasattr(ltts_event, "reactor"):
            ltts_event.reactor.signal(
                "control_vector_processed",
                {
                    "layer_path": layer_path,
                    "vector_strength": vector_strength,
                    "event_count": _visualization_count,
                },
            )

        return {
            "status": "ok",
            "event_type": ltts_event.event_type,
            "trained": _trained_vector is not None,
            "vector_enabled": bool(vector_enabled),
            "vector_strength": float(vector_strength),
        }

    except Exception as e:
        logger.error(f"Error in control vector processing: {e}")


def generate_control_vector_visualization_data(vector_strength, trained_vector):
    """Describe the actual trained vector; never fabricate an activation value."""
    directions = getattr(trained_vector, "directions", None)
    norms = []
    if isinstance(directions, dict):
        for direction in directions.values():
            try:
                norms.append(float(np.linalg.norm(np.asarray(direction))))
            except (TypeError, ValueError):
                continue
    mean_norm = float(np.mean(norms)) if norms else 0.0
    effective_norm = mean_norm * abs(float(vector_strength))

    return {
        "type": "scalar",
        "value": effective_norm,
        "description": (
            "Mean per-layer control-vector L2 norm multiplied by |strength|"
            if norms
            else "No trained control vector is currently applied"
        ),
        "unit": "l2_norm",
        "trained": bool(norms),
        "layers": len(norms),
        "coefficient": float(vector_strength),
    }


async def send_vector_status(status_data):
    """Send control vector status to client."""
    global _message_bus

    if _message_bus is None:
        return

    try:
        message = {
            "type": "control_vector_status",
            "module": "model_control_vector",
            "timestamp": datetime.now().isoformat(),
            "data": status_data,
        }

        await _message_bus.publish("control_vector_events", message)

    except Exception as e:
        logger.error(f"Error sending control vector status: {e}")


def _publish_vector_status(status_data):
    """Publish status safely from either the app loop or a worker thread."""
    if _message_bus is None:
        return

    message = {
        "type": "control_vector_status",
        "module": "model_control_vector",
        "timestamp": datetime.now().isoformat(),
        "data": status_data,
    }
    try:
        if hasattr(_message_bus, "publish_from_thread"):
            _message_bus.publish_from_thread("control_vector_events", message)
        elif hasattr(_message_bus, "publish_sync"):
            _message_bus.publish_sync("control_vector_events", message)
        else:
            asyncio.get_running_loop().create_task(
                _message_bus.publish("control_vector_events", message)
            )
    except Exception as exc:
        logger.error("Failed to publish control vector status: %s", exc)


async def send_websocket_message(
    message_type, data, module_name="model_control_vector"
):
    """Send a message via WebSocket to connected clients."""
    global _message_bus

    logger.debug(f"send_websocket_message called - module_name = {module_name}")
    logger.debug(f"_message_bus = {_message_bus}")

    if _message_bus is None:
        return

    try:
        message = {
            "type": message_type,
            "module": module_name,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }

        logger.debug(f"Constructed message: {message}")

        # Send via message bus
        await _message_bus.publish("control_vector_events", message)

    except Exception as e:
        import traceback

        traceback.print_exc()


def handle_training_request(params, model, tokenizer):
    """Handle control vector training request."""
    global _training_data, _training_active, _trained_vector

    if not REPNG_AVAILABLE or DatasetEntry is None:
        return {"error": "repeng library not available"}

    positive_examples = params.get("positive_examples", "").strip().split("\n")
    negative_examples = params.get("negative_examples", "").strip().split("\n")

    # Filter out empty lines
    positive_examples = [ex.strip() for ex in positive_examples if ex.strip()]
    negative_examples = [ex.strip() for ex in negative_examples if ex.strip()]

    if not positive_examples or not negative_examples:
        return {"error": "Both positive and negative examples are required"}

    # Create dataset
    dataset = []
    min_len = min(len(positive_examples), len(negative_examples))

    for i in range(min_len):
        dataset.append(
            DatasetEntry(positive=positive_examples[i], negative=negative_examples[i])
        )

    _training_active = True
    try:
        logger.info("Training control vector with %s examples", len(dataset))
        trained_vector = _train_control_vector(dataset, model, tokenizer, params)
        _trained_vector = trained_vector
        _training_active = False

        vector_info = {}
        for attr in ("model_type", "directions", "layers"):
            if hasattr(trained_vector, attr):
                vector_info[attr] = getattr(trained_vector, attr)

        return {
            "success": True,
            "message": f"Trained control vector with {len(dataset)} examples",
            "vector_info": vector_info,
        }
    except Exception as e:
        _training_active = False
        logger.error("Error training control vector: %s", e)
        return {"error": str(e)}


def _send_control_vector_message(
    vector_strength,
    context_or_env,
    layer_path=None,
    layer_type=None,
    module_name=None,
    trained_vector=None,
):
    """Helper function to send WebSocket message."""
    logger.debug(f"Sending control vector visualization message...")

    # Generate visualization data
    viz_data = generate_control_vector_visualization_data(
        vector_strength, trained_vector
    )

    # Extract layer info
    if layer_path is None:
        layer_path = getattr(context_or_env, "layer_path", "unknown")
    if layer_type is None:
        layer_type = getattr(context_or_env, "layer_type", "unknown")
    if module_name is None:
        module_name = getattr(context_or_env, "module_name", "model_control_vector")

    logger.debug(f"Using module name: {module_name}")

    # Create the complete message structure
    message_data = {
        "subtype": "control_vector_activation",
        "layer_info": {"path": layer_path, "type": layer_type},
        "parameters": {
            "vector_strength": vector_strength,
            "timestamp": datetime.now().isoformat(),
        },
        "visualization": viz_data,
    }

    # Send WebSocket message asynchronously
    logger.debug(f"About to send WebSocket message...")
    try:
        # Create event loop if none exists
        try:
            loop = asyncio.get_event_loop()
            logger.debug(f"Got existing event loop: {loop}")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            logger.debug(f"Created new event loop: {loop}")

        # Schedule the coroutine
        if loop.is_running():
            # If loop is already running, create a task
            logger.debug(f"Loop is running, creating task...")
            asyncio.create_task(
                send_websocket_message("visualization", message_data, module_name)
            )
        else:
            # If loop is not running, run the coroutine
            logger.debug(f"Loop not running, running coroutine...")
            loop.run_until_complete(
                send_websocket_message("visualization", message_data, module_name)
            )

    except Exception as e:
        import traceback

        traceback.print_exc()


def reset_control_vector():
    """Reset trained control vector."""
    global _trained_vector, _control_model

    _trained_vector = None
    if _control_model is not None:
        _control_model.reset()

    return {"success": True, "message": "Control vector reset"}


# Event handlers for UI actions
def handle_ui_action(action_name, params):
    """Handle UI actions from client."""
    if action_name == "train_control_vector":
        return handle_training_request(params)
    elif action_name == "reset_control_vector":
        return reset_control_vector()
    else:
        return {"error": f"Unknown action: {action_name}"}


# Module interface functions
def ctor(context, **kwargs):
    """Module constructor."""
    return initialize_control_vector(context)


def ntor(ltts_event, **kwargs):
    """Module event processor."""
    return process_model_event(ltts_event)


def utor(**kwargs):
    """Module UI provider."""
    return get_control_vector_ui()


# For backward compatibility
def interceptor(ltts_event):
    """Compatibility interface."""
    return process_model_event(ltts_event)


def cvector_module_ui():
    """Compatibility UI interface."""
    return get_control_vector_ui()
