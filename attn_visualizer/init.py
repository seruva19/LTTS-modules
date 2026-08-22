"""
Attention visualizer module for LTTS.

Provides visualization utilities for transformer attention patterns including
heatmaps, head analysis, similarity matrices, and comprehensive layer comparisons.
"""

import logging
import time

try:
    from .visuals import validate_and_set_defaults
    from . import visuals as _visuals
except Exception:
    import os
    import sys

    _CURRENT_DIR = os.path.dirname(__file__)
    if _CURRENT_DIR not in sys.path:
        sys.path.append(_CURRENT_DIR)
    from visuals import validate_and_set_defaults
    import visuals as _visuals

logger = logging.getLogger(__name__)

METADATA = {
    "name": "attn_visualizer",
    "version": "0.0.1",
    "description": "Attention visualization with heatmaps, head analysis, pattern detection, and similarity matrices",
    "author": "LTTS",
    "event_types": ["layer_after", "layer_before"],
    "dependencies": ["requirements.txt"],
    "methods": {
        "ctor": "initialize_module",
        "dtor": "cleanup_module",
        "ntor": "process_event",
        "utor": "get_ui_schema",
    },
}


def initialize_module(context, **config):
    logger.info("Initializing attention visualizer module")
    return {"status": "initialized"}


def get_ltts_model_from_context_or_global(ltts_event):
    return getattr(ltts_event.context, "ltts_model", None)


def cleanup_module():
    logger.info("Cleaning up attention visualizer module")
    return {"status": "cleaned_up"}


def get_ui_schema():
    schema = {
        "parameters": [
            {
                "name": "visualization_type",
                "type": "select",
                "label": "Visualization Type",
                "default": "attention_heatmap",
                "options": [
                    "attention_heatmap",
                    "similarity_matrix",
                    "head_analysis",
                    "pattern_matrix",
                    "entropy_plot",
                    "comprehensive_analysis",
                    "multi_layer_comparison",
                    "token_level_analysis",
                ],
                "help": "Type of attention visualization to generate",
            },
            {
                "name": "layer_index",
                "type": "number",
                "label": "Layer Index",
                "default": -1,
                "min": -1,
                "help": "Target layer (-1 for current layer)",
            },
            {
                "name": "head_index",
                "type": "number",
                "label": "Attention Head",
                "default": -1,
                "min": -1,
                "max": 32,
                "help": "Which attention head to visualize (-1 for all heads average)",
            },
            {
                "name": "similarity_metric",
                "type": "select",
                "label": "Similarity Metric",
                "default": "cosine",
                "options": ["cosine", "dot", "euclidean"],
                "help": "Metric for similarity matrix calculation",
            },
            {
                "name": "token_range",
                "type": "text",
                "label": "Token Range",
                "default": "0:50",
                "help": "Token range to visualize (start:end)",
            },
            {
                "name": "normalize",
                "type": "boolean",
                "label": "Normalize Weights",
                "default": True,
                "help": "Normalize attention weights",
            },
            {
                "name": "threshold",
                "type": "number",
                "label": "Weight Threshold",
                "default": 0.01,
                "min": 0,
                "max": 1,
                "step": 0.01,
                "help": "Minimum attention weight to display",
            },
            {
                "name": "colormap",
                "type": "select",
                "label": "Color Scheme",
                "default": "viridis",
                "options": [
                    "viridis",
                    "plasma",
                    "inferno",
                    "magma",
                    "blues",
                    "reds",
                    "coolwarm",
                ],
                "help": "Color scheme for visualization",
            },
            {
                "name": "show_annotations",
                "type": "boolean",
                "label": "Show Value Annotations",
                "default": False,
                "help": "Display numerical values on heatmap",
            },
        ],
        "layout": {"type": "grid", "columns": 2},
        "real_time": {"enabled": True, "interval": 200},
    }

    return schema


def process_event(ltts_event):
    try:
        if not isinstance(ltts_event.module_state, dict):
            ltts_event.module_state = {}
        state = ltts_event.module_state
        stats = state.get("stats")
        if not isinstance(stats, dict):
            stats = {
                "processed_events": 0,
                "visualizations_created": 0,
                "skipped_emits": 0,
            }
            state["stats"] = stats
        stats["processed_events"] += 1

        logger.info(
            f"ATTN_VISUALIZER: Processing event {ltts_event.event_type} from {ltts_event.context.module_path}"
        )

        params = ltts_event.module_state or {}
        params = validate_and_set_defaults(params)
        viz_type = params["visualization_type"]

        actual_layer_path = getattr(ltts_event.context, "module_path", None)
        requested_layer_path = (
            ltts_event.environment.get("requested_layer_path")
            if hasattr(ltts_event, "environment")
            else None
        )
        target_layer_path = (
            requested_layer_path or params.get("layer_path") or actual_layer_path
        )

        module_name = METADATA.get("name", "attn_visualizer")
        emission_info = ltts_event.should_emit(module_name, actual_layer_path, params)
        logger.debug(
            "ATTN_VISUALIZER: Layer paths resolved actual=%s target=%s",
            actual_layer_path,
            target_layer_path,
        )

        if not emission_info["emit"]:
            stats["skipped_emits"] += 1
            logger.debug(
                "ATTN_VISUALIZER: Skipping emission for %s (mode=%s, pass=%s)",
                actual_layer_path or target_layer_path or "unknown_layer",
                emission_info["mode"],
                emission_info["forward_pass"],
            )
            return

        result = None
        error_message = None

        try:
            if viz_type == "attention_heatmap":
                result = _visuals.create_attention_heatmap(ltts_event, params)
            elif viz_type == "similarity_matrix":
                result = _visuals.create_similarity_matrix(ltts_event, params)
            elif viz_type == "head_analysis":
                result = _visuals.create_head_analysis(ltts_event, params)
            elif viz_type == "pattern_matrix":
                result = _visuals.create_pattern_matrix(ltts_event, params)
            elif viz_type == "entropy_plot":
                result = _visuals.create_entropy_plot(ltts_event, params)
            elif viz_type == "comprehensive_analysis":
                result = _visuals.create_comprehensive_analysis(ltts_event, params)
            elif viz_type == "multi_layer_comparison":
                result = _visuals.create_multi_layer_comparison(ltts_event, params)
            elif viz_type == "token_level_analysis":
                result = _visuals.create_token_level_analysis(ltts_event, params)
            else:
                result = _visuals.create_attention_heatmap(ltts_event, params)
        except ValueError as err:
            error_message = str(err)

        if result:
            stats["visualizations_created"] += 1

            logger.info(
                "ATTN_VISUALIZER: Created visualization %s for layer %s (mode=%s, pass=%s)",
                viz_type,
                actual_layer_path or target_layer_path,
                emission_info["mode"],
                emission_info["forward_pass"],
            )

            payload = {
                "type": "attention_visualization",
                "subtype": viz_type,
                "module": module_name,
                "data": result,
                "params": params,
                "event_type": ltts_event.event_type,
                "module_path": target_layer_path,
                "layer_path": target_layer_path,
                "layer_info": (
                    {"path": target_layer_path} if target_layer_path else None
                ),
                "actual_layer_path": actual_layer_path,
                "resolved_layer_path": actual_layer_path,
                "stats": stats,
                "emit_mode": emission_info["mode"],
                "forward_pass": emission_info["forward_pass"],
                "emit_stride": emission_info.get("stride"),
                "deferred": emission_info.get("deferred", False),
                "timestamp": time.time(),
            }

            ltts_event.reactor.signal("visualization", payload)
        else:
            logger.warning(
                f"ATTN_VISUALIZER: No visualization result generated for {viz_type}"
            )
            if error_message and viz_type == "attention_heatmap":
                stats["visualizations_created"] += 1
                text_payload = {
                    "type": "attention_visualization",
                    "subtype": viz_type,
                    "module": module_name,
                    "data": error_message,
                    "params": params,
                    "event_type": ltts_event.event_type,
                    "module_path": target_layer_path,
                    "layer_path": target_layer_path,
                    "layer_info": (
                        {"path": target_layer_path} if target_layer_path else None
                    ),
                    "actual_layer_path": actual_layer_path,
                    "resolved_layer_path": actual_layer_path,
                    "stats": stats,
                    "emit_mode": emission_info["mode"],
                    "forward_pass": emission_info["forward_pass"],
                    "emit_stride": emission_info.get("stride"),
                    "deferred": emission_info.get("deferred", False),
                    "timestamp": time.time(),
                    "is_error": True,
                }
                ltts_event.reactor.signal("visualization", text_payload)
            else:
                logger.warning(
                    f"ATTN_VISUALIZER: No visualization result generated for {viz_type}"
                )
                ltts_event.reactor.signal(
                    "error",
                    {
                        "message": "Failed to generate visualization",
                        "module": "attn_visualizer",
                    },
                )

    except Exception as e:
        logger.error(f"Error in unified attention visualizer: {e}", exc_info=True)
        ltts_event.reactor.signal(
            "error",
            {
                "message": f"Error in attention visualization: {str(e)}",
                "module": "attn_visualizer",
                "event_type": getattr(ltts_event, "event_type", "unknown"),
            },
        )


def interceptor(ltts_event):
    return process_event(ltts_event)
