import logging
from typing import Dict, Any, List
import threading
import time
import base64
import io
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from scripts.core.module_data_sender import get_data_sender


METADATA = {
    "name": "memory_monitor",
    "version": "0.2.0",
    "description": "Real-time memory monitoring with GPU and CPU tracking",
    "author": "LTTS Team",
    "event_types": ["model_before", "model_after"],
    "methods": {
        "ctor": "initialize",
        "dtor": "cleanup",
        "ntor": "process_event",
        "utor": "get_ui_schema",
    },
    "dependencies": ["psutil", "matplotlib"],
}


_SENDER = None
_MONITOR_THREAD = None
_STOP_EVENT = threading.Event()
_HISTORY_SIZE = 60
_UPDATE_INTERVAL = 1.0

_gpu_memory_history = deque(maxlen=_HISTORY_SIZE)
_cpu_memory_history = deque(maxlen=_HISTORY_SIZE)
_timestamp_history = deque(maxlen=_HISTORY_SIZE)


def get_ui_schema() -> Dict[str, Any]:
    return {
        "parameters": [
            {
                "name": "update_interval",
                "type": "number",
                "label": "Update Interval (seconds)",
                "default": 1.0,
                "min": 0.5,
                "max": 10.0,
                "step": 0.5,
                "help": "How often to update memory stats",
            },
            {
                "name": "history_size",
                "type": "number",
                "label": "History Size",
                "default": 60,
                "min": 10,
                "max": 300,
                "step": 10,
                "help": "Number of data points to keep in history",
            },
            {
                "name": "show_graphs",
                "type": "boolean",
                "label": "Show Memory Graphs",
                "default": True,
                "help": "Generate and send memory usage graphs",
            },
            {
                "name": "include_cpu",
                "type": "boolean",
                "label": "Monitor CPU Memory",
                "default": True,
                "help": "Track system RAM usage",
            },
            {
                "name": "include_gpu",
                "type": "boolean",
                "label": "Monitor GPU Memory",
                "default": True,
                "help": "Track GPU VRAM usage (if available)",
            },
        ]
    }


def _get_memory_stats() -> Dict[str, Any]:
    stats = {
        "timestamp": datetime.now().isoformat(),
        "cpu": {},
        "gpu": {},
        "available": {
            "psutil": _HAS_PSUTIL,
            "torch": _HAS_TORCH,
            "matplotlib": _HAS_MPL,
        },
    }

    if _HAS_PSUTIL:
        vm = psutil.virtual_memory()
        stats["cpu"] = {
            "total_mb": vm.total / (1024**2),
            "used_mb": vm.used / (1024**2),
            "available_mb": vm.available / (1024**2),
            "percent": vm.percent,
            "free_mb": vm.free / (1024**2),
        }

    if _HAS_TORCH and torch.cuda.is_available():
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            allocated = torch.cuda.memory_allocated(0) / (1024**2)
            reserved = torch.cuda.memory_reserved(0) / (1024**2)
            total = gpu_props.total_memory / (1024**2)

            stats["gpu"] = {
                "device_name": gpu_props.name,
                "total_mb": total,
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "free_mb": total - reserved,
                "percent": (reserved / total) * 100 if total > 0 else 0,
            }
        except Exception as e:
            stats["gpu"] = {"error": str(e)}

    return stats


def _generate_memory_graph() -> str:
    if not _HAS_MPL or len(_timestamp_history) < 2:
        return ""

    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle("Memory Usage Over Time", fontsize=14, fontweight="bold")

        timestamps = list(_timestamp_history)
        time_labels = [t.strftime("%H:%M:%S") for t in timestamps]

        if _gpu_memory_history:
            gpu_data = list(_gpu_memory_history)
            ax1.plot(
                time_labels,
                gpu_data,
                "r-",
                linewidth=2,
                marker="o",
                markersize=4,
                label="GPU Memory",
            )
            ax1.fill_between(range(len(gpu_data)), gpu_data, alpha=0.3, color="red")
            ax1.set_ylabel("GPU Memory (MB)", fontsize=10)
            ax1.set_title("GPU Memory Usage", fontsize=11, fontweight="bold")
            ax1.grid(True, alpha=0.3, linestyle="--")
            ax1.legend(loc="upper left")

            every_n = max(1, len(time_labels) // 10)
            ax1.set_xticks(range(0, len(time_labels), every_n))
            ax1.set_xticklabels(
                [time_labels[i] for i in range(0, len(time_labels), every_n)],
                rotation=45,
            )

        if _cpu_memory_history:
            cpu_data = list(_cpu_memory_history)
            ax2.plot(
                time_labels,
                cpu_data,
                "b-",
                linewidth=2,
                marker="s",
                markersize=4,
                label="CPU Memory",
            )
            ax2.fill_between(range(len(cpu_data)), cpu_data, alpha=0.3, color="blue")
            ax2.set_ylabel("CPU Memory (MB)", fontsize=10)
            ax2.set_xlabel("Time", fontsize=10)
            ax2.set_title("CPU Memory Usage", fontsize=11, fontweight="bold")
            ax2.grid(True, alpha=0.3, linestyle="--")
            ax2.legend(loc="upper left")

            every_n = max(1, len(time_labels) // 10)
            ax2.set_xticks(range(0, len(time_labels), every_n))
            ax2.set_xticklabels(
                [time_labels[i] for i in range(0, len(time_labels), every_n)],
                rotation=45,
            )

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        image_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return image_b64

    except Exception as e:
        logger.error("Error generating memory graph: %s", e)
        return ""


def _format_memory_text(stats: Dict[str, Any]) -> str:
    lines = [
        "=" * 60,
        "📊 MEMORY MONITOR - Real-time Stats",
        "=" * 60,
        f"⏰ Timestamp: {stats['timestamp']}",
        "",
    ]

    if stats.get("cpu"):
        cpu = stats["cpu"]
        lines.extend(
            [
                "💻 CPU/RAM Memory:",
                f"  Total:     {cpu['total_mb']:.1f} MB",
                f"  Used:      {cpu['used_mb']:.1f} MB ({cpu['percent']:.1f}%)",
                f"  Available: {cpu['available_mb']:.1f} MB",
                f"  Free:      {cpu['free_mb']:.1f} MB",
                "",
            ]
        )

    if stats.get("gpu") and not stats["gpu"].get("error"):
        gpu = stats["gpu"]
        lines.extend(
            [
                f"🎮 GPU Memory ({gpu.get('device_name', 'Unknown')}):",
                f"  Total:     {gpu['total_mb']:.1f} MB",
                f"  Allocated: {gpu['allocated_mb']:.1f} MB",
                f"  Reserved:  {gpu['reserved_mb']:.1f} MB ({gpu['percent']:.1f}%)",
                f"  Free:      {gpu['free_mb']:.1f} MB",
                "",
            ]
        )
    elif stats.get("gpu", {}).get("error"):
        lines.extend(["🎮 GPU Memory: Error", f"  {stats['gpu']['error']}", ""])
    elif not _HAS_TORCH or not torch.cuda.is_available():
        lines.extend(
            ["🎮 GPU Memory: Not Available", "  (No CUDA-capable GPU detected)", ""]
        )

    lines.extend(
        [
            "=" * 60,
        ]
    )

    return "\n".join(lines)


def _monitoring_loop(params: Dict[str, Any]):
    global _SENDER

    update_interval = params.get("update_interval", _UPDATE_INTERVAL)
    show_graphs = params.get("show_graphs", True)
    include_cpu = params.get("include_cpu", True)
    include_gpu = params.get("include_gpu", True)

    iteration = 0

    while not _STOP_EVENT.is_set():
        try:
            stats = _get_memory_stats()

            if include_gpu and stats.get("gpu") and not stats["gpu"].get("error"):
                _gpu_memory_history.append(stats["gpu"]["reserved_mb"])

            if include_cpu and stats.get("cpu"):
                _cpu_memory_history.append(stats["cpu"]["used_mb"])

            _timestamp_history.append(datetime.now())

            if _SENDER:
                text_stats = _format_memory_text(stats)
                _SENDER.send_text(
                    text_stats,
                    preformatted=True,
                    layer="memory_monitor",
                    title="Memory Stats",
                )

                _SENDER.send_visualization(
                    stats, subtype="json", layer="memory_monitor", label="raw_stats"
                )

                if show_graphs and _HAS_MPL and iteration % 5 == 0:
                    graph_b64 = _generate_memory_graph()
                    if graph_b64:
                        _SENDER.send_image(
                            graph_b64,
                            image_type="base64",
                            layer="memory_monitor",
                            description="Memory usage over time",
                        )

            iteration += 1
            _STOP_EVENT.wait(update_interval)

        except Exception as e:
            logger.error("Error in monitoring loop: %s", e)
            _STOP_EVENT.wait(update_interval)


def initialize(context, **kwargs):
    global _SENDER, _MONITOR_THREAD

    _SENDER = get_data_sender(METADATA["name"])
    _SENDER.set_context(context)

    if not _HAS_PSUTIL:
        _SENDER.send_text(
            "⚠️ Warning: psutil not installed. CPU memory monitoring unavailable.\n"
            "Install with: pip install psutil",
            preformatted=False,
            layer="memory_monitor",
        )

    initial_stats = _get_memory_stats()
    _SENDER.send_text(
        f"✅ Memory Monitor Initialized\n\n{_format_memory_text(initial_stats)}",
        preformatted=True,
        layer="memory_monitor",
    )

    return {
        "status": "initialized",
        "capabilities": {
            "cpu_monitoring": _HAS_PSUTIL,
            "gpu_monitoring": _HAS_TORCH and torch.cuda.is_available(),
            "graphing": _HAS_MPL,
        },
    }


def cleanup(context=None, **kwargs):
    global _MONITOR_THREAD, _SENDER

    _STOP_EVENT.set()

    if _MONITOR_THREAD and _MONITOR_THREAD.is_alive():
        _MONITOR_THREAD.join(timeout=2.0)

    if _SENDER:
        _SENDER.send_text(
            "🛑 Memory Monitor Stopped", preformatted=False, layer="memory_monitor"
        )

    _MONITOR_THREAD = None
    _gpu_memory_history.clear()
    _cpu_memory_history.clear()
    _timestamp_history.clear()

    return {"status": "cleaned"}


def process_event(ltts_event, **kwargs):
    global _SENDER, _MONITOR_THREAD

    if _SENDER is None:
        _SENDER = get_data_sender(METADATA["name"])
        if hasattr(ltts_event, "context"):
            _SENDER.set_context(ltts_event.context)

    params = getattr(ltts_event, "module_state", {}) or {}

    try:
        if _MONITOR_THREAD is None or not _MONITOR_THREAD.is_alive():
            _STOP_EVENT.clear()
            _MONITOR_THREAD = threading.Thread(
                target=_monitoring_loop,
                args=(params,),
                daemon=True,
                name="MemoryMonitorThread",
            )
            _MONITOR_THREAD.start()

            _SENDER.send_text(
                "🚀 Memory monitoring started! Updates every second...",
                preformatted=False,
                layer="memory_monitor",
            )

        return {"status": "monitoring", "thread_alive": _MONITOR_THREAD.is_alive()}

    except Exception as e:
        if _SENDER:
            _SENDER.send_text(
                f"❌ Error starting memory monitor: {e}",
                preformatted=False,
                layer="memory_monitor",
            )
        return {"status": "error", "message": str(e)}


def interceptor(ltts_event, **kwargs):
    return process_event(ltts_event, **kwargs)
