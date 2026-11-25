from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import torch


class _Sender:
    def __init__(self):
        self.messages = []

    def set_context(self, _context):
        pass

    def _record(self, subtype, data, kwargs):
        self.messages.append({"subtype": subtype, "data": data, **kwargs})

    def send_scalar(self, value, **kwargs):
        self._record("scalar", {"type": "scalar", "value": value}, kwargs)

    def send_table(self, headers, rows, **kwargs):
        self._record("table", {"type": "table", "headers": headers, "rows": rows}, kwargs)

    def send_chart(self, data, chart_type="line", **kwargs):
        self._record(
            "chart",
            {"type": "chart", "chart_type": chart_type, "data": data},
            kwargs,
        )

    def send_heatmap(self, values, dimensions=None, **kwargs):
        self._record(
            "heatmap",
            {"type": "heatmap", "values": values, "dimensions": dimensions},
            kwargs,
        )


def _load_module(sender):
    package = types.ModuleType("scripts")
    core = types.ModuleType("scripts.core")
    data_sender = types.ModuleType("scripts.core.module_data_sender")
    data_sender.get_data_sender = lambda _name: sender
    sys.modules["scripts"] = package
    sys.modules["scripts.core"] = core
    sys.modules["scripts.core.module_data_sender"] = data_sender
    path = Path(__file__).parents[1] / "lookback_lens" / "init.py"
    spec = importlib.util.spec_from_file_location("lookback_lens_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lookback_ratio_distinguishes_context_and_self_heads():
    sender = _Sender()
    module = _load_module(sender)
    module.initialize_module(types.SimpleNamespace(message_bus=object()))

    # 2 heads, 4 queries, 4 keys, context_tokens=2:
    # head 0 attends 0.9 to context (:2) and 0.1 to new tokens (2:)
    # head 1 attends 0.1 to context (:2) and 0.9 to new tokens (2:)
    attn = torch.zeros(1, 2, 4, 4)
    attn[0, 0, :, :2] = 0.45
    attn[0, 0, :, 2:] = 0.05
    attn[0, 1, :, :2] = 0.05
    attn[0, 1, :, 2:] = 0.45

    results = []
    for layer in range(2):
        context = types.SimpleNamespace(
            layer_type="attention",
            module_path=f"model.layers.{layer}.self_attn",
            attention_weights=attn,
        )
        event = types.SimpleNamespace(
            event_type="layer_after",
            context=context,
            module_state={"emit_mode": "all", "context_tokens": 2},
            environment={},
            should_emit=lambda *_args, **_kwargs: {"emit": True, "forward_pass": "run-1"},
            mark_emission_finalized=lambda *_args, **_kwargs: None,
        )
        results.append(module.process_event(event))

    assert results[0]["context_split"] == 2
    assert set(results[0]["emitted"]) == {"scalar", "table"}
    assert set(results[1]["emitted"]) == {"scalar", "table", "chart", "heatmap"}

    latest = {msg["subtype"]: msg for msg in sender.messages}
    rows = latest["table"]["data"]["rows"]
    assert rows[0][1] > 0.8
    assert rows[1][1] < 0.2
    assert latest["heatmap"]["data"]["dimensions"] == {"rows": 2, "cols": 2}
