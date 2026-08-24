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

    def send_heatmap(self, values, dimensions=None, **kwargs):
        self._record(
            "heatmap",
            {"type": "heatmap", "values": values, "dimensions": dimensions},
            kwargs,
        )

    def send_chart(self, data, chart_type="line", **kwargs):
        self._record(
            "chart",
            {"type": "chart", "chart_type": chart_type, "data": data},
            kwargs,
        )

    def send_table(self, headers, rows, **kwargs):
        self._record(
            "table",
            {"type": "table", "headers": headers, "rows": rows},
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
    path = Path(__file__).parents[1] / "state_tracking_diagnostics" / "init.py"
    spec = importlib.util.spec_from_file_location("state_tracking_diagnostics_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multilayer_trace_emits_three_bounded_renderable_artifacts():
    sender = _Sender()
    module = _load_module(sender)
    module.initialize_module(types.SimpleNamespace(message_bus=object()))
    torch.manual_seed(5)
    base = torch.randn(1, 4, 6)

    results = []
    for layer, scale in enumerate((0.0, 0.1, 0.35)):
        context = types.SimpleNamespace(
            outputs=base + scale * torch.randn_like(base),
            module_path=f"transformer.h.{layer}",
            module_role="block",
            module_class="GPT2Block",
            layer_type="block",
        )
        event = types.SimpleNamespace(
            event_type="layer_after",
            context=context,
            module_state={
                "emit_mode": "all",
                "max_positions": 4,
                "top_tokens": 3,
            },
            environment={
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "tokenizer": types.SimpleNamespace(
                    convert_ids_to_tokens=lambda ids: [f"tok-{item}" for item in ids]
                ),
            },
            should_emit=lambda *_args, **_kwargs: {
                "emit": True,
                "forward_pass": "run-1",
            },
        )
        results.append(module.process_event(event))

    assert results[0]["emitted"] == []
    assert results[-1]["layers_accumulated"] == 3
    assert set(results[-1]["emitted"]) == {"heatmap", "chart", "table"}
    latest = {message["subtype"]: message for message in sender.messages}
    assert set(latest) == {"heatmap", "chart", "table"}
    assert latest["heatmap"]["data"]["dimensions"] == {"rows": 4, "cols": 2}
    assert len(latest["table"]["data"]["rows"]) == 3
    assert all(
        len(series["points"]) == 2
        for series in latest["chart"]["data"]["data"]["series"]
    )
    assert {message["emit_id"] for message in latest.values()} == {
        "state_tracking_diagnostics:heatmap",
        "state_tracking_diagnostics:chart",
        "state_tracking_diagnostics:table",
    }
