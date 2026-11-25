from __future__ import annotations

import importlib.util
import math
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


def _load_module(sender):
    package = types.ModuleType("scripts")
    core = types.ModuleType("scripts.core")
    data_sender = types.ModuleType("scripts.core.module_data_sender")
    data_sender.get_data_sender = lambda _name: sender
    sys.modules["scripts"] = package
    sys.modules["scripts.core"] = core
    sys.modules["scripts.core.module_data_sender"] = data_sender
    path = Path(__file__).parents[1] / "intrinsic_dimensionality" / "init.py"
    spec = importlib.util.spec_from_file_location("intrinsic_dimensionality_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rank1_vs_orthogonal_representations():
    sender = _Sender()
    module = _load_module(sender)
    module.initialize_module(types.SimpleNamespace(message_bus=object()))

    # Layer 0: rank-1 hidden states -> PR ~ 1.0, eRank ~ 1.0
    coeffs = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    direction = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    rank1 = (coeffs @ direction).unsqueeze(0)

    # Layer 1: 4 orthogonal unit directions centered around 0 -> PR ~ 3.0, eRank ~ 3.0
    # (4 points in R^4 forming a regular simplex after centering have 3 equal eigenvalues)
    ortho = torch.eye(4).unsqueeze(0)

    res0 = module.process_event(
        types.SimpleNamespace(
            event_type="layer_after",
            context=types.SimpleNamespace(outputs=rank1, module_path="transformer.h.0"),
            module_state={"emit_mode": "all"},
            environment={},
            should_emit=lambda *_args, **_kwargs: {"emit": True, "forward_pass": "run-1"},
            mark_emission_finalized=lambda *_args, **_kwargs: None,
        )
    )
    res1 = module.process_event(
        types.SimpleNamespace(
            event_type="layer_after",
            context=types.SimpleNamespace(outputs=ortho, module_path="transformer.h.1"),
            module_state={"emit_mode": "all"},
            environment={},
            should_emit=lambda *_args, **_kwargs: {"emit": True, "forward_pass": "run-1"},
            mark_emission_finalized=lambda *_args, **_kwargs: None,
        )
    )

    assert math.isclose(res0["participation_ratio"], 1.0, rel_tol=1e-4)
    assert math.isclose(res0["effective_rank"], 1.0, rel_tol=1e-4)
    assert math.isclose(res1["participation_ratio"], 3.0, rel_tol=1e-4)
    assert math.isclose(res1["effective_rank"], 3.0, rel_tol=1e-4)
    assert set(res0["emitted"]) == {"scalar", "table"}
    assert set(res1["emitted"]) == {"scalar", "table", "chart"}
