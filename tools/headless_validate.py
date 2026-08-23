#!/usr/bin/env python3
"""Validate every LTTS module without starting the LTTS web application."""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import os
import signal
import subprocess
import sys
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REQUIRED_METADATA = {
    "name": str,
    "version": str,
    "description": str,
    "event_types": list,
    "methods": dict,
}
METHOD_ROLES = ("ctor", "ntor", "utor")


def module_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "init.py").is_file() and path.name != "tests"
    )


def literal_metadata(init_path: Path) -> dict:
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "METADATA"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                raise TypeError("METADATA must be a dictionary")
            return value
    raise ValueError("METADATA assignment not found")


def validate_static(root: Path, selected: set[str] | None = None) -> list[dict]:
    failures: list[dict] = []
    discovered: dict[str, dict] = {}
    for directory in module_dirs(root):
        if selected and directory.name not in selected:
            continue
        try:
            metadata = literal_metadata(directory / "init.py")
            for key, expected_type in REQUIRED_METADATA.items():
                if not isinstance(metadata.get(key), expected_type):
                    raise TypeError(f"METADATA[{key!r}] must be {expected_type.__name__}")
            if not metadata["event_types"] or not all(
                isinstance(item, str) and item for item in metadata["event_types"]
            ):
                raise ValueError("event_types must contain at least one event name")
            for role in METHOD_ROLES:
                method = metadata["methods"].get(role)
                if not isinstance(method, str) or not method:
                    raise ValueError(f"methods.{role} is required")
            for dependency_file in metadata.get("dependencies", []):
                if dependency_file.endswith((".txt", ".in")) and not (
                    directory / dependency_file
                ).is_file():
                    raise FileNotFoundError(f"missing dependency file: {dependency_file}")
            discovered[directory.name] = metadata
        except Exception as exc:  # noqa: BLE001 - report every module in one run
            failures.append({"module": directory.name, "phase": "static", "error": str(exc)})

    if selected:
        missing = selected - set(discovered) - {item["module"] for item in failures}
        failures.extend(
            {"module": name, "phase": "discovery", "error": "module not found"}
            for name in sorted(missing)
        )
        return failures

    registry_path = root / "registry.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        entries = registry.get("modules", [])
        indexed = {entry.get("name"): entry for entry in entries}
        if set(indexed) != set(discovered):
            raise ValueError(
                f"registry modules differ: missing={sorted(set(discovered) - set(indexed))}, "
                f"extra={sorted(set(indexed) - set(discovered))}"
            )
        for name, metadata in discovered.items():
            entry = indexed[name]
            for key in ("version", "description", "event_types", "dependencies"):
                if entry.get(key) != metadata.get(key, []):
                    raise ValueError(f"registry {name}.{key} is out of date")
    except Exception as exc:  # noqa: BLE001
        failures.append({"module": "registry.json", "phase": "registry", "error": str(exc)})
    return failures


class _Bus:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def publish_from_thread(self, _token, message) -> None:
        self.messages.append(message)

    def publish_sync(self, _token, message) -> None:
        self.messages.append(message)

    async def publish(self, _token, message) -> None:
        self.messages.append(message)


def _invoke(function, primary=None):
    signature = inspect.signature(function)
    if not signature.parameters:
        return function()
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is inspect.Parameter.empty
    ]
    if positional:
        return function(primary)
    return function()


def validate_worker(root: Path, ltts_root: Path, name: str) -> dict:
    sys.path.insert(0, str(ltts_root))
    package = types.ModuleType("ltts_modules")
    package.__path__ = [str(root)]
    sys.modules["ltts_modules"] = package

    import torch
    from scripts.core.ltts_event import LTTSContext, LTTSEvent, LTTSReactor

    module = importlib.import_module(f"ltts_modules.{name}.init")
    metadata = module.METADATA
    methods = metadata["methods"]
    for role in METHOD_ROLES:
        function_name = methods[role]
        if not callable(getattr(module, function_name, None)):
            raise TypeError(f"{role} target {function_name!r} is not callable")

    bus = _Bus()
    hidden = torch.randn(1, 6, 8, requires_grad=True)
    attention = torch.softmax(torch.randn(1, 2, 6, 6), dim=-1)
    tiny_model = torch.nn.Module()
    tiny_model.config = types.SimpleNamespace(
        model_type="gpt2",
        hidden_size=8,
        n_embd=8,
        vocab_size=16,
        num_hidden_layers=1,
    )
    tiny_model.lm_head = torch.nn.Linear(8, 16, bias=False)
    ltts_model = types.SimpleNamespace(
        original_model=tiny_model,
        _tokenizer=types.SimpleNamespace(
            decode=lambda ids, **_kwargs: str(ids[0] if ids else ""),
            convert_ids_to_tokens=lambda ids: [str(item) for item in ids],
        ),
        architecture={
            "topology": "decoder_only",
            "family": "gpt2",
            "capabilities": {
                "hidden_states": True,
                "self_attention": True,
                "gradients": True,
                "tokenizer": True,
                "unembedding": True,
            },
        },
        layer_wrappers={"transformer.h.0": object()},
    )
    runtime_context = types.SimpleNamespace(
        message_bus=bus,
        model_config=tiny_model.config,
        models=types.SimpleNamespace(),
        module_loader=types.SimpleNamespace(get_all_modules=lambda: {}),
    )
    ctor_result = _invoke(getattr(module, methods["ctor"]), runtime_context)

    ui_result = _invoke(getattr(module, methods["utor"]), None)
    if ui_result is not None:
        json.dumps(ui_result, default=str)

    context = LTTSContext(
        inputs=hidden.detach().clone(),
        outputs=hidden,
        module_path="transformer.h.0",
        module_class="GPT2Block",
        layer_type="block",
        module_role="block",
        architecture_stage="decoder",
        layer_index=0,
        attention_kind="self",
        architecture=ltts_model.architecture,
        attention_weights=attention,
        message_bus=bus,
        ltts_model=ltts_model,
        previous_outputs=[hidden.detach().clone()],
    )
    state = {
        "enabled": True,
        "emit_mode": "all",
        "emit_summary": True,
        "visualization_type": "table",
        "method": "saliency",
    }
    event = LTTSEvent(
        event_type=metadata["event_types"][0],
        context=context,
        reactor=LTTSReactor(message_bus=bus),
        module_state=state,
        environment={"model": tiny_model, "tokenizer": ltts_model._tokenizer},
    )
    interceptor = getattr(module, "interceptor", None)
    if not callable(interceptor):
        raise TypeError("module must expose callable interceptor(ltts_event)")
    ntor_result = interceptor(event)
    if ntor_result is not None:
        json.dumps(ntor_result, default=str)
    for message in bus.messages:
        if not isinstance(message, dict):
            raise TypeError("artifact message must be a dictionary")
        json.dumps(message, default=str)

    dtor_name = methods.get("dtor")
    if dtor_name and callable(getattr(module, dtor_name, None)):
        _invoke(getattr(module, dtor_name), ctor_result)
    return {
        "module": name,
        "constructor": type(ctor_result).__name__,
        "runtime": type(ntor_result).__name__,
        "artifacts": len(bus.messages),
    }


def _run_one(root: Path, ltts_root: Path, name: str, timeout: int) -> tuple[dict | None, dict | None]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        name,
        "--root",
        str(root),
        "--ltts-root",
        str(ltts_root),
    ]
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "LTTS_SKIP_MODULE_VENVS": "1"},
        creationflags=creationflags,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        if process.returncode:
            return None, {
                "module": name,
                "phase": "runtime",
                "error": (stderr or stdout).strip()[-2000:],
            }
        return json.loads(stdout.strip().splitlines()[-1]), None
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        return None, {
            "module": name,
            "phase": "runtime",
            "error": f"timeout after {timeout}s",
        }


def run_parent(
    root: Path,
    ltts_root: Path,
    selected: set[str] | None,
    timeout: int,
    jobs: int,
) -> int:
    failures = validate_static(root, selected)
    names = [path.name for path in module_dirs(root) if not selected or path.name in selected]
    results = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        futures = {
            executor.submit(_run_one, root, ltts_root, name, timeout): name
            for name in names
        }
        for future in as_completed(futures):
            result, failure = future.result()
            if result:
                results.append(result)
            if failure:
                failures.append(failure)

    for result in results:
        print(
            f"PASS {result['module']:<32} runtime={result['runtime']:<10} "
            f"artifacts={result['artifacts']}"
        )
    for failure in failures:
        print(
            f"FAIL {failure['module']:<32} phase={failure['phase']}: "
            f"{failure['error']}",
            file=sys.stderr,
        )
    print(f"\n{len(results)}/{len(names)} module runtime checks passed")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("modules", nargs="*")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--ltts-root", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--worker", metavar="MODULE")
    args = parser.parse_args()
    root = args.root.resolve()
    ltts_root = (args.ltts_root or (root.parent / "LTTS")).resolve()
    if args.worker:
        try:
            print(json.dumps(validate_worker(root, ltts_root, args.worker)))
            return 0
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            return 1
    if not ltts_root.is_dir():
        print(f"LTTS source directory not found: {ltts_root}", file=sys.stderr)
        return 2
    return run_parent(root, ltts_root, set(args.modules) or None, args.timeout, args.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
