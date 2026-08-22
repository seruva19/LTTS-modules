#!/usr/bin/env python3
"""Regenerate registry.json from module METADATA.

Run from the repo root:  python generate_registry.py
"""

import ast
import json
import re
from datetime import datetime
from pathlib import Path

REPO = "LTTS-modules"
OWNER = "seruva19"

ROOT = Path(__file__).parent.parent


def parse_metadata(init_path: Path) -> dict:
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "METADATA":
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:
                        return {}
    return {}


def folder_size(path: Path) -> int:
    import os
    total = 0
    skip = {"venv", "__pycache__", ".git"}
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for filename in filenames:
            total += os.path.getsize(os.path.join(dirpath, filename))
    return total


def main():
    modules = []
    for module_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        init = module_dir / "init.py"
        if not init.exists():
            continue
        meta = parse_metadata(init)
        if not meta:
            print(f"  ! no METADATA: {module_dir.name}")
            continue
        versions = [meta.get("version", "0.0.0")]
        modules.append({
            "name": module_dir.name,
            "version": meta.get("version", "0.0.0"),
            "description": meta.get("description", ""),
            "author": meta.get("author", "LTTS"),
            "event_types": meta.get("event_types", []),
            "dependencies": meta.get("dependencies", []),
            "tags": meta.get("tags", [t for t in [meta.get("category", "")] if t]),
            "repository_url": f"https://github.com/{OWNER}/{REPO}",
            "module_path": module_dir.name,
            "size_bytes": folder_size(module_dir),
            "created_at": meta.get("created_at", "2026-01-01T00:00:00"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        print(f"  + {module_dir.name} {meta.get('version')}")

    registry = {"modules": modules}
    out = ROOT / "registry.json"
    out.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(f"\nregistry.json: {len(modules)} modules -> {out}")


if __name__ == "__main__":
    main()
