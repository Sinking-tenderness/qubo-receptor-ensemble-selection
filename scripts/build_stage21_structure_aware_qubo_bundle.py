"""Build a deterministic core bundle for the Stage 21 structural QUBO screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config_path = root / "configs/stage21_structure_aware_qubo.json"
    config = read_json(config_path)
    paths = {
        "configs/stage21_structure_aware_qubo.json",
        "scripts/run_stage21_structure_aware_qubo.py",
        "scripts/audit_stage21_structure_aware_qubo.py",
        "scripts/__init__.py",
        "tests/test_stage21_structure_aware_qubo.py",
        "data/stage21_structure_aware_qubo_result.json",
        "data/stage21_structure_aware_qubo_audit.json",
        "data/stage21_structure_aware_qubo_model_record.json",
        "reports/stage-21/structure_aware_qubo.md",
        "pyproject.toml",
    }
    for spec in config["targets"].values():
        for relative in spec["inputs"].values():
            paths.add(str(relative).replace("\\", "/"))
    for relative in config["outputs"].values():
        paths.add(str(relative).replace("\\", "/"))
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_bundle(args.root, args.output, bundle_paths(args.root))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
