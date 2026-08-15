"""Build the deterministic Stage37 robust-functional-QUBO core bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = root / "configs/stage37_cross_target_robust_functional_qubo.json"
    config = json.loads(config_path.read_text(encoding="ascii"))
    result = json.loads((root / config["outputs"]["result_json"]).read_text(encoding="ascii"))
    audit = json.loads((root / config["outputs"]["audit_json"]).read_text(encoding="ascii"))
    if result.get("status") != "stage37_cross_target_robust_functional_qubo_complete":
        raise ValueError("Stage37 result is incomplete")
    if audit.get("status") != "stage37_cross_target_robust_functional_qubo_audit_ok":
        raise ValueError("Stage37 audit is incomplete")
    paths = {
        "configs/stage37_cross_target_robust_functional_qubo.json",
        "scripts/run_stage37_cross_target_robust_functional_qubo.py",
        "scripts/audit_stage37_cross_target_robust_functional_qubo.py",
        "scripts/build_stage37_cross_target_robust_functional_qubo_bundle.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/diagnose_stage19e_cross_target_qubo_v2.py",
        "scripts/screen_stage10_mk14_expanded16_qubo_greedy.py",
        "scripts/run_stage05_mk14_method_gate.py",
        "scripts/prepare_receptor.py",
        "scripts/__init__.py",
        "tests/test_stage37_cross_target_robust_functional_qubo.py",
        "pyproject.toml",
    }
    paths.update(str(value["path"]).replace("\\", "/") for value in config["inputs"].values())
    paths.update(str(value).replace("\\", "/") for value in config["outputs"].values())
    bundle = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
