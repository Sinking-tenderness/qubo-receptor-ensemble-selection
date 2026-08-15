"""Build the deterministic Stage33 sparse PPARG hardware-QUBO core bundle."""

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
    config_path = root / "configs/stage33_pparg_sparse_hardware_qubo.json"
    config = json.loads(config_path.read_text(encoding="ascii"))
    result = json.loads((root / config["outputs"]["result_json"]).read_text(encoding="ascii"))
    audit = json.loads((root / config["outputs"]["audit_json"]).read_text(encoding="ascii"))
    if result.get("status") != "stage33_pparg_sparse_hardware_qubo_complete":
        raise ValueError("Stage33 result is incomplete")
    if audit.get("status") != "stage33_pparg_sparse_hardware_qubo_audit_ok":
        raise ValueError("Stage33 audit is incomplete")
    paths = {
        "configs/stage33_pparg_sparse_hardware_qubo.json",
        "scripts/run_stage33_pparg_sparse_hardware_qubo.py",
        "scripts/audit_stage33_pparg_sparse_hardware_qubo.py",
        "scripts/build_stage33_pparg_sparse_hardware_qubo_bundle.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/run_stage21_structure_aware_qubo.py",
        "scripts/run_stage29_pparg_md_qubo_scaling.py",
        "scripts/run_stage30_pparg_group_balanced_state_qubo.py",
        "scripts/__init__.py",
        "tests/test_stage33_pparg_sparse_hardware_qubo.py",
        "pyproject.toml",
        "results/runs/stage30_pparg_group_balanced_state_qubo/solver_results.csv",
    }
    paths.update(str(value).replace("\\", "/") for value in config["inputs"].values())
    paths.update(str(value).replace("\\", "/") for value in config["outputs"].values())
    bundle = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
