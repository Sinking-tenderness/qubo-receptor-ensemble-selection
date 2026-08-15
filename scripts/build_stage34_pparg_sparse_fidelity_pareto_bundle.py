"""Build the deterministic Stage34 sparse-fidelity Pareto core bundle."""

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
    config_path = root / "configs/stage34_pparg_sparse_fidelity_pareto.json"
    config = json.loads(config_path.read_text(encoding="ascii"))
    result = json.loads((root / config["outputs"]["result_json"]).read_text(encoding="ascii"))
    audit = json.loads((root / config["outputs"]["audit_json"]).read_text(encoding="ascii"))
    if result.get("status") != "stage34_pparg_sparse_fidelity_pareto_complete":
        raise ValueError("Stage34 result is incomplete")
    if audit.get("status") != "stage34_pparg_sparse_fidelity_pareto_audit_ok":
        raise ValueError("Stage34 audit is incomplete")
    paths = {
        "configs/stage34_pparg_sparse_fidelity_pareto.json",
        "scripts/run_stage34_pparg_sparse_fidelity_pareto.py",
        "scripts/audit_stage34_pparg_sparse_fidelity_pareto.py",
        "scripts/build_stage34_pparg_sparse_fidelity_pareto_bundle.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/run_stage21_structure_aware_qubo.py",
        "scripts/run_stage29_pparg_md_qubo_scaling.py",
        "scripts/run_stage30_pparg_group_balanced_state_qubo.py",
        "scripts/run_stage33_pparg_sparse_hardware_qubo.py",
        "scripts/__init__.py",
        "tests/test_stage34_pparg_sparse_fidelity_pareto.py",
        "pyproject.toml",
    }
    paths.update(str(value).replace("\\", "/") for value in config["inputs"].values())
    paths.update(str(value).replace("\\", "/") for value in config["outputs"].values())
    bundle = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
