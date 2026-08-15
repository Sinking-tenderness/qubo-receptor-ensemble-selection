"""Build the deterministic Stage26 core bundle."""

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
    config_path = root / "configs/stage26_variable_budget_consensus_qubo.json"
    config = json.loads(config_path.read_text(encoding="ascii"))
    paths = {
        "configs/stage26_variable_budget_consensus_qubo.json",
        "scripts/run_stage26_variable_budget_consensus_qubo.py",
        "scripts/audit_stage26_variable_budget_consensus_qubo.py",
        "scripts/build_stage26_variable_budget_consensus_qubo_bundle.py",
        "scripts/run_stage22_structural_state_coverage_qubo.py",
        "scripts/run_stage21_structure_aware_qubo.py",
        "scripts/__init__.py",
        "tests/test_stage26_variable_budget_consensus_qubo.py",
        "data/stage26_variable_budget_consensus_qubo_result.json",
        "data/stage26_variable_budget_consensus_qubo_audit.json",
        "reports/stage-26/variable_budget_consensus_qubo.md",
        "pyproject.toml",
    }
    for spec in config["targets"].values():
        paths.update(str(value).replace("\\", "/") for value in spec["inputs"].values())
    paths.update(str(value).replace("\\", "/") for value in config["outputs"].values())
    result = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
