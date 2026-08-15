from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_stage05_mk14_remote_bundle import write_bundle


FILES = (
    "configs/stage101_marginal_transfer_gate.json",
    "data/stage101_marginal_transfer_gate_result.json",
    "data/stage101_marginal_transfer_gate_audit.json",
    "results/runs/stage101_marginal_transfer_gate/marginal_edges.csv",
    "results/runs/stage101_marginal_transfer_gate/policy_target_summary.csv",
    "results/runs/stage101_marginal_transfer_gate/policy_fold_decisions.csv",
    "reports/stage-101/marginal_transfer_gate.md",
    "reports/stage-101/next_optimization_spec.md",
    "scripts/run_stage101_marginal_transfer_gate.py",
    "scripts/audit_stage101_marginal_transfer_gate.py",
    "tests/test_stage101_marginal_transfer_gate.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_bundle(args.root.resolve(), args.output, list(FILES))
    result.update({"operation": "Stage101 marginal-transfer gate", "new_docking_jobs": 0, "quantum_jobs": 0, "hardware_authorized": False})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
