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
    "configs/stage99_qubo_objective_repair_screen.json",
    "data/stage99_qubo_objective_repair_screen_result.json",
    "data/stage99_qubo_objective_repair_screen_audit.json",
    "results/runs/stage99_qubo_objective_repair_screen/fold_metrics.csv",
    "results/runs/stage99_qubo_objective_repair_screen/target_summary.csv",
    "results/runs/stage99_qubo_objective_repair_screen/solver_diagnostics.csv",
    "results/runs/stage99_qubo_objective_repair_screen/adaptive_k_metrics.csv",
    "reports/stage-99/qubo_objective_repair_screen.md",
    "reports/stage-99/qubo_failure_diagnosis_and_objective_repair.md",
    "scripts/run_stage99_qubo_objective_repair_screen.py",
    "scripts/audit_stage99_qubo_objective_repair_screen.py",
    "tests/test_stage99_qubo_objective_repair_screen.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_bundle(args.root.resolve(), args.output, list(FILES))
    result.update({
        "operation": "Stage99 QUBO failure diagnosis and objective-repair screen",
        "targets": ["MK14", "PPARG", "BACE1", "PPARA", "PPARD"],
        "new_docking_jobs": 0,
        "quantum_jobs": 0,
        "go_gate_passes": False,
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
