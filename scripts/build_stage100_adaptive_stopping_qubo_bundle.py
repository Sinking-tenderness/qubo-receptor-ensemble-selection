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
    "configs/stage100_adaptive_stopping_qubo.json",
    "data/stage100_adaptive_stopping_qubo_result.json",
    "data/stage100_adaptive_stopping_qubo_audit.json",
    "results/runs/stage100_adaptive_stopping_qubo/fold_metrics.csv",
    "results/runs/stage100_adaptive_stopping_qubo/inner_k_profiles.csv",
    "results/runs/stage100_adaptive_stopping_qubo/target_summary.csv",
    "reports/stage-100/adaptive_stopping_qubo.md",
    "scripts/run_stage100_adaptive_stopping_qubo.py",
    "scripts/audit_stage100_adaptive_stopping_qubo.py",
    "tests/test_stage100_adaptive_stopping_qubo.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_bundle(args.root.resolve(), args.output, list(FILES))
    result.update({"operation": "Stage100 adaptive stopping QUBO", "new_docking_jobs": 0, "quantum_jobs": 0, "go_gate_passes": False})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
