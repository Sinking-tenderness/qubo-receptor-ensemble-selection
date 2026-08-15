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
    "configs/stage102_prospective_marginal_learning.json",
    "data/stage102_prospective_marginal_learning_readiness.json",
    "reports/stage-102/prospective_marginal_learning_plan.md",
    "scripts/audit_stage102_prospective_marginal_learning_readiness.py",
    "tests/test_stage102_prospective_marginal_learning.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_bundle(args.root.resolve(), args.output, list(FILES))
    result.update({
        "operation": "Stage102 prospective marginal-learning preregistration and readiness",
        "phase_a_pair_budget": 45000,
        "phase_b_released": False,
        "new_docking_jobs_started": 0,
        "quantum_jobs": 0,
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
