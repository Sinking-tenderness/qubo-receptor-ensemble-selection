from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_stage05_mk14_remote_bundle import write_bundle


STATIC_FILES = (
    "configs/stage96_multitarget_adaptive_docking_replay.json",
    "data/processed/stage96_multitarget_balanced_chemistry_batches.csv",
    "data/stage96_multitarget_balanced_chemistry_batches_summary.json",
    "data/stage96_multitarget_adaptive_docking_replay_result.json",
    "results/runs/stage96_multitarget_adaptive_docking_replay/trajectories.csv",
    "results/runs/stage96_multitarget_adaptive_docking_replay/checkpoints.csv",
    "results/runs/stage96_multitarget_adaptive_docking_replay/qubo_solver_comparisons.csv",
    "reports/stage-96/multitarget_adaptive_docking_replay.md",
    "scripts/run_stage96_multitarget_adaptive_docking_replay.py",
    "scripts/audit_stage96_multitarget_adaptive_docking_replay.py",
    "scripts/build_stage96_balanced_chemistry_batches.py",
    "tests/test_stage96_multitarget_adaptive_docking_replay.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deliverables/stage96_multitarget_adaptive_docking_replay_core_v1.tar.gz"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_bundle(root, args.output, list(STATIC_FILES))
    result.update(
        {
            "operation": "Stage96 hidden-matrix adaptive docking replay",
            "targets": ["PPARG", "BACE1"],
            "synthetic_scores": 0,
            "new_docking_jobs": 0,
            "fresh_validation_rows": 0,
            "quantum_jobs": 0,
            "policy_gate_passes": json.loads(
                (root / "data/stage96_multitarget_adaptive_docking_replay_result.json").read_text(encoding="utf-8")
            )["policy_gate"]["passes"],
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
