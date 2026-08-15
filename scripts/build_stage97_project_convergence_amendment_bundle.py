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
    "configs/stage89_project_convergence.json",
    "configs/stage97_project_convergence_amendment_stage96.json",
    "data/stage89_project_convergence_result.json",
    "data/stage89_project_convergence_audit.json",
    "results/runs/stage89_project_convergence/claim_evidence_matrix.csv",
    "reports/stage-89/project_convergence.md",
    "data/stage96_multitarget_adaptive_docking_replay_result.json",
    "data/stage96_multitarget_adaptive_docking_replay_audit.json",
    "reports/stage-96/multitarget_adaptive_docking_replay.md",
    "data/stage97_project_convergence_amendment_stage96_result.json",
    "data/stage97_project_convergence_amendment_stage96_audit.json",
    "results/runs/stage97_project_convergence_amendment_stage96/claim_evidence_matrix.csv",
    "reports/stage-97/project_convergence_amendment_stage96.md",
    "scripts/run_stage97_project_convergence_amendment_stage96.py",
    "scripts/audit_stage96_multitarget_adaptive_docking_replay.py",
    "tests/test_stage96_multitarget_adaptive_docking_replay.py",
    "tests/test_stage97_project_convergence_amendment_stage96.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_bundle(args.root.resolve(), args.output, list(FILES))
    result.update({"operation": "Stage97 project convergence amendment with Stage96 evidence", "new_docking_jobs": 0, "quantum_jobs": 0, "policy_gate_passes": False})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
