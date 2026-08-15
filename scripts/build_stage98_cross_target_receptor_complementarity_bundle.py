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
    "configs/stage98_cross_target_receptor_complementarity.json",
    "data/stage98_cross_target_receptor_complementarity_result.json",
    "data/stage98_cross_target_receptor_complementarity_audit.json",
    "results/runs/stage98_cross_target_receptor_complementarity/fold_metrics.csv",
    "results/runs/stage98_cross_target_receptor_complementarity/target_summary.csv",
    "results/runs/stage98_cross_target_receptor_complementarity/receptor_pair_complementarity.csv",
    "reports/stage-98/cross_target_receptor_complementarity.md",
    "scripts/run_stage98_cross_target_receptor_complementarity.py",
    "scripts/audit_stage98_cross_target_receptor_complementarity.py",
    "tests/test_stage98_cross_target_receptor_complementarity.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_bundle(args.root.resolve(), args.output, list(FILES))
    result.update({"operation": "Stage98 cross-target receptor complementarity analysis", "targets": ["MK14", "PPARG", "BACE1", "PPARA", "PPARD"], "new_docking_jobs": 0, "quantum_jobs": 0, "go_gate_passes": False})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
