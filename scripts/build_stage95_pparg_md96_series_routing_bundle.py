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
    "configs/stage95_pparg_md96_series_routing_scaling.json",
    "data/processed/stage95_pparg_active_series_manifest.csv",
    "data/stage95_pparg_active_series_summary.json",
    "data/stage95_pparg_md96_series_routing_scaling_result.json",
    "data/stage95_pparg_md96_series_routing_bound_adjudication.json",
    "data/stage95_pparg_md96_series_routing_scaling_audit.json",
    "results/runs/stage95_pparg_md96_series_routing_scaling/scale_results.csv",
    "results/runs/stage95_pparg_md96_series_routing_scaling/solutions.csv",
    "reports/stage-95/pparg_md96_series_routing_scaling.md",
    "scripts/build_stage95_pparg_active_series.py",
    "scripts/run_stage95_pparg_md96_series_routing_scaling.py",
    "scripts/adjudicate_stage95_solver_bounds.py",
    "scripts/audit_stage95_pparg_md96_series_routing_scaling.py",
    "tests/test_stage95_pparg_md96_series_routing_scaling.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "deliverables/stage95_pparg_md96_series_routing_scaling_core_v1.tar.gz"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    checkpoints = sorted(
        path.relative_to(root).as_posix()
        for path in (
            root
            / "results/runs/stage95_pparg_md96_series_routing_scaling/checkpoints"
        ).glob("*.json")
    )
    result = write_bundle(root, args.output, list(STATIC_FILES) + checkpoints)
    result.update(
        {
            "operation": "Stage95 PPARG MD-96 real-matrix series-routing solver scaling",
            "real_score_rows": 46080,
            "synthetic_scores": 0,
            "exact_scale_count": 4,
            "bounded_nonexact_scale_count": 1,
            "one_percent_gap_excluded_scale_count": 5,
            "quantum_hardware_authorized": False,
            "new_docking_jobs": 0,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
