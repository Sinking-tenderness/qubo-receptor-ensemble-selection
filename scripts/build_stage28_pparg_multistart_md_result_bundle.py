"""Build the Stage28 core result bundle after all remote MD runs pass audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/stage28_pparg_multistart_md_ensemble.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = root / args.config
    config = json.loads(config_path.read_text(encoding="ascii"))
    with (root / config["runtime"]["start_manifest"]).open("r", encoding="utf-8", newline="") as handle:
        starts = list(csv.DictReader(handle))
    audit_path = root / config["outputs"]["audit_json"]
    audit = json.loads(audit_path.read_text(encoding="ascii"))
    if audit.get("status") != "stage28_pparg_multistart_md_ensemble_audit_ok":
        raise ValueError("Stage28 result bundle requires a passing audit")
    paths = {
        config_path.relative_to(root).as_posix(),
        config["runtime"]["start_manifest"],
        config["outputs"]["preparation_result_json"],
        config["outputs"]["frame_manifest_csv"],
        config["outputs"]["feature_archive_npz"],
        config["outputs"]["distance_archive_npz"],
        config["outputs"]["ensemble_summary_json"],
        config["outputs"]["audit_json"],
        "scripts/collect_stage28_pparg_md_ensemble.py",
        "scripts/audit_stage28_pparg_multistart_md_ensemble.py",
        "scripts/build_stage28_pparg_multistart_md_result_bundle.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/run_stage28_pparg_multistart_md.py",
        "scripts/run_stage21_structure_aware_qubo.py",
        "scripts/cluster_md_pocket_frames.py",
        "scripts/__init__.py",
        "reports/stage-28/pparg_multistart_md_remote_execution.md",
        "pyproject.toml",
    }
    for row in starts:
        for key in ("protocol_config", "equilibration_config", "production_config", "trajectory_qc_config", "system_manifest", "equilibration_manifest", "production_manifest", "trajectory_qc_summary"):
            paths.add(row[key])
        base = Path(row["trajectory_qc_summary"]).parent
        paths.add((base / "frame_metrics.csv").as_posix())
        paths.add((base / "residue_ca_rmsf.csv").as_posix())
    result = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
