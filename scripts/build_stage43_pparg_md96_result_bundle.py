"""Build the pose-free Stage43 PPARG MD-96 result bundle."""

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
    config_path = root / "configs/stage43_pparg_md96_rank_sensitive_replication.json"
    config = json.loads(config_path.read_text(encoding="ascii"))
    summary = root / config["outputs"]["summary_json"]
    audit = root / config["outputs"]["audit_json"]
    if json.loads(summary.read_text(encoding="ascii")).get("status") != "stage43_pparg_md96_unidock_matrix_ok":
        raise ValueError("Stage43 matrix is incomplete")
    if json.loads(audit.read_text(encoding="ascii")).get("status") != "stage43_pparg_md96_unidock_matrix_independent_audit_ok":
        raise ValueError("Stage43 independent audit did not pass")
    paths = {
        "configs/stage43_pparg_md96_rank_sensitive_replication.json",
        "configs/stage43_pparg_md96_technical_rescue_amendment01.json",
        "scripts/prepare_stage43_pparg_md96_inputs.py",
        "scripts/experimental/unidock/run_stage43_pparg_md96_production.py",
        "scripts/audit_stage43_pparg_md96_production.py",
        "data/processed/stage43_pparg_md96_frame_manifest.csv",
        "data/processed/stage43_pparg_md96_prepared_receptor_manifest.csv",
        "data/stage43_pparg_md96_input_preparation_result.json",
        config["outputs"]["summary_json"], config["outputs"]["audit_json"],
        config["outputs"]["scores_csv"], config["outputs"]["batch_runs_csv"],
        config["outputs"]["median_matrix_csv"], config["outputs"]["minimum_matrix_csv"],
        config["outputs"]["progress_json"],
    }
    run_directory = root / config["outputs"]["run_directory"]
    environment_directory = run_directory / "environment"
    if environment_directory.is_dir():
        paths.update(
            path.relative_to(root).as_posix()
            for path in environment_directory.rglob("*")
            if path.is_file()
        )
    for pattern in ("batch_summary.json", "scores.csv", "unidock.log"):
        paths.update(path.relative_to(root).as_posix() for path in (run_directory / "batches").glob(f"*/*/{pattern}"))
    result = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
