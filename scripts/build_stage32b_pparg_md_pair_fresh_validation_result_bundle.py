"""Build the pose-free Stage32b PPARG fresh-validation result bundle."""

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
    config_path = root / "configs/stage32b_pparg_md_pair_fresh_validation.json"
    config = json.loads(config_path.read_text(encoding="ascii"))
    summary = json.loads((root / config["outputs"]["summary_json"]).read_text(encoding="ascii"))
    evaluation = json.loads((root / config["outputs"]["evaluation_json"]).read_text(encoding="ascii"))
    if summary.get("status") != "stage32b_pparg_md_pair_fresh_validation_matrix_ok":
        raise ValueError("Stage32b matrix is not complete")
    if evaluation.get("status") != "stage32b_pparg_md_pair_fresh_validation_evaluation_complete":
        raise ValueError("Stage32b evaluation is not complete")
    paths = {
        "configs/stage32b_pparg_md_pair_fresh_validation.json",
        "scripts/evaluate_stage32b_pparg_md_pair_fresh_validation.py",
        "scripts/experimental/unidock/run_stage32b_pparg_md_pair_fresh_validation.py",
        "scripts/build_stage32b_pparg_md_pair_fresh_validation_result_bundle.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/stage32b_common.py",
        "scripts/__init__.py",
        "scripts/experimental/__init__.py",
        "scripts/experimental/unidock/__init__.py",
        "tests/test_stage32b_pparg_md_pair_fresh_validation.py",
        "pyproject.toml",
        config["inputs"]["stage32_scores_csv"],
        config["inputs"]["stage32_median_matrix_csv"],
        config["inputs"]["stage32_minimum_matrix_csv"],
        config["inputs"]["stage32_train_ligand_manifest"],
        config["inputs"]["stage32_prepared_receptor_manifest"],
        config["outputs"]["train_selection_json"],
        config["outputs"]["fresh_validation_source_manifest"],
        config["outputs"]["prepared_ligand_manifest"],
        config["outputs"]["selected_receptor_manifest"],
        config["outputs"]["preparation_result"],
        config["outputs"]["scores_csv"],
        config["outputs"]["batch_runs_csv"],
        config["outputs"]["median_matrix_csv"],
        config["outputs"]["minimum_matrix_csv"],
        config["outputs"]["progress_json"],
        config["outputs"]["summary_json"],
        config["outputs"]["evaluation_json"],
        config["outputs"]["report_md"],
    }
    batch_root = root / config["outputs"]["run_directory"] / "batches"
    for pattern in ("batch_summary.json", "scores.csv", "unidock.log"):
        paths.update(path.relative_to(root).as_posix() for path in batch_root.glob(f"*/*/{pattern}"))
    result = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
