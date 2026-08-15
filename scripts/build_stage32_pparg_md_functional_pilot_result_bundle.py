"""Build a pose-free Stage32 PPARG MD functional-pilot result bundle."""

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
    config_path = root / "configs/stage32_pparg_md_functional_complementarity_pilot.json"
    config = json.loads(config_path.read_text(encoding="ascii"))
    outputs = config["outputs"]
    summary_path = root / outputs["summary_json"]
    if not summary_path.is_file() or json.loads(summary_path.read_text(encoding="ascii")).get("status") != "stage32_pparg_md_functional_pilot_matrix_ok":
        raise ValueError("Stage32 matrix is not complete")
    paths = {
        "configs/stage32_pparg_md_functional_complementarity_pilot.json",
        "scripts/prepare_stage32_pparg_md_functional_pilot.py",
        "scripts/experimental/unidock/run_stage32_pparg_md_functional_pilot.py",
        "scripts/build_stage32_pparg_md_functional_pilot_result_bundle.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "data/processed/stage32_pparg_md_selected16_frame_manifest.csv",
        "data/processed/stage32_pparg_train160_ligand_manifest.csv",
        "data/processed/stage32_pparg_md_selected16_prepared_receptor_manifest.csv",
        "data/stage32_pparg_md_functional_pilot_input_preparation_result.json",
    }
    for key in ("scores_csv", "batch_runs_csv", "median_matrix_csv", "minimum_matrix_csv", "progress_json", "summary_json"):
        paths.add(outputs[key])
    run_directory = root / outputs["run_directory"] / "batches"
    for pattern in ("batch_summary.json", "scores.csv", "unidock.log"):
        paths.update(path.relative_to(root).as_posix() for path in run_directory.glob(f"*/*/{pattern}"))
    result = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
