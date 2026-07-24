"""Package compact deterministic-batch Vina-GPU bridge evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from .run_vinagpu_equivalence import read_json, rooted_path
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from run_vinagpu_equivalence import read_json, rooted_path


CONFIG = "configs/stage06_mk14_vinagpu21_deterministic_batch_bridge.json"


def result_paths(root: Path) -> list[str]:
    config = read_json(root / CONFIG)
    outputs = config["outputs"]
    paths = [
        CONFIG,
        "data/stage06_mk14_vinagpu21_train160_v1_result_summary.json",
    ]
    for key in (
        "runtime_lock_json",
        "preflight_summary_json",
        "batch_scores_csv",
        "batch_runs_csv",
        "batch_summary_json",
        "bridge_summary_json",
    ):
        paths.append(str(outputs[key]))
    run_directory = rooted_path(root, str(outputs["run_directory"]))
    preflight = config["preflight"]
    preflight_directory = (
        run_directory
        / "chunks"
        / str(preflight["seed_id"])
        / str(preflight["receptor_id"])
        / f"chunk_{int(preflight['chunk_index']):03d}"
    )
    paths.extend(
        path.relative_to(root).as_posix()
        for path in sorted(preflight_directory.rglob("*"))
        if path.is_file() and not path.is_symlink()
    )
    environment_directory = run_directory / "environment"
    if environment_directory.is_dir():
        paths.extend(
            path.relative_to(root).as_posix()
            for path in sorted(environment_directory.iterdir())
            if path.is_file()
        )
    run_summary = read_json(rooted_path(root, str(outputs["batch_summary_json"])))
    bridge = read_json(rooted_path(root, str(outputs["bridge_summary_json"])))
    if run_summary.get("status") != "ok" or int(run_summary["pair_count"]) != 2400:
        raise ValueError("deterministic batch execution is incomplete")
    if bridge.get("status") not in {
        "deterministic_batch_bridge_passed",
        "deterministic_batch_bridge_failed",
    }:
        raise ValueError("deterministic batch bridge audit is invalid")
    return sorted(set(paths))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_bundle(root, args.output, result_paths(root))
    result.update(
        {
            "operation": "compact deterministic Vina-GPU batch bridge evidence",
            "pair_count": 2400,
            "all_raw_chunk_logs_and_poses_included": False,
            "preflight_chunk_evidence_included": True,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
