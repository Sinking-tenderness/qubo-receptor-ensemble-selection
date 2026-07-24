"""Package compact Vina-GPU search-depth diagnostic evidence."""

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


CONFIG = "configs/stage06_mk14_vinagpu21_search_depth_diagnostic.json"


def result_paths(root: Path) -> tuple[list[str], dict[str, object]]:
    config = read_json(root / CONFIG)
    outputs = config["outputs"]
    diagnostic_path = rooted_path(root, str(outputs["diagnostic_summary_json"]))
    diagnostic = read_json(diagnostic_path)
    if diagnostic.get("status") not in {
        "search_depth_candidate_selected",
        "search_depth_diagnostic_failed",
    }:
        raise ValueError("search-depth diagnostic summary is invalid")
    paths = [
        CONFIG,
        "data/stage06_mk14_vinagpu21_train160_v1_result_summary.json",
        "data/stage06_mk14_vinagpu21_deterministic_batch_bridge_result_summary.json",
        "data/stage06_mk14_vinagpu21_deterministic_batch_runtime_lock.json",
        str(outputs["runtime_lock_json"]),
        str(outputs["preflight_summary_json"]),
        str(outputs["diagnostic_summary_json"]),
    ]
    run_directory = rooted_path(root, str(outputs["run_directory"]))
    for item in diagnostic["executed_profiles"]:
        profile_directory = run_directory / "profiles" / str(item["profile_id"])
        for name in (
            "profile_scores.csv",
            "chunk_runs.csv",
            "group_metrics.csv",
            "profile_summary.json",
        ):
            path = profile_directory / name
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path.relative_to(root).as_posix())
        profile_summary = read_json(profile_directory / "profile_summary.json")
        if int(profile_summary["pair_count"]) != 320:
            raise ValueError("packaged search-depth profile is incomplete")

    preflight = config["preflight"]
    preflight_directory = (
        run_directory
        / "profiles"
        / str(preflight["profile_id"])
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
    return sorted(set(paths)), diagnostic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    paths, diagnostic = result_paths(root)
    result = write_bundle(root, args.output, paths)
    result.update(
        {
            "operation": "compact targeted Vina-GPU search-depth diagnostic evidence",
            "diagnostic_status": diagnostic["status"],
            "selected_profile": diagnostic["selected_profile"],
            "executed_profile_count": len(diagnostic["executed_profiles"]),
            "all_raw_chunk_logs_and_poses_included": False,
            "preflight_chunk_evidence_included": True,
            "validation_rows": 0,
            "test_rows": 0,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
