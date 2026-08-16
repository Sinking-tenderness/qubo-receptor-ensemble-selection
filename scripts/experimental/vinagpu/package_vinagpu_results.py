"""Package the compact audited Stage 6 Vina-GPU result evidence."""

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


CONFIG = "configs/stage06_mk14_vinagpu21_train160_equivalence.json"


def result_paths(root: Path) -> list[str]:
    config = read_json(root / CONFIG)
    outputs = config["outputs"]
    required_keys = (
        "runtime_lock_json",
        "gpu_scores_csv",
        "gpu_pair_runs_csv",
        "gpu_summary_json",
        "pairwise_comparison_csv",
        "group_metrics_csv",
        "equivalence_summary_json",
    )
    paths = [CONFIG]
    paths.extend(str(outputs[key]) for key in required_keys)
    run_directory = rooted_path(root, str(outputs["run_directory"]))
    smoke_directory = run_directory / "compatibility_smoke"
    paths.extend(
        relative.as_posix()
        for relative in (
            (run_directory / "compatibility_smoke_summary.json").relative_to(root),
            (smoke_directory / "pair_summary.json").relative_to(root),
            (smoke_directory / "vinagpu.log").relative_to(root),
            (smoke_directory / "pose_out.pdbqt").relative_to(root),
        )
    )
    diagnostic_directory = rooted_path(
        root, str(outputs["diagnostic_pose_directory"])
    )
    paths.extend(
        path.relative_to(root).as_posix()
        for path in sorted(diagnostic_directory.glob("*.pdbqt"))
    )
    environment_directory = run_directory / "environment"
    if environment_directory.is_dir():
        paths.extend(
            path.relative_to(root).as_posix()
            for path in sorted(environment_directory.iterdir())
            if path.is_file()
        )

    gpu_summary = read_json(rooted_path(root, str(outputs["gpu_summary_json"])))
    if gpu_summary.get("status") != "ok" or int(gpu_summary["gpu_pair_count"]) != 2400:
        raise ValueError("Vina-GPU execution is not complete")
    equivalence = read_json(
        rooted_path(root, str(outputs["equivalence_summary_json"]))
    )
    allowed_statuses = {
        "gpu_equivalence_gate_passed",
        "gpu_equivalence_gate_failed",
    }
    if equivalence.get("status") not in allowed_statuses:
        raise ValueError("Vina-GPU equivalence audit is missing or invalid")
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
            "operation": "compact audited Stage 6 Vina-GPU Train-160 evidence",
            "gpu_pair_count": 2400,
            "raw_pair_logs_and_all_poses_included": False,
            "largest_delta_diagnostic_poses_included": True,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
