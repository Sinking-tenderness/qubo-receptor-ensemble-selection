import json
from pathlib import Path

from scripts.experimental.vinagpu.build_vinagpu_train160_bundle import (
    CONFIG,
    FIXED_PATHS,
    bundle_paths,
)
from scripts.experimental.vinagpu.package_vinagpu_results import result_paths


def test_vinagpu_bundle_contains_frozen_inputs_and_remote_entrypoint():
    paths = bundle_paths(Path.cwd())

    assert CONFIG in paths
    assert (
        "scripts/experimental/vinagpu/run_vinagpu_train160_remote.sh"
        in FIXED_PATHS
    )
    assert (
        "data/processed/stage05_mk14_unidock_rigid_train160_pdbqt_manifest.csv"
        in paths
    )
    assert sum(path.endswith(".pdbqt") for path in paths) == 165
    assert not any("fresh_validation" in path for path in paths)


def test_result_packager_keeps_core_and_targeted_poses_only(tmp_path: Path):
    run = tmp_path / "results/runs/stage06"
    smoke = run / "compatibility_smoke"
    diagnostic = run / "diagnostic_poses"
    environment = run / "environment"
    for directory in (smoke, diagnostic, environment):
        directory.mkdir(parents=True)
    output_names = {
        "runtime_lock_json": "runtime_lock.json",
        "gpu_scores_csv": "gpu_scores.csv",
        "gpu_pair_runs_csv": "gpu_pair_runs.csv",
        "gpu_summary_json": "gpu_run_summary.json",
        "pairwise_comparison_csv": "equivalence_pairwise.csv",
        "group_metrics_csv": "equivalence_group_metrics.csv",
        "equivalence_summary_json": "equivalence_summary.json",
    }
    outputs = {
        "run_directory": "results/runs/stage06",
        "diagnostic_pose_directory": "results/runs/stage06/diagnostic_poses",
        **{
            key: f"results/runs/stage06/{name}"
            for key, name in output_names.items()
        },
    }
    config_path = (
        tmp_path / "configs/stage06_mk14_vinagpu21_train160_equivalence.json"
    )
    config_path.parent.mkdir()
    config_path.write_text(json.dumps({"outputs": outputs}), encoding="ascii")
    for name in output_names.values():
        (run / name).write_text("placeholder\n", encoding="ascii")
    (run / "gpu_run_summary.json").write_text(
        json.dumps({"status": "ok", "gpu_pair_count": 2400}),
        encoding="ascii",
    )
    (run / "equivalence_summary.json").write_text(
        json.dumps({"status": "gpu_equivalence_gate_failed"}),
        encoding="ascii",
    )
    for name in (
        "compatibility_smoke_summary.json",
        "compatibility_smoke/pair_summary.json",
        "compatibility_smoke/vinagpu.log",
        "compatibility_smoke/pose_out.pdbqt",
        "diagnostic_poses/outlier.pdbqt",
        "environment/nvidia_smi.csv",
    ):
        path = run / name
        path.write_text("evidence\n", encoding="ascii")
    raw_pose = run / "pairs/seed0/R/L/pose_out.pdbqt"
    raw_pose.parent.mkdir(parents=True)
    raw_pose.write_text("raw\n", encoding="ascii")

    paths = result_paths(tmp_path.resolve())

    assert "results/runs/stage06/diagnostic_poses/outlier.pdbqt" in paths
    assert "results/runs/stage06/environment/nvidia_smi.csv" in paths
    assert "results/runs/stage06/pairs/seed0/R/L/pose_out.pdbqt" not in paths
