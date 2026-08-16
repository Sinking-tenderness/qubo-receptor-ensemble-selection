import json
from pathlib import Path

from scripts.experimental.vinagpu.build_vinagpu_search_depth_bundle import (
    CONFIG,
    FIXED_PATHS,
    bundle_paths,
)
from scripts.experimental.vinagpu.package_vinagpu_search_depth_results import (
    result_paths,
)


def test_search_depth_bundle_is_self_contained_and_train_only():
    paths = bundle_paths(Path.cwd())

    assert CONFIG in paths
    assert "data/stage06_mk14_vinagpu21_deterministic_batch_runtime_lock.json" in paths
    assert (
        "scripts/experimental/vinagpu/run_vinagpu_search_depth_remote.sh" in FIXED_PATHS
    )
    assert sum(path.endswith(".pdbqt") for path in paths) == 165
    assert not any("fresh_validation" in path for path in paths)


def test_result_bundle_keeps_profile_tables_and_preflight_raw_evidence(
    tmp_path: Path,
):
    run = tmp_path / "results/runs/search_depth"
    profile = run / "profiles/fixed_depth_16"
    preflight = profile / "chunks/seed1/R1/chunk_000"
    other = profile / "chunks/seed1/R1/chunk_001"
    environment = run / "environment"
    for directory in (preflight, other, environment):
        directory.mkdir(parents=True)
    outputs = {
        "run_directory": "results/runs/search_depth",
        "runtime_lock_json": "results/runs/search_depth/runtime_lock.json",
        "preflight_summary_json": "results/runs/search_depth/preflight_summary.json",
        "diagnostic_summary_json": "results/runs/search_depth/diagnostic_summary.json",
    }
    config = {
        "outputs": outputs,
        "preflight": {
            "profile_id": "fixed_depth_16",
            "seed_id": "seed1",
            "receptor_id": "R1",
            "chunk_index": 0,
        },
    }
    config_path = (
        tmp_path / "configs/stage06_mk14_vinagpu21_search_depth_diagnostic.json"
    )
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(config), encoding="ascii")
    data = tmp_path / "data"
    data.mkdir()
    for name in (
        "stage06_mk14_vinagpu21_train160_v1_result_summary.json",
        "stage06_mk14_vinagpu21_deterministic_batch_bridge_result_summary.json",
        "stage06_mk14_vinagpu21_deterministic_batch_runtime_lock.json",
    ):
        (data / name).write_text("{}\n", encoding="ascii")
    for name in (
        "runtime_lock.json",
        "preflight_summary.json",
        "diagnostic_summary.json",
    ):
        (run / name).write_text("{}\n", encoding="ascii")
    diagnostic = {
        "status": "search_depth_candidate_selected",
        "selected_profile": {
            "profile_id": "fixed_depth_16",
            "search_depth": 16,
        },
        "executed_profiles": [{"profile_id": "fixed_depth_16"}],
    }
    (run / "diagnostic_summary.json").write_text(
        json.dumps(diagnostic), encoding="ascii"
    )
    for name in (
        "profile_scores.csv",
        "chunk_runs.csv",
        "group_metrics.csv",
    ):
        (profile / name).write_text("evidence\n", encoding="ascii")
    (profile / "profile_summary.json").write_text(
        json.dumps({"pair_count": 320}), encoding="ascii"
    )
    (preflight / "chunk_summary.json").write_text("{}\n", encoding="ascii")
    (preflight / "pose_out.pdbqt").write_text("pose\n", encoding="ascii")
    (other / "raw_pose.pdbqt").write_text("raw\n", encoding="ascii")
    (environment / "nvidia_smi.csv").write_text("gpu\n", encoding="ascii")

    paths, packaged = result_paths(tmp_path.resolve())

    assert packaged["status"] == "search_depth_candidate_selected"
    assert (
        "results/runs/search_depth/profiles/fixed_depth_16/profile_scores.csv" in paths
    )
    assert (
        "results/runs/search_depth/profiles/fixed_depth_16/chunks/seed1/R1/"
        "chunk_000/pose_out.pdbqt" in paths
    )
    assert not any("chunk_001/raw_pose.pdbqt" in path for path in paths)
    assert "results/runs/search_depth/environment/nvidia_smi.csv" in paths
