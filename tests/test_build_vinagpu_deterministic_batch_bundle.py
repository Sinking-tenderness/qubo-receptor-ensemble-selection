import json
from pathlib import Path

from scripts.experimental.vinagpu.build_vinagpu_deterministic_batch_bundle import (
    CONFIG,
    FIXED_PATHS,
    bundle_paths,
)
from scripts.experimental.vinagpu.package_vinagpu_deterministic_batch_results import (
    result_paths,
)


def test_deterministic_batch_bundle_is_self_contained_and_train_only():
    paths = bundle_paths(Path.cwd())

    assert CONFIG in paths
    assert "data/stage06_mk14_vinagpu21_train160_v1_gpu_reference.csv" in paths
    assert (
        "scripts/experimental/vinagpu/apply_deterministic_batch_patch.py"
        in FIXED_PATHS
    )
    assert (
        "scripts/experimental/vinagpu/run_vinagpu_deterministic_batch_remote.sh"
        in FIXED_PATHS
    )
    assert sum(path.endswith(".pdbqt") for path in paths) == 165
    assert not any("fresh_validation" in path for path in paths)


def test_result_bundle_keeps_preflight_chunk_but_not_all_raw_chunks(tmp_path: Path):
    run = tmp_path / "results/runs/bridge"
    preflight = run / "chunks/seed0/R1/chunk_000"
    other = run / "chunks/seed0/R1/chunk_001"
    environment = run / "environment"
    for directory in (preflight, other, environment):
        directory.mkdir(parents=True)
    output_names = {
        "runtime_lock_json": "runtime_lock.json",
        "preflight_summary_json": "preflight_summary.json",
        "batch_scores_csv": "batch_scores.csv",
        "batch_runs_csv": "batch_runs.csv",
        "batch_summary_json": "batch_run_summary.json",
        "bridge_summary_json": "bridge_summary.json",
    }
    outputs = {
        "run_directory": "results/runs/bridge",
        **{
            key: f"results/runs/bridge/{name}"
            for key, name in output_names.items()
        },
    }
    config = {
        "outputs": outputs,
        "preflight": {
            "seed_id": "seed0",
            "receptor_id": "R1",
            "chunk_index": 0,
        },
    }
    config_path = (
        tmp_path
        / "configs/stage06_mk14_vinagpu21_deterministic_batch_bridge.json"
    )
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(config), encoding="ascii")
    data = tmp_path / "data"
    data.mkdir()
    (data / "stage06_mk14_vinagpu21_train160_v1_result_summary.json").write_text(
        "{}\n", encoding="ascii"
    )
    for name in output_names.values():
        (run / name).write_text("evidence\n", encoding="ascii")
    (run / "batch_run_summary.json").write_text(
        json.dumps({"status": "ok", "pair_count": 2400}), encoding="ascii"
    )
    (run / "bridge_summary.json").write_text(
        json.dumps({"status": "deterministic_batch_bridge_passed"}),
        encoding="ascii",
    )
    (preflight / "chunk_summary.json").write_text("{}\n", encoding="ascii")
    (preflight / "pose_out.pdbqt").write_text("pose\n", encoding="ascii")
    (other / "raw_pose.pdbqt").write_text("raw\n", encoding="ascii")
    (environment / "nvidia_smi.csv").write_text("gpu\n", encoding="ascii")

    paths = result_paths(tmp_path.resolve())

    assert "results/runs/bridge/chunks/seed0/R1/chunk_000/pose_out.pdbqt" in paths
    assert "results/runs/bridge/chunks/seed0/R1/chunk_001/raw_pose.pdbqt" not in paths
    assert "results/runs/bridge/environment/nvidia_smi.csv" in paths
