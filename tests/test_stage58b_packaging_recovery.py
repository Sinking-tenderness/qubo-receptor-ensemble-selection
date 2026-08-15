import json
import shutil
from pathlib import Path

from scripts.experimental.unidock.build_stage58b_packaging_recovery_bundle import (
    bundle_paths,
)
from scripts.experimental.unidock.repair_stage58b_ppard_pilot96_packaging import (
    FROZEN_CONFIG_SHA256,
    run,
)
from scripts.experimental.unidock.run_stage58b_ppard_pilot96_production import common


ROOT = Path(__file__).resolve().parents[1]


def test_stage58b_recovery_bundle_is_packaging_only():
    paths = bundle_paths(ROOT)
    assert len(paths) == 4
    assert not any(path.startswith("configs/") for path in paths)
    assert not any(path.startswith("data/") for path in paths)
    assert not any(path.startswith("results/") for path in paths)


def test_stage58b_recovery_launcher_does_not_invoke_docking():
    script = (
        ROOT
        / "scripts/experimental/unidock/run_stage58b_ppard_packaging_recovery_remote.sh"
    ).read_text()
    assert "repair_stage58b_ppard_pilot96_packaging" in script
    assert "audit_stage58b_ppard_pilot96_production" in script
    assert "--unidock" not in script
    assert "nvidia-smi" not in script
    assert FROZEN_CONFIG_SHA256 not in script


def test_stage58b_recovery_repairs_only_stale_progress_descriptor(tmp_path):
    config_source = ROOT / "configs/stage58b_ppard_pilot96_unidock113_production.json"
    config_target = tmp_path / "configs/stage58b_ppard_pilot96_unidock113_production.json"
    adapter_source = (
        ROOT / "scripts/experimental/unidock/run_stage58b_ppard_pilot96_production.py"
    )
    adapter_target = (
        tmp_path / "scripts/experimental/unidock/run_stage58b_ppard_pilot96_production.py"
    )
    config_target.parent.mkdir(parents=True)
    adapter_target.parent.mkdir(parents=True)
    shutil.copy2(config_source, config_target)
    shutil.copy2(adapter_source, adapter_target)
    config = json.loads(config_target.read_text())
    outputs = config["outputs"]
    paths = {key: tmp_path / value for key, value in outputs.items()}

    for key in ("scores_csv", "batch_runs_csv", "median_matrix_csv", "minimum_matrix_csv"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        paths[key].write_text(f"frozen-{key}\n", encoding="ascii")
    progress = {
        "status": "stage52b_production_complete",
        "completed_batch_count": 87,
        "completed_pair_count": 8352,
        "missing_batch_count": 0,
    }
    common.write_json(paths["progress_json"], progress)
    stale = common.output_descriptor(tmp_path, paths["progress_json"])
    progress["status"] = "stage58b_production_complete"
    common.write_json(paths["progress_json"], progress)
    frozen_hashes = {
        key: common.file_sha256(paths[key])
        for key in ("scores_csv", "batch_runs_csv", "median_matrix_csv", "minimum_matrix_csv")
    }
    summary = {
        "status": "stage58b_ppard_pilot96_unidock_matrix_ok",
        "config": {"sha256": FROZEN_CONFIG_SHA256},
        "batch_count": 87,
        "pair_count": 8352,
        "unresolved_warning_event_count": 0,
        "pose_integrity_failure_count": 0,
        "outputs": {
            key: common.output_descriptor(tmp_path, paths[key])
            for key in ("scores_csv", "batch_runs_csv", "median_matrix_csv", "minimum_matrix_csv")
        },
    }
    summary["outputs"]["progress_json"] = stale
    common.write_json(paths["summary_json"], summary)

    result = run(config_target, tmp_path)
    repaired = common.read_json(paths["summary_json"])
    assert result["docking_batches_rerun"] == 0
    assert result["summary_progress_descriptor_changed"] is True
    assert repaired["outputs"]["progress_json"] == common.output_descriptor(
        tmp_path, paths["progress_json"]
    )
    assert {
        key: common.file_sha256(paths[key]) for key in frozen_hashes
    } == frozen_hashes
