import json
from pathlib import Path

from scripts.build_stage58b_ppard_passing29_receptor_manifest import run as freeze
from scripts.experimental.unidock.build_stage58b_ppard_pilot96_production_bundle import (
    bundle_paths,
)
from scripts.experimental.unidock.run_stage58b_ppard_pilot96_production import run


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage58b_ppard_pilot96_unidock113_production.json"


def test_stage58b_receptor_manifest_is_all_stage57_three_of_three_passes():
    result = freeze(
        ROOT,
        Path("data/processed/stage58b_ppard_stage57_passing29_receptor_manifest.csv"),
        Path("data/stage58b_ppard_stage57_passing29_receptor_manifest_summary.json"),
    )
    assert result["status"] == "stage58b_ppard_passing29_receptor_manifest_ok"
    assert result["receptor_count"] == 29
    assert result["all_receptors_passed_three_of_three_seeds"] is True
    assert result["data_boundary"]["pilot_docking_scores_read"] == 0


def test_stage58b_audit_only_freezes_complete_pilot_grid():
    result = run(CONFIG, ROOT, None, True, False, None, None, False)
    assert result["status"] == "audit_only_ok"
    assert result["receptor_count"] == 29
    assert result["ligand_count"] == 96
    assert result["label_counts"] == {"active": 48, "decoy": 48}
    assert result["selected_batch_count"] == 87
    assert result["selected_pair_count"] == 8352
    assert result["fresh_validation_rows"] == 0
    assert result["locked_test_rows"] == 0


def test_stage58b_config_preserves_preregistered_pilot_boundary():
    config = json.loads(CONFIG.read_text())
    assert "prospective outcome-blind" in config["decision_boundary"]
    assert config["expected"]["pair_count"] == 8352
    assert config["unidock"]["required_package_version"] == "1.1.3"
    assert config["unidock"]["exhaustiveness"] == 1024
    assert config["unidock"]["max_step"] == 80
    assert config["data_boundary"]["required_pilot_role"] == "development_train_pilot"


def test_stage58b_adapter_finalize_status_is_ppard_specific(monkeypatch, tmp_path):
    from scripts.experimental.unidock import run_stage58b_ppard_pilot96_production as adapter

    progress_path = tmp_path / "progress.json"
    summary_path = tmp_path / "summary.json"
    adapter.common.write_json(progress_path, {"status": "stage52b_production_complete"})
    config = {
        "outputs": {
            "progress_json": str(progress_path),
            "summary_json": str(summary_path),
        }
    }
    source = {
        "status": "stage52b_ppara_train374_unidock_matrix_ok",
        "stage51_gate_status": "must_be_removed",
        "data_boundary": {"fresh_validation_rows_read": 0, "locked_test_rows_read": 0},
    }
    monkeypatch.setattr(adapter, "ORIGINAL_FINALIZE", lambda *args, **kwargs: source)
    result = adapter.finalize(tmp_path, None, config)
    assert result["status"] == "stage58b_ppard_pilot96_unidock_matrix_ok"
    assert result["experiment_class"] == "prospective outcome-blind development pilot"
    assert "stage51_gate_status" not in result
    assert adapter.common.read_json(progress_path)["status"] == "stage58b_production_complete"


def test_stage58b_remote_supports_resume_partition_and_poweroff():
    launcher = (
        ROOT
        / "scripts/experimental/unidock/run_stage58b_ppard_pilot96_production_remote.sh"
    ).read_text()
    for token in (
        "--resume", "SEED_IDS", "RECEPTOR_IDS", "FINALIZE_ONLY", "AUTO_POWEROFF",
        "stage58b_ppard_pilot96_unidock113_production_core_v1.tar.gz",
    ):
        assert token in launcher


def test_stage58b_bundle_is_pilot_only_and_complete():
    paths = bundle_paths(ROOT)
    lowered = [path.lower() for path in paths]
    assert not any("fresh_validation" in path for path in lowered)
    assert not any("locked_test" in path for path in lowered)
    assert not any("data/protected" in path for path in lowered)
    assert sum(path.endswith("_receptor.pdbqt") for path in paths) == 29
    assert sum(
        path.startswith("results/runs/stage58a_ppard_pilot96_unidock_inputs/pdbqt/")
        and path.endswith(".pdbqt")
        for path in paths
    ) == 96
