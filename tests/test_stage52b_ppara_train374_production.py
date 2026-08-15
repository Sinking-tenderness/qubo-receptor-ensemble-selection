import json
from pathlib import Path

from scripts.build_stage52b_ppara_passing20_receptor_manifest import run as freeze
from scripts.experimental.unidock.build_stage52b_ppara_train374_production_bundle import (
    bundle_paths,
)
from scripts.experimental.unidock.run_stage52b_ppara_train374_production import (
    run,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage52b_ppara_train374_unidock113_production.json"


def test_stage52b_receptor_manifest_matches_frozen_passing_pool():
    result = freeze(ROOT)
    assert result["status"] == "stage52b_ppara_passing20_receptor_manifest_ok"
    assert result["receptor_count"] == 20
    assert result["stable_three_of_three_count"] == 18
    assert result["two_of_three_count"] == 2
    assert result["data_boundary"]["docking_jobs_started"] == 0


def test_stage52b_audit_only_freezes_complete_train_grid():
    result = run(CONFIG, ROOT, None, True, False, None, None, False)
    assert result["status"] == "audit_only_ok"
    assert result["receptor_count"] == 20
    assert result["ligand_count"] == 374
    assert result["label_counts"] == {"active": 187, "decoy": 187}
    assert result["selected_batch_count"] == 60
    assert result["selected_pair_count"] == 22440
    assert result["fresh_validation_rows"] == 0
    assert result["locked_test_rows"] == 0


def test_stage52b_config_preserves_failed_confirmation_boundary():
    config = json.loads(CONFIG.read_text())
    assert "Stage51 remains failed" in config["decision_boundary"]
    assert config["expected"]["pair_count"] == 22440
    assert config["unidock"]["required_package_version"] == "1.1.3"
    assert config["unidock"]["exhaustiveness"] == 1024
    assert config["unidock"]["max_step"] == 80


def test_stage52b_remote_supports_resume_partition_and_poweroff():
    launcher = (
        ROOT
        / "scripts/experimental/unidock/run_stage52b_ppara_train374_production_remote.sh"
    ).read_text()
    for token in (
        "--resume",
        "SEED_IDS",
        "RECEPTOR_IDS",
        "FINALIZE_ONLY",
        "AUTO_POWEROFF",
        "stage52b_ppara_train374_unidock113_production_core_v1.tar.gz",
    ):
        assert token in launcher


def test_stage52b_bundle_is_train_only_and_complete():
    paths = bundle_paths(ROOT)
    lowered = [path.lower() for path in paths]
    assert not any("fresh_validation" in path for path in lowered)
    assert not any("locked_test" in path for path in lowered)
    assert not any("data/protected" in path for path in lowered)
    assert sum(path.endswith("_receptor.pdbqt") for path in paths) == 20
    assert sum(
        path.startswith("results/runs/stage52a_ppara_train374_unidock_inputs/pdbqt/")
        and path.endswith(".pdbqt")
        for path in paths
    ) == 374
