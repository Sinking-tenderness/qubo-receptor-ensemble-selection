import json
from pathlib import Path

from scripts.experimental.unidock.build_stage61b_ppard_remaining144_production_bundle import (
    bundle_paths,
)
from scripts.experimental.unidock.run_stage61b_ppard_remaining144_production import (
    common,
    validate_config,
    validate_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage61b_ppard_remaining144_unidock113_production.json"


def test_stage61b_frozen_input_audit_dimensions():
    config = common.read_json(CONFIG)
    validate_config(config)
    receptors, ligands, audit = validate_inputs(ROOT, config)
    assert len(receptors) == 29
    assert len(ligands) == 144
    assert audit["label_counts"] == {"active": 72, "decoy": 72}
    assert audit["expected_batch_count"] == 87
    assert audit["expected_pair_count"] == 12528
    assert audit["macrocycle_closure_pseudoatom_ligand_count"] == 0
    assert {row["pilot_selected"] for row in ligands} == {"False"}


def test_stage61b_protocol_matches_pilot_without_repeating_it():
    config = json.loads(CONFIG.read_text())
    protocol = config["unidock"]
    assert protocol["required_package_version"] == "1.1.3"
    assert protocol["profile_id"] == "enhanced"
    assert protocol["exhaustiveness"] == 1024
    assert protocol["max_step"] == 80
    assert [row["base_seed"] for row in config["inputs"]["seeds"]] == [
        20260801, 20260802, 20260803
    ]
    assert config["expected"]["pilot_pair_count_not_repeated"] == 8352
    assert config["expected"]["full_development_pair_count_after_merge"] == 20880


def test_stage61b_requires_stage60_freeze_and_preserves_boundaries():
    config = json.loads(CONFIG.read_text())
    stage60 = json.loads(
        (ROOT / config["inputs"]["stage60_result"]["path"]).read_text()
    )
    assert stage60["decision"]["remaining_development_docking_authorized"] is True
    assert stage60["decision"]["fresh_validation_authorized"] is False
    assert config["data_boundary"]["fresh_validation_rows_permitted"] == 0
    assert config["data_boundary"]["locked_test_rows_permitted"] == 0


def test_stage61b_bundle_contains_no_protected_paths():
    paths = bundle_paths(ROOT)
    assert paths
    assert all("fresh_validation" not in path.lower() for path in paths)
    assert all("locked_test" not in path.lower() for path in paths)
    assert all("data/protected" not in path.lower() for path in paths)
