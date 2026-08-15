import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text())


def test_stage55_selects_ppard_without_outcomes():
    result = read_json("data/stage55_ppard_small_pilot_preregistration_result.json")
    assert result["status"] == (
        "stage55_ppard_small_pilot_source_and_preregistration_ok"
    )
    assert result["selection"]["eligible_target_order"] == [
        "PPARA",
        "PPARD",
        "ESR2",
    ]
    assert result["selection"]["selected_target"] == "PPARD"
    assert result["selection"]["selection_was_outcome_blind"] is True


def test_stage55_source_and_reference_identity_pass():
    result = read_json("data/stage55_ppard_small_pilot_preregistration_result.json")
    assert result["dude_source"]["actives"]["row_count"] == 240
    assert result["dude_source"]["decoys"]["row_count"] == 12250
    identity = result["reference_structure"]["ligand_identity"]
    assert identity["heavy_atom_count"] == 33
    assert identity["fixed_frame_symmetry_corrected_rmsd_angstrom"] == 0.0
    assert identity["heavy_atom_coordinate_identity"] is True


def test_stage55_inherits_stage54_gate_and_caps_pilot_cost():
    result = read_json("data/stage55_ppard_small_pilot_preregistration_result.json")
    future = read_json("data/stage54_future_target_intake_criteria.json")
    assert result["frozen_protocol"]["functional_gate"] == future["criteria"]
    assert result["pilot_budget"]["maximum_receptor_ligand_pairs"] == 4896
    assert result["pilot_budget"]["maximum_seeded_docking_jobs"] == 14688
    assert (
        result["pilot_budget"]["full_training_matrix_before_gate_permitted"]
        is False
    )


def test_stage55_keeps_protected_and_expensive_work_locked():
    result = read_json("data/stage55_ppard_small_pilot_preregistration_result.json")
    for key in (
        "pilot_production_docking_authorized",
        "full_training_matrix_authorized",
        "fresh_validation_authorized",
        "quantum_hardware_authorized",
    ):
        assert result["decision"][key] is False
    assert result["data_boundary"]["docking_scores_read"] == 0
    assert result["data_boundary"]["fresh_validation_rows_read"] == 0
    assert result["data_boundary"]["locked_test_rows_read"] == 0
    assert result["data_boundary"]["new_docking_jobs"] == 0
