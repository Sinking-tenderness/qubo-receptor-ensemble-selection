import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="ascii"))


def test_stage91_freezes_a_nontrivial_group_robust_problem():
    result = read_json(
        ROOT / "data/stage91_bace1_group_robust_rescue_preregistration_result.json"
    )
    assert result["status"] == "stage91_bace1_group_robust_rescue_preregistered"
    assert result["receptor_count"] == 34
    assert result["primary_k"] == 6
    assert result["primary_k_state_count"] == 1344904
    assert result["role_summary"]["development"]["molecule_count"] == 365
    assert result["role_summary"]["development"]["core_series_count"] == 6


def test_stage91_core_series_all_have_high_and_low_examples():
    result = read_json(
        ROOT / "data/stage91_bace1_group_robust_rescue_preregistration_result.json"
    )
    series = result["development_core_series"]
    assert len(series) == 6
    assert sum(row["molecule_count"] for row in series) == 258
    assert all(row["high_count"] > 0 for row in series)
    assert all(row["low_count"] > 0 for row in series)


def test_stage91_keeps_all_outcome_generating_work_locked():
    result = read_json(
        ROOT / "data/stage91_bace1_group_robust_rescue_preregistration_result.json"
    )
    authorization = result["authorization"]
    assert authorization["development_ligand_input_preparation_bundle_authorized"] is True
    assert authorization["development_docking_authorized"] is False
    assert authorization["confirmation_or_test_preparation_authorized"] is False
    assert authorization["quantum_simulation_or_hardware_authorized"] is False
    assert result["data_boundary"] == {
        "new_docking_jobs": 0,
        "confirmation_docking_scores_read": 0,
        "locked_test_docking_scores_read": 0,
        "quantum_hardware_jobs": 0,
    }
