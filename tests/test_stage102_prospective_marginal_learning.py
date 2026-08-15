import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage102_readiness_and_budget():
    result = json.loads((ROOT / "data/stage102_prospective_marginal_learning_readiness.json").read_text(encoding="utf-8"))
    assert result["status"] == "stage102_phase_a_readiness_ok"
    assert result["phase_a_targets"]["EGFR"]["passing_receptor_count"] == 12
    assert result["phase_a_targets"]["FA10"]["passing_receptor_count"] == 13
    assert result["phase_a_expected_receptor_ligand_seed_pairs"]["total"] == 45000


def test_stage102_keeps_parp1_and_hardware_locked():
    result = json.loads((ROOT / "data/stage102_prospective_marginal_learning_readiness.json").read_text(encoding="utf-8"))
    assert result["phase_b_target"] == "PARP1"
    assert result["phase_b_released"] is False
    assert result["parp1_fresh_validation_rows_read"] == 0
    assert result["quantum_hardware_jobs"] == 0
