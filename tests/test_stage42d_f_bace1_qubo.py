from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text())


def test_stage42d_old_objective_is_frozen_no_go() -> None:
    result = load("data/stage42d_bace1_large_pool_qubo_screen_result.json")
    assert result["status"] == "stage42d_bace1_large_pool_qubo_screen_complete"
    assert result["input_statistics"]["full_state_count_k1_to_k6"] == 1676115
    assert result["decision"]["positive_train_gap_fold_count"] == 0
    assert result["decision"]["frozen_objective_supported_on_bace1"] is False
    assert result["decision"]["fresh_validation_authorized"] is False


def test_stage42e_diagnoses_cardinality_pressure() -> None:
    result = load("data/stage42e_bace1_qubo_rank_alignment_diagnosis_result.json")
    diagnosis = result["diagnosis"]
    assert diagnosis["objective_monotonically_increases_k1_to_k6"] is True
    assert diagnosis["objective_optimal_k"] == 6
    assert diagnosis["bedroc_optimal_k"] == 2
    assert diagnosis["cardinality_pressure_detected"] is True
    assert diagnosis["old_objective_retuning_authorized"] is False


def test_stage42f_improves_combination_but_not_solver_gap() -> None:
    result = load("data/stage42f_bace1_rank_sensitive_pair_qubo_result.json")
    decision = result["decision"]
    assert result["status"] == "stage42f_bace1_rank_sensitive_pair_qubo_complete"
    assert result["input_statistics"]["state_count_k1_to_k6"] == 1676115
    assert decision["best_combination_over_single_bedroc_gain"] > 0.019
    assert decision["positive_gap_cell_count"] == 0
    assert decision["full_data_positive_gap_k_values"] == []
    assert decision["rank_sensitive_pair_qubo_supported"] is False
    assert decision["quantum_hardware_authorized"] is False


def test_stage42f_independent_audit_passed() -> None:
    audit = load("data/stage42f_bace1_rank_sensitive_pair_qubo_audit.json")
    assert audit["status"] == "stage42d_f_bace1_qubo_independent_audit_ok"
    assert audit["stage42c_pair_count"] == 27132
    assert audit["stage42f_fold_k_cell_count"] == 24
    assert audit["stage42f_exact_classical_positive_gap_cell_count"] == 0
