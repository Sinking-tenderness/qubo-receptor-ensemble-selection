import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from scripts.run_stage71_qubo_coefficient_noise_robustness import (
    build_feasible_states,
    exact_feasible_optimum,
    perturb_coefficients,
    reconstruct_model,
)


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="ascii"))


def read_csv(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def summary_row(noise_model: str, level: float, scope: str = "ALL"):
    rows = read_csv(
        "results/runs/stage71_qubo_coefficient_noise_robustness/noise_summary.csv"
    )
    matches = [
        row
        for row in rows
        if row["noise_model"] == noise_model
        and float(row["noise_level"]) == pytest.approx(level)
        and row["scope"] == scope
    ]
    assert len(matches) == 1
    return matches[0]


def test_stage71_implementation_and_sources_are_hash_locked():
    config = read_json("configs/stage71_qubo_coefficient_noise_robustness.json")
    for section in ("implementation", "inputs"):
        for value in config[section].values():
            assert sha256(ROOT / value["path"]) == value["sha256"]


def test_stage71_reconstructs_all_frozen_qubos_exactly():
    model_record = read_json(
        "data/stage70_constraint_aware_qubo_encoding_model_record.json"
    )
    assert model_record["model_count"] == 16
    models = [reconstruct_model(record) for record in model_record["models"]]
    assert len(models) == 16
    assert all(
        model["source_qubo_sha256"]
        == model["record"]["qubo_summary"]["qubo_sha256"]
        for model in models
    )
    assert max(len(model["variables"]) for model in models) == 100


def test_stage71_exact_feasible_landscape_is_frozen():
    result = read_json("data/stage71_qubo_coefficient_noise_robustness_result.json")
    landscape = result["exact_landscape"]
    assert landscape["model_count"] == 16
    assert landscape["unique_source_optimum_count"] == 16
    assert landscape["finite_second_energy_count"] == 14
    assert landscape["maximum_feasible_receptor_subset_count"] == 17
    assert landscape["minimum_normalized_feasible_energy_gap"] == pytest.approx(
        1.097160305749828e-08
    )
    assert landscape["maximum_normalized_feasible_energy_gap"] == pytest.approx(
        2.1406160766179738e-07
    )


def test_stage71_zero_noise_sampler_calibration_is_separate_and_frozen():
    result = read_json("data/stage71_qubo_coefficient_noise_robustness_result.json")
    calibration = result["sampler_calibration"]
    assert calibration["model_count"] == 16
    assert calibration["best_feasible_count"] == 13
    assert calibration["best_exact_selected_count"] == 11
    assert calibration["best_feasible_rate"] == pytest.approx(0.8125)
    assert calibration["best_exact_selected_rate"] == pytest.approx(0.6875)
    assert calibration["calibration_gate_passed"] is True


def test_stage71_noise_grid_and_baseline_are_complete():
    rows = read_csv(
        "results/runs/stage71_qubo_coefficient_noise_robustness/noise_trials.csv"
    )
    assert len(rows) == 1312
    assert len(
        {
            (row["target_id"], int(row["outer_fold"]))
            for row in rows
        }
    ) == 16
    assert sum(row["noise_model"] == "none" for row in rows) == 16
    assert sum(
        row["noise_model"] == "round_to_nearest_full_scale" for row in rows
    ) == 144
    assert sum(row["noise_model"] == "iid_gaussian_full_scale" for row in rows) == 1152
    baseline = summary_row("none", 0.0)
    assert float(baseline["exact_selected_unique_rate"]) == pytest.approx(1.0)
    assert float(baseline["local_feasible_rate"]) == pytest.approx(1.0)


def test_stage71_reference_noise_gates_fail_for_identified_precision_reason():
    quantized = summary_row("round_to_nearest_full_scale", 1e-6)
    gaussian = summary_row("iid_gaussian_full_scale", 1e-6)
    assert float(quantized["exact_selected_unique_rate"]) == pytest.approx(0.25)
    assert float(quantized["local_feasible_rate"]) == pytest.approx(1.0)
    assert float(gaussian["exact_selected_unique_rate"]) == pytest.approx(0.421875)
    assert float(gaussian["local_feasible_rate"]) == pytest.approx(0.9296875)
    result = read_json("data/stage71_qubo_coefficient_noise_robustness_result.json")
    assert result["robustness_envelopes"]["round_to_nearest_full_scale"][
        "largest_tested_level_passing_project_gate"
    ] == pytest.approx(3e-9)
    assert result["robustness_envelopes"]["iid_gaussian_full_scale"][
        "largest_tested_level_passing_project_gate"
    ] == pytest.approx(1e-9)
    assert result["robustness_gate"][
        "coefficient_robust_logical_bqm_gate_passed"
    ] is False


def test_stage71_perturbation_is_deterministic_and_exactly_enumerable():
    record = read_json(
        "data/stage70_constraint_aware_qubo_encoding_model_record.json"
    )["models"][0]
    model = reconstruct_model(record)
    states = build_feasible_states(model)
    first = perturb_coefficients(model, "iid_gaussian_full_scale", 1e-8, 202671000)
    second = perturb_coefficients(model, "iid_gaussian_full_scale", 1e-8, 202671000)
    assert first[2]["perturbed_coefficients_sha256"] == second[2][
        "perturbed_coefficients_sha256"
    ]
    optimum = exact_feasible_optimum(model, states, first[0], first[1], 1e-12)
    assert optimum["selected_remains_unique"] is True
    assert math.isfinite(optimum["best_energy"])


def test_stage71_audit_and_decision_boundary_are_frozen():
    result = read_json("data/stage71_qubo_coefficient_noise_robustness_result.json")
    audit = read_json("data/stage71_qubo_coefficient_noise_robustness_audit.json")
    assert result["status"] == "stage71_qubo_coefficient_noise_robustness_complete"
    assert result["noise_trial_count"] == 1312
    assert result["noise_summary_count"] == 95
    assert result["decision"]["direct_qpu_execution_authorized"] is False
    assert result["decision"]["constraint_native_reformulation_authorized"] is True
    assert result["decision"]["new_target_preregistration_remains_authorized"] is True
    assert result["decision"]["quantum_advantage_claim_authorized"] is False
    assert result["data_boundary"]["fresh_validation_rows_read"] == 0
    assert result["data_boundary"]["new_docking_jobs"] == 0
    assert result["data_boundary"]["quantum_hardware_jobs"] == 0
    assert audit["status"] == (
        "stage71_qubo_coefficient_noise_robustness_independent_audit_ok"
    )
    assert audit["logical_models_independently_reconstructed"] == 16
    assert audit["noise_trials_independently_recomputed"] == 1312
    assert audit["noise_summaries_independently_recomputed"] == 95
    assert audit["source_result"]["sha256"] == sha256(
        ROOT / "data/stage71_qubo_coefficient_noise_robustness_result.json"
    )
