import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_stage72_constraint_native_cqm import (
    build_feasible_states,
    build_native_model,
    exact_optimum,
    perturb_coefficients,
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
    rows = read_csv("results/runs/stage72_constraint_native_cqm/noise_summary.csv")
    matches = [
        row
        for row in rows
        if row["noise_model"] == noise_model
        and float(row["noise_level"]) == pytest.approx(level)
        and row["scope"] == scope
    ]
    assert len(matches) == 1
    return matches[0]


def test_stage72_implementation_and_sources_are_hash_locked():
    config = read_json("configs/stage72_constraint_native_cqm.json")
    for section in ("implementation", "inputs"):
        for value in config[section].values():
            assert sha256(ROOT / value["path"]) == value["sha256"]


def test_stage72_midpoint_centering_preserves_exact_objective():
    records = read_json(
        "data/stage70_constraint_aware_qubo_encoding_model_record.json"
    )["models"]
    for record in records:
        model = build_native_model(record)
        states = build_feasible_states(model)
        exact = exact_optimum(model, states, model["normalized_coefficients"])
        assert exact["selected_remains_unique"] is True
        selected = states["selected_index"]
        normalized_energy = (
            model["normalized_offset"]
            + states["features"][selected] @ model["normalized_coefficients"]
        )
        assert normalized_energy * model["normalization_scale"] == pytest.approx(
            states["raw_objectives"][selected], abs=1e-10
        )
        assert len(model["cqm"].constraints) == 2


def test_stage72_formulation_summary_is_frozen():
    result = read_json("data/stage72_constraint_native_cqm_result.json")
    summary = result["formulation_summary"]
    assert summary["model_count"] == 16
    assert summary["exact_source_optimum_count"] == 16
    assert summary["connected_feasible_swap_graph_count"] == 16
    assert summary["single_local_minimum_model_count"] == 15
    assert summary["minimum_best_improvement_global_recovery_rate"] == pytest.approx(
        4 / 7
    )
    assert summary["minimum_native_normalized_feasible_energy_gap"] == pytest.approx(
        0.045439974405263595
    )
    assert summary[
        "minimum_normalized_gap_improvement_factor_vs_stage71"
    ] == pytest.approx(1742186.5076166491)
    assert summary[
        "maximum_normalized_gap_improvement_factor_vs_stage71"
    ] == pytest.approx(6982180.925817009)


def test_stage72_reduces_penalty_variables_and_couplings():
    result = read_json("data/stage72_constraint_native_cqm_result.json")
    summary = result["formulation_summary"]
    assert summary["maximum_native_logical_variable_count"] == 96
    assert summary["maximum_logical_variable_reduction"] == 9
    assert summary["maximum_quadratic_coefficient_reduction"] == 390
    rows = read_csv("results/runs/stage72_constraint_native_cqm/model_metrics.csv")
    assert len(rows) == 16
    assert all(int(row["explicit_constraint_count"]) == 2 for row in rows)
    assert all(row["feasible_swap_graph_connected"] == "True" for row in rows)


def test_stage72_noise_grid_is_complete_and_baseline_is_exact():
    rows = read_csv("results/runs/stage72_constraint_native_cqm/noise_trials.csv")
    assert len(rows) == 2736
    assert sum(row["noise_model"] == "none" for row in rows) == 16
    assert sum(
        row["noise_model"] == "round_to_nearest_full_scale" for row in rows
    ) == 160
    assert sum(row["noise_model"] == "iid_gaussian_full_scale" for row in rows) == 2560
    baseline = summary_row("none", 0.0)
    assert float(baseline["exact_selected_unique_rate"]) == pytest.approx(1.0)
    assert float(baseline["exact_selected_optimal_rate"]) == pytest.approx(1.0)


def test_stage72_matched_and_stress_noise_gates_pass_pre_registered_thresholds():
    for level in (1e-6, 1e-3):
        quantized = summary_row("round_to_nearest_full_scale", level)
        gaussian = summary_row("iid_gaussian_full_scale", level)
        assert float(quantized["exact_selected_unique_rate"]) == pytest.approx(1.0)
        assert float(gaussian["exact_selected_unique_rate"]) == pytest.approx(1.0)
    result = read_json("data/stage72_constraint_native_cqm_result.json")
    assert all(
        value["gate_passed"]
        for value in result["matched_stage71_reference_gate"].values()
    )
    assert all(
        value["gate_passed"] for value in result["stress_reference_gate"].values()
    )
    assert result["robustness_envelopes"]["round_to_nearest_full_scale"][
        "largest_tested_level_passing_project_gate"
    ] == pytest.approx(1e-2)
    assert result["robustness_envelopes"]["iid_gaussian_full_scale"][
        "largest_tested_level_passing_project_gate"
    ] == pytest.approx(3e-2)


def test_stage72_noise_is_deterministic_on_constraint_native_objective():
    record = read_json(
        "data/stage70_constraint_aware_qubo_encoding_model_record.json"
    )["models"][0]
    model = build_native_model(record)
    states = build_feasible_states(model)
    first = perturb_coefficients(model, "iid_gaussian_full_scale", 1e-3, 202672000)
    second = perturb_coefficients(model, "iid_gaussian_full_scale", 1e-3, 202672000)
    assert first[1]["perturbed_coefficients_sha256"] == second[1][
        "perturbed_coefficients_sha256"
    ]
    assert exact_optimum(model, states, first[0])["selected_remains_unique"] is True


def test_stage72_audit_and_claim_boundary_are_frozen():
    result = read_json("data/stage72_constraint_native_cqm_result.json")
    audit = read_json("data/stage72_constraint_native_cqm_audit.json")
    assert result["status"] == "stage72_constraint_native_cqm_complete"
    assert result["formulation_gate"][
        "constraint_native_formulation_gate_passed"
    ] is True
    assert result["decision"]["constraint_native_formulation_freeze_authorized"] is True
    assert result["decision"]["solver_scaling_benchmark_authorized"] is True
    assert result["decision"]["direct_qpu_execution_authorized"] is False
    assert result["decision"]["quantum_advantage_claim_authorized"] is False
    assert result["data_boundary"]["new_docking_jobs"] == 0
    assert result["data_boundary"]["quantum_hardware_jobs"] == 0
    assert audit["status"] == "stage72_constraint_native_cqm_independent_audit_ok"
    assert audit["constraint_native_models_independently_rebuilt"] == 16
    assert audit["noise_trials_independently_recomputed"] == 2736
    assert audit["noise_summaries_independently_recomputed"] == 105
    assert audit["source_result"]["sha256"] == sha256(
        ROOT / "data/stage72_constraint_native_cqm_result.json"
    )
