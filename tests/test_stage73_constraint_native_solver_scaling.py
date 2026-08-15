import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_stage73_constraint_native_solver_scaling import (
    budgeted_multistart_greedy,
    constraint_preserving_annealing,
    enumerate_pool,
    load_model,
    make_cell,
    pool_sizes,
)


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="ascii"))


def read_csv(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_stage73_implementation_and_sources_are_hash_locked():
    config = read_json("configs/stage73_constraint_native_solver_scaling.json")
    for section in ("implementation", "inputs"):
        for value in config[section].values():
            path = ROOT / value["path"]
            assert sha256(path) == value["sha256"]
            if "size_bytes" in value:
                assert path.stat().st_size == value["size_bytes"]


def test_stage73_nested_pool_and_quality_workload_grid_is_complete():
    rows = read_csv(
        "results/runs/stage73_constraint_native_solver_scaling/workload_metrics.csv"
    )
    assert len(rows) == 276
    assert len(
        {
            (
                row["target_id"],
                row["outer_fold"],
                row["pool_size"],
                row["quality_regime"],
            )
            for row in rows
        }
    ) == 276
    expected_sizes = {
        "BACE1": {8, 12, 16, 24, 32, 34},
        "PPARG": {8, 12, 16, 24, 32, 48, 64, 96},
        "PPARA": {8, 12, 16, 20},
        "PPARD": {8, 12, 16, 24, 29},
    }
    for target, sizes in expected_sizes.items():
        assert {int(row["pool_size"]) for row in rows if row["target_id"] == target} == sizes
    assert {row["quality_regime"] for row in rows} == {
        "frozen_quality_floor",
        "relaxed_10pct_quality_floor",
        "no_quality_floor",
    }


def test_stage73_solver_trial_grid_and_exact_oracle_are_complete():
    rows = read_csv(
        "results/runs/stage73_constraint_native_solver_scaling/solver_trials.csv"
    )
    assert len(rows) == 7176
    counts = {}
    for row in rows:
        counts[row["method"]] = counts.get(row["method"], 0) + 1
    assert counts == {
        "exact_enumeration": 276,
        "single_start_best_improvement": 276,
        "budgeted_random_feasible": 2208,
        "budgeted_multistart_greedy": 2208,
        "constraint_preserving_annealing": 2208,
    }
    exact = [row for row in rows if row["method"] == "exact_enumeration"]
    assert all(row["exact_optimum_match"] == "True" for row in exact)


def test_stage73_frozen_k3_problem_is_uniformly_easy_for_classical_methods():
    result = read_json("data/stage73_constraint_native_solver_scaling_result.json")
    for metrics in result["full_pool_frozen_performance"].values():
        assert metrics["exact_optimum_success_rate"] == pytest.approx(1.0)
        assert metrics["maximum_normalized_objective_regret"] == pytest.approx(0.0)
    assert result["scaling_summary"][
        "maximum_full_pool_frozen_feasible_subset_count"
    ] == 17


def test_stage73_expanded_feasible_regions_expose_classical_failures():
    result = read_json("data/stage73_constraint_native_solver_scaling_result.json")
    assert result["hardness_summary"] == {
        "single_start_greedy_failure_cell_count": 39,
        "budgeted_random_feasible_failure_trial_count": 258,
        "budgeted_multistart_greedy_failure_trial_count": 51,
        "constraint_preserving_annealing_failure_trial_count": 67,
    }
    rows = read_csv(
        "results/runs/stage73_constraint_native_solver_scaling/solver_trials.csv"
    )
    frozen = [row for row in rows if row["quality_regime"] == "frozen_quality_floor"]
    assert all(row["exact_optimum_match"] == "True" for row in frozen)
    assert any(
        row["quality_regime"] == "no_quality_floor"
        and row["method"] == "budgeted_multistart_greedy"
        and row["exact_optimum_match"] == "False"
        for row in rows
    )


def test_stage73_stochastic_methods_are_seed_deterministic_and_feasible():
    config = read_json("configs/stage73_constraint_native_solver_scaling.json")
    record = read_json("data/stage72_constraint_native_cqm_model_record.json")[
        "models"
    ][0]
    model = load_model(record)
    assert pool_sizes(model, config)[-1] == 34
    pool = enumerate_pool(model, 16, config)
    cell = make_cell(model, pool, "relaxed_10pct_quality_floor")
    first = budgeted_multistart_greedy(cell, 128, np.random.default_rng(731))
    second = budgeted_multistart_greedy(cell, 128, np.random.default_rng(731))
    assert first == second
    annealed = constraint_preserving_annealing(
        cell, 128, 0.1, 100.0, np.random.default_rng(732)
    )
    assert first["subset"] in cell["feasible_lookup"]
    assert annealed["subset"] in cell["feasible_lookup"]


def test_stage73_scale_and_route_decision_are_frozen():
    result = read_json("data/stage73_constraint_native_solver_scaling_result.json")
    scale = result["scaling_summary"]
    assert scale["model_count"] == 16
    assert scale["workload_cell_count"] == 276
    assert scale["solver_trial_count"] == 7176
    assert scale["maximum_pool_size"] == 96
    assert scale["maximum_total_fixed_k_subset_count"] == 142880
    assert scale["maximum_feasible_subset_count"] == 142880
    assert scale["total_exact_enumeration_state_checks"] == 2783448
    assert result["route_gate"] == {
        "current_k3_exact_enumeration_tractable": True,
        "constraint_native_solver_gate_passed": True,
    }
    assert result["decision"]["larger_k_scaling_study_authorized"] is True
    assert result["decision"]["direct_qpu_execution_authorized"] is False
    assert result["decision"]["quantum_scaling_claim_authorized"] is False
    assert result["decision"]["quantum_advantage_claim_authorized"] is False


def test_stage73_independent_audit_and_data_boundary_are_frozen():
    result = read_json("data/stage73_constraint_native_solver_scaling_result.json")
    audit = read_json("data/stage73_constraint_native_solver_scaling_audit.json")
    assert result["status"] == "stage73_constraint_native_solver_scaling_complete"
    assert audit["status"] == (
        "stage73_constraint_native_solver_scaling_independent_audit_ok"
    )
    assert audit["stage72_models_independently_rebuilt"] == 16
    assert audit["workload_cells_independently_enumerated"] == 276
    assert audit["solver_trials_deterministically_replayed"] == 7176
    assert audit["solver_summaries_independently_recomputed"] == 510
    assert audit["source_result"]["sha256"] == sha256(
        ROOT / "data/stage73_constraint_native_solver_scaling_result.json"
    )
    assert result["data_boundary"] == {
        "historical_development_targets_read": 4,
        "fresh_validation_rows_read": 0,
        "locked_test_rows_read": 0,
        "new_docking_jobs": 0,
        "cloud_cqm_jobs": 0,
        "quantum_hardware_jobs": 0,
    }
