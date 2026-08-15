import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.run_stage74_larger_k_solver_scaling import (
    budgeted_tabu_search,
    constraint_preserving_annealing,
    deficit_distributions,
    feasible,
    k_schedule,
    load_model,
    quality_thresholds,
    state,
)


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="ascii"))


def read_csv(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_stage74_implementation_and_sources_are_hash_locked():
    config = read_json("configs/stage74_larger_k_solver_scaling.json")
    for section in ("implementation", "inputs"):
        for value in config[section].values():
            path = ROOT / value["path"]
            assert sha256(path) == value["sha256"]
            if "size_bytes" in value:
                assert path.stat().st_size == value["size_bytes"]


def test_stage74_larger_k_workload_grid_and_exact_envelope_are_complete():
    rows = read_csv(
        "results/runs/stage74_larger_k_solver_scaling/workload_metrics.csv"
    )
    assert len(rows) == 300
    assert len(
        {
            (
                row["target_id"],
                row["outer_fold"],
                row["k"],
                row["quality_regime"],
            )
            for row in rows
        }
    ) == 300
    expected_k = {
        "BACE1": {3, 4, 6, 8, 10, 12, 16},
        "PPARG": {3, 4, 6, 8, 10, 12, 16},
        "PPARA": {3, 4, 6, 8, 10},
        "PPARD": {3, 4, 6, 8, 10, 12},
    }
    for target, values in expected_k.items():
        assert {int(row["k"]) for row in rows if row["target_id"] == target} == values
    assert sum(row["exact_oracle_available"] == "True" for row in rows) == 120
    assert {row["quality_regime"] for row in rows} == {
        "strict_1pct_quality",
        "balanced_10pct_quality",
        "unconstrained",
    }


def test_stage74_quality_density_uses_exact_integer_subset_counts():
    rows = read_csv(
        "results/runs/stage74_larger_k_solver_scaling/workload_metrics.csv"
    )
    for row in rows:
        total = int(row["total_fixed_k_subset_count"])
        feasible_count = int(row["feasible_subset_count"])
        requested = float(row["requested_feasible_density"])
        assert feasible_count >= math.ceil(requested * total - 1e-9)
        assert float(row["feasible_subset_fraction"]) == pytest.approx(
            feasible_count / total
        )
    largest = max(rows, key=lambda row: int(row["total_fixed_k_subset_count"]))
    assert int(largest["total_fixed_k_subset_count"]) == 662252084388541314
    unconstrained = [
        row
        for row in rows
        if row["target_id"] == "PPARG"
        and row["outer_fold"] == "0"
        and row["k"] == "16"
        and row["quality_regime"] == "unconstrained"
    ][0]
    assert int(unconstrained["feasible_subset_count"]) == 662252084388541314


def test_stage74_solver_trial_grid_and_exact_validation_are_frozen():
    rows = read_csv(
        "results/runs/stage74_larger_k_solver_scaling/solver_trials.csv"
    )
    assert len(rows) == 7620
    counts = {}
    for row in rows:
        counts[row["method"]] = counts.get(row["method"], 0) + 1
    assert counts == {
        "exact_enumeration": 120,
        "deterministic_best_improvement": 300,
        "budgeted_multistart_greedy": 2400,
        "budgeted_tabu_search": 2400,
        "constraint_preserving_annealing": 2400,
    }
    result = read_json("data/stage74_larger_k_solver_scaling_result.json")
    assert result["exact_validation"][
        "strong_classical_exact_cell_success_rate"
    ] == pytest.approx(1.0)
    assert result["exact_validation"]["sampler_exact_cell_success_rate"] == pytest.approx(
        1.0
    )


def test_stage74_large_cells_expose_solver_disagreement_without_overclaiming():
    result = read_json("data/stage74_larger_k_solver_scaling_result.json")
    hardness = result["hardness_summary"]
    assert hardness["nonexact_workload_cell_count"] == 180
    assert hardness["nonexact_solver_disagreement_cell_count"] == 108
    assert hardness["nonexact_solver_disagreement_fraction"] == pytest.approx(0.6)
    assert hardness["sampler_strict_win_cell_count"] == 19
    assert hardness["strong_classical_strict_win_cell_count"] == 41
    assert hardness["sampler_tie_within_tolerance_cell_count"] == 120
    assert hardness["maximum_solver_best_mean_pair_spread"] == pytest.approx(
        0.1634287503748418
    )


def test_stage74_stochastic_methods_are_seed_deterministic_and_feasible():
    config = read_json("configs/stage74_larger_k_solver_scaling.json")
    record = read_json("data/stage72_constraint_native_cqm_model_record.json")[
        "models"
    ][4]
    model = load_model(record)
    assert k_schedule(model, config)[-1] == 16
    distributions = deficit_distributions(model["deficits"], 16)
    total = math.comb(model["count"], 16)
    quality = quality_thresholds(distributions[16], total, config)[
        "balanced_10pct_quality"
    ]
    cell = state(model, 16, "balanced_10pct_quality", quality, None)
    first = budgeted_tabu_search(
        cell, 128, 32, 7, np.random.default_rng(7401), 100000
    )
    second = budgeted_tabu_search(
        cell, 128, 32, 7, np.random.default_rng(7401), 100000
    )
    assert first == second
    annealed = constraint_preserving_annealing(
        cell, 128, 0.1, 100.0, np.random.default_rng(7402), 100000
    )
    assert feasible(cell, first["subset"])
    assert feasible(cell, annealed["subset"])


def test_stage74_route_authorizes_design_but_not_quantum_claims():
    result = read_json("data/stage74_larger_k_solver_scaling_result.json")
    scale = result["scaling_summary"]
    assert scale["model_count"] == 16
    assert scale["model_k_count"] == 100
    assert scale["workload_cell_count"] == 300
    assert scale["solver_trial_count"] == 7620
    assert scale["exact_oracle_cell_count"] == 120
    assert scale["maximum_k"] == 16
    assert scale["maximum_quadratic_coupler_count"] == 4560
    assert scale["maximum_total_fixed_k_subset_count"] == 662252084388541314
    assert all(result["route_gate"].values())
    assert result["decision"]["explicit_variable_k_cqm_design_authorized"] is True
    assert result["decision"]["hardware_shaped_sampler_poc_authorized"] is True
    assert result["decision"]["direct_qpu_execution_authorized"] is False
    assert result["decision"]["quantum_scaling_claim_authorized"] is False
    assert result["decision"]["quantum_advantage_claim_authorized"] is False


def test_stage74_independent_audit_and_data_boundary_are_frozen():
    result = read_json("data/stage74_larger_k_solver_scaling_result.json")
    audit = read_json("data/stage74_larger_k_solver_scaling_audit.json")
    assert result["status"] == "stage74_larger_k_solver_scaling_complete"
    assert audit["status"] == "stage74_larger_k_solver_scaling_independent_audit_ok"
    assert audit["stage72_models_independently_rebuilt"] == 16
    assert audit["workload_cells_independently_recomputed"] == 300
    assert audit["solver_trials_deterministically_replayed"] == 7620
    assert audit["cell_comparisons_independently_recomputed"] == 300
    assert audit["solver_summaries_independently_recomputed"] == 429
    assert audit["source_result"]["sha256"] == sha256(
        ROOT / "data/stage74_larger_k_solver_scaling_result.json"
    )
    assert result["data_boundary"] == {
        "historical_development_targets_read": 4,
        "fresh_validation_rows_read": 0,
        "locked_test_rows_read": 0,
        "new_docking_jobs": 0,
        "cloud_cqm_jobs": 0,
        "quantum_hardware_jobs": 0,
    }
