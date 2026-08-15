import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_stage75_explicit_variable_k_cqm import (
    assignment,
    budgeted_variable_tabu,
    build_cqm,
    fixed_candidate,
    load_model,
    reward_order_statistic,
    source_frontiers,
    valid,
    variable_annealing,
    variable_energy,
)


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="ascii"))


def read_csv(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build_first_cell(proposal_budget: int = 128):
    config = read_json("configs/stage75_explicit_variable_k_cqm.json")
    source = read_json("data/stage72_constraint_native_cqm_model_record.json")
    model = load_model(source["models"][0])
    frontiers = source_frontiers(
        model,
        read_csv("results/runs/stage74_larger_k_solver_scaling/workload_metrics.csv"),
        read_csv("results/runs/stage74_larger_k_solver_scaling/cell_comparison.csv"),
        read_csv("results/runs/stage74_larger_k_solver_scaling/solver_trials.csv"),
        config["variable_k_cqm"]["quality_regime"],
    )
    reward = reward_order_statistic(model, 0.5)["reward"]
    protocol = dict(config["solver_protocol"])
    protocol["proposal_budget"] = proposal_budget
    return {
        "model": model,
        "frontiers": frontiers,
        "reward": reward,
        "solver_protocol": protocol,
    }


def test_stage75_implementation_and_sources_are_hash_locked():
    config = read_json("configs/stage75_explicit_variable_k_cqm.json")
    for section in ("implementation", "inputs"):
        for value in config[section].values():
            path = ROOT / value["path"]
            assert sha256(path) == value["sha256"]
            assert path.stat().st_size == value["size_bytes"]


def test_stage75_explicit_cqm_grid_and_encoding_are_complete():
    rows = read_csv("results/runs/stage75_explicit_variable_k_cqm/cqm_metrics.csv")
    assert len(rows) == 80
    assert len(
        {
            (row["target_id"], row["outer_fold"], row["reward_quantile"])
            for row in rows
        }
    ) == 80
    assert {float(row["reward_quantile"]) for row in rows} == {
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
    }
    assert {int(row["explicit_constraint_count"]) for row in rows} == {3}
    assert max(int(row["total_logical_variable_count"]) for row in rows) == 103
    assert max(int(row["quadratic_coupler_count"]) for row in rows) == 4560
    assert max(float(row["frontier_energy_encoding_residual"]) for row in rows) < 1e-9
    assert all(row["frontier_assignment_feasible"] == "True" for row in rows)


def test_stage75_reward_paths_are_monotonic_and_use_all_candidate_budgets():
    rows = read_csv("results/runs/stage75_explicit_variable_k_cqm/cell_comparison.csv")
    paths = {}
    counts = {}
    for row in rows:
        key = (row["target_id"], int(row["outer_fold"]))
        paths.setdefault(key, []).append(row)
        k = int(row["frozen_frontier_selected_k"])
        counts[k] = counts.get(k, 0) + 1
    assert len(paths) == 16
    for path in paths.values():
        selected = [
            int(row["frozen_frontier_selected_k"])
            for row in sorted(path, key=lambda row: float(row["reward_quantile"]))
        ]
        assert selected == sorted(selected)
    assert counts == {3: 16, 4: 4, 6: 3, 8: 5, 10: 16, 12: 12, 16: 24}


def test_stage75_solver_trial_grid_and_matched_budgets_are_frozen():
    rows = read_csv("results/runs/stage75_explicit_variable_k_cqm/solver_trials.csv")
    assert len(rows) == 1440
    counts = {}
    for row in rows:
        counts[row["method"]] = counts.get(row["method"], 0) + 1
        assert int(row["selected_k"]) in {3, 4, 6, 8, 10, 12, 16}
    assert counts == {
        "fixed_k_frontier_reference": 80,
        "decomposed_deterministic_baseline": 80,
        "budgeted_variable_tabu": 640,
        "constraint_native_variable_annealing": 640,
    }
    stochastic = [row for row in rows if "variable" in row["method"]]
    assert {int(row["proposal_count"]) for row in stochastic} == {8192}


def test_stage75_solver_result_freezes_encoding_success_and_sampler_limit():
    result = read_json("data/stage75_explicit_variable_k_cqm_result.json")
    encoding = result["encoding_summary"]
    performance = result["solver_performance"]
    assert encoding["cqm_model_count"] == 80
    assert encoding["monotonic_reward_path_count"] == 16
    assert encoding["distinct_frontier_selected_k"] == [3, 4, 6, 8, 10, 12, 16]
    assert performance["exact_frontier_cell_count"] == 20
    assert performance["joint_classical_exact_frontier_match_rate"] == pytest.approx(1.0)
    assert performance["sampler_exact_frontier_match_rate"] == pytest.approx(0.65)
    assert performance["sampler_joint_classical_competitive_fraction"] == pytest.approx(0.3)
    assert performance["sampler_frozen_frontier_competitive_fraction"] == pytest.approx(0.2875)
    assert performance["frozen_frontier_refined_cell_count"] == 11
    assert result["route_gate"] == {
        "explicit_variable_k_cqm_encoding_passed": True,
        "exact_frontier_solver_validation_passed": False,
        "variable_k_sampler_competitiveness_passed": False,
    }


def test_stage75_cqm_energy_identity_and_stochastic_solvers_are_deterministic():
    cell = build_first_cell()
    subset, frontier_energy, _ = fixed_candidate(cell, "reference")
    cqm = build_cqm(cell["model"], cell["frontiers"], cell["reward"])
    sample = assignment(cell["model"], cell["frontiers"], subset)
    assert cqm.check_feasible(sample)
    assert cqm.objective.energy(sample) == pytest.approx(frontier_energy, abs=1e-10)
    assert frontier_energy == pytest.approx(
        variable_energy(cell["model"], subset, cell["reward"])
    )
    first = budgeted_variable_tabu(cell, np.random.default_rng(7501))
    second = budgeted_variable_tabu(cell, np.random.default_rng(7501))
    annealed = variable_annealing(cell, np.random.default_rng(7502))
    assert first == second
    assert valid(cell, first["subset"])
    assert valid(cell, annealed["subset"])


def test_stage75_independent_audit_replays_every_output_cell():
    audit = read_json("data/stage75_explicit_variable_k_cqm_audit.json")
    assert audit["status"] == "stage75_explicit_variable_k_cqm_independent_audit_ok"
    assert audit["stage72_models_independently_rebuilt"] == 16
    assert audit["cqm_models_independently_rebuilt"] == 80
    assert audit["solver_trials_deterministically_replayed"] == 1440
    assert audit["cell_comparisons_independently_recomputed"] == 80
    assert audit["solver_summaries_independently_recomputed"] == 100
    assert audit["source_result"]["sha256"] == sha256(
        ROOT / "data/stage75_explicit_variable_k_cqm_result.json"
    )


def test_stage75_outputs_boundaries_and_claims_are_frozen():
    result = read_json("data/stage75_explicit_variable_k_cqm_result.json")
    assert result["status"] == "stage75_explicit_variable_k_cqm_complete"
    for value in result["outputs"].values():
        path = ROOT / value["path"]
        assert sha256(path) == value["sha256"]
        assert path.stat().st_size == value["size_bytes"]
    assert result["decision"]["explicit_variable_k_cqm_freeze_authorized"] is True
    assert result["decision"]["local_hardware_shaped_emulation_authorized"] is False
    assert result["decision"]["cloud_cqm_execution_authorized"] is False
    assert result["decision"]["direct_qpu_execution_authorized"] is False
    assert result["decision"]["quantum_scaling_claim_authorized"] is False
    assert result["decision"]["quantum_advantage_claim_authorized"] is False
    assert result["data_boundary"] == {
        "historical_development_targets_read": 4,
        "fresh_validation_rows_read": 0,
        "locked_test_rows_read": 0,
        "new_docking_jobs": 0,
        "cloud_cqm_jobs": 0,
        "quantum_hardware_jobs": 0,
    }
