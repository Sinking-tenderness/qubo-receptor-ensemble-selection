import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.run_stage75_explicit_variable_k_cqm as s75
from scripts.run_stage76_variable_k_sampler_repair import (
    adjacent_annealing,
    parallel_tempering,
)


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="ascii"))


def read_csv(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build_smoke_cell():
    config = read_json("configs/stage76_variable_k_sampler_repair.json")
    source = read_json("data/stage72_constraint_native_cqm_model_record.json")
    model = s75.load_model(source["models"][0])
    frontiers = s75.source_frontiers(
        model,
        read_csv("results/runs/stage74_larger_k_solver_scaling/workload_metrics.csv"),
        read_csv("results/runs/stage74_larger_k_solver_scaling/cell_comparison.csv"),
        read_csv("results/runs/stage74_larger_k_solver_scaling/solver_trials.csv"),
        config["variable_k_cqm"]["quality_regime"],
    )
    protocol = dict(config["solver_protocol"])
    protocol["proposal_budget"] = 128
    return {
        "model": model,
        "frontiers": frontiers,
        "reward_quantile": 0.5,
        "reward": s75.reward_order_statistic(model, 0.5)["reward"],
        "solver_protocol": protocol,
    }


def test_stage76_implementation_and_sources_are_hash_locked():
    config = read_json("configs/stage76_variable_k_sampler_repair.json")
    for section in ("implementation", "inputs"):
        for value in config[section].values():
            path = ROOT / value["path"]
            assert sha256(path) == value["sha256"]
            assert path.stat().st_size == value["size_bytes"]


def test_stage76_preserves_all_stage75_cqm_identities():
    rows = read_csv("results/runs/stage76_variable_k_sampler_repair/cqm_identity.csv")
    assert len(rows) == 80
    assert len(
        {
            (row["target_id"], row["outer_fold"], row["reward_quantile"])
            for row in rows
        }
    ) == 80
    assert all(row["cqm_hash_match"] == "True" for row in rows)
    assert all(row["assignment_feasible"] == "True" for row in rows)
    assert max(float(row["energy_residual"]) for row in rows) < 1e-9
    result = read_json("data/stage76_variable_k_sampler_repair_result.json")
    assert result["encoding_summary"]["objective_or_constraint_changes"] == 0
    assert result["encoding_summary"]["cqm_hash_match_count"] == 80


def test_stage76_trial_grid_and_matched_proposal_budgets_are_complete():
    rows = read_csv("results/runs/stage76_variable_k_sampler_repair/solver_trials.csv")
    assert len(rows) == 3200
    counts = {}
    for row in rows:
        counts[row["method"]] = counts.get(row["method"], 0) + 1
        assert int(row["proposal_count"]) == 8192
    assert counts == {
        "stage75_cold_global_annealing_reference": 640,
        "cold_adjacent_variable_annealing": 640,
        "decomposed_warm_adjacent_annealing": 640,
        "decomposed_warm_parallel_tempering": 640,
        "frontier_warm_parallel_tempering": 640,
    }
    tempering = [row for row in rows if "parallel_tempering" in row["method"]]
    assert all(int(row["exchange_attempt_count"]) > 0 for row in tempering)
    assert all(int(row["accepted_exchange_count"]) > 0 for row in tempering)


def test_stage76_method_fidelity_progression_and_gates_are_frozen():
    result = read_json("data/stage76_variable_k_sampler_repair_result.json")
    summary = {row["method"]: row for row in result["method_summaries"]}
    source = summary["stage75_cold_global_annealing_reference"]
    cold = summary["cold_adjacent_variable_annealing"]
    warm = summary["decomposed_warm_adjacent_annealing"]
    tempering = summary["decomposed_warm_parallel_tempering"]
    frontier = summary["frontier_warm_parallel_tempering"]
    assert source["exact_frontier_match_rate"] == pytest.approx(0.65)
    assert source["joint_competitive_fraction"] == pytest.approx(0.3)
    assert cold["exact_frontier_match_rate"] == pytest.approx(0.7)
    assert cold["joint_competitive_fraction"] == pytest.approx(0.3375)
    assert warm["exact_frontier_match_rate"] == pytest.approx(0.85)
    assert warm["joint_competitive_fraction"] == pytest.approx(0.5125)
    assert tempering["exact_frontier_match_rate"] == pytest.approx(1.0)
    assert tempering["joint_competitive_fraction"] == pytest.approx(0.6)
    assert frontier["exact_frontier_match_rate"] == pytest.approx(1.0)
    assert frontier["joint_competitive_fraction"] == pytest.approx(0.875)
    assert frontier["frontier_competitive_fraction"] == pytest.approx(1.0)
    assert frontier["strict_frontier_improvement_cell_count"] == 5
    assert result["route_gate"] == {
        "stage75_cqm_identity_preserved": True,
        "cold_start_sampler_repair_passed": False,
        "warm_start_parallel_tempering_fidelity_passed": True,
    }


def test_stage76_mechanism_ablation_is_not_a_success_only_summary():
    result = read_json("data/stage76_variable_k_sampler_repair_result.json")
    ablations = result["ablation_summary"]
    assert [row["right_strict_win_cell_count"] for row in ablations] == [
        31,
        45,
        23,
        34,
    ]
    assert [row["left_strict_win_cell_count"] for row in ablations] == [
        29,
        10,
        1,
        0,
    ]
    assert [row["tie_cell_count"] for row in ablations] == [20, 25, 56, 46]
    rows = read_csv(
        "results/runs/stage76_variable_k_sampler_repair/cell_method_comparison.csv"
    )
    improved = [
        row
        for row in rows
        if row["method"] == "frontier_warm_parallel_tempering"
        and row["strict_frontier_improvement"] == "True"
    ]
    assert len(improved) == 5
    assert {row["target_id"] for row in improved} == {"PPARG"}


def test_stage76_new_samplers_are_seed_deterministic_and_feasible():
    cell = build_smoke_cell()
    first = adjacent_annealing(
        cell, np.random.default_rng(7601), "cold_random"
    )
    second = adjacent_annealing(
        cell, np.random.default_rng(7601), "cold_random"
    )
    warm_first = parallel_tempering(
        cell, np.random.default_rng(7602), "reference"
    )
    warm_second = parallel_tempering(
        cell, np.random.default_rng(7602), "reference"
    )
    assert first == second
    assert warm_first == warm_second
    assert s75.valid(cell, first["subset"])
    assert s75.valid(cell, warm_first["subset"])
    assert warm_first["exchange_attempt_count"] == 14


def test_stage76_independent_audit_replays_all_solver_trajectories():
    audit = read_json("data/stage76_variable_k_sampler_repair_audit.json")
    assert audit["status"] == "stage76_variable_k_sampler_repair_independent_audit_ok"
    assert audit["stage75_cqm_models_independently_rebuilt"] == 80
    assert audit["solver_trials_deterministically_replayed"] == 3200
    assert audit["cell_method_comparisons_independently_recomputed"] == 400
    assert audit["method_summaries_independently_recomputed"] == 5
    assert audit["source_result"]["sha256"] == sha256(
        ROOT / "data/stage76_variable_k_sampler_repair_result.json"
    )


def test_stage76_hybrid_route_and_claim_boundaries_are_explicit():
    result = read_json("data/stage76_variable_k_sampler_repair_result.json")
    decision = result["decision"]
    assert decision["objective_redesign_required"] is False
    assert decision["explicit_variable_k_cqm_remains_frozen"] is True
    assert decision["standalone_cold_sampler_ready"] is False
    assert decision["local_warm_start_hardware_shaped_emulation_authorized"] is True
    assert decision["cloud_cqm_execution_authorized"] is False
    assert decision["direct_qpu_execution_authorized"] is False
    assert decision["quantum_scaling_claim_authorized"] is False
    assert decision["quantum_advantage_claim_authorized"] is False
    for value in result["outputs"].values():
        path = ROOT / value["path"]
        assert sha256(path) == value["sha256"]
        assert path.stat().st_size == value["size_bytes"]
    assert result["data_boundary"] == {
        "historical_development_targets_read": 4,
        "fresh_validation_rows_read": 0,
        "locked_test_rows_read": 0,
        "new_docking_jobs": 0,
        "cloud_cqm_jobs": 0,
        "quantum_hardware_jobs": 0,
    }
