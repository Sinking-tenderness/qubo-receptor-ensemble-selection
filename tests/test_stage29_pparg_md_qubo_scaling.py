import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage29_pparg_md_qubo_scaling import (
    balanced_temporal_order,
    best_one_swap,
    exact_oracle,
    greedy,
    qubo_record,
    subset_components,
    subset_objective,
    temporal_maximin_order,
)


def toy_model() -> dict[str, object]:
    pair = np.array(
        [
            [0.0, 0.01, 0.05, 0.06],
            [0.01, 0.0, 0.04, 0.05],
            [0.05, 0.04, 0.0, 0.01],
            [0.06, 0.05, 0.01, 0.0],
        ]
    )
    return {
        "indices": np.arange(4),
        "frame_ids": ("F0", "F1", "F2", "F3"),
        "conformer_ids": ("A", "A", "B", "B"),
        "linear": np.array([0.02, 0.03, 0.01, 0.02]),
        "pair": pair,
        "distance": pair / 0.6,
        "redundancy": np.zeros((4, 4)),
        "k": 2,
    }


def test_temporal_maximin_order_is_complete_and_deterministic() -> None:
    assert temporal_maximin_order(range(7)) == [3, 0, 6, 1, 2, 4, 5]
    assert sorted(temporal_maximin_order(range(150))) == list(range(150))


def test_balanced_order_round_robins_across_starts() -> None:
    rows = []
    for start in range(2):
        for local in range(3):
            rows.append(
                {
                    "start_index": str(start),
                    "local_frame_index": str(local),
                    "global_frame_index": str(start * 3 + local),
                }
            )
    order = balanced_temporal_order(rows)
    assert order[:2] == [1, 4]
    assert sorted(order) == list(range(6))


def test_fixed_k_objective_and_qubo_are_exactly_equivalent() -> None:
    model = toy_model()
    selected = (0, 3)
    value = subset_objective(selected, model)
    assert abs(value - (0.02 + 0.02 + 0.06)) < 1e-12
    objective = {
        "selected_count": 2,
        "centrality_weight": 0.4,
        "pair_diversity_weight": 0.6,
        "temporal_redundancy_weight": 0.1,
        "cardinality_penalty": 2.0,
    }
    gate = {
        "direct_qpu_max_logical_variables": 200,
        "direct_qpu_max_quadratic_couplers": 20000,
        "direct_qpu_max_coefficient_dynamic_range": 1000.0,
    }
    record = qubo_record(model, selected, objective, gate)
    assert record["equivalence_residual"] < 1e-12
    assert record["logical_variable_count"] == 4
    assert record["quadratic_coupler_count"] == 6


def test_greedy_swap_matches_exact_toy_optimum() -> None:
    model = toy_model()
    selected, value, _ = best_one_swap(greedy(model), model)
    exact = exact_oracle(model, 100)
    assert exact is not None
    assert abs(value - exact["objective"]) < 1e-12


def test_component_sum_reproduces_objective() -> None:
    model = toy_model()
    objective = {
        "pair_diversity_weight": 0.6,
        "temporal_redundancy_weight": 0.1,
    }
    components = subset_components((0, 3), model, objective)
    assert abs(components["objective"] - subset_objective((0, 3), model)) < 1e-12


def test_stage29_config_is_frozen_before_solver_outcomes() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/stage29_pparg_md_qubo_solver_scaling.json").read_text(encoding="ascii"))
    assert config["evidence_timing"]["stage29_solver_outcomes_known_before_freeze"] is False
    assert config["objective"]["selected_count"] == 8
    assert config["candidate_pools"]["primary_scaling_sizes"] == [16, 32, 64, 120, 240, 480, 800, 1200]
    assert config["evidence_timing"]["docking_scores_permitted"] is False
    posthoc = {row["pool_id"]: row["posthoc"] for row in config["candidate_pools"]["sensitivity_pools"]}
    assert posthoc["exclude_3d6d_n1050"] is True


def test_stage29_result_and_audit_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    result_path = root / "data/stage29_pparg_md_qubo_solver_scaling_result.json"
    audit_path = root / "data/stage29_pparg_md_qubo_solver_scaling_audit.json"
    if not result_path.exists() or not audit_path.exists():
        return
    result = json.loads(result_path.read_text(encoding="ascii"))
    audit = json.loads(audit_path.read_text(encoding="ascii"))
    assert result["status"] == "stage29_pparg_md_qubo_solver_scaling_complete"
    assert audit["status"] == "stage29_pparg_md_qubo_solver_scaling_audit_ok"
    assert audit["checks"]["data_boundary_zero"] is True
