import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage30_pparg_group_balanced_state_qubo import (
    coordinate_descent,
    cyclic_orders,
    exact_oracle,
    greedy_for_order,
    qubo_record,
    subset_objective,
    validate_selected,
)


def toy_model() -> dict[str, object]:
    pair = np.array(
        [
            [0.0, 0.0, 0.01, 0.05, 0.03, 0.06],
            [0.0, 0.0, 0.04, 0.02, 0.05, 0.01],
            [0.01, 0.04, 0.0, 0.0, 0.02, 0.06],
            [0.05, 0.02, 0.0, 0.0, 0.05, 0.02],
            [0.03, 0.05, 0.02, 0.05, 0.0, 0.0],
            [0.06, 0.01, 0.06, 0.02, 0.0, 0.0],
        ]
    )
    return {
        "per_start": 2,
        "global_indices": np.arange(6),
        "groups": [(0, 1), (2, 3), (4, 5)],
        "linear": np.array([0.03, 0.02, 0.01, 0.03, 0.02, 0.01]),
        "pair": pair,
        "k": 3,
        "frame_ids": tuple(f"F{i}" for i in range(6)),
        "source_ids": ("A", "A", "B", "B", "C", "C"),
        "distance": pair,
        "state_separation": np.zeros_like(pair),
    }


def test_exactly_one_group_constraint_is_enforced() -> None:
    model = toy_model()
    assert validate_selected((0, 2, 4), model) == (0, 2, 4)
    try:
        validate_selected((0, 1, 4), model)
    except ValueError as error:
        assert "exactly-one-per-start" in str(error)
    else:
        raise AssertionError("invalid group assignment was accepted")


def test_coordinate_descent_reaches_toy_exact_value() -> None:
    model = toy_model()
    exact = exact_oracle(model, 100)
    assert exact is not None
    best_value = -np.inf
    for order in cyclic_orders(3):
        _, value, _ = coordinate_descent(greedy_for_order(model, order), model)
        best_value = max(best_value, value)
    assert abs(best_value - exact["objective"]) < 1e-12


def test_group_qubo_energy_matches_feasible_objective() -> None:
    model = toy_model()
    selected = (0, 3, 5)
    objective = {"exactly_one_group_penalty": 2.0}
    gate = {
        "direct_qpu_max_logical_variables": 200,
        "direct_qpu_max_quadratic_couplers": 20000,
        "direct_qpu_max_coefficient_dynamic_range": 10000.0,
    }
    record = qubo_record(model, selected, objective, gate)
    assert record["equivalence_residual"] < 1e-12
    assert record["logical_variable_count"] == 6
    assert record["quadratic_coupler_count"] == 15
    assert subset_objective(selected, model) > 0


def test_stage30_config_is_frozen_before_solver_outcomes() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/stage30_pparg_group_balanced_state_qubo.json").read_text(encoding="ascii"))
    assert config["evidence_timing"]["stage30_solver_outcomes_known_before_freeze"] is False
    assert config["candidate_scaling"]["frames_per_start"] == [2, 4, 8, 16, 32, 64, 100, 150]
    assert config["structural_states"]["cluster_counts"] == [8, 16, 32]
    assert config["objective"]["group_count"] == 8
    assert config["evidence_timing"]["docking_scores_permitted"] is False


def test_stage30_result_and_audit_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    result_path = root / "data/stage30_pparg_group_balanced_state_qubo_result.json"
    audit_path = root / "data/stage30_pparg_group_balanced_state_qubo_audit.json"
    if not result_path.exists() or not audit_path.exists():
        return
    result = json.loads(result_path.read_text(encoding="ascii"))
    audit = json.loads(audit_path.read_text(encoding="ascii"))
    assert result["status"] == "stage30_pparg_group_balanced_state_qubo_complete"
    assert audit["status"] == "stage30_pparg_group_balanced_state_qubo_audit_ok"
    assert audit["checks"]["data_boundary_zero"] is True
