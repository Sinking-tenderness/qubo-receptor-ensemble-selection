import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage31_pparg_objective_landscape_screen import (
    all_assignments,
    cyclic_orders,
    objective_values,
    successor_and_local_metrics,
)


def test_exact_assignment_space_is_complete_and_ordered() -> None:
    assignments = all_assignments()
    assert assignments.shape == (4**8, 8)
    assert assignments.dtype == np.int8
    assert np.array_equal(assignments[0], np.zeros(8, dtype=np.int8))
    assert np.array_equal(assignments[-1], np.full(8, 3, dtype=np.int8))
    encoded = assignments @ np.asarray([4 ** (7 - group) for group in range(8)])
    assert np.array_equal(encoded, np.arange(4**8))


def test_objective_formulas_match_frozen_coefficients() -> None:
    values = {
        "single_q05": np.array([0.1]),
        "single_q10": np.array([0.2]),
        "single_q20": np.array([0.3]),
        "double_q10": np.array([0.4]),
        "worst_start_single_q10": np.array([0.5]),
        "worst_two_start_single_q10": np.array([0.6]),
        "mean_pair_distance": np.array([0.7]),
        "mean_within_start_centrality": np.array([0.8]),
        "mean_multiscale_state_separation": np.array([0.9]),
    }
    assert np.isclose(objective_values("robust_coverage_pair", values)[0], 0.43)
    assert np.isclose(objective_values("multiscale_robust_coverage", values)[0], 0.30)
    assert np.isclose(objective_values("worst_two_single_double", values)[0], 0.42)
    assert np.isclose(objective_values("worst_start_single", values)[0], 0.50)
    assert np.isclose(objective_values("global_single_double", values)[0], 0.35)
    assert np.isclose(objective_values("smooth_pair_control", values)[0], 0.77)


def test_exact_landscape_recovers_single_global_basin() -> None:
    assignments = all_assignments()
    scores = -np.square(assignments.astype(float)).sum(axis=1)
    landscape = successor_and_local_metrics(scores, assignments, 1e-12)
    assert landscape["optimum_states"] == {0}
    assert int(landscape["strict_local"].sum()) == 1
    assert int(landscape["weak_local"].sum()) == 1
    assert np.all(landscape["endpoints"] == 0)
    assert np.isclose(landscape["optimum_basin_fraction"], 1.0)


def test_sixteen_cyclic_orders_are_unique() -> None:
    orders = cyclic_orders()
    assert len(orders) == 16
    assert len(set(orders)) == 16
    assert all(sorted(order) == list(range(8)) for order in orders)


def test_stage31_config_is_frozen_before_landscape_outcomes() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/stage31_pparg_objective_landscape_screen.json").read_text(encoding="ascii"))
    assert config["evidence_timing"]["stage31_landscape_outcomes_known_before_freeze"] is False
    assert len(config["exact_candidate_cohorts"]) == 3
    assert len(config["objective_families_in_priority_order"]) == 6
    assert config["landscape"]["state_count_per_cohort"] == 4**8
    assert config["difficulty_gate"]["minimum_passing_cohorts_per_objective"] == 2
    assert config["evidence_timing"]["docking_scores_permitted"] is False


def test_stage31_result_and_audit_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    result_path = root / "data/stage31_pparg_objective_landscape_screen_result.json"
    audit_path = root / "data/stage31_pparg_objective_landscape_screen_audit.json"
    if not result_path.exists() or not audit_path.exists():
        return
    result = json.loads(result_path.read_text(encoding="ascii"))
    audit = json.loads(audit_path.read_text(encoding="ascii"))
    assert result["status"] == "stage31_pparg_objective_landscape_screen_complete"
    assert audit["status"] == "stage31_pparg_objective_landscape_screen_audit_ok"
    assert audit["checks"]["data_boundary_zero"] is True
