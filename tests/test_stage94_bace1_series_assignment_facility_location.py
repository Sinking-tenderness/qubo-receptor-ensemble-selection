import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="ascii"))


def test_stage94_has_real_combination_gain_but_fails_hardness_gate():
    result = read_json(
        "data/stage94_bace1_series_assignment_facility_location_result.json"
    )
    assert result["status"] == "stage94_bace1_series_assignment_gate_failed"
    assert result["milp"]["mip_gap"] == 0
    assert result["checks"]["milp_optimum_strictly_improves_greedy_plus_one_swap"]
    assert not result["checks"][
        "milp_optimum_strictly_improves_best_256_random_restart_one_swap"
    ]
    assert result["best_one_swap"]["replacement_distance_to_milp"] == 0


def test_stage94_strong_classical_methods_match_the_milp_optimum():
    path = (
        ROOT
        / "results/runs/stage94_bace1_series_assignment_facility_location/classical_baselines.csv"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    for method in (
        "all_pair_starts_plus_one_swap",
        "random256_plus_one_swap",
        "simulated_annealing",
        "milp_exact",
    ):
        assert rows[method]["matches_milp_open_set"].lower() == "true"
        assert abs(float(rows[method]["objective_gap_to_milp"])) < 1e-9


def test_stage94_routes_each_series_to_two_open_receptors():
    result = read_json(
        "data/stage94_bace1_series_assignment_facility_location_result.json"
    )
    opened = set(result["milp"]["open_receptor_ids"])
    path = (
        ROOT
        / "results/runs/stage94_bace1_series_assignment_facility_location/exact_assignments.csv"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert len({row["series_id"] for row in rows}) == 6
    assert all(row["receptor_a"] in opened and row["receptor_b"] in opened for row in rows)
    assert min(float(row["minimum_coverage"]) for row in rows) > 0


def test_stage94_keeps_protected_data_docking_and_quantum_locked():
    result = read_json(
        "data/stage94_bace1_series_assignment_facility_location_result.json"
    )
    assert all(value == 0 for value in result["data_boundary"].values())
    assert not any(result["authorization"].values())
    assert result["stop_rule_applies"] is True
