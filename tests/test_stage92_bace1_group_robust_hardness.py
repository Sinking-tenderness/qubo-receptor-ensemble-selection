import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="ascii"))


def test_stage92_reports_real_combination_effect_but_failed_hardness_gate():
    result = read_json(
        "data/stage92_bace1_group_robust_hardness_adjudication_result.json"
    )
    assert result["status"] == "stage92_bace1_group_robust_hardness_gate_failed"
    assert result["state_count"] == 1344904
    assert result["comparisons"]["exact_minus_direct_greedy"] > 0
    assert result["comparisons"]["exact_minus_greedy_swap"] == 0
    assert result["checks"]["exact_solution_differs_from_direct_greedy"] is True
    assert result["checks"]["reproducible_multi_move_local_trap"] is False


def test_stage92_strong_classical_methods_reach_exact_solution():
    path = (
        ROOT
        / "results/runs/stage92_bace1_group_robust_hardness_adjudication/classical_baselines.csv"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    for method in (
        "greedy_plus_all_one_swaps",
        "all_singleton_greedy_plus_swaps",
        "multistart_tabu",
        "simulated_annealing",
        "exact_enumeration",
    ):
        assert rows[method]["matches_exact_selection"].lower() == "true"
        assert float(rows[method]["objective_gap_to_exact"]) == 0


def test_stage92_keeps_confirmation_and_quantum_locked():
    result = read_json(
        "data/stage92_bace1_group_robust_hardness_adjudication_result.json"
    )
    assert result["authorization"] == {
        "confirmation_a_preparation_or_docking_authorized": False,
        "confirmation_b_authorized": False,
        "locked_test_authorized": False,
        "quantum_simulation_or_hardware_authorized": False,
    }
    assert result["data_boundary"] == {
        "confirmation_scores_read": 0,
        "locked_test_scores_read": 0,
        "new_docking_jobs": 0,
        "quantum_jobs": 0,
    }
