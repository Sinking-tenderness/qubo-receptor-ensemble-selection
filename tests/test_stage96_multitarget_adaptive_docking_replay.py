import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_stage96_is_real_posthoc_replay_without_new_jobs():
    result = read_json("data/stage96_multitarget_adaptive_docking_replay_result.json")
    assert result["status"] == "stage96_replay_complete"
    assert result["audit"] == {
        "labels_used_by_selector": False,
        "docking_scores_revealed_only_after_task_selection": True,
        "synthetic_scores": 0,
        "new_docking_jobs": 0,
        "fresh_validation_rows": 0,
    }


def test_stage96_has_all_targets_policies_seeds_and_checkpoints():
    path = ROOT / "results/runs/stage96_multitarget_adaptive_docking_replay/checkpoints.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 108
    assert {row["target_id"] for row in rows} == {"PPARG", "BACE1"}
    assert {row["policy"] for row in rows} == {
        "random",
        "predicted_mean",
        "predictive_uncertainty",
        "qubo_direct_greedy",
        "qubo_greedy_one_swap",
        "qubo_exact_milp",
    }
    assert {float(row["checkpoint_fraction"]) for row in rows} == {0.1, 0.2, 0.3}
    assert {int(row["replay_seed"]) for row in rows} == {20260824, 20260825, 20260826}


def test_stage96_policy_gate_and_solver_value_fail_conservatively():
    result = read_json("data/stage96_multitarget_adaptive_docking_replay_result.json")
    assert result["policy_gate"]["passes"] is False
    assert result["solver_value"]["passes"] is False
    assert result["hardware_authorization"]["authorized"] is False


def test_stage96_exact_solver_matches_one_swap_in_this_instance():
    path = ROOT / "results/runs/stage96_multitarget_adaptive_docking_replay/qubo_solver_comparisons.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 387
    assert max(abs(float(row["exact_minus_one_swap"])) for row in rows) < 1e-9


def test_stage96_chemistry_batches_are_structure_only_and_balanced():
    summary = read_json("data/stage96_multitarget_balanced_chemistry_batches_summary.json")
    assert summary["status"] == "stage96_balanced_chemistry_batches_structure_only_ok"
    assert summary["labels_read"] == 0
    assert summary["docking_score_rows_read"] == 0
    for target in summary["targets"].values():
        assert target["capacity_difference"] <= 1
