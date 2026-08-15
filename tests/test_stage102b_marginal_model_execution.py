import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage102b_is_a_conservative_no_go():
    result = json.loads(
        (ROOT / "data/stage102b_marginal_model_execution_result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "stage102b_marginal_model_execution_complete"
    assert result["decision"]["phase_a_gate_passes"] is False
    assert result["selected_candidate"] is None
    assert all(not decision["passes"] for decision in result["candidate_decisions"].values())
    assert result["decision"]["parp1_released"] is False
    assert result["decision"]["quantum_hardware_authorized"] is False
    assert all(value == 0 for value in result["data_boundary"].values())


def test_stage102b_has_complete_nested_coverage_without_candidate_leakage():
    run = ROOT / "results/runs/stage102b_marginal_model_execution"
    with (run / "marginal_edge_features.csv").open(encoding="utf-8", newline="") as handle:
        edges = list(csv.DictReader(handle))
    with (run / "fold_decisions.csv").open(encoding="utf-8", newline="") as handle:
        folds = list(csv.DictReader(handle))
    assert len(edges) == 70
    assert len(folds) == 700
    assert {row["target_id"] for row in edges} == {
        "BACE1", "EGFR", "FA10", "MK14", "PPARA", "PPARD", "PPARG"
    }
    assert all(row["target_id"] not in row["ridge_training_targets"].split("|") for row in edges)
    candidate_rows = [
        row
        for row in folds
        if row["policy"] in {"mechanistic_bootstrap_lcb", "target_held_out_l2_ridge"}
    ]
    assert len(candidate_rows) == 210
    assert all(row["uses_outer_labels_for_selection"] == "False" for row in candidate_rows)


def test_stage102b_independent_audit_is_present_and_passing():
    audit = json.loads(
        (ROOT / "data/stage102b_marginal_model_execution_audit.json").read_text(encoding="utf-8")
    )
    assert audit["status"] == "stage102b_independent_audit_ok"
    assert audit["marginal_edge_count"] == 70
    assert audit["fold_decision_count"] == 700
    assert audit["outer_labels_used_by_candidate_selectors"] is False
    assert all(not value for value in audit["candidate_passes"].values())
