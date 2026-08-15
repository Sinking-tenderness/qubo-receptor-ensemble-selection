import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_stage62_ppard_train240_nested_qubo import one_standard_error


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="ascii"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_stage61c_changes_only_the_inherited_target_id():
    result = read_json("data/stage61c_ppard_target_id_amendment_result.json")
    audit = read_json("data/stage61c_ppard_target_id_amendment_audit.json")
    assert result["status"] == "stage61c_ppard_target_id_amendment_ok"
    assert audit["status"] == "stage61c_ppard_target_id_amendment_independent_audit_ok"
    assert result["row_count"] == 12528
    assert result["changed_field_count"] == 12528
    assert audit["changed_columns"] == ["target_id"]
    assert audit["docking_scores_changed"] == 0
    assert audit["pose_fields_changed"] == 0
    assert audit["target_id_before"] == ["MK14"]
    assert audit["target_id_after"] == ["PPARD"]


def test_stage62_implementation_and_frozen_inputs_are_hash_locked():
    config = read_json("configs/stage62_ppard_train240_nested_qubo.json")
    for value in config["implementation"].values():
        path = ROOT / value["path"]
        assert sha256(path) == value["sha256"]
    for value in config["inputs"].values():
        path = ROOT / value["path"]
        assert sha256(path) == value["sha256"]
    stage60 = read_json("data/stage60_ppard_transferred_qubo_freeze_result.json")
    assert config["objective"] == stage60["transferred_objective"]
    assert config["nested_cv"] == stage60["nested_cv"]


def test_stage62_result_and_full_recomputation_audit_agree():
    result = read_json("data/stage62_ppard_train240_nested_qubo_result.json")
    audit = read_json("data/stage62_ppard_train240_nested_qubo_audit.json")
    assert result["status"] == "stage62_ppard_train240_frozen_nested_qubo_complete"
    assert audit["status"] == "stage62_ppard_train240_full_recomputation_audit_ok"
    assert audit["analysis_payload_sha256"] == result["analysis_payload_sha256"]
    assert audit["all_csv_rows_exact"] is True
    assert audit["model_record_exact"] is True
    assert audit["row_counts"]["merged_scores_csv"] == 20880
    assert audit["row_counts"]["objective_gap_cells_csv"] == 102
    assert result["final_k_selection"]["selected_k"] == 1
    assert result["decision"]["transferred_qubo_application_supported"] is False
    assert result["decision"]["fresh_validation_authorized"] is False
    assert result["decision"]["optimization_novelty_supported"] is False
    assert result["decision"]["positive_objective_gap_cells_over_strong_classical"] == 0
    assert result["data_boundary"]["fresh_validation_rows_read"] == 0
    assert result["data_boundary"]["locked_test_rows_read"] == 0


def test_stage62_reported_performance_and_gap_landscape():
    result = read_json("data/stage62_ppard_train240_nested_qubo_result.json")
    performance = result["performance"]
    assert performance["mean_nested_outer_rank_pair_qubo_exact_robust_bedroc"] == pytest.approx(
        0.7774504219886386
    )
    assert performance["mean_gain_over_best_single"] == pytest.approx(
        -0.02269229016353755
    )
    assert performance["mean_gain_over_linear_topk"] == pytest.approx(
        -0.03053952241422861
    )
    assert performance["mean_gain_over_nested_bedroc_greedy"] == pytest.approx(
        0.004990096059487581
    )
    with (
        ROOT
        / "results/runs/stage62_ppard_train240_nested_qubo/objective_gap_cells.csv"
    ).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 102
    assert sum(float(row["exact_minus_strong_gap"]) > 1e-12 for row in rows) == 0
    assert sum(float(row["exact_minus_direct_greedy_gap"]) > 1e-12 for row in rows) == 32


def test_one_standard_error_rule_prefers_the_smallest_eligible_k():
    choice = one_standard_error(
        {1: [0.80, 0.81, 0.79], 2: [0.82, 0.84, 0.83], 3: [0.83, 0.84, 0.82]}
    )
    assert choice["best_k"] == 2
    assert choice["selected_k"] == 2
