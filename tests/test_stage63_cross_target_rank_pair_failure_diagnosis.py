import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_stage63_cross_target_rank_pair_failure_diagnosis import (
    pairwise_jaccard,
    spearman,
)


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="ascii"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_stage61b_diagnostics_archive_local_stream_audit_passed():
    audit = read_json("data/stage61b_ppard_diagnostics_archive_local_audit.json")
    assert audit["status"] == "stage61b_ppard_diagnostics_archive_local_stream_audit_ok"
    assert audit["dimensions"]["pose_count"] == 12528
    assert audit["technical_gate"]["unresolved_warning_event_count"] == 0
    assert audit["technical_gate"]["pose_integrity_failure_count"] == 0
    assert audit["independent_recomputations"]["median_matrix_exact"] is True
    assert audit["independent_recomputations"]["minimum_matrix_exact"] is True
    assert audit["metadata_adjudication"]["raw_target_id_values"] == ["MK14"]
    assert audit["metadata_adjudication"]["scientific_score_or_pose_effect"] is False


def test_stage63_implementation_and_inputs_are_hash_locked():
    config = read_json("configs/stage63_cross_target_rank_pair_failure_diagnosis.json")
    for value in config["implementation"].values():
        assert sha256(ROOT / value["path"]) == value["sha256"]
    for value in config["inputs"].values():
        assert sha256(ROOT / value["path"]) == value["sha256"]


def test_stage63_primary_failure_mechanism_and_solver_counts():
    result = read_json("data/stage63_cross_target_rank_pair_failure_diagnosis_result.json")
    primary = result["primary_diagnosis"]
    assert result["status"] == "stage63_cross_target_rank_pair_failure_diagnosis_complete"
    assert primary["target_count"] == 4
    assert primary["outer_fold_count"] == 16
    assert primary["fixed_k_cell_count"] == 96
    assert primary["training_objective_best_k2_fold_count"] == 16
    assert primary["holdout_k2_beats_k1_fold_count"] == 2
    assert primary["k2_pair_reward_direction_conflict_fold_count"] == 14
    assert primary["mean_k2_minus_k1_holdout_bedroc"] == pytest.approx(
        -0.14408553531192292
    )
    assert primary["negative_train_holdout_spearman_fold_count"] == 15
    assert primary["exact_certified_solver_cell_count"] == 171
    assert primary["exact_over_strong_positive_gap_count"] == 0
    assert primary["exact_strong_subset_difference_count"] == 0
    assert primary["strong_or_exact_over_weak_greedy_positive_gap_count"] == 48
    assert result["mechanism"]["primary_failure_mode"] == (
        "pair_complementarity_generalization_failure"
    )
    assert result["mechanism"]["solver_search_bottleneck_supported"] is False


def test_stage63_ppard_and_claim_boundaries_are_conservative():
    result = read_json("data/stage63_cross_target_rank_pair_failure_diagnosis_result.json")
    primary = result["primary_diagnosis"]
    assert primary["ppard_mean_inner_outer_k_curve_spearman"] == pytest.approx(
        0.01428571428571429
    )
    assert primary["ppard_mean_selected_k_outer_regret"] == pytest.approx(
        0.043385115850299416
    )
    assert result["next_objective_design"]["status"] == (
        "design_requirements_only_not_frozen"
    )
    assert result["decision"]["same_target_ppard_retuning_authorized"] is False
    assert result["decision"]["fresh_validation_authorized"] is False
    assert result["decision"]["quantum_hardware_authorized"] is False
    assert result["decision"]["new_cross_target_objective_development_authorized"] is True


def test_stage63_independent_audit_passed():
    result = read_json("data/stage63_cross_target_rank_pair_failure_diagnosis_result.json")
    audit = read_json("data/stage63_cross_target_rank_pair_failure_diagnosis_audit.json")
    assert audit["status"] == (
        "stage63_cross_target_rank_pair_failure_diagnosis_independent_audit_ok"
    )
    assert audit["primary_diagnosis_independently_recomputed"] is True
    assert audit["decision_boundary_exact"] is True
    assert audit["source_result"]["sha256"] == sha256(
        ROOT / "data/stage63_cross_target_rank_pair_failure_diagnosis_result.json"
    )
    assert result["analysis_payload_sha256"] == (
        "6117CEF89D04F0B59382E0337ADBDCEC6E07476A3E271FC48A63782C43CC1755"
    )


def test_stage63_small_statistical_helpers():
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert pairwise_jaccard(["A+B", "A+C"]) == pytest.approx(1.0 / 3.0)
