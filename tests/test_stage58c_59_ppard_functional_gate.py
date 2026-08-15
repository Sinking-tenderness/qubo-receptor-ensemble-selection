import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text())


def read_csv(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_stage58c_changes_only_inherited_target_id():
    result = read_json("data/stage58c_ppard_target_id_amendment_result.json")
    audit = read_json("data/stage58c_ppard_target_id_amendment_audit.json")
    assert result["status"] == "stage58c_ppard_target_id_amendment_ok"
    assert audit["status"] == "stage58c_ppard_target_id_amendment_independent_audit_ok"
    assert result["changed_field_count"] == 8352
    assert result["non_target_row_fingerprints_exact"] is True
    assert audit["docking_scores_changed"] == 0
    assert audit["pose_fields_changed"] == 0
    assert result["target_id_before"] == ["MK14"]
    assert result["target_id_after"] == ["PPARD"]


def test_stage59_preregistered_gate_passes_all_five_checks():
    result = read_json("data/stage59_ppard_functional_complementarity_gate_result.json")
    audit = read_json("data/stage59_ppard_functional_complementarity_gate_audit.json")
    assert result["status"] == "stage59_ppard_functional_complementarity_gate_complete"
    assert audit["status"] == "stage59_ppard_functional_complementarity_independent_audit_ok"
    assert result["decision"]["functional_complementarity_gate_passed"] is True
    assert all(result["gate"]["observed_checks"].values())
    assert audit["preregistered_gate_checks_exact"] is True
    assert audit["functional_complementarity_gate_passed"] is True


def test_stage59_exhausts_singles_pairs_and_frozen_folds():
    result = read_json("data/stage59_ppard_functional_complementarity_gate_result.json")
    pairs = read_csv("results/runs/stage59_ppard_functional_complementarity_gate/pair_metrics.csv")
    folds = read_csv("results/runs/stage59_ppard_functional_complementarity_gate/fold_gate.csv")
    assert result["coverage"] == {
        "receptor_count": 29,
        "ligand_count": 96,
        "seed_count": 3,
        "single_receptor_count": 29,
        "pair_count": 406,
        "outer_fold_count": 4,
    }
    assert len(pairs) == 406
    assert len({row["pair"] for row in pairs}) == 406
    assert len(folds) == 4
    assert [int(row["holdout_ligand_count"]) for row in folds] == [24, 24, 24, 24]


def test_stage59_metrics_are_frozen_regression_values():
    diagnosis = read_json(
        "data/stage59_ppard_functional_complementarity_gate_result.json"
    )["diagnosis"]
    assert diagnosis["dominant_single_receptor"] == "PPARD_2ZNQ_aligned"
    assert diagnosis["best_pair"] == "PPARD_2XYX_aligned+PPARD_3TKM_aligned"
    assert abs(diagnosis["best_pair_gain_over_best_member"] - 0.06890861779430912) < 1e-12
    assert abs(diagnosis["mean_holdout_pair_gain"] - 0.06777438817203263) < 1e-12
    assert diagnosis["positive_holdout_pair_gain_fold_count"] == 2
    assert diagnosis["positive_rescue_balance_dominant_addition_count"] == 22
    assert abs(diagnosis["median_pair_rank_spearman"] - 0.7801638191424729) < 1e-12


def test_stage59_authorizes_only_next_development_step():
    result = read_json("data/stage59_ppard_functional_complementarity_gate_result.json")
    decision = result["decision"]
    assert decision["transferred_qubo_objective_freeze_authorized"] is True
    assert decision["remaining_development_train_docking_authorized"] is True
    assert decision["same_pilot_objective_or_threshold_retuning_authorized"] is False
    assert decision["fresh_validation_authorized"] is False
    assert decision["locked_test_authorized"] is False
    assert decision["quantum_hardware_authorized"] is False
    assert result["data_boundary"] == {
        "pilot_train_rows_read": 96,
        "remaining_development_train_rows_read": 0,
        "fresh_validation_rows_read": 0,
        "locked_test_rows_read": 0,
        "new_docking_jobs": 0,
        "quantum_hardware_jobs": 0,
    }
