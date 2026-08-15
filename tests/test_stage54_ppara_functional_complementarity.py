import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text())


def read_csv(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_stage54_resolves_ppara_as_single_receptor_dominated():
    result = read_json(
        "data/stage54_ppara_functional_complementarity_diagnosis_result.json"
    )
    diagnosis = result["diagnosis"]
    assert result["status"] == (
        "stage54_ppara_functional_complementarity_diagnosis_complete"
    )
    assert diagnosis["dominant_single_receptor"] == "PPARA_5HYK_aligned"
    assert diagnosis["best_pair_gain_over_best_member"] < 0
    assert diagnosis["positive_dominant_addition_bedroc_count"] == 0
    assert diagnosis["single_receptor_dominance_confirmed"] is True
    assert diagnosis["decoy_promotion_failure_confirmed"] is True


def test_stage54_exhausts_all_pairs_and_scaffold_folds():
    pairs = read_csv(
        "results/runs/stage54_ppara_functional_complementarity_diagnosis/"
        "pair_diagnostics.csv"
    )
    folds = read_csv(
        "results/runs/stage54_ppara_functional_complementarity_diagnosis/"
        "fold_oracle_diagnostics.csv"
    )
    assert len(pairs) == 190
    assert len({row["pair"] for row in pairs}) == 190
    assert len(folds) == 4
    assert sum(float(row["holdout_pair_gain"]) for row in folds) / 4 < 0


def test_stage54_freezes_prospective_intake_gate_without_retuning_ppara():
    result = read_json(
        "data/stage54_ppara_functional_complementarity_diagnosis_result.json"
    )
    future = read_json("data/stage54_future_target_intake_criteria.json")
    assert future["status"] == "stage54_future_target_intake_criteria_frozen"
    assert future["ppara_pass"] is False
    assert result["decision"]["future_target_intake_criteria_frozen"] is True
    assert result["decision"]["small_pilot_required_before_full_matrix"] is True
    assert result["decision"]["ppara_same_data_retuning_authorized"] is False
    assert result["decision"]["ppara_fresh_validation_authorized"] is False
    assert result["decision"]["quantum_hardware_authorized"] is False


def test_stage54_never_reads_protected_rows_or_runs_new_jobs():
    result = read_json(
        "data/stage54_ppara_functional_complementarity_diagnosis_result.json"
    )
    boundary = result["data_boundary"]
    assert boundary["fresh_validation_rows_read"] == 0
    assert boundary["locked_test_rows_read"] == 0
    assert boundary["new_docking_jobs"] == 0
    assert boundary["quantum_hardware_jobs"] == 0
