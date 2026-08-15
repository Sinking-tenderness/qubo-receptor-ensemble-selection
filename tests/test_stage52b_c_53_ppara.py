import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text())


def read_csv(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_stage52b_downloaded_archives_pass_with_metadata_only_defect():
    audit = read_json("data/stage52b_ppara_train374_downloaded_result_audit.json")
    assert audit["technical_integrity"]["batch_count"] == 60
    assert audit["technical_integrity"]["pair_count"] == 22440
    assert audit["technical_integrity"]["pose_hash_mismatch_count"] == 0
    assert audit["metadata_adjudication"]["observed_target_ids"] == ["MK14"]
    assert audit["metadata_adjudication"]["score_or_pose_effect"] is False


def test_stage52c_changes_only_target_id():
    result = read_json("data/stage52c_ppara_target_id_amendment_result.json")
    assert result["status"] == "stage52c_ppara_target_id_amendment_ok"
    assert result["changed_field_count"] == 22440
    assert result["non_target_row_fingerprints_exact"] is True
    assert result["target_id_before"] == ["MK14"]
    assert result["target_id_after"] == ["PPARA"]
    rows = read_csv("results/runs/stage52c_ppara_target_id_amendment/scores.csv")
    assert len(rows) == 22440
    assert {row["target_id"] for row in rows} == {"PPARA"}


def test_stage53_records_complete_frozen_comparison_and_no_go():
    result = read_json("data/stage53_ppara_large_pool_qubo_transfer_result.json")
    assert result["status"] == "stage53_ppara_large_pool_qubo_transfer_complete"
    assert result["input_statistics"]["receptor_count"] == 20
    assert result["input_statistics"]["ligand_count"] == 374
    assert result["input_statistics"]["state_count_k1_to_k6"] == 60459
    assert len(result["full_data_methods"]) == 12
    assert result["decision"]["frozen_qubo_application_transfer_supported"] is False
    assert result["decision"]["solver_novelty_detected"] is False
    assert result["decision"]["same_data_weight_retuning_authorized"] is False


def test_stage53_fixed_k_ladder_identifies_single_receptor_regime():
    rows = read_csv("results/runs/stage53_ppara_large_pool_qubo_transfer/fixed_k_landscape.csv")
    exact = [row for row in rows if row["method"] == "rank_pair_qubo_exact"]
    means = {
        size: sum(
            float(row["evaluation_robust_bedroc"])
            for row in exact
            if int(row["subset_size"]) == size
        )
        / 4
        for size in range(1, 7)
    }
    assert means[1] > max(means[size] for size in range(2, 7))


def test_stage53_never_reads_protected_rows():
    result = read_json("data/stage53_ppara_large_pool_qubo_transfer_result.json")
    assert result["data_boundary"]["fresh_validation_rows_read"] == 0
    assert result["data_boundary"]["locked_test_rows_read"] == 0
    assert result["data_boundary"]["quantum_hardware_jobs"] == 0
