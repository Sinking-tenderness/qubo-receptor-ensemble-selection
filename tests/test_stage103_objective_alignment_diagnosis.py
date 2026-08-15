import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage103_is_diagnostic_and_keeps_boundaries_closed():
    result = json.loads((ROOT / "data/stage103_objective_alignment_diagnosis_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "stage103_objective_alignment_diagnosis_complete"
    assert result["summary"]["k2_target_median_qubo_vs_train_primary_spearman"] == 0.6481339474340302
    assert result["summary"]["k2_target_median_qubo_vs_outer_primary_spearman"] == 0.18993141001641606
    assert result["decision"]["replacement_objective_authorized"] is False
    assert result["decision"]["parp1_released"] is False
    assert result["decision"]["quantum_hardware_authorized"] is False
    assert all(value == 0 for value in result["data_boundary"].values())


def test_stage103_has_full_fixed_k_coverage_without_outer_label_selection():
    path = ROOT / "results/runs/stage103_objective_alignment_diagnosis/fold_alignment.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 105
    assert {(row["target_id"], row["outer_fold"], row["k"]) for row in rows} == {
        (target, str(fold), str(k))
        for target in ("BACE1", "EGFR", "FA10", "MK14", "PPARA", "PPARD", "PPARG")
        for fold in range(1, 6)
        for k in range(1, 4)
    }
    assert all(row["uses_outer_labels_for_selection"] == "False" for row in rows)


def test_stage103_independent_audit_passed():
    audit = json.loads((ROOT / "data/stage103_objective_alignment_diagnosis_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "stage103_independent_audit_ok"
    assert audit["fold_alignment_count"] == 105
    assert audit["outer_labels_used_by_selector"] is False
