import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage100_receptor_count_is_actually_adaptive():
    path = ROOT / "results/runs/stage100_adaptive_stopping_qubo/fold_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["method"] == "one_standard_error_smallest_k"]
    assert len(rows) == 25
    assert {int(row["selected_k"]) for row in rows} == {1, 2, 3}
    assert sum(int(row["selected_k"]) > 1 for row in rows) == 7
    assert all(row["selector_used_outer_test_labels"] == "False" for row in rows)


def test_stage100_stopping_reduces_overselection_but_does_not_pass():
    result = json.loads((ROOT / "data/stage100_adaptive_stopping_qubo_result.json").read_text(encoding="utf-8"))
    rows = {row["target_id"]: row for row in result["primary_target_rows"]}
    assert result["gate"]["passes"] is False
    assert result["gate"]["mean_gain_over_single"] < 0.0
    assert rows["PPARA"]["gain"] > -0.02
    assert rows["MK14"]["gain"] > 0.0
    assert rows["PPARD"]["gain"] == 0.0


def test_stage100_audit_and_compute_boundary():
    result = json.loads((ROOT / "data/stage100_adaptive_stopping_qubo_result.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "data/stage100_adaptive_stopping_qubo_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "stage100_independent_audit_ok"
    assert result["data_boundary"]["outer_test_labels_used_by_selector"] is False
    assert result["data_boundary"]["new_docking_jobs"] == 0
    assert result["data_boundary"]["quantum_hardware_jobs"] == 0
