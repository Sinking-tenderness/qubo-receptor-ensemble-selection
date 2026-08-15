import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage98_cross_target_gate_is_audited_negative():
    result = json.loads((ROOT / "data/stage98_cross_target_receptor_complementarity_result.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "data/stage98_cross_target_receptor_complementarity_audit.json").read_text(encoding="utf-8"))
    assert result["status"] == "stage98_cross_target_receptor_complementarity_complete"
    assert result["gate"]["passes"] is False
    assert result["gate"]["positive_target_count"] == 1
    assert audit["status"] == "stage98_audit_ok"
    assert audit["selector_labels_used"] is False


def test_stage98_has_five_targets_three_sizes_and_four_methods():
    path = ROOT / "results/runs/stage98_cross_target_receptor_complementarity/fold_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 300
    assert {row["target_id"] for row in rows} == {"MK14", "PPARG", "BACE1", "PPARA", "PPARD"}
    assert {row["method"] for row in rows} == {"mean_score", "complementarity", "oracle_train", "random"}
    assert {int(row["ensemble_size"]) for row in rows} == {1, 2, 3}


def test_stage98_result_has_no_new_compute_boundary_violations():
    result = json.loads((ROOT / "data/stage98_cross_target_receptor_complementarity_result.json").read_text(encoding="utf-8"))
    assert result["audit"]["new_docking_jobs"] == 0
    assert result["audit"]["quantum_hardware_jobs"] == 0
    assert result["audit"]["synthetic_scores"] == 0
