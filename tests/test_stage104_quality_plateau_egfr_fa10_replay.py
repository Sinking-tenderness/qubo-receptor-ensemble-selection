import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage104_is_posthoc_transfer_only_and_keeps_boundaries_closed():
    result = json.loads((ROOT / "data/stage104_quality_plateau_egfr_fa10_replay_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "stage104_quality_plateau_egfr_fa10_replay_complete"
    assert result["transfer_checks"]["passing_cell_count"] == 2
    assert result["transfer_checks"]["cell_count"] == 4
    assert result["transfer_checks"]["all_target_k_cells_quality_noninferior_and_redundancy_nonnegative"] is False
    assert result["decision"]["new_target_protocol_authorized"] is False
    assert result["decision"]["parp1_released"] is False
    assert result["decision"]["quantum_hardware_authorized"] is False
    assert all(value == 0 for value in result["data_boundary"].values())


def test_stage104_has_complete_certified_coverage_without_outer_label_selection():
    path = ROOT / "results/runs/stage104_quality_plateau_egfr_fa10_replay/fold_metrics.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 40
    exact = [row for row in rows if row["solver_id"] == "continuous_milp_certificate"]
    assert len(exact) == 20
    assert all(row["uses_outer_labels_for_selection"] == "False" for row in rows)
    assert all(float(row["train_quality_margin"]) >= -1e-10 for row in rows)
    assert all(float(row["milp_gap"]) == 0.0 for row in exact)


def test_stage104_independent_audit_passed():
    audit = json.loads((ROOT / "data/stage104_quality_plateau_egfr_fa10_replay_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "stage104_independent_audit_ok"
    assert audit["fold_metric_count"] == 40
    assert audit["transfer_passing_cell_count"] == 2
    assert audit["outer_labels_used_by_selector"] is False
