import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage105_collapses_to_the_baseline_under_all_scenario_constraints():
    result = json.loads((ROOT / "data/stage105_scenario_robust_portfolio_diagnosis_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "stage105_scenario_robust_portfolio_diagnosis_complete"
    assert result["certificate_count"] == 52
    assert all(row["gain"] == 0.0 and row["redundancy"] == 0.0 for row in result["target_mean_gain_and_redundancy"].values())
    assert result["diagnostic_gate"]["passes"] is False
    assert result["diagnostic_gate"]["checks"]["mean_redundancy_reduction"] is False
    assert result["decision"]["replacement_objective_authorized"] is False
    assert result["decision"]["parp1_released"] is False
    assert result["decision"]["quantum_hardware_authorized"] is False
    assert all(value == 0 for value in result["data_boundary"].values())


def test_stage105_has_complete_certified_coverage_without_outer_label_selection():
    path = ROOT / "results/runs/stage105_scenario_robust_portfolio_diagnosis/fold_metrics.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    baseline = {
        (row["target_id"], row["outer_fold"], row["subset_size"]): row
        for row in rows
        if row["solver_id"] == "pair_off_baseline"
    }
    certificates = [row for row in rows if row["solver_id"] == "scenario_robust_milp_certificate"]
    assert len(rows) == 104
    assert len(certificates) == 52
    assert all(row["uses_outer_labels_for_selection"] == "False" for row in rows)
    assert all(float(row["minimum_jackknife_quality_margin"]) >= -1e-10 for row in rows)
    assert all(float(row["milp_gap"]) == 0.0 for row in certificates)
    assert all(row["selected_subset"] == baseline[(row["target_id"], row["outer_fold"], row["subset_size"])]["selected_subset"] for row in certificates)


def test_stage105_independent_audit_passed():
    audit = json.loads((ROOT / "data/stage105_scenario_robust_portfolio_diagnosis_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "stage105_independent_audit_ok"
    assert audit["fold_metric_count"] == 104
    assert audit["certificate_count"] == 52
    assert audit["changed_subset_certificate_count"] == 0
    assert audit["all_scenario_constraints_satisfied"] is True
