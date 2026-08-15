import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage99_gate_and_data_boundary_are_conservative():
    result = json.loads((ROOT / "data/stage99_qubo_objective_repair_screen_result.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "data/stage99_qubo_objective_repair_screen_audit.json").read_text(encoding="utf-8"))
    assert result["status"] == "stage99_qubo_objective_repair_screen_complete"
    assert result["gate"]["passes"] is False
    assert result["gate"]["fixed_k3_passes"] is False
    assert result["data_boundary"]["test_labels_used_by_selector"] is False
    assert result["data_boundary"]["historical_consumed_fresh_validation_rows_read_posthoc"] == 1576
    assert result["data_boundary"]["protected_fresh_validation_rows_read"] == 0
    assert result["data_boundary"]["new_docking_jobs"] == 0
    assert result["data_boundary"]["quantum_hardware_jobs"] == 0
    assert audit["status"] == "stage99_independent_audit_ok"


def test_stage99_fixed_repair_has_real_but_insufficient_improvement():
    result = json.loads((ROOT / "data/stage99_qubo_objective_repair_screen_result.json").read_text(encoding="utf-8"))
    rows = {row["target_id"]: row for row in result["fixed_k3_target_gain_rows"]}
    assert rows["MK14"]["repair_k3_minus_single"] > 0.02
    assert rows["PPARG"]["repair_k3_minus_single"] > 0.02
    assert rows["PPARD"]["repair_k3_minus_single"] > 0.02
    assert rows["PPARA"]["repair_k3_minus_single"] < -0.05
    assert 0.0 < result["gate"]["mean_gain_over_single"] < 0.02


def test_stage99_has_full_nested_and_solver_coverage():
    run = ROOT / "results/runs/stage99_qubo_objective_repair_screen"
    with (run / "fold_metrics.csv").open(newline="", encoding="utf-8") as handle:
        folds = list(csv.DictReader(handle))
    with (run / "solver_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        solvers = list(csv.DictReader(handle))
    with (run / "adaptive_k_metrics.csv").open(newline="", encoding="utf-8") as handle:
        adaptive = list(csv.DictReader(handle))
    assert len(folds) == 375
    assert len(solvers) == 75
    assert len(adaptive) == 25
    assert sum(row["exact_differs"] == "True" for row in solvers) == 5
    assert all(row["selector_used_outer_test_labels"] == "False" for row in adaptive)
