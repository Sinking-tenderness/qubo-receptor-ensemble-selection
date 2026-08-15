import csv
import json
from pathlib import Path


def test_stage44_freezes_bace_objective_and_primary_k3() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/stage44_pparg_md96_rank_sensitive_qubo.json").read_text(encoding="ascii"))
    assert config["objective"]["source_objective_id"] == "bedroc20_rank_sensitive_pair_complementarity_v1"
    assert config["objective"]["primary_replication_subset_size"] == 3
    assert config["evidence_timing"]["same_data_weight_search_permitted"] is False
    assert config["evidence_timing"]["fresh_validation_rows_permitted"] is False
    assert config["evidence_timing"]["test_rows_permitted"] is False


def test_stage44_result_and_audit_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    result_path = root / "data/stage44_pparg_md96_rank_sensitive_qubo_result.json"
    audit_path = root / "data/stage44_pparg_md96_rank_sensitive_qubo_audit.json"
    if not result_path.is_file() or not audit_path.is_file():
        return
    result = json.loads(result_path.read_text(encoding="ascii"))
    audit = json.loads(audit_path.read_text(encoding="ascii"))
    assert result["status"] == "stage44_pparg_md96_rank_sensitive_qubo_complete"
    assert audit["status"] == "stage44_pparg_md96_rank_sensitive_qubo_independent_audit_ok"
    assert result["decision"]["full_k3_over_single_robust_bedroc_gain"] > 0.02
    assert result["decision"]["mean_outer_holdout_k3_over_single_gain"] < 0
    assert result["decision"]["positive_solver_gap_cell_count"] == 0
    assert result["decision"]["same_data_retuning_authorized"] is False


def test_stage44_exact_cells_match_all_solvers() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "results/runs/stage44_pparg_md96_rank_sensitive_qubo/solver_comparison.csv"
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    exact = [row for row in rows if int(row["subset_size"]) <= 3]
    assert len(exact) == 15
    assert all(abs(float(row["classical_exact_gap"])) <= 1e-12 for row in exact)
    assert all(abs(float(row["annealing_exact_gap"])) <= 1e-12 for row in exact)
