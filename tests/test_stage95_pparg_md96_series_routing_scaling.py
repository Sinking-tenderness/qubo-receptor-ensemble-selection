import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="ascii"))


def test_stage95_uses_only_real_existing_scores():
    result = read_json("data/stage95_pparg_md96_series_routing_scaling_result.json")
    assert result["real_data_dimensions"] == {
        "actives": 80,
        "decoys": 80,
        "hit_count": 16,
        "ligands": 160,
        "receptors": 96,
        "score_rows": 46080,
        "seeds": 3,
        "synthetic_scores": 0,
    }
    assert all(value == 0 for value in result["data_boundary"].values())


def test_stage95_structure_series_did_not_read_docking_scores():
    summary = read_json("data/stage95_pparg_active_series_summary.json")
    assert summary["status"] == "stage95_pparg_active_series_structure_only_ok"
    assert summary["active_ligand_count"] == 80
    assert summary["decoy_ligand_count_used_for_clustering"] == 0
    assert summary["docking_score_rows_read"] == 0
    assert summary["linkage"] == "complete"


def test_stage95_exact_and_bounded_scales_exclude_one_percent_gap():
    path = (
        ROOT
        / "results/runs/stage95_pparg_md96_series_routing_scaling/scale_results.csv"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert sum(row["milp_optimal"].lower() == "true" for row in rows) == 4
    assert sum(row["milp_optimal"].lower() != "true" for row in rows) == 1
    assert all(
        row["one_percent_gap_mathematically_excluded"].lower() == "true"
        for row in rows
    )
    assert max(float(row["maximum_possible_relative_gap"]) for row in rows) < 0.01


def test_stage95_does_not_authorize_quantum_or_retuning():
    result = read_json("data/stage95_pparg_md96_series_routing_scaling_result.json")
    assert result["status"] == "stage95_pparg_md96_series_routing_hardness_not_supported"
    assert result["passing_scale_ids"] == []
    assert result["hardness_supported"] is False
    assert result["quantum_hardware_authorized"] is False
    assert result["same_matrix_objective_retuning_authorized"] is False
