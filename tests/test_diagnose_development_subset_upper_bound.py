from scripts.diagnose_development_subset_upper_bound import (
    rank_subset_candidates,
)


def test_rank_subset_candidates_returns_bedroc_ordered_development_scan():
    rows = [
        {"target_id": "T", "ligand_id": "A1", "label": "active", "R1": "-9", "R2": "-8"},
        {"target_id": "T", "ligand_id": "A2", "label": "active", "R1": "-7", "R2": "-6"},
        {"target_id": "T", "ligand_id": "D1", "label": "decoy", "R1": "-5", "R2": "-4"},
        {"target_id": "T", "ligand_id": "D2", "label": "decoy", "R1": "-3", "R2": "-2"},
    ]
    candidates = rank_subset_candidates(rows, ["R1", "R2"], 2, "mean_score")

    assert len(candidates) == 3
    assert candidates[0]["metrics"]["bedroc_alpha_20"] >= candidates[-1][
        "metrics"
    ]["bedroc_alpha_20"]
    assert all(row["target_size"] in {1, 2} for row in candidates)
