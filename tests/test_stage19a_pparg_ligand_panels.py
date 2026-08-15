import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def test_stage19a_panels_are_exact_and_scaffold_disjoint() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = json.loads(
        (root / "data/stage19a_pparg_ligand_panel_allocation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    with (
        root / "data/processed/stage19a_pparg_selected_ligand_panel_manifest.csv"
    ).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (root / "data/processed/stage19a_pparg_train668_ligand_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        train = list(csv.DictReader(handle))

    assert summary["status"] == "stage19a_pparg_ligand_panels_frozen"
    assert len(rows) == 3820
    assert len(train) == 668
    assert Counter((row["split"], row["label"]) for row in rows) == Counter(
        {
            ("train", "active"): 334,
            ("train", "decoy"): 334,
            ("validation", "active"): 75,
            ("validation", "decoy"): 1501,
            ("test", "active"): 75,
            ("test", "decoy"): 1501,
        }
    )
    assert {row["split"] for row in train} == {"train"}
    assert {row["selection_role"] for row in train} == {"development_train"}

    for key in ("canonical_smiles", "scaffold_smiles", "split_group_id"):
        splits: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            splits[row[key]].add(row["split"])
        assert all(len(value) == 1 for value in splits.values())
    assert summary["data_boundary"]["docking_scores_read"] == 0
    assert summary["data_boundary"]["fresh_validation_docking_scores_read"] == 0
    assert summary["data_boundary"]["test_docking_scores_read"] == 0
