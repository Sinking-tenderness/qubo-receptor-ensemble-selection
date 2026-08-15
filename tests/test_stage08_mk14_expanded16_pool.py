import csv
from pathlib import Path

import pytest

from scripts.audit_stage08_mk14_expanded16_pool import independent_maxmin
from scripts.select_stage08_mk14_expanded16_pool import (
    deterministic_maxmin,
    load_complete_distances,
)


def pair_rows(values: dict[tuple[str, str], float]) -> list[dict[str, str]]:
    return [
        {
            "conformer_id_a": first,
            "conformer_id_b": second,
            "standardized_pocket_distance": str(distance),
        }
        for (first, second), distance in values.items()
    ]


def test_maxmin_uses_lexical_tie_break_and_updates_pool() -> None:
    ids = ["S", "A", "B", "C"]
    values = {
        ("A", "S"): 2.0,
        ("B", "S"): 2.0,
        ("C", "S"): 1.0,
        ("A", "B"): 0.5,
        ("A", "C"): 3.0,
        ("B", "C"): 3.0,
    }
    distances = {tuple(sorted(pair)): value for pair, value in values.items()}
    result = deterministic_maxmin(ids, ["S"], distances, 2)
    assert [row["conformer_id"] for row in result] == ["A", "C"]
    assert [row["minimum_standardized_distance_to_selected_pool"] for row in result] == [
        2.0,
        1.0,
    ]


def test_independent_implementation_reconstructs_same_result() -> None:
    ids = ["S1", "S2", "A", "B"]
    values = {
        ("S1", "S2"): 0.2,
        ("S1", "A"): 1.0,
        ("S2", "A"): 0.8,
        ("S1", "B"): 1.2,
        ("S2", "B"): 1.1,
        ("A", "B"): 0.4,
    }
    result = independent_maxmin(ids, ["S1", "S2"], pair_rows(values), 2)
    assert [row["conformer_id"] for row in result] == ["B", "A"]


def test_distance_loader_rejects_incomplete_table() -> None:
    rows = pair_rows({("A", "B"): 1.0, ("A", "C"): 2.0})
    with pytest.raises(ValueError, match="incomplete"):
        load_complete_distances(rows, ["A", "B", "C"])


def test_repository_stage08_selection_is_expected_extension() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "data/processed/stage08_mk14_expanded16_structural_selection.csv"
    if not path.is_file():
        pytest.skip("Stage 08 structural outputs have not been materialized")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    additions = [row for row in rows if row["pool_role"] == "new_maxmin_addition"]
    assert [row["conformer_id"] for row in additions] == [
        "MK14_2BAJ_aligned",
        "MK14_4F9W_aligned",
        "MK14_3OCG_aligned",
        "MK14_3MPT_aligned",
        "MK14_3ZSI_aligned",
        "MK14_5N65_aligned",
        "MK14_3ZSH_aligned",
        "MK14_3ZSG_aligned",
        "MK14_3BV2_aligned",
        "MK14_2GFS_aligned",
        "MK14_4FA2_aligned",
        "MK14_4AAC_aligned",
    ]
