import pytest

from scripts.merge_ligand_manifests import merge_rows


def test_merge_rows_rejects_duplicate_ligand_ids() -> None:
    with pytest.raises(ValueError, match="occur in multiple inputs"):
        merge_rows(
            [
                [{"ligand_id": "same", "label": "active"}],
                [{"ligand_id": "same", "label": "decoy"}],
            ]
        )


def test_merge_rows_is_deterministic() -> None:
    rows = merge_rows(
        [
            [{"ligand_id": "b", "label": "decoy", "selection_role": "train"}],
            [{"ligand_id": "a", "label": "active", "selection_role": "train"}],
        ]
    )

    assert [row["ligand_id"] for row in rows] == ["a", "b"]
