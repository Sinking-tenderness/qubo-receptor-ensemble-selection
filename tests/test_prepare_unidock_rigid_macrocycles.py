import pytest

from scripts.experimental.unidock.prepare_unidock_rigid_macrocycles import select_source_rows


def test_select_source_rows_preserves_index_and_hash():
    rows = [
        {"ligand_id": "A", "label": "active", "pdbqt_sha256": "AA"},
        {"ligand_id": "B", "label": "decoy", "pdbqt_sha256": "BB"},
    ]
    expected = [
        {
            "ligand_id": "B",
            "label": "decoy",
            "source_manifest_index": 1,
            "source_pdbqt_sha256": "bb",
        }
    ]

    selected = select_source_rows(rows, expected)

    assert selected[0][0] == 1
    assert selected[0][1]["ligand_id"] == "B"


def test_select_source_rows_rejects_index_drift():
    rows = [{"ligand_id": "A", "label": "active", "pdbqt_sha256": "AA"}]
    expected = [
        {
            "ligand_id": "A",
            "label": "active",
            "source_manifest_index": 2,
            "source_pdbqt_sha256": "AA",
        }
    ]

    with pytest.raises(ValueError, match="source manifest index differs"):
        select_source_rows(rows, expected)
