import pytest

from scripts.batch_vina_docking_parallel import ligand_seed, replace_ligand_rows


def test_replace_ligand_rows_replaces_only_requested_ligand():
    rows = [
        {"ligand_id": "A", "status": "ok"},
        {"ligand_id": "B", "status": "ok"},
    ]
    result = replace_ligand_rows(rows, "A", [{"ligand_id": "A", "status": "failed"}])
    assert result == [
        {"ligand_id": "B", "status": "ok"},
        {"ligand_id": "A", "status": "failed"},
    ]


def test_ligand_seed_preserves_explicit_source_manifest_offset():
    row = {"ligand_id": "L1", "seed_offset": "117"}

    assert ligand_seed(row, index=0, base_seed=20260801) == 20260918


def test_ligand_seed_defaults_to_selected_row_index():
    assert ligand_seed({"ligand_id": "L1"}, 3, 100) == 103

    with pytest.raises(ValueError, match="negative seed offset"):
        ligand_seed({"ligand_id": "L2", "seed_offset": "-1"}, 0, 100)
