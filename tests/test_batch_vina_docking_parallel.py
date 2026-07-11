from scripts.batch_vina_docking_parallel import replace_ligand_rows


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
