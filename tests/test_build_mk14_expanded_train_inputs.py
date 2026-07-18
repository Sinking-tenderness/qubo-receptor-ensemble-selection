from scripts.build_mk14_expanded_train_inputs import normalize_receptor_row


def test_normalize_receptor_row_accepts_existing_manifest_schema(tmp_path):
    pdbqt = tmp_path / "receptor.pdbqt"
    pdbqt.write_text("ATOM\n", encoding="ascii")
    from scripts.build_mk14_expanded_train_inputs import file_sha256

    row = {
        "conformer_id": "R1",
        "input_pdb": "input.pdb",
        "input_pdb_sha256": "ABC",
        "chain": "A",
        "residue_count": "10",
        "receptor_atom_count": "100",
        "hydrogen_like_atom_count": "20",
        "autodock_atom_types": "A;C",
        "charge_min": "-0.5",
        "charge_max": "0.3",
        "receptor_pdbqt": pdbqt.as_posix(),
        "receptor_pdbqt_sha256": file_sha256(pdbqt),
        "status": "ok",
    }

    result = normalize_receptor_row(row, "existing")

    assert result["input_structure"] == "input.pdb"
    assert result["receptor_atom_count"] == "100"
    assert result["source_pool"] == "existing"
