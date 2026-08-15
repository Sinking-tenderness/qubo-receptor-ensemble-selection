from scripts.audit_stage14_fa10_source import pdb_atom_audit


def test_dude_receptor_has_protein_atoms_and_stripped_chain_id(tmp_path) -> None:
    path = tmp_path / "receptor.pdb"
    path.write_text(
        "ATOM      1  N   ILE     1      10.000  10.000  10.000\n"
        "ATOM      2  HN  ILE     1      10.100  10.100  10.100\n",
        encoding="ascii",
    )
    result = pdb_atom_audit(path)

    assert result["protein_atom_record_count"] == 2
    assert result["protein_chain_ids"] == [""]
    assert result["hydrogen_atom_count"] == 1
