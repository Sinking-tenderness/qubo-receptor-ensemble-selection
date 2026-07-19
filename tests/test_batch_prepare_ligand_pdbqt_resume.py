from scripts.batch_prepare_ligand_pdbqt import validated_existing_pdbqt


def test_validated_existing_pdbqt_reconstructs_audit_fields(tmp_path) -> None:
    path = tmp_path / "ligand.pdbqt"
    path.write_text(
        "ROOT\n"
        "ATOM      1  C   LIG A   1       1.000   2.000   3.000  1.00  0.00     0.125 C\n"
        "ENDROOT\n"
        "TORSDOF 0\n",
        encoding="ascii",
    )

    result = validated_existing_pdbqt(path)

    assert result is not None
    assert result["pdbqt_status"] == "ok"
    assert result["pdbqt_message"] == "meeko_existing_validated"
    assert result["pdbqt_atom_count"] == 1
    assert result["torsdof"] == "0"
    assert len(result["pdbqt_sha256"]) == 64


def test_validated_existing_pdbqt_rejects_partial_file(tmp_path) -> None:
    path = tmp_path / "partial.pdbqt"
    path.write_text("ROOT\nENDROOT\n", encoding="ascii")

    assert validated_existing_pdbqt(path) is None
