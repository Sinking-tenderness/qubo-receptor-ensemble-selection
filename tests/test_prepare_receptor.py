from pathlib import Path

from scripts.prepare_receptor import audit_pdbqt


def test_audit_pdbqt_uses_fixed_charge_and_atom_type_columns(tmp_path: Path) -> None:
    pdbqt = tmp_path / "receptor.pdbqt"
    pdbqt.write_text(
        "ATOM      1  C   MET A   1     -12.439  28.623  -4.635  1.00  0.00     0.242 C \n"
        "ATOM      2  H   MET A   1     -13.000  29.000  -4.000  1.00  0.00     0.164 HD\n",
        encoding="ascii",
    )

    audit = audit_pdbqt(pdbqt)

    assert audit["coordinate_record_count"] == 2
    assert audit["residue_count"] == 1
    assert audit["hydrogen_like_atom_count"] == 1
    assert audit["charge_min"] == 0.164
    assert audit["charge_max"] == 0.242
    assert audit["autodock_atom_types"] == ["C", "HD"]
