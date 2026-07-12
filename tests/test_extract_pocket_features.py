from pathlib import Path

from scripts.extract_pocket_features import parse_pdb, reference_pocket


def test_reference_pocket_selects_nearby_protein_residue(tmp_path: Path):
    pdb = tmp_path / "reference.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  CB  ALA A   1       1.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      3  CA  GLY A   2      20.000   0.000   0.000  1.00 20.00           C\n"
        "HETATM    4  C1  STU A 299       2.000   0.000   0.000  1.00 20.00           C\n",
        encoding="ascii",
    )
    pocket, ligand_coords = reference_pocket(parse_pdb(pdb), "A", "STU", "A", 5.0)
    assert pocket == [("A", 1, "", "ALA")]
    assert ligand_coords.shape == (1, 3)
