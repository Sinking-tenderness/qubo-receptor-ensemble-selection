from pathlib import Path

import numpy as np

from qubo_receptor_ensemble.pdb import parse_pdb, rmsd
from qubo_receptor_ensemble.raw_preparation import (
    align_pdb_file,
    calculate_ligand_box,
    discover_structure_files,
)


def test_calculate_ligand_box_uses_coordinates_padding_and_minimum_size(
    tmp_path: Path,
) -> None:
    ligand = tmp_path / "ligand.mol2"
    ligand.write_text(
        """@<TRIPOS>MOLECULE
LIG
  2 0 0 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
      1 C1          0.0000   1.0000   2.0000 C.3       1 LIG 0.0
      2 C2          2.0000   4.0000   5.0000 C.3       1 LIG 0.0
""",
        encoding="ascii",
    )

    box = calculate_ligand_box(ligand, padding=1.0, minimum_size=(10.0, 11.0, 12.0))

    assert box == {
        "center_x": 1.0,
        "center_y": 2.5,
        "center_z": 3.5,
        "size_x": 10.0,
        "size_y": 11.0,
        "size_z": 12.0,
    }


def test_discover_structure_files_recurses_and_deduplicates_pdb_cif_pairs(
    tmp_path: Path,
) -> None:
    (tmp_path / "1ABC.cif").write_text("data_1ABC\n", encoding="ascii")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "2DEF.cif").write_text("data_2DEF\n", encoding="ascii")
    (tmp_path / "nested" / "2DEF.pdb").write_text("END\n", encoding="ascii")

    discovered = discover_structure_files(tmp_path)

    assert [path.name for path in discovered] == ["1ABC.cif", "2DEF.cif"]


def test_align_pdb_file_writes_current_run_alignment_and_reduces_ca_rmsd(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.pdb"
    mobile = tmp_path / "mobile.pdb"
    aligned = tmp_path / "aligned.pdb"
    reference.write_text(
        """ATOM      1  CA  ALA A   1       0.000   0.000   0.000
ATOM      2  CA  GLY A   2       1.000   0.000   0.000
ATOM      3  CA  SER A   3       0.000   1.000   0.000
END
""",
        encoding="ascii",
    )
    mobile.write_text(
        """ATOM      1  CA  ALA B   1       4.000  -2.000   7.000
ATOM      2  CA  GLY B   2       4.000  -1.000   7.000
ATOM      3  CA  SER B   3       3.000  -2.000   7.000
END
""",
        encoding="ascii",
    )

    summary = align_pdb_file(
        reference,
        mobile,
        aligned,
        reference_chain="A",
        mobile_chain="B",
    )

    _, reference_atoms = parse_pdb(reference)
    _, aligned_atoms = parse_pdb(aligned)
    ref = np.vstack([atom.coord for atom in reference_atoms])
    got = np.vstack([atom.coord for atom in aligned_atoms])
    assert aligned.is_file()
    assert summary["matched_ca_count"] == 3
    assert summary["rmsd_after_angstrom"] < 1e-6
    assert rmsd(got, ref) < 1e-6


def test_align_pdb_file_matches_same_sequence_when_residue_numbers_are_offset(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference_offset.pdb"
    mobile = tmp_path / "mobile_offset.pdb"
    aligned = tmp_path / "aligned_offset.pdb"
    reference.write_text(
        """ATOM      1  CA  ALA A   1       0.000   0.000   0.000
ATOM      2  CA  GLY A   2       1.000   0.000   0.000
ATOM      3  CA  SER A   3       0.000   1.000   0.000
END
""",
        encoding="ascii",
    )
    mobile.write_text(
        """ATOM      1  CA  ALA B  16       4.000  -2.000   7.000
ATOM      2  CA  GLY B  17       4.000  -1.000   7.000
ATOM      3  CA  SER B  18       3.000  -2.000   7.000
END
""",
        encoding="ascii",
    )

    summary = align_pdb_file(
        reference,
        mobile,
        aligned,
        reference_chain="A",
        mobile_chain="B",
    )

    assert summary["matched_ca_count"] == 3
    assert summary["rmsd_after_angstrom"] < 1e-6
