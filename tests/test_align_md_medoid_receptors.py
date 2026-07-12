from pathlib import Path

import numpy as np

from scripts.align_md_medoid_receptors import (
    audit_heavy_protein,
    is_hydrogen_pdb_atom,
    temporal_support_role,
    write_aligned_heavy_protein,
)
from scripts.align_receptor_structure import parse_pdb


def pdb_line(
    record: str,
    serial: int,
    atom_name: str,
    residue_name: str,
    chain: str,
    residue_number: int,
    element: str,
) -> str:
    return (
        f"{record:<6}{serial:>5} {atom_name:^4} {residue_name:>3} {chain}{residue_number:>4}    "
        f"{float(serial):8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2}"
    )


def test_write_aligned_heavy_protein_removes_hydrogen_and_hetatm(tmp_path: Path):
    input_path = tmp_path / "input.pdb"
    input_path.write_text(
        "\n".join([
            pdb_line("ATOM", 1, "CA", "ALA", "A", 1, "C"),
            pdb_line("ATOM", 2, "HA", "ALA", "A", 1, "H"),
            pdb_line("ATOM", 3, "CA", "ALA", "B", 1, "C"),
            pdb_line("HETATM", 4, "O", "HOH", "A", 2, "O"),
            "END",
        ]) + "\n",
        encoding="ascii",
    )
    lines, atoms = parse_pdb(input_path)
    hydrogen = next(atom for atom in atoms if atom.atom_name == "HA")
    assert is_hydrogen_pdb_atom(hydrogen, lines[hydrogen.line_index])
    output_path = tmp_path / "output.pdb"
    write_aligned_heavy_protein(
        output_path,
        lines,
        atoms,
        "A",
        np.eye(3),
        np.zeros(3),
    )
    audit = audit_heavy_protein(output_path, "A")
    assert audit == {
        "coordinate_record_count": 1,
        "atom_record_count": 1,
        "hetatm_record_count": 0,
        "hydrogen_count": 0,
        "residue_count": 1,
        "chains": ["A"],
    }


def test_temporal_support_role_preserves_short_unrevisited_candidate():
    assert temporal_support_role({"cluster_size": "2", "revisited_after_exit": "False"}) == (
        "exploratory_low_temporal_support"
    )
    assert temporal_support_role({"cluster_size": "7", "revisited_after_exit": "True"}) == (
        "revisited_primary_candidate"
    )
