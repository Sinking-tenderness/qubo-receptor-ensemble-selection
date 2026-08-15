from pathlib import Path

import numpy as np

from scripts.select_stage13_egfr_coordinate_pool import (
    AtomRecord,
    altloc_rank,
    derive_reference_residues,
    incomplete_standard_residues,
)


AMENDMENT_PATH = Path(
    "configs/stage13_egfr_external_target_pilot_preregistration_amendment01.json"
)


def atom(
    atom_name: str,
    coord: tuple[float, float, float],
    *,
    kind: str = "protein",
    resname: str = "ALA",
    resseq: int = 1,
    altloc: str = "",
    occupancy: float = 1.0,
) -> AtomRecord:
    return AtomRecord(
        kind=kind,
        atom_name=atom_name,
        altloc=altloc,
        resname=resname,
        resseq=resseq,
        icode="",
        coord=np.array(coord, dtype=float),
        occupancy=occupancy,
        b_iso=20.0,
        element="N" if atom_name == "N" else "O" if atom_name == "O" else "C",
    )


def test_altloc_rank_prefers_occupancy_then_blank() -> None:
    blank = atom("CA", (0.0, 0.0, 0.0), occupancy=0.6)
    alternate = atom("CA", (1.0, 0.0, 0.0), altloc="B", occupancy=0.6)
    lower = atom("CA", (2.0, 0.0, 0.0), altloc="A", occupancy=0.4)

    assert min([alternate, lower, blank], key=altloc_rank) is blank


def test_reference_pocket_derivation_uses_heavy_atom_distance() -> None:
    values = [
        atom("CA", (0.0, 0.0, 0.0), resseq=10),
        atom("CB", (1.0, 0.0, 0.0), resseq=10),
        atom("CA", (9.0, 0.0, 0.0), resseq=20),
        atom(
            "C1",
            (3.0, 0.0, 0.0),
            kind="hetero",
            resname="LIG",
            resseq=1,
        ),
    ]

    assert derive_reference_residues(values, "LIG", 2.1) == [10]


def test_incomplete_standard_residue_is_recorded_without_imputation() -> None:
    values = [
        atom(name, (float(index), 0.0, 0.0), resseq=10)
        for index, name in enumerate(["N", "CA", "C", "O"], start=1)
    ]

    assert incomplete_standard_residues(values) == ["ALA:10[CB]"]


def test_amendment_freezes_sixteen_receptors_and_hidden_covalency() -> None:
    import json

    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="ascii"))

    assert amendment["structural_selection"]["target_receptor_count"] == 16
    assert len(amendment["reference"]["reference_pocket_residue_numbers"]) == 34
    assert len(amendment["reference"]["required_anchor_residue_numbers"]) == 20
    assert amendment["coordinate_gate"]["hidden_covalency"][
        "exclude_if_minimum_protein_ligand_heavy_atom_distance_at_or_below_angstrom"
    ] == 2.0
