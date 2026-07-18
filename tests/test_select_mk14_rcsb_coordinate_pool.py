from pathlib import Path

import numpy as np

from scripts.select_mk14_rcsb_coordinate_pool import (
    Atom,
    kabsch,
    load_config,
    maxmin_select,
    preparation_readiness_metrics,
    select_chain_atoms,
)


CONFIG_PATH = Path("configs/stage05_mk14_rcsb_coordinate_selection.json")


def atom(altloc, occupancy, serial=1):
    return Atom(
        line=(
            f"ATOM  {serial:5d}  CA {altloc or ' '}ALA A  30      "
            "  0.000   0.000   0.000  1.00 20.00           C"
        ),
        record="ATOM",
        serial=serial,
        atom_name="CA",
        altloc=altloc,
        resname="ALA",
        chain="A",
        resseq=30,
        icode="",
        coord=np.zeros(3),
        occupancy=occupancy,
        element="C",
    )


def named_atom(name, record="ATOM", resname="ALA", serial=1):
    return Atom(
        line="",
        record=record,
        serial=serial,
        atom_name=name,
        altloc="",
        resname=resname,
        chain="A",
        resseq=30,
        icode="",
        coord=np.array([float(serial), 0.0, 0.0]),
        occupancy=1.0,
        element="N" if name == "N" else "O" if name == "O" else "C",
    )


def test_coordinate_config_freezes_eight_receptor_pool():
    config = load_config(CONFIG_PATH)

    assert config["coordinate_gate"]["minimum_matched_ca_count"] == 300
    assert config["coordinate_gate"]["maximum_aligned_global_ca_rmsd_angstrom"] == 3.0
    assert config["outputs"]["selected_expansion_manifest_csv"].endswith(
        "stage05_mk14_expanded8_structural_selection.csv"
    )


def test_altloc_selection_prefers_occupancy_then_blank_then_a():
    selected = select_chain_atoms(
        [atom("A", 0.4, 1), atom("B", 0.6, 2), atom("", 0.6, 3)], "A"
    )

    assert len(selected) == 1
    assert selected[0].altloc == ""


def test_kabsch_always_returns_a_proper_rotation():
    mobile = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    reference = np.array([[2.0, 3.0, 4.0], [2.0, 4.0, 4.0], [1.0, 3.0, 4.0]])
    rotation, translation = kabsch(mobile, reference)

    assert np.linalg.det(rotation) > 0.999999
    assert np.allclose(mobile @ rotation + translation, reference)


def test_maxmin_selection_is_deterministic_and_seeded_by_existing_pool():
    ids = ["E1", "E2", "N1", "N2", "N3"]
    values = {
        ("E1", "E2"): 1.0,
        ("E1", "N1"): 4.0,
        ("E2", "N1"): 3.0,
        ("E1", "N2"): 2.0,
        ("E2", "N2"): 5.0,
        ("E1", "N3"): 3.0,
        ("E2", "N3"): 3.0,
        ("N1", "N2"): 4.0,
        ("N1", "N3"): 1.0,
        ("N2", "N3"): 5.0,
    }
    selected = maxmin_select(ids, ["E1", "E2"], values, 2)

    assert [row["conformer_id"] for row in selected] == ["N1", "N2"]


def test_preparation_readiness_detects_ca_only_structure():
    reference = [
        named_atom(name, serial=index)
        for index, name in enumerate(["N", "CA", "C", "O", "CB"], start=1)
    ]
    candidate = [named_atom("CA")]

    metrics = preparation_readiness_metrics(
        candidate, reference, [30], {"N", "CA", "C", "O"}
    )

    assert metrics["chain_heavy_atoms_per_ca"] == 1.0
    assert metrics["pocket_heavy_atom_completeness_fraction"] == 0.2


def test_preparation_readiness_detects_modified_polymer_residue():
    reference = [
        named_atom(name, serial=index)
        for index, name in enumerate(["N", "CA", "C", "O", "CB"], start=1)
    ]
    modified = [
        named_atom(name, record="HETATM", resname="TPO", serial=index)
        for index, name in enumerate(["N", "CA", "C", "O"], start=10)
    ]

    metrics = preparation_readiness_metrics(
        reference + modified, reference, [30], {"N", "CA", "C", "O"}
    )

    assert metrics["polymer_like_hetero_residue_count"] == 1
    assert metrics["polymer_like_hetero_residues"] == "TPO:30"


def test_preparation_readiness_detects_incomplete_standard_side_chain():
    reference = [
        named_atom(name, resname="TYR", serial=index)
        for index, name in enumerate(
            ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
            start=1,
        )
    ]
    candidate = reference[:5]

    metrics = preparation_readiness_metrics(
        candidate, reference, [30], {"N", "CA", "C", "O"}
    )

    assert metrics["incomplete_standard_amino_acid_residue_count"] == 1
    assert metrics["incomplete_standard_amino_acid_residues"] == (
        "TYR:30[CD1,CD2,CE1,CE2,CG,CZ,OH]"
    )
