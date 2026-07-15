import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from scripts.build_expanded_structural_receptor_pool import (
    audit_pdbqt,
    audit_pocket,
    deterministic_cluster_labels,
    load_config,
)


CONFIG_PATH = Path("configs/stage04_cdk2_expanded_structural_pool.json")


def pdb_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    chain: str,
    residue_number: int,
    x: float,
    element: str,
) -> str:
    return (
        f"{'ATOM':<6}{serial:>5} {atom_name:^4} {residue_name:>3} {chain}{residue_number:>4}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2}"
    )


def write_pocket(path: Path, residue_names: dict[int, str]) -> None:
    lines: list[str] = []
    serial = 1
    for number, residue_name in residue_names.items():
        for atom_name, offset, element in (("N", 0.0, "N"), ("CA", 0.5, "C"), ("CB", 1.0, "C")):
            lines.append(
                pdb_line(serial, atom_name, residue_name, "A", number, number + offset, element)
            )
            serial += 1
    path.write_text("\n".join([*lines, "END"]) + "\n", encoding="ascii")


def test_audit_pocket_reports_missing_residue(tmp_path: Path):
    path = tmp_path / "missing.pdb"
    write_pocket(path, {10: "ALA", 12: "SER"})

    audit, geometry = audit_pocket(
        path,
        "A",
        [10, 11, 12],
        {10: "ALA", 11: "GLY", 12: "SER"},
    )

    assert audit["missing_pocket_residues"] == [11]
    assert geometry is None


def test_audit_pocket_reports_reference_residue_name_mismatch(tmp_path: Path):
    path = tmp_path / "mismatch.pdb"
    write_pocket(path, {10: "ALA", 11: "VAL", 12: "SER"})

    audit, geometry = audit_pocket(
        path,
        "A",
        [10, 11, 12],
        {10: "ALA", 11: "GLY", 12: "SER"},
    )

    assert audit["residue_name_mismatches"] == ["11:GLY!=VAL"]
    assert geometry is not None
    assert geometry["ca"].shape == (3, 3)


def test_audit_pdbqt_counts_records_charges_and_atom_types(tmp_path: Path):
    path = tmp_path / "receptor.pdbqt"
    path.write_text(
        "\n".join([
            "ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00  0.00     0.242 C",
            "ATOM      2  H   ALA A   1       0.000   0.000   1.000  1.00  0.00     0.100 HD",
            "HETATM    3  O   HOH A   2       0.000   1.000   0.000  1.00  0.00    -0.400 OA",
        ])
        + "\n",
        encoding="ascii",
    )

    audit = audit_pdbqt(path)

    assert audit == {
        "atom_count": 3,
        "atom_record_count": 2,
        "hetatm_count": 1,
        "hydrogen_like_atom_count": 1,
        "charge_min": -0.4,
        "charge_max": 0.242,
        "atom_types": ["C", "HD", "OA"],
    }


def test_deterministic_clustering_uses_conformer_id_for_medoid_ties():
    scaled = np.array([[0.0], [0.1], [10.0], [10.1]])
    distances = squareform(pdist(scaled))
    conformer_ids = ["B", "A", "D", "C"]

    first_labels, first_medoids = deterministic_cluster_labels(
        scaled, distances, conformer_ids, 2
    )
    second_labels, second_medoids = deterministic_cluster_labels(
        scaled, distances, conformer_ids, 2
    )

    assert first_labels.tolist() == [0, 0, 1, 1]
    assert first_medoids == {0: 1, 1: 3}
    assert np.array_equal(first_labels, second_labels)
    assert first_medoids == second_medoids


def test_load_config_rejects_duplicate_cluster_counts(tmp_path: Path):
    config = json.loads(CONFIG_PATH.read_text(encoding="ascii"))
    config["cluster_counts"] = [1, 2, 2]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="ascii")

    with pytest.raises(ValueError, match="cluster_counts"):
        load_config(path)
