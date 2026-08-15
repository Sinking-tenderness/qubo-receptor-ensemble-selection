from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rdkit import Chem

from scripts import prepare_stage50_ppara_large_pool_redocking_inputs as prep_base
from scripts.build_stage57_ppard_cognate_redocking_input_bundle import bundle_paths
from scripts.prepare_stage57_ppard_redocking_inputs import (
    alignment_audit_ppard,
    box_overflow,
    validate_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
PREP_CONFIG = ROOT / "configs/stage57_ppard_cognate_redocking_input_preparation.json"
DOCK_CONFIG = ROOT / "configs/stage57_ppard_cognate_redocking.json"


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def test_stage57_static_input_audit_is_outcome_blind_and_complete() -> None:
    config, _, rows, audit = validate_inputs(PREP_CONFIG, ROOT)
    assert audit["status"] == "audit_only_ok"
    assert audit["frozen_receptor_count"] == 51
    assert audit["cognate_ligand_count"] == 51
    assert all(int(value) == 0 for value in audit["data_boundary"].values())
    assert rows[0]["conformer_id"] == "PPARD_2ZNP_reference"
    assert len({row["conformer_id"] for row in rows}) == 51
    assert config["expected"]["minimum_prepared_receptor_count"] == 24


def test_stage57_box_matches_frozen_k55_geometry() -> None:
    config = read_json(PREP_CONFIG)
    with (ROOT / "data/raw/rcsb/ppard/2ZNP_K55_A922.sdf").open("rb") as handle:
        molecule = next(Chem.ForwardSDMolSupplier(handle, removeHs=False))
    coordinates = []
    conformer = molecule.GetConformer()
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() > 1:
            point = conformer.GetAtomPosition(atom.GetIdx())
            coordinates.append((point.x, point.y, point.z))
    values = np.asarray(coordinates)
    expected_center = np.round(values.mean(axis=0), 2)
    expected_size = np.clip(np.ceil((np.ptp(values, axis=0) + 12.0) / 2.0) * 2.0, 22.0, 30.0)
    box = config["frozen_common_box"]
    assert np.allclose(
        expected_center, [box["center_x"], box["center_y"], box["center_z"]]
    )
    assert np.allclose(expected_size, [box["size_x"], box["size_y"], box["size_z"]])
    assert box_overflow(values, box) <= 0.0


def test_stage57_gate_and_budget_remain_frozen() -> None:
    config = read_json(DOCK_CONFIG)
    gate = config["redocking_gate"]
    protocol = config["unidock"]
    assert gate["maximum_rmsd_angstrom"] == 2.0
    assert gate["minimum_successful_seeds_per_receptor"] == 2
    assert gate["minimum_passing_receptor_count"] == 24
    assert protocol["required_package_version"] == "1.1.3"
    assert protocol["exhaustiveness"] == 1024
    assert protocol["max_step"] == 80
    assert config["expected"]["maximum_redocking_pair_count"] == 153


def test_stage57_sequence_scoped_alignment_recomputes_all_51_cases() -> None:
    config, inputs, rows, _ = validate_inputs(PREP_CONFIG, ROOT)
    prep_base.load_dependencies()
    anchors = config["reference"]["anchor_residue_numbers"]
    window = config["preparation_protocol"]["target_sequence_residue_window"]
    maximum_error = 0.0
    for row in rows:
        evidence, _, _, _ = alignment_audit_ppard(
            inputs["reference_mmcif"],
            ROOT / row["mmcif_path"],
            row["chain"],
            ROOT / row["aligned_protein_pdb_path"],
            anchors,
            window,
        )
        assert evidence["matched_ca_count"] == int(row["matched_ca_count"])
        maximum_error = max(
            maximum_error,
            evidence["maximum_coordinate_difference_from_selected_aligned_pdb_angstrom"],
        )
    assert maximum_error <= 0.0011


def test_stage57_bundle_contains_coordinates_but_no_outcomes() -> None:
    paths = bundle_paths(ROOT)
    lowered = [path.lower() for path in paths]
    assert "scripts/select_stage13c_egfr_local_pocket_pool.py" in paths
    assert len([path for path in paths if path.endswith("_to_2ZNP_A.pdb")]) == 51
    assert len([path for path in paths if path.startswith("results/runs/stage56b_ppard_sequence_remapped_coordinates/mmcif/")]) == 50
    assert not any("stage57_ppard_cognate_redocking_results.csv" in path for path in lowered)
    assert not any("stage57_ppard_receptor_gate_results.csv" in path for path in lowered)
    assert not any("ppard_fresh_validation" in path for path in lowered)
    assert not any("ppard_locked_test" in path for path in lowered)
