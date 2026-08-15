from __future__ import annotations

from pathlib import Path

from scripts.build_stage41b_bace1_large_pool_input_bundle import bundle_paths
from scripts.prepare_stage41b_bace1_large_pool_redocking_inputs import (
    checkpoint,
    validate_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage41b_bace1_large_pool_redocking_input_preparation.json"


def test_stage41b_input_audit_freezes_all_49_cases() -> None:
    config, _, rows, audit = validate_inputs(CONFIG, ROOT)
    assert audit["status"] == "audit_only_ok"
    assert audit["receptor_count"] == 49
    assert audit["cognate_ligand_count"] == 49
    assert rows[0]["conformer_id"] == "BACE1_3L5D_reference"
    assert len({row["conformer_id"] for row in rows}) == 49
    assert len({row["pdb_id"] for row in rows}) == 49
    assert all(value == 0 for value in audit["data_boundary"].values())
    override = config["receptor_preparation_overrides"]["BACE1_4I12_aligned"]
    assert override["set_template"] == "A:216,420=CYX"
    assert override["blunt_ends"] is None
    assert override["evidence"]["cyx_only_return_code"] == 0
    assert override["evidence"]["sg_sg_distance_angstrom"] < 2.1
    assert config["receptor_preparation_overrides"]["BACE1_4I1C_aligned"]["set_template"] == "A:216,420=CYX"
    assert config["expected"]["minimum_prepared_receptor_count"] == 40


def test_stage41b_bundle_contains_all_coordinates_without_protected_data() -> None:
    paths = bundle_paths(ROOT)
    assert len([path for path in paths if path.endswith(".cif")]) == 49
    assert len([path for path in paths if path.endswith("_to_3L5D_A.pdb")]) == 49
    assert "environment/stage41b_bace1_input_preparation.yml" in paths
    assert "scripts/freeze_stage41a_bace1_large_pool.py" in paths
    assert "scripts/prepare_receptor_stage41b.py" in paths
    assert "scripts/select_mk14_rcsb_coordinate_pool.py" in paths
    assert "scripts/select_stage13b_egfr_expanded_coordinate_pool.py" in paths
    assert "scripts/run_stage41b_bace1_large_pool_input_preparation_remote.sh" in paths
    assert not any("fresh_validation" in path.lower() for path in paths)
    assert not any("locked_test" in path.lower() for path in paths)


def test_stage41b_missing_checkpoint_is_not_resumed(tmp_path: Path) -> None:
    assert checkpoint(ROOT, tmp_path / "missing.json") is None
