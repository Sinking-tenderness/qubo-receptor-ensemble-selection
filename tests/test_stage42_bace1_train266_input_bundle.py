from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_stage42_preregistration_freezes_expected_scale() -> None:
    config = json.loads(
        (ROOT / "configs/stage42_bace1_redocking_qualified_development_preregistration.json").read_text()
    )

    assert config["receptor_pool"]["redocking_qualified_count"] == 34
    assert len(config["receptor_pool"]["receptor_ids"]) == 34
    assert config["ligand_panel"]["development_train"] == {
        "active_count": 133,
        "decoy_count": 133,
    }
    assert config["production_protocol"]["receptor_ligand_seed_pair_count"] == 27132
    assert config["evidence_timing"]["stage41c_gate_failure_known"] is True


def test_stage42_configs_reference_frozen_files() -> None:
    for relative in (
        "configs/stage42a_bace1_ligand_panel_allocation.json",
        "configs/stage42b_bace1_train266_unidock_input_preparation.json",
    ):
        config = json.loads((ROOT / relative).read_text())
        for section in ("implementation", "inputs"):
            for descriptor in dict(config.get(section, {})).values():
                if not isinstance(descriptor, dict) or "path" not in descriptor or "sha256" not in descriptor:
                    continue
                path = ROOT / descriptor["path"]
                assert path.is_file()
                assert sha256(path) == descriptor["sha256"]


def test_stage42_scripts_parse_and_bundle_has_no_docking_job() -> None:
    for relative in (
        "scripts/allocate_stage42a_bace1_ligand_panels.py",
        "scripts/experimental/unidock/prepare_stage42b_bace1_train266_inputs.py",
        "scripts/experimental/unidock/build_stage42b_bace1_train266_input_bundle.py",
    ):
        ast.parse((ROOT / relative).read_text())
    summary = json.loads(
        (ROOT / "data/stage42b_bace1_train266_input_bundle_summary.json").read_text()
    )
    assert summary["status"] == "ok"
    assert summary["development_ligand_count"] == 266
    assert summary["gpu_docking_jobs_in_this_bundle"] == 0


def test_stage42_achiral_scaffold_clears_stereo_before_serialization() -> None:
    rdkit = pytest.importorskip("rdkit")
    del rdkit
    from rdkit import Chem

    from scripts.allocate_stage42a_bace1_ligand_panels import scaffold_for

    molecule = Chem.MolFromSmiles("C/C=C(/c1ccccc1)c1ncccc1")
    assert molecule is not None
    scaffold = scaffold_for(molecule)
    assert scaffold
    assert "/" not in scaffold
    assert "\\" not in scaffold


def test_stage42b_independent_input_audit_passed() -> None:
    audit = json.loads(
        (ROOT / "data/stage42b_bace1_train266_unidock_input_independent_audit.json").read_text()
    )
    assert audit["status"] == "independent_stage42b_bace1_train266_input_audit_ok"
    assert audit["ligand_count"] == 266
    assert audit["label_counts"] == {"active": 133, "decoy": 133}
    assert audit["verified_prepared_file_count"] == 532
    assert audit["failed_ligand_count"] == 0
    assert audit["macrocycle_closure_pseudoatom_ligand_count"] == 0


def test_stage42c_production_bundle_freezes_complete_development_grid() -> None:
    config = json.loads(
        (ROOT / "configs/stage42c_bace1_train266_unidock113_production.json").read_text()
    )
    assert config["expected"]["receptor_count"] == 34
    assert config["expected"]["ligand_count"] == 266
    assert config["expected"]["seed_count"] == 3
    assert config["expected"]["batch_count"] == 102
    assert config["expected"]["pair_count"] == 27132
    assert config["expected"]["validation_rows"] == 0
    assert config["expected"]["test_rows"] == 0
    assert config["unidock"]["exhaustiveness"] == 1024

    summary = json.loads(
        (ROOT / "data/stage42c_bace1_train266_unidock113_production_bundle_summary.json").read_text()
    )
    assert summary["status"] == "ok"
    assert summary["gpu_pair_count"] == 27132
    assert summary["validation_rows"] == 0
    assert summary["test_rows"] == 0
