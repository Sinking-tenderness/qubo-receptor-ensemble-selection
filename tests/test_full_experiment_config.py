import json
from pathlib import Path

import pytest

from qubo_receptor_ensemble.full_workflow import (
    FULL_WORKFLOW_STAGES,
    ConfigError,
    load_full_experiment_config,
)


def _write_config(tmp_path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "schema_version": "3.0",
        "experiment_id": "test-full",
        "target_id": "TEST",
        "workflow_mode": "full",
        "selection": {
            "receptor_count": 2,
            "ligand_count": 4,
            "label_counts": {"active": 2, "decoy": 2},
            "ordering": "manifest_order",
        },
        "sources": {
            "active_ism": "raw/active.ism",
            "decoy_ism": "raw/decoy.ism",
            "receptor_manifest": "processed/receptors.csv",
        },
        "docking": {
            "redock": True,
            "engine": "unidock",
            "seeds": [11, 12, 13],
            "executable": "unidock",
            "box": {
                "center_x": 1,
                "center_y": 2,
                "center_z": 3,
                "size_x": 20,
                "size_y": 20,
                "size_z": 20,
            },
            "parameters": {},
        },
        "problem": {
            "type": "receptor_subset",
            "strategy": "qubo",
            "target_size": 1,
            "weights": {"redundancy": 0.25, "count": 0.1, "size": 10.0},
        },
        "solve": {"backend": "exact"},
        "evaluate": {"metrics": ["roc_auc"]},
        "paths": {
            "run_directory": "results/test-full",
            "prepared_ligand_manifest": "results/test-full/prepared_ligands.csv",
            "selected_receptor_manifest": "results/test-full/receptors.csv",
            "score_tables": "results/test-full/scores",
            "matrices": "results/test-full/matrices",
            "problem": "results/test-full/problem.json",
            "selection": "results/test-full/selection.json",
            "evaluation": "results/test-full/evaluation.json",
        },
    }
    for key, value in overrides.items():
        payload[key] = value
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_full_config_defaults_to_source_data_redocking_and_unidock(tmp_path: Path) -> None:
    path = _write_config(tmp_path)

    config = load_full_experiment_config(path, data_root=tmp_path)

    assert config.workflow_mode == "full"
    assert config.start_stage == "prepare"
    assert config.end_stage == "persist"
    assert config.stages == FULL_WORKFLOW_STAGES
    assert config.data["docking"]["redock"] is True
    assert config.data["docking"]["engine"] == "unidock"
    assert config.data["selection"]["label_counts"] == {"active": 2, "decoy": 2}


def test_full_config_defaults_problem_selection_to_bedroc20(tmp_path: Path) -> None:
    path = _write_config(tmp_path)

    config = load_full_experiment_config(path, data_root=tmp_path)

    assert config.data["problem"]["utility_metric"] == "bedroc"
    assert config.data["problem"]["bedroc_alpha"] == 20.0


def test_full_config_resolves_external_data_root_without_sha_requirements(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path)

    config = load_full_experiment_config(path, data_root=tmp_path / "external")

    assert config.data_root == (tmp_path / "external").resolve()
    assert config.paths["active_ism"] == (
        tmp_path / "external" / "raw" / "active.ism"
    ).resolve()


def test_partial_config_requires_front_input_paths(tmp_path: Path) -> None:
    paths = {
        "run_directory": "results/test-full",
        "prepared_ligand_manifest": "results/test-full/prepared_ligands.csv",
        "selected_receptor_manifest": "results/test-full/receptors.csv",
        "matrices": "results/test-full/matrices",
        "problem": "results/test-full/problem.json",
        "selection": "results/test-full/selection.json",
        "evaluation": "results/test-full/evaluation.json",
    }
    path = _write_config(tmp_path, start_stage="aggregate", paths=paths)

    with pytest.raises(ConfigError, match="score_tables"):
        load_full_experiment_config(path, data_root=tmp_path)


def test_reference_replay_must_be_explicit(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        workflow_mode="reference_replay",
        docking={"redock": False, "engine": "unidock", "seeds": [11]},
    )

    config = load_full_experiment_config(path, data_root=tmp_path)

    assert config.workflow_mode == "reference_replay"
    assert config.data["docking"]["redock"] is False


def test_full_config_rejects_unknown_docking_engine(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        docking={"redock": True, "engine": "remote_unidock", "seeds": [11]},
    )

    with pytest.raises(ConfigError, match="docking.engine"):
        load_full_experiment_config(path, data_root=tmp_path)


def test_preselected_manifest_is_required_for_preselected_ordering(tmp_path: Path) -> None:
    payload = json.loads(_write_config(tmp_path).read_text(encoding="utf-8"))
    payload["selection"]["ordering"] = "preselected_manifest"
    path = tmp_path / "preselected.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match="ligand_manifest"):
        load_full_experiment_config(path, data_root=tmp_path)


def test_full_config_accepts_raw_sources_and_computed_box_policy(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        selection={
            "receptor_count": 2,
            "ligand_count": 4,
            "label_counts": {"active": 2, "decoy": 2},
            "ordering": "manifest_order",
        },
        sources={
            "active_ism": "raw/active.ism",
            "decoy_ism": "raw/decoy.ism",
            "reference_receptor_pdb": "raw/receptor.pdb",
            "crystal_ligand": "raw/crystal_ligand.mol2",
            "rcsb_directory": "raw/rcsb",
        },
        docking={
            "redock": True,
            "engine": "unidock",
            "seeds": [11, 12, 13],
            "box": {
                "method": "ligand_bounds",
                "padding": 5.0,
                "minimum_size": [22.0, 22.0, 28.0],
            },
            "parameters": {},
        },
    )

    config = load_full_experiment_config(path, data_root=tmp_path)

    assert config.paths["rcsb_directory"] == (tmp_path / "raw" / "rcsb").resolve()
    assert config.paths["crystal_ligand"] == (
        tmp_path / "raw" / "crystal_ligand.mol2"
    ).resolve()


def test_full_config_accepts_scaffold_hash_allocation_and_preserves_manifest_option(
    tmp_path: Path,
) -> None:
    payload = json.loads(_write_config(tmp_path).read_text(encoding="utf-8"))
    payload["selection"]["ordering"] = "scaffold_hash_allocation"
    payload["selection"]["allocation"] = {
        "hash_namespace": "STAGE102A",
        "outer_fold_count": 2,
        "minimum_label_counts_per_outer_fold": {"active": 1, "decoy": 1},
    }
    path = tmp_path / "scaffold-hash.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_full_experiment_config(path, data_root=tmp_path)

    assert config.data["selection"]["ordering"] == "scaffold_hash_allocation"
    assert config.data["selection"]["allocation"]["outer_fold_count"] == 2


def test_raw_full_config_rejects_fixed_box_policy(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        sources={
            "active_ism": "raw/active.ism",
            "decoy_ism": "raw/decoy.ism",
            "reference_receptor_pdb": "raw/receptor.pdb",
            "crystal_ligand": "raw/crystal_ligand.mol2",
            "rcsb_directory": "raw/rcsb",
        },
        docking={
            "redock": True,
            "engine": "unidock",
            "seeds": [11, 12, 13],
            "box": {
                "center_x": 1.0,
                "center_y": 2.0,
                "center_z": 3.0,
                "size_x": 22.0,
                "size_y": 22.0,
                "size_z": 28.0,
            },
            "parameters": {},
        },
    )

    with pytest.raises(ConfigError, match="ligand_bounds"):
        load_full_experiment_config(path, data_root=tmp_path)
