import json
from pathlib import Path

import pytest

from qubo_receptor_ensemble import experiment as experiment_module
from qubo_receptor_ensemble.experiment import _load_problem_payload
from qubo_receptor_ensemble.full_workflow import (
    FULL_WORKFLOW_STAGES,
    ConfigError,
    front_input_keys,
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


def test_full_config_accepts_mechanistic_adaptive_k_policy(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selection"]["receptor_count"] = 3
    payload["problem"]["k_policy"] = {
        "mode": "adaptive",
        "selector": "mechanistic_bootstrap_lcb",
        "candidates": [1, 2, 3],
        "scaffold_field": "scaffold_smiles",
        "inner_fold_count": 3,
        "bootstrap_iterations": 100,
        "lower_quantile": 0.025,
        "rescue_fractions": [0.01, 0.05],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_full_experiment_config(path, data_root=tmp_path)

    assert config.data["problem"]["k_policy"]["selector"] == (
        "mechanistic_bootstrap_lcb"
    )


def test_full_config_rejects_unknown_adaptive_k_policy(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["problem"]["k_policy"] = {
        "mode": "adaptive",
        "selector": "unknown",
        "candidates": [1, 2],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match="selector"):
        load_full_experiment_config(path, data_root=tmp_path)


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


def test_full_config_accepts_manual_receptor_selection_without_manifest(
    tmp_path: Path,
) -> None:
    payload = json.loads(_write_config(tmp_path).read_text(encoding="utf-8"))
    payload["selection"]["receptor_selection"] = {
        "mode": "manual",
        "receptors": [
            {
                "conformer_id": "R1",
                "receptor_pdbqt": "prepared/R1.pdbqt",
            },
            {
                "conformer_id": "R2",
                "receptor_pdbqt": "prepared/R2.pdbqt",
            },
        ],
    }
    payload["sources"].pop("receptor_manifest")
    payload["paths"].pop("selected_receptor_manifest")
    path = tmp_path / "manual-receptors.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_full_experiment_config(path, data_root=tmp_path)

    assert config.data["selection"]["receptor_selection"]["mode"] == "manual"
    assert "receptor_manifest" not in config.paths
    assert "selected_receptor_manifest" not in config.paths
    assert front_input_keys(config.data, "dock") == ("prepared_ligand_manifest",)
    assert front_input_keys(config.data, "aggregate") == (
        "prepared_ligand_manifest",
        "score_tables",
    )
    assert front_input_keys(config.data, "build_problem") == ("primary_matrix",)


def test_manual_receptors_are_loaded_from_config_when_building_problem(
    tmp_path: Path,
) -> None:
    payload = json.loads(_write_config(tmp_path).read_text(encoding="utf-8"))
    payload["selection"]["receptor_selection"] = {
        "mode": "manual",
        "receptors": [
            {"conformer_id": "R1", "receptor_pdbqt": "prepared/R1.pdbqt"},
            {"conformer_id": "R2", "receptor_pdbqt": "prepared/R2.pdbqt"},
        ],
    }
    payload["sources"].pop("receptor_manifest")
    payload["paths"].pop("selected_receptor_manifest")
    path = tmp_path / "manual-receptors-problem.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "prepared").mkdir()
    (tmp_path / "prepared" / "R1.pdbqt").write_text("R1", encoding="ascii")
    (tmp_path / "prepared" / "R2.pdbqt").write_text("R2", encoding="ascii")
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("ligand_id,label\nL1,active\n", encoding="utf-8")

    config = load_full_experiment_config(path, data_root=tmp_path)
    result = _load_problem_payload(config, matrix)

    assert result["problem_config"]["receptor_ids"] == ["R1", "R2"]


def test_manual_receptor_selection_requires_exact_receptor_count(tmp_path: Path) -> None:
    payload = json.loads(_write_config(tmp_path).read_text(encoding="utf-8"))
    payload["selection"]["receptor_selection"] = {
        "mode": "manual",
        "receptors": [
            {"conformer_id": "R1", "receptor_pdbqt": "prepared/R1.pdbqt"}
        ],
    }
    payload["sources"].pop("receptor_manifest")
    path = tmp_path / "manual-receptors-invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match="receptor_count"):
        load_full_experiment_config(path, data_root=tmp_path)


def test_manual_receptors_can_reuse_audit_from_another_run_directory(
    tmp_path: Path,
) -> None:
    payload = json.loads(_write_config(tmp_path).read_text(encoding="utf-8"))
    payload["selection"]["receptor_selection"] = {
        "mode": "manual",
        "receptors": [{"conformer_id": "R1", "rcsb_id": "R1"}],
    }
    payload["selection"]["receptor_count"] = 1
    payload["paths"]["receptor_preparation_audit"] = (
        "results/source-run/receptor_preparation_audit.json"
    )
    path = _write_config(tmp_path, selection=payload["selection"], paths=payload["paths"])
    config = load_full_experiment_config(path, data_root=tmp_path)

    source_run = tmp_path / "results" / "source-run"
    source_run.mkdir(parents=True)
    (source_run / "receptor_preparation_audit.json").write_text(
        json.dumps(
            {
                "selected": [
                    {"rcsb_id": "R1", "receptor_pdbqt": "prepared/R1.pdbqt"}
                ]
            }
        ),
        encoding="ascii",
    )
    receptor_path = tmp_path / "prepared" / "R1.pdbqt"
    receptor_path.parent.mkdir(parents=True)
    receptor_path.write_text("receptor", encoding="ascii")

    rows = experiment_module._resolve_manual_receptor_rows(config)

    assert rows == [
        {
            "conformer_id": "R1",
            "rcsb_id": "R1",
            "receptor_pdbqt": "prepared/R1.pdbqt",
        }
    ]


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


def test_raw_full_config_accepts_fixed_box_policy(tmp_path: Path) -> None:
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

    config = load_full_experiment_config(path, data_root=tmp_path)

    assert config.data["docking"]["box"]["center_x"] == 1.0


def test_full_config_accepts_manual_rcsb_receptors_with_fixed_snapshot_box(
    tmp_path: Path,
) -> None:
    payload = json.loads(_write_config(tmp_path).read_text(encoding="utf-8"))
    payload["selection"]["ordering"] = "manual_ids"
    payload["selection"]["ligand_ids"] = [
        "TEST_active_L000001",
        "TEST_active_L000002",
        "TEST_decoy_L000001",
        "TEST_decoy_L000002",
    ]
    payload["selection"]["receptor_selection"] = {
        "mode": "manual",
        "receptors": [
            {"conformer_id": "FA10_R1", "rcsb_id": "R1"},
            {"conformer_id": "FA10_R2", "rcsb_id": "R2"},
        ],
    }
    payload["sources"] = {
        "active_ism": "active.ism",
        "decoy_ism": "decoy.ism",
        "reference_receptor_pdb": "reference.pdb",
        "crystal_ligand": "ligand.mol2",
        "rcsb_directory": "rcsb",
    }
    path = tmp_path / "manual-raw-fixed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_full_experiment_config(path, data_root=tmp_path)

    assert config.data["selection"]["receptor_selection"]["receptors"][0]["rcsb_id"] == "R1"
    assert "ligand_manifest" not in config.paths
    assert "docking_box" not in config.paths


def test_raw_manual_receptors_preserve_configured_names_and_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = json.loads(_write_config(tmp_path).read_text(encoding="utf-8"))
    payload["selection"]["receptor_selection"] = {
        "mode": "manual",
        "receptors": [
            {"conformer_id": "FA10_R1", "rcsb_id": "R1"},
            {"conformer_id": "FA10_R2", "rcsb_id": "R2"},
        ],
    }
    payload["sources"] = {
        "active_ism": "active.ism",
        "decoy_ism": "decoy.ism",
        "reference_receptor_pdb": "reference.pdb",
        "crystal_ligand": "ligand.mol2",
        "rcsb_directory": "rcsb",
    }
    path = tmp_path / "manual-raw-order.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_full_experiment_config(path, data_root=tmp_path)
    observed: dict[str, object] = {}
    audit_path = config.paths["run_directory"] / "receptor_preparation_audit.json"
    assert not audit_path.exists()

    def fake_prepare_raw_receptors(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        records = []
        for rcsb_id in ("R2", "R1"):
            records.append(
                {
                    "conformer_id": rcsb_id,
                    "rcsb_id": rcsb_id,
                    "source_structure": f"rcsb/{rcsb_id}.cif",
                    "source_sha256": "source-hash",
                    "source_pdb": f"run/{rcsb_id}.pdb",
                    "receptor_pdb": f"run/{rcsb_id}_aligned.pdb",
                    "receptor_pdbqt": f"run/{rcsb_id}.pdbqt",
                    "alignment": {
                        "reference_chain": "A",
                        "mobile_chain": "A",
                        "matched_ca_count": 100,
                        "rmsd_after_angstrom": 0.1,
                    },
                }
            )
        return {"selected": records, "candidate_count": 2}

    monkeypatch.setattr(experiment_module, "prepare_raw_receptors", fake_prepare_raw_receptors)

    rows, _ = experiment_module._prepare_raw_receptor_manifest(config)

    assert observed["candidate_ids"] == ["R1", "R2"]
    assert [row["conformer_id"] for row in rows] == ["FA10_R1", "FA10_R2"]
    assert [row["rcsb_id"] for row in rows] == ["R1", "R2"]
    assert audit_path.is_file()
