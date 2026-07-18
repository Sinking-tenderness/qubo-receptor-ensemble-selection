from pathlib import Path

import pytest

from scripts.run_stage05_mk14_method_gate import (
    audit_inputs,
    check_runtime,
    checked_input_paths,
    exact_random_subset_tables,
    gate_decision,
    load_execution_config,
    make_frozen_group_folds,
    model_from_preregistration,
    read_json,
    validate_preregistration,
)


CONFIG_PATH = Path(
    "configs/stage05_mk14_development_method_gate_execution.json"
)


def test_execution_config_preserves_train_validation_and_test_lock():
    config = load_execution_config(CONFIG_PATH)
    preregistration = read_json(Path(config["preregistration"]["path"]))
    validate_preregistration(preregistration)
    model = model_from_preregistration(preregistration)

    assert preregistration["data_roles"]["train"]["active"] == 80
    assert preregistration["data_roles"]["validation"]["active"] == 40
    assert preregistration["data_roles"]["test"]["status"] == "locked_unreleased"
    assert model["families"] == ["coverage_qubo", "discriminative_qubo"]
    assert model["subset_sizes"] == [1, 2, 3]
    assert config["runtime"] == {
        "conda_environment": "qubo-receptor-ensemble",
        "python_version": "3.11.15",
        "numpy_version": "1.26.4",
        "scipy_version": "1.17.1",
    }


def test_frozen_group_folds_keep_groups_and_scaffolds_together():
    rows = [
        {
            "ligand_id": "A1",
            "label": "active",
            "split_group_id": "GA",
            "scaffold_smiles": "SA",
        },
        {
            "ligand_id": "A2",
            "label": "active",
            "split_group_id": "GA",
            "scaffold_smiles": "SA",
        },
        {
            "ligand_id": "A3",
            "label": "active",
            "split_group_id": "GB",
            "scaffold_smiles": "SB",
        },
        {
            "ligand_id": "A4",
            "label": "active",
            "split_group_id": "GC",
            "scaffold_smiles": "SC",
        },
        {
            "ligand_id": "D1",
            "label": "decoy",
            "split_group_id": "GD",
            "scaffold_smiles": "SD",
        },
        {
            "ligand_id": "D2",
            "label": "decoy",
            "split_group_id": "GE",
            "scaffold_smiles": "SE",
        },
        {
            "ligand_id": "D3",
            "label": "decoy",
            "split_group_id": "GF",
            "scaffold_smiles": "SF",
        },
        {
            "ligand_id": "D4",
            "label": "decoy",
            "split_group_id": "GG",
            "scaffold_smiles": "SG",
        },
    ]
    assignments = make_frozen_group_folds(rows, 2, 7)

    assert assignments["A1"] == assignments["A2"]
    for fold in range(2):
        assert {
            row["label"]
            for row in rows
            if assignments[row["ligand_id"]] == fold
        } == {"active", "decoy"}


def test_exact_random_subset_table_enumerates_without_random_sampling():
    rows = [
        {"ligand_id": "A", "label": "active", "R1": 0.0, "R2": 1.0, "R3": 0.5},
        {"ligand_id": "D", "label": "decoy", "R1": 1.0, "R2": 0.0, "R3": 0.5},
    ]
    detail, summary = exact_random_subset_tables(
        {"primary": rows}, ["R1", "R2", "R3"], [1, 2], ["min_score", "mean_score"]
    )

    assert len(detail) == 3 + 3 * 2
    assert len(summary) == 3
    assert {row["subset_count"] for row in summary} == {3}


def test_gate_requires_every_preregistered_check():
    selected = {
        "primary": {
            "bedroc_alpha_20": 0.72,
            "roc_auc": 0.71,
            "pr_auc_average_precision": 0.70,
        },
        "sensitivity": {"bedroc_alpha_20": 0.68},
    }
    baseline = {
        "primary": {
            "bedroc_alpha_20": 0.70,
            "roc_auc": 0.70,
            "pr_auc_average_precision": 0.69,
        },
        "sensitivity": {"bedroc_alpha_20": 0.67},
    }
    bootstrap = {"bedroc_alpha_20": {"ci95_low": -0.001}}
    gate = {
        "minimum_primary_bedroc_delta": 0.02,
        "minimum_primary_roc_auc_delta": 0.0,
        "minimum_primary_pr_auc_delta": 0.0,
        "minimum_sensitivity_bedroc_delta": 0.0,
        "minimum_primary_bedroc_bootstrap_ci95_low": 0.0,
    }
    _, checks, passed = gate_decision(selected, baseline, bootstrap, gate)

    assert checks["primary_bedroc_delta"]
    assert not checks["primary_bedroc_bootstrap_ci95_low"]
    assert not passed


def test_execution_config_hash_is_current():
    config = load_execution_config(CONFIG_PATH)
    assert config["preregistration"]["sha256"] == (
        "1F2D81BE000A13F4B1E0EA867299073663C784B3BE1E41BBA52AAABCCB1CCD23"
    )
    assert config["inputs"]["primary_matrix"]["sha256"] == (
        "921348DF346C0104438C245CD116A51222850E7E801E8E27884B7968E9786612"
    )


def test_current_mapk14_inputs_pass_pre_metric_audit():
    config = load_execution_config(CONFIG_PATH)
    preregistration_path, input_paths = checked_input_paths(config)
    preregistration = read_json(preregistration_path)
    audited = audit_inputs(preregistration, input_paths)

    assert len(audited["train_manifest_rows"]) == 160
    assert len(audited["validation_manifest_rows"]) == 80
    assert audited["nonnegative_score_counts"] == {
        "primary": 0,
        "sensitivity": 0,
    }


def test_method_gate_runtime_matches_project_environment():
    config = load_execution_config(CONFIG_PATH)
    actual = check_runtime(config)

    assert actual["conda_environment"] == "qubo-receptor-ensemble"
    assert actual["python_version"] == "3.11.15"
