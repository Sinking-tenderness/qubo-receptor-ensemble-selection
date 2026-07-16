from pathlib import Path

import pytest

from scripts.run_receptor_selection_validation_gate import (
    load_config,
    normalize_from_train,
    solve_qubo_grid,
    validate_dataset,
)


CONFIG_PATH = Path(
    "configs/stage03_cdk2_af2_md_100a100d_selection_validation_gate.json"
)


def test_fixed_validation_gate_keeps_test_locked():
    config = load_config(CONFIG_PATH)
    assert config["selection"]["selection_split"] == "train"
    assert config["selection"]["tuning_split"] == "validation"
    assert config["selection"]["locked_split"] == "test"
    assert config["selection"]["evaluate_locked_test"] is False
    assert config["selection"]["subset_sizes"] == [2, 3]


def test_train_minmax_does_not_fit_on_validation_values():
    train = [
        {"ligand_id": "A", "label": "active", "R1": "-10", "R2": "-8"},
        {"ligand_id": "D", "label": "decoy", "R1": "-5", "R2": "-4"},
    ]
    validation = [
        {"ligand_id": "V", "label": "active", "R1": "-15", "R2": "0"}
    ]
    normalized_train, normalized_validation, bounds = normalize_from_train(
        train, validation, ["R1", "R2"]
    )
    assert bounds["R1"] == {"minimum": -10.0, "maximum": -5.0}
    assert normalized_train[0]["R1"] == pytest.approx(0.0)
    assert normalized_train[1]["R1"] == pytest.approx(1.0)
    assert normalized_validation[0]["R1"] == pytest.approx(-1.0)
    assert normalized_validation[0]["R2"] == pytest.approx(2.0)


def matrix_rows():
    rows = []
    specifications = [
        ("TA", "active", "train", -9.0, -8.0),
        ("TD", "decoy", "train", -5.0, -4.0),
        ("VA", "active", "validation", -8.0, -7.0),
        ("VD", "decoy", "validation", -4.0, -3.0),
        ("XA", "active", "test", -7.0, -6.0),
        ("XD", "decoy", "test", -3.0, -2.0),
    ]
    for ligand_id, label, split, r1, r2 in specifications:
        rows.append(
            {
                "target_id": "T",
                "ligand_id": ligand_id,
                "label": label,
                "R1": str(r1),
                "R2": str(r2),
                "split": split,
            }
        )
    return rows


def test_dataset_audit_partitions_visible_splits_without_returning_test_scores():
    rows = matrix_rows()
    primary = [
        {key: value for key, value in row.items() if key != "split"} for row in rows
    ]
    sensitivity = [dict(row) for row in primary]
    split_rows = [
        {
            "ligand_id": row["ligand_id"],
            "label": row["label"],
            "split": row["split"],
        }
        for row in rows
    ]
    warning_rows = [
        {"ligand_id": "TA", "receptor_id": "R1", "label": "active"}
    ]
    expected = {
        "ligand_count": 6,
        "label_counts": {"active": 3, "decoy": 3},
        "split_label_counts": {
            "train": {"active": 1, "decoy": 1},
            "validation": {"active": 1, "decoy": 1},
            "test": {"active": 1, "decoy": 1},
        },
        "seed_warning_count": 1,
    }
    audit = validate_dataset(
        primary,
        sensitivity,
        split_rows,
        warning_rows,
        ["R1", "R2"],
        expected,
    )
    assert set(audit["primary_visible"]) == {"train", "validation"}
    assert [row["ligand_id"] for row in audit["primary_visible"]["train"]] == [
        "TA",
        "TD",
    ]
    assert audit["warning_by_split"] == {"train": 1}


def test_dataset_audit_accepts_development_only_score_matrices():
    rows = matrix_rows()
    development = [row for row in rows if row["split"] != "test"]
    primary = [
        {key: value for key, value in row.items() if key != "split"}
        for row in development
    ]
    split_rows = [
        {
            "ligand_id": row["ligand_id"],
            "label": row["label"],
            "split": row["split"],
        }
        for row in rows
    ]
    expected = {
        "ligand_count": 6,
        "label_counts": {"active": 3, "decoy": 3},
        "split_label_counts": {
            "train": {"active": 1, "decoy": 1},
            "validation": {"active": 1, "decoy": 1},
            "test": {"active": 1, "decoy": 1},
        },
        "seed_warning_count": 1,
    }

    audit = validate_dataset(
        primary,
        [dict(row) for row in primary],
        split_rows,
        [{"ligand_id": "TA", "receptor_id": "R1"}],
        ["R1", "R2"],
        expected,
        {"train", "validation"},
    )

    assert sum(len(rows) for rows in audit["primary_visible"].values()) == 4
    assert audit["warning_by_split"] == {"train": 1}


def qubo_rows():
    labels = ["active"] * 4 + ["decoy"] * 4
    score_columns = {
        "R1": [-9.0, -8.8, -8.6, -8.4, -6.0, -5.8, -5.6, -5.4],
        "R2": [-8.7, -8.5, -6.2, -6.0, -8.4, -5.7, -5.5, -5.3],
        "R3": [-5.4, -5.6, -5.8, -6.0, -8.4, -8.6, -8.8, -9.0],
        "R4": [-8.0, -7.8, -7.6, -7.4, -6.4, -6.2, -6.0, -5.8],
    }
    return [
        {
            "ligand_id": f"L{index}",
            "label": label,
            **{
                receptor_id: values[index]
                for receptor_id, values in score_columns.items()
            },
        }
        for index, label in enumerate(labels)
    ]


def test_qubo_grid_enforces_budget_and_returns_only_visible_metrics():
    selection = {
        "subset_sizes": [2],
        "aggregation_methods": ["min_score", "mean_score"],
        "utility_metric": "bedroc",
        "validation_metric": "bedroc_alpha_20",
        "validation_tie_breakers": ["pr_auc_average_precision", "roc_auc"],
        "coverage_fraction": 0.25,
        "size_penalty": 10.0,
        "weight_grids": {
            "coverage": [0.0, 0.5],
            "overlap": [0.0],
            "redundancy": [0.0],
        },
    }
    rows = qubo_rows()
    trials, chosen = solve_qubo_grid(
        rows,
        rows,
        ["R1", "R2", "R3", "R4"],
        selection,
    )
    assert len(trials) == 4
    assert all(len(row["subset"]) == 2 for row in trials)
    assert len(chosen["subset"]) == 2
    assert "validation_metrics" in chosen
    assert all("test" not in key for key in chosen)
