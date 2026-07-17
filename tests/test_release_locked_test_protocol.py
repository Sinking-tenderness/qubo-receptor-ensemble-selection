import csv
import json
import sys
from pathlib import Path

import pytest

from scripts.release_locked_test_protocol import (
    file_sha256,
    load_config,
    main,
    stratified_bootstrap_metrics,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_repository_release_config_is_fixed_and_descriptive() -> None:
    config = load_config(
        Path("configs/stage04_cdk2_single_best_locked_test_release.json")
    )
    assert config["authorization"]["approved"] is True
    assert config["fixed_protocol"]["receptor_id"] == (
        "CDK2_AF2_MD2NS_C06_F077"
    )
    assert config["evaluation"]["acceptance_threshold"] is None
    assert config["evaluation"]["bootstrap_iterations"] == 5000


def test_stratified_bootstrap_is_deterministic() -> None:
    records = {
        "A1": {"label": "active", "score": -9.0},
        "A2": {"label": "active", "score": -8.0},
        "D1": {"label": "decoy", "score": -6.0},
        "D2": {"label": "decoy", "score": -5.0},
    }
    first = stratified_bootstrap_metrics(records, 20, 17)
    second = stratified_bootstrap_metrics(records, 20, 17)
    assert first == second
    assert first["roc_auc"]["mean"] == 1.0


def test_release_runs_once_and_refuses_overwrite(
    tmp_path: Path, monkeypatch
) -> None:
    receptor = tmp_path / "R1.pdbqt"
    receptor.write_text("ATOM R1\n", encoding="ascii")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "status": "single_best_protocol_materialized_test_locked",
                "selection_rule": {
                    "protocol_id": "single_best_fail_closed_v1"
                },
                "selected_receptor": {
                    "receptor_id": "CDK2_AF2_MD2NS_C06_F077",
                    "artifact": {
                        "receptor_pdbqt": {
                            "path": receptor.as_posix(),
                            "sha256": file_sha256(receptor),
                        }
                    },
                },
                "test_lock": {
                    "scores_evaluated": False,
                    "release_authorized": False,
                },
            }
        ),
        encoding="ascii",
    )
    split = tmp_path / "split.csv"
    write_csv(
        split,
        [
            {"ligand_id": "DEV_A", "label": "active", "split": "train"},
            {"ligand_id": "DEV_D", "label": "decoy", "split": "validation"},
            {"ligand_id": "TEST_A1", "label": "active", "split": "test"},
            {"ligand_id": "TEST_A2", "label": "active", "split": "test"},
            {"ligand_id": "TEST_D1", "label": "decoy", "split": "test"},
            {"ligand_id": "TEST_D2", "label": "decoy", "split": "test"},
        ],
    )
    primary = tmp_path / "primary.csv"
    sensitivity = tmp_path / "sensitivity.csv"
    score_rows = [
        {"ligand_id": "DEV_A", "label": "active", "CDK2_AF2_MD2NS_C06_F077": -7.0, "other": -1.0},
        {"ligand_id": "DEV_D", "label": "decoy", "CDK2_AF2_MD2NS_C06_F077": -6.0, "other": -2.0},
        {"ligand_id": "TEST_A1", "label": "active", "CDK2_AF2_MD2NS_C06_F077": -9.0, "other": 999.0},
        {"ligand_id": "TEST_A2", "label": "active", "CDK2_AF2_MD2NS_C06_F077": -8.0, "other": 999.0},
        {"ligand_id": "TEST_D1", "label": "decoy", "CDK2_AF2_MD2NS_C06_F077": -6.0, "other": -999.0},
        {"ligand_id": "TEST_D2", "label": "decoy", "CDK2_AF2_MD2NS_C06_F077": -5.0, "other": -999.0},
    ]
    write_csv(primary, score_rows)
    write_csv(sensitivity, score_rows)
    source_summary = tmp_path / "source.json"
    source_summary.write_text(
        json.dumps({"ligand_count": 6, "receptor_count": 2}),
        encoding="ascii",
    )
    outputs = {
        "primary_rankings_csv": (tmp_path / "primary_rank.csv").as_posix(),
        "sensitivity_rankings_csv": (
            tmp_path / "sensitivity_rank.csv"
        ).as_posix(),
        "release_marker_json": (tmp_path / "marker.json").as_posix(),
        "summary_json": (tmp_path / "summary.json").as_posix(),
    }
    inputs = {
        "materialized_protocol": protocol.as_posix(),
        "primary_score_matrix": primary.as_posix(),
        "sensitivity_score_matrix": sensitivity.as_posix(),
        "source_summary": source_summary.as_posix(),
        "split_manifest": split.as_posix(),
    }
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment_id": "synthetic-release",
                "purpose": "test",
                "authorization": {
                    "approved": True,
                    "approved_by": "project_owner",
                    "approved_on": "2026-07-17",
                    "authorization_id": "synthetic",
                    "scope": "one_time_fixed_protocol_test_evaluation",
                },
                "inputs": inputs,
                "input_sha256": {
                    key: file_sha256(Path(value)) for key, value in inputs.items()
                },
                "fixed_protocol": {
                    "protocol_id": "single_best_fail_closed_v1",
                    "method": "single_best",
                    "receptor_id": "CDK2_AF2_MD2NS_C06_F077",
                    "receptor_pdbqt_sha256": file_sha256(receptor),
                },
                "expected": {
                    "total_ligand_count": 6,
                    "development_ligand_count": 2,
                    "locked_test_ligand_count": 4,
                    "locked_test_label_counts": {"active": 2, "decoy": 2},
                    "source_receptor_count": 2,
                },
                "evaluation": {
                    "locked_split": "test",
                    "ranking_order": [
                        "docking_score_ascending",
                        "ligand_id_ascending",
                    ],
                    "metrics": [
                        "roc_auc",
                        "pr_auc_average_precision",
                        "bedroc_alpha_20",
                        "EF1%",
                        "EF5%",
                        "EF10%",
                    ],
                    "bootstrap_method": "stratified_active_decoy_resampling_with_replacement",
                    "bootstrap_iterations": 20,
                    "bootstrap_seed": 17,
                    "primary_matrix_role": "median",
                    "sensitivity_matrix_role": "minimum",
                    "acceptance_threshold": None,
                },
                "outputs": outputs,
                "interpretation_boundary": "synthetic test",
            }
        ),
        encoding="ascii",
    )
    argv = [
        "release",
        "--config",
        str(config),
        "--authorize-locked-test-release",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 0
    summary = json.loads(Path(outputs["summary_json"]).read_text(encoding="ascii"))
    assert summary["metrics"]["primary"]["roc_auc"] == 1.0
    assert summary["test_data"]["receptor_score_column_count"] == 1
    assert summary["test_release"]["repeat_or_overwrite_allowed"] is False
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileExistsError):
        main()
