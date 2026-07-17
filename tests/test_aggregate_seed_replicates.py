import json
import sys

import pytest

from scripts.aggregate_seed_replicates import (
    aggregate_seed_rows,
    audit_ligand_manifest,
    build_matrix,
    file_sha256,
    main,
    write_csv,
)


def ligand_rows():
    return [
        {
            "target_id": "MK14",
            "ligand_id": "A",
            "label": "active",
            "split": "train",
            "selection_role": "development_train",
        },
        {
            "target_id": "MK14",
            "ligand_id": "D",
            "label": "decoy",
            "split": "validation",
            "selection_role": "development_validation",
        },
    ]


def seed_rows(offset: float):
    rows = []
    for ligand_id, label, base in (("A", "active", -9.0), ("D", "decoy", -7.0)):
        for receptor_id, receptor_offset in (("R1", 0.0), ("R2", 0.5)):
            rows.append(
                {
                    "ligand_id": ligand_id,
                    "label": label,
                    "receptor_id": receptor_id,
                    "representative_score": str(base + receptor_offset + offset),
                    "representative_method": "pose_rank_1",
                    "status": "ok",
                }
            )
    return rows


def audited_ligands():
    return audit_ligand_manifest(
        ligand_rows(),
        expected_count=2,
        expected_role_label_counts={
            "development_train:active": 1,
            "development_validation:decoy": 1,
        },
        allowed_roles={"development_train", "development_validation"},
    )


def test_three_seed_aggregation_uses_frozen_median_and_minimum() -> None:
    rows = aggregate_seed_rows(
        [
            ("seed0", seed_rows(0.0)),
            ("seed1", seed_rows(-0.3)),
            ("seed2", seed_rows(0.2)),
        ],
        audited_ligands(),
        expected_receptor_count=2,
        representative_method="pose_rank_1",
    )

    active_r1 = next(
        row for row in rows if row["ligand_id"] == "A" and row["receptor_id"] == "R1"
    )
    assert active_r1["median_representative_score"] == pytest.approx(-9.0)
    assert active_r1["minimum_representative_score"] == pytest.approx(-9.3)
    assert active_r1["seed_score_range"] == pytest.approx(0.5)
    matrix = build_matrix(rows, "median_representative_score")
    assert len(matrix) == 2
    assert matrix[0]["R1"] == pytest.approx(-9.0)


def test_aggregation_rejects_incomplete_seed() -> None:
    with pytest.raises(ValueError, match="expected 4 rows"):
        aggregate_seed_rows(
            [("seed0", seed_rows(0.0)), ("seed1", seed_rows(0.1)[:-1])],
            audited_ligands(),
            expected_receptor_count=2,
            representative_method="pose_rank_1",
        )


def test_ligand_audit_rejects_locked_test_rows() -> None:
    rows = ligand_rows()
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="locked test"):
        audit_ligand_manifest(
            rows,
            expected_count=2,
            expected_role_label_counts={
                "development_train:active": 1,
                "development_validation:decoy": 1,
            },
            allowed_roles={"development_train", "development_validation"},
        )


def test_cli_audits_three_seed_outputs_and_writes_matrices(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ligand_path = tmp_path / "data" / "ligands.csv"
    write_csv(ligand_path, ligand_rows())
    seed_specs = []
    for index, offset in enumerate((0.0, -0.3, 0.2)):
        seed_id = f"seed{index}"
        run_config = tmp_path / "configs" / f"{seed_id}.json"
        run_config.parent.mkdir(parents=True, exist_ok=True)
        run_config.write_text("{}\n", encoding="ascii")
        representative = tmp_path / "results" / seed_id / "representative.csv"
        write_csv(representative, seed_rows(offset))
        summary = tmp_path / "results" / seed_id / "summary.json"
        summary.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "docking_parameters": {"base_seed": 100 + index},
                    "failed_receptor_ligand_pairs": 0,
                    "observed_receptor_ligand_pairs": 4,
                    "search_quality_warning_count": 0,
                    "outputs": {
                        "representative_long_csv": {
                            "path": representative.relative_to(tmp_path).as_posix(),
                            "sha256": file_sha256(representative),
                        }
                    },
                }
            )
            + "\n",
            encoding="ascii",
        )
        seed_specs.append(
            {
                "seed_id": seed_id,
                "base_seed": 100 + index,
                "config_path": run_config.relative_to(tmp_path).as_posix(),
                "config_sha256": file_sha256(run_config),
                "summary_path": summary.relative_to(tmp_path).as_posix(),
                "representative_scores_path": representative.relative_to(
                    tmp_path
                ).as_posix(),
            }
        )

    config = {
        "schema_version": "1.0",
        "experiment_id": "test-aggregation",
        "purpose": "test",
        "ligand_manifest": {
            "path": ligand_path.relative_to(tmp_path).as_posix(),
            "sha256": file_sha256(ligand_path),
        },
        "seed_runs": seed_specs,
        "expected": {
            "seed_count": 3,
            "receptor_count": 2,
            "ligand_count": 2,
            "receptor_ligand_pairs_per_seed": 4,
            "allowed_selection_roles": [
                "development_train",
                "development_validation",
            ],
            "role_label_counts": {
                "development_train:active": 1,
                "development_validation:decoy": 1,
            },
        },
        "aggregation": {
            "representative_method": "pose_rank_1",
            "primary": "median",
            "sensitivity": "minimum",
        },
        "outputs": {
            "aggregated_long_csv": "out/long.csv",
            "primary_median_matrix_csv": "out/median.csv",
            "sensitivity_minimum_matrix_csv": "out/minimum.csv",
            "summary_json": "out/summary.json",
        },
        "interpretation_boundary": "test only",
    }
    config_path = tmp_path / "aggregation.json"
    config_path.write_text(json.dumps(config) + "\n", encoding="ascii")
    monkeypatch.setattr(sys, "argv", ["aggregate", "--config", str(config_path)])

    assert main() == 0
    median_lines = (tmp_path / "out" / "median.csv").read_text().splitlines()
    assert len(median_lines) == 3
    assert len((tmp_path / "out" / "minimum.csv").read_text().splitlines()) == 3
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["locked_test_manifest_rows"] == 0
