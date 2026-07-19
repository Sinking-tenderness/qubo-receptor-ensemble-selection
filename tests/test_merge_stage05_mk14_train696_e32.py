import json
import sys
from pathlib import Path

import pytest

from scripts.merge_stage05_mk14_train696_e32 import file_sha256, main, write_csv


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")


def file_spec(path: Path, include_hash: bool = True) -> dict[str, str]:
    output = {"path": path.as_posix()}
    if include_hash:
        output["sha256"] = file_sha256(path)
    return output


def build_source(
    root: Path,
    name: str,
    ligand_manifest: Path,
    ligand: dict[str, str],
    receptor_ids: list[str],
    protocol: dict,
    base_score: float,
) -> tuple[dict, Path]:
    seed_runs = []
    seed_evidence = []
    for index, seed in enumerate((101, 102, 103)):
        seed_id = f"seed{index}"
        seed_dir = root / "results" / name / seed_id
        representative = seed_dir / "representative_scores.csv"
        write_csv(
            representative,
            [
                {
                    "ligand_id": ligand["ligand_id"],
                    "receptor_id": receptor_id,
                    "representative_score": base_score + index * 0.1,
                }
                for receptor_id in receptor_ids
            ],
        )
        seed_summary = seed_dir / "summary.json"
        write_json(
            seed_summary,
            {
                "status": "ok",
                "docking_parameters": {"base_seed": seed},
                "observed_receptor_ligand_pairs": len(receptor_ids),
                "failed_receptor_ligand_pairs": 0,
                "outputs": {
                    "representative_long_csv": {
                        "path": representative.as_posix(),
                        "sha256": file_sha256(representative),
                    }
                },
            },
        )
        seed_config = root / "configs" / f"{name}_{seed_id}.json"
        input_paths = {
            "receptor_manifest": protocol["receptor_manifest"]["path"],
            "ligand_manifest": ligand_manifest.as_posix(),
            "vina_executable": protocol["vina_executable"]["path"],
            "vina_config": protocol["vina_config"]["path"],
            "parallel_runner": protocol["parallel_runner"]["path"],
            "score_matrix_module": protocol["score_matrix_module"]["path"],
        }
        input_hashes = {
            key: (
                file_sha256(ligand_manifest)
                if key == "ligand_manifest"
                else protocol[key]["sha256"]
            )
            for key in input_paths
        }
        write_json(
            seed_config,
            {
                "inputs": input_paths,
                "input_sha256": input_hashes,
                "expected_receptor_count": len(receptor_ids),
                "expected_ligand_count": 1,
                "expected_label_counts": {ligand["label"]: 1},
                "docking": {
                    "base_seed": seed,
                    "representative_method": "pose_rank_1",
                },
            },
        )
        seed_runs.append(
            {
                "seed_id": seed_id,
                "base_seed": seed,
                "config_path": seed_config.as_posix(),
                "config_sha256": file_sha256(seed_config),
                "summary_path": seed_summary.as_posix(),
                "representative_scores_path": representative.as_posix(),
            }
        )
        seed_evidence.append(
            {
                "seed_id": seed_id,
                "base_seed": seed,
                "summary_path": seed_summary.as_posix(),
                "summary_sha256": file_sha256(seed_summary),
                "representative_scores_path": representative.as_posix(),
                "representative_scores_sha256": file_sha256(representative),
                "search_quality_warning_count": 0,
            }
        )

    aggregate_dir = root / "results" / name / "aggregate"
    long_path = aggregate_dir / "aggregated.csv"
    median_path = aggregate_dir / "median.csv"
    minimum_path = aggregate_dir / "minimum.csv"
    long_rows = []
    for receptor_index, receptor_id in enumerate(receptor_ids):
        values = [
            base_score + receptor_index * 0.2,
            base_score + receptor_index * 0.2 - 0.1,
            base_score + receptor_index * 0.2 + 0.2,
        ]
        long_rows.append(
            {
                "target_id": "MK14",
                "ligand_id": ligand["ligand_id"],
                "label": ligand["label"],
                "selection_role": ligand["selection_role"],
                "receptor_id": receptor_id,
                "seed_count": 3,
                "seed0_representative_score": values[0],
                "seed1_representative_score": values[1],
                "seed2_representative_score": values[2],
                "median_representative_score": values[0],
                "minimum_representative_score": values[1],
                "maximum_representative_score": values[2],
                "seed_score_range": values[2] - values[1],
                "primary_ranking_score": -values[0],
                "sensitivity_ranking_score": -values[1],
                "representative_method": "pose_rank_1",
                "status": "ok",
            }
        )
    write_csv(long_path, long_rows)
    metadata = {
        "target_id": "MK14",
        "ligand_id": ligand["ligand_id"],
        "label": ligand["label"],
        "selection_role": ligand["selection_role"],
    }
    write_csv(
        median_path,
        [
            {
                **metadata,
                **{
                    row["receptor_id"]: row["median_representative_score"]
                    for row in long_rows
                },
            }
        ],
    )
    write_csv(
        minimum_path,
        [
            {
                **metadata,
                **{
                    row["receptor_id"]: row["minimum_representative_score"]
                    for row in long_rows
                },
            }
        ],
    )
    aggregate_config = root / "configs" / f"{name}_aggregation.json"
    aggregate_outputs = {
        "aggregated_long_csv": long_path.as_posix(),
        "primary_median_matrix_csv": median_path.as_posix(),
        "sensitivity_minimum_matrix_csv": minimum_path.as_posix(),
        "summary_json": (aggregate_dir / "summary.json").as_posix(),
    }
    role = ligand["selection_role"]
    write_json(
        aggregate_config,
        {
            "ligand_manifest": file_spec(ligand_manifest),
            "seed_runs": seed_runs,
            "expected": {
                "seed_count": 3,
                "receptor_count": len(receptor_ids),
                "ligand_count": 1,
                "receptor_ligand_pairs_per_seed": len(receptor_ids),
                "allowed_selection_roles": [role],
                "role_label_counts": {f"{role}:{ligand['label']}": 1},
            },
            "aggregation": {"representative_method": "pose_rank_1"},
            "outputs": aggregate_outputs,
        },
    )
    summary_path = aggregate_dir / "summary.json"
    write_json(
        summary_path,
        {
            "status": "ok",
            "config": {
                "path": aggregate_config.as_posix(),
                "sha256": file_sha256(aggregate_config),
            },
            "ligand_count": 1,
            "receptor_count": len(receptor_ids),
            "seed_count": 3,
            "aggregated_pair_count": len(receptor_ids),
            "locked_test_manifest_rows": 0,
            "aggregation": {"representative_method": "pose_rank_1"},
            "seed_evidence": seed_evidence,
            "outputs": {
                "aggregated_long_csv": file_spec(long_path),
                "primary_median_matrix_csv": file_spec(median_path),
                "sensitivity_minimum_matrix_csv": file_spec(minimum_path),
            },
        },
    )
    return {
        "aggregation_config": file_spec(aggregate_config),
        "summary": file_spec(summary_path),
    }, long_path


def build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    authorization = tmp_path / "configs" / "authorization.json"
    write_json(authorization, {"status": "authorized"})
    receptor_manifest = tmp_path / "data" / "receptors.csv"
    receptor_ids = ["R1", "R2"]
    write_csv(
        receptor_manifest,
        [{"conformer_id": receptor_id, "status": "ok"} for receptor_id in receptor_ids],
    )
    protocol = {"receptor_manifest": file_spec(receptor_manifest)}
    for key, suffix in (
        ("vina_executable", "vina"),
        ("vina_config", "vina.txt"),
        ("parallel_runner", "parallel.py"),
        ("score_matrix_module", "matrix.py"),
    ):
        path = tmp_path / "inputs" / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key + "\n", encoding="ascii")
        protocol[key] = file_spec(path)
    protocol.update(
        {"representative_method": "pose_rank_1", "base_seeds": [101, 102, 103]}
    )

    reused = {
        "target_id": "MK14",
        "ligand_id": "A",
        "label": "active",
        "split": "train",
        "selection_role": "development_train",
        "pdbqt_status": "ok",
        "pdbqt_sha256": "AAA",
    }
    new = {
        "target_id": "MK14",
        "ligand_id": "D",
        "label": "decoy",
        "split": "train",
        "selection_role": "development_train_expanded",
        "pdbqt_status": "ok",
        "pdbqt_sha256": "DDD",
    }
    reused_manifest = tmp_path / "data" / "reused.csv"
    new_manifest = tmp_path / "data" / "new.csv"
    full_manifest = tmp_path / "data" / "full.csv"
    write_csv(reused_manifest, [reused])
    write_csv(new_manifest, [new])
    write_csv(
        full_manifest,
        [{**reused, "selection_role": "development_train_expanded"}, new],
    )
    sources = {}
    sources["reused"], reused_long = build_source(
        tmp_path, "reused", reused_manifest, reused, receptor_ids, protocol, -9.0
    )
    sources["new"], _ = build_source(
        tmp_path, "new", new_manifest, new, receptor_ids, protocol, -7.0
    )
    output_dir = tmp_path / "results" / "merged"
    merge_config = tmp_path / "configs" / "merge.json"
    write_json(
        merge_config,
        {
            "schema_version": "1.0",
            "experiment_id": "test-merge",
            "purpose": "test",
            "authorization": file_spec(authorization),
            "protocol": protocol,
            "manifests": {
                "full": {
                    **file_spec(full_manifest),
                    "selection_role": "development_train_expanded",
                },
                "reused": {
                    **file_spec(reused_manifest),
                    "selection_role": "development_train",
                },
                "new": {
                    **file_spec(new_manifest),
                    "selection_role": "development_train_expanded",
                },
            },
            "sources": sources,
            "expected": {
                "receptor_count": 2,
                "seed_count": 3,
                "reused_ligand_count": 1,
                "new_ligand_count": 1,
                "full_ligand_count": 2,
                "full_label_counts": {"active": 1, "decoy": 1},
                "aggregated_pair_count": 4,
                "reused_seed_cells": 6,
                "new_seed_cells": 6,
                "complete_seed_cells": 12,
            },
            "outputs": {
                "aggregated_long_csv": (output_dir / "aggregated.csv").as_posix(),
                "primary_median_matrix_csv": (output_dir / "median.csv").as_posix(),
                "sensitivity_minimum_matrix_csv": (output_dir / "minimum.csv").as_posix(),
                "summary_json": (output_dir / "summary.json").as_posix(),
            },
            "interpretation_boundary": "test only",
        },
    )
    return merge_config, reused_long


def test_cli_merges_exact_e32_sources_and_normalizes_legacy_role(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config, _ = build_fixture(Path("."))
    monkeypatch.setattr(sys, "argv", ["merge", "--config", str(config)])

    assert main() == 0
    output = list(
        __import__("csv").DictReader(
            Path("results/merged/median.csv").open(encoding="utf-8", newline="")
        )
    )
    assert len(output) == 2
    assert {row["selection_role"] for row in output} == {
        "development_train_expanded"
    }
    summary = json.loads(Path("results/merged/summary.json").read_text())
    assert summary["evidence_cells"] == {
        "reused_e32": 6,
        "new_e32": 6,
        "complete_e32": 12,
        "diagnostic_e64_used": 0,
    }
    assert summary["locked_validation_manifest_rows"] == 0
    assert summary["locked_test_manifest_rows"] == 0


def test_cli_rejects_tampered_reused_aggregate(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config, reused_long = build_fixture(Path("."))
    reused_long.write_text(reused_long.read_text() + "tampered\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["merge", "--config", str(config)])

    with pytest.raises(ValueError, match="reused aggregated_long_csv SHA-256 differs"):
        main()
