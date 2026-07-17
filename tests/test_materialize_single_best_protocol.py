import csv
import json
import sys
from pathlib import Path

from scripts.materialize_single_best_protocol import (
    file_sha256,
    load_config,
    main,
    rank_receptors,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_repository_config_is_fail_closed() -> None:
    config = load_config(
        Path("configs/stage04_cdk2_single_best_protocol_materialization.json")
    )
    assert config["selection"]["method"] == "single_best"
    assert config["selection"]["locked_split"] == "test"
    assert config["selection"]["development_splits"] == [
        "train",
        "validation",
    ]


def test_rank_receptors_prefers_early_active_ranking() -> None:
    rows = [
        {"ligand_id": "A1", "label": "active", "good": -9.0, "bad": -5.0},
        {"ligand_id": "A2", "label": "active", "good": -8.0, "bad": -6.0},
        {"ligand_id": "D1", "label": "decoy", "good": -6.0, "bad": -9.0},
        {"ligand_id": "D2", "label": "decoy", "good": -5.0, "bad": -8.0},
    ]
    ranking = rank_receptors(rows, ["bad", "good"])
    assert ranking[0]["receptor_id"] == "good"
    assert ranking[0]["selected"] is True
    assert ranking[1]["selected"] is False


def test_main_materializes_without_test_score_cells(
    tmp_path: Path, monkeypatch
) -> None:
    selector = tmp_path / "selector.json"
    selector.write_text(
        json.dumps(
            {
                "status": "fallback_selected_no_reliable_qubo_candidate",
                "selected_protocol": {
                    "protocol_id": "single_best_fail_closed_v1",
                    "method": "single_best",
                },
                "test_lock": {"scores_evaluated": False},
            }
        ),
        encoding="ascii",
    )
    manifest = tmp_path / "split.csv"
    write_csv(
        manifest,
        [
            {"ligand_id": "A1", "label": "active", "split": "train"},
            {"ligand_id": "A2", "label": "active", "split": "validation"},
            {"ligand_id": "D1", "label": "decoy", "split": "train"},
            {"ligand_id": "D2", "label": "decoy", "split": "validation"},
            {"ligand_id": "T1", "label": "active", "split": "test"},
        ],
    )
    matrix = tmp_path / "matrix.csv"
    write_csv(
        matrix,
        [
            {"target_id": "T", "ligand_id": "A1", "label": "active", "R1": -9.0, "R2": -5.0},
            {"target_id": "T", "ligand_id": "A2", "label": "active", "R1": -8.0, "R2": -6.0},
            {"target_id": "T", "ligand_id": "D1", "label": "decoy", "R1": -6.0, "R2": -9.0},
            {"target_id": "T", "ligand_id": "D2", "label": "decoy", "R1": -5.0, "R2": -8.0},
        ],
    )
    sensitivity = tmp_path / "sensitivity.csv"
    sensitivity.write_bytes(matrix.read_bytes())
    receptor_rows = []
    for receptor_id in ("R1", "R2"):
        pdb = tmp_path / f"{receptor_id}.pdb"
        pdbqt = tmp_path / f"{receptor_id}.pdbqt"
        pdb.write_text(f"ATOM {receptor_id}\n", encoding="ascii")
        pdbqt.write_text(f"ATOM {receptor_id}\n", encoding="ascii")
        receptor_rows.append(
            {
                "conformer_id": receptor_id,
                "source_type": "synthetic",
                "source_identifier": receptor_id,
                "chain": "A",
                "selected_altloc": "A",
                "preparation_status": "ok",
                "pdb_path": pdb.as_posix(),
                "pdb_sha256": file_sha256(pdb),
                "receptor_pdbqt_path": pdbqt.as_posix(),
                "receptor_pdbqt_sha256": file_sha256(pdbqt),
                "pocket_residue_count": 1,
                "pdbqt_atom_count": 1,
                "pdbqt_hetatm_count": 0,
                "pdbqt_hydrogen_like_atom_count": 0,
                "pdbqt_charge_min": 0,
                "pdbqt_charge_max": 0,
                "pdbqt_autodock_atom_types": "C",
            }
        )
    receptor_manifest = tmp_path / "receptors.csv"
    write_csv(receptor_manifest, receptor_rows)
    outer = tmp_path / "outer.csv"
    write_csv(
        outer,
        [
            {"outer_fold": 0, "method": "single_best", "subset": "R1"},
            {"outer_fold": 1, "method": "single_best", "subset": "R2"},
        ],
    )
    source_summary = tmp_path / "source_summary.json"
    source_summary.write_text(
        json.dumps(
            {
                "test_lock": {"scores_evaluated": False},
                "outputs": {
                    "outer_fold_results_csv": {
                        "path": outer.as_posix(),
                        "sha256": file_sha256(outer),
                    }
                },
            }
        ),
        encoding="ascii",
    )
    repeated = tmp_path / "repeated.json"
    repeated.write_text(
        json.dumps(
            {
                "requested_repeat_count": 1,
                "successful_repeat_count": 1,
                "test_lock": {
                    "scores_evaluated": False,
                    "all_successful_repeat_audits_passed": True,
                },
                "source_runs": [
                    {
                        "fold_seed": 7,
                        "summary": {
                            "path": source_summary.as_posix(),
                            "sha256": file_sha256(source_summary),
                        },
                    }
                ],
            }
        ),
        encoding="ascii",
    )
    ranking = tmp_path / "ranking.csv"
    protocol = tmp_path / "protocol.json"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment_id": "synthetic",
                "purpose": "test",
                "upstream_selector": {
                    "path": selector.as_posix(),
                    "sha256": file_sha256(selector),
                    "protocol_id": "single_best_fail_closed_v1",
                },
                "stability_evidence": {
                    "path": repeated.as_posix(),
                    "sha256": file_sha256(repeated),
                },
                "inputs": {
                    "primary_matrix": matrix.as_posix(),
                    "receptor_manifest": receptor_manifest.as_posix(),
                    "sensitivity_matrix": sensitivity.as_posix(),
                    "split_manifest": manifest.as_posix(),
                },
                "input_sha256": {
                    "primary_matrix": file_sha256(matrix),
                    "receptor_manifest": file_sha256(receptor_manifest),
                    "sensitivity_matrix": file_sha256(sensitivity),
                    "split_manifest": file_sha256(manifest),
                },
                "receptor_ids": ["R1", "R2"],
                "expected": {
                    "development_ligand_count": 4,
                    "locked_test_ligand_count": 1,
                    "development_label_counts": {"active": 2, "decoy": 2},
                    "locked_test_label_counts": {"active": 1},
                    "stability_repeat_count": 1,
                    "stability_outer_fold_count": 2,
                },
                "selection": {
                    "protocol_id": "single_best_fail_closed_v1",
                    "method": "single_best",
                    "aggregation": "min_score",
                    "selection_metric": "bedroc_alpha_20",
                    "tie_breakers": [
                        "pr_auc_average_precision",
                        "roc_auc",
                        "receptor_id",
                    ],
                    "score_direction": "lower_is_better",
                    "score_matrix_role": "synthetic",
                    "development_splits": ["train", "validation"],
                    "locked_split": "test",
                },
                "outputs": {
                    "receptor_ranking_csv": ranking.as_posix(),
                    "protocol_json": protocol.as_posix(),
                },
                "interpretation_boundary": "synthetic test",
            }
        ),
        encoding="ascii",
    )
    monkeypatch.setattr(sys, "argv", ["materialize", "--config", str(config)])
    assert main() == 0
    result = json.loads(protocol.read_text(encoding="ascii"))
    assert result["selected_receptor"]["receptor_id"] == "R1"
    assert result["selected_receptor"]["artifact"]["receptor_pdbqt"][
        "sha256"
    ] == file_sha256(tmp_path / "R1.pdbqt")
    assert (
        result["data_audit"]["primary_matrix_locked_test_overlap_count"]
        == 0
    )
    assert (
        result["data_audit"]["sensitivity_matrix_locked_test_overlap_count"]
        == 0
    )
    assert result["test_lock"]["scores_evaluated"] is False
    assert result["test_lock"]["score_cells_read"] == 0
