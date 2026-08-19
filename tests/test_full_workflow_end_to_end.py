import csv
import json
from pathlib import Path

from qubo_receptor_ensemble.experiment import FullExperimentRunner
from qubo_receptor_ensemble.full_workflow import load_full_experiment_config


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_partial_run_from_aggregate_reaches_persist_without_reference_matrix(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    ligands = [
        {
            "target_id": "TEST",
            "ligand_id": "A1",
            "label": "active",
            "selection_role": "development",
            "split": "train",
            "pdbqt_path": "run/A1.pdbqt",
        },
        {
            "target_id": "TEST",
            "ligand_id": "D1",
            "label": "decoy",
            "selection_role": "development",
            "split": "train",
            "pdbqt_path": "run/D1.pdbqt",
        },
    ]
    _write_csv(run / "prepared_ligands.csv", ligands)
    (run / "A1.pdbqt").write_text("placeholder", encoding="ascii")
    (run / "D1.pdbqt").write_text("placeholder", encoding="ascii")
    _write_csv(
        run / "selected_receptors.csv",
        [
            {"conformer_id": "R1", "receptor_pdbqt": "R1.pdbqt"},
            {"conformer_id": "R2", "receptor_pdbqt": "R2.pdbqt"},
        ],
    )
    scores = run / "score_tables"
    for seed, shift in ((11, 0.0), (12, 0.2)):
        _write_csv(
            scores / f"seed_{seed}.csv",
            [
                {
                    "target_id": "TEST",
                    "receptor_id": receptor,
                    "ligand_id": ligand,
                    "label": label,
                    "pose_rank": 1,
                    "docking_score": score + shift,
                    "status": "ok",
                    "seed": seed,
                }
                for receptor, ligand, label, score in (
                    ("R1", "A1", "active", -9.0),
                    ("R1", "D1", "decoy", -5.0),
                    ("R2", "A1", "active", -8.0),
                    ("R2", "D1", "decoy", -4.0),
                )
            ],
        )

    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "3.0",
                "experiment_id": "test-partial",
                "target_id": "TEST",
                "workflow_mode": "full",
                "start_stage": "aggregate",
                "selection": {
                    "receptor_count": 2,
                    "ligand_count": 2,
                    "label_counts": {"active": 1, "decoy": 1},
                },
                "sources": {
                    "active_ism": "active.ism",
                    "decoy_ism": "decoy.ism",
                    "receptor_manifest": "source_receptors.csv",
                },
                "docking": {"redock": True, "engine": "unidock", "seeds": [11, 12]},
                "problem": {
                    "type": "receptor_subset",
                    "strategy": "qubo",
                    "target_size": 1,
                    "weights": {"redundancy": 0.25, "count": 0.1, "size": 10.0},
                },
                "solve": {"backend": "exact"},
                "evaluate": {"aggregation": "mean_score", "metrics": ["roc_auc"]},
                "paths": {
                    "run_directory": "run",
                    "prepared_ligand_manifest": "run/prepared_ligands.csv",
                    "selected_receptor_manifest": "run/selected_receptors.csv",
                    "score_tables": "run/score_tables",
                    "matrices": "run/matrices",
                    "primary_matrix": "run/matrices/primary_median_matrix.csv",
                    "problem": "run/problem.json",
                    "selection": "run/selection.json",
                    "evaluation": "run/evaluation.json",
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_full_experiment_config(config_path, data_root=tmp_path)

    summary = FullExperimentRunner(config).run(end_stage="persist")

    assert summary["status"] == "completed"
    assert summary["stages"]["aggregate"]["status"] == "completed"
    assert (run / "selection.json").is_file()
    assert (run / "evaluation.json").is_file()
    assert (run / "manifest.json").is_file()
