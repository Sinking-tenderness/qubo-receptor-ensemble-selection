import csv
import json
from pathlib import Path

import pytest

from qubo_receptor_ensemble.experiment import (
    aggregate_score_tables,
    build_problem_stage,
    evaluate_stage,
    solve_stage,
    validate_front_inputs,
)
from qubo_receptor_ensemble.full_workflow import (
    FULL_WORKFLOW_STAGES,
    ConfigError,
    FullExperimentConfig,
    load_full_experiment_config,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_aggregate_score_tables_builds_primary_and_sensitivity_matrices(
    tmp_path: Path,
) -> None:
    ligand_manifest = _write_csv(
        tmp_path / "ligands.csv",
        [
            {
                "target_id": "TEST",
                "ligand_id": "A1",
                "label": "active",
                "selection_role": "development",
                "split": "train",
                "pdbqt_path": "A1.pdbqt",
            },
            {
                "target_id": "TEST",
                "ligand_id": "D1",
                "label": "decoy",
                "selection_role": "development",
                "split": "train",
                "pdbqt_path": "D1.pdbqt",
            },
        ],
    )
    scores = tmp_path / "scores"
    for seed, offset in ((11, 0.0), (12, 0.2)):
        _write_csv(
            scores / f"seed_{seed}.csv",
            [
                {
                    "target_id": "TEST",
                    "receptor_id": receptor,
                    "ligand_id": ligand,
                    "label": label,
                    "pose_rank": 1,
                    "docking_score": -8.0 + offset + index,
                    "status": "ok",
                    "seed": seed,
                }
                for index, (receptor, ligand, label) in enumerate(
                    (("R1", "A1", "active"), ("R1", "D1", "decoy"),
                     ("R2", "A1", "active"), ("R2", "D1", "decoy"))
                )
            ],
        )

    result = aggregate_score_tables(
        score_directory=scores,
        ligand_manifest=ligand_manifest,
        receptor_count=2,
        seed_count=2,
        matrices_directory=tmp_path / "matrices",
    )

    assert result["summary"]["seed_count"] == 2
    assert result["summary"]["receptor_count"] == 2
    assert result["primary_matrix"].is_file()
    assert result["sensitivity_matrix"].is_file()


def test_aggregate_start_requires_the_configured_score_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """{
          "schema_version": "3.0",
          "experiment_id": "partial",
          "target_id": "TEST",
          "workflow_mode": "full",
          "start_stage": "aggregate",
          "selection": {"receptor_count": 1, "ligand_count": 2, "label_counts": {"active": 1, "decoy": 1}},
          "sources": {"active_ism": "active.ism", "decoy_ism": "decoy.ism", "receptor_manifest": "receptors.csv"},
          "docking": {"redock": true, "engine": "unidock", "seeds": [11, 12]},
          "problem": {"type": "receptor_subset", "strategy": "qubo", "target_size": 1},
          "solve": {"backend": "exact"},
          "evaluate": {"metrics": ["roc_auc"]},
          "paths": {
            "run_directory": "run",
            "prepared_ligand_manifest": "run/ligands.csv",
            "selected_receptor_manifest": "run/receptors.csv",
            "score_tables": "missing-scores",
            "matrices": "run/matrices",
            "primary_matrix": "run/matrices/primary.csv",
            "problem": "run/problem.json",
            "selection": "run/selection.json",
            "evaluation": "run/evaluation.json"
          }
        }""",
        encoding="utf-8",
    )
    config = load_full_experiment_config(config_path, data_root=tmp_path)

    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "ligands.csv").write_text("placeholder", encoding="ascii")
    (tmp_path / "run" / "receptors.csv").write_text("placeholder", encoding="ascii")

    with pytest.raises(FileNotFoundError, match="missing-scores"):
        validate_front_inputs(config)


def test_compare_problem_runs_share_matrix_and_write_method_outputs(tmp_path: Path) -> None:
    matrix = _write_csv(
        tmp_path / "matrices" / "primary.csv",
        [
            {"ligand_id": "A1", "label": "active", "R1": -10.0, "R2": -9.0},
            {"ligand_id": "A2", "label": "active", "R1": -9.0, "R2": -8.0},
            {"ligand_id": "D1", "label": "decoy", "R1": -5.0, "R2": -4.0},
            {"ligand_id": "D2", "label": "decoy", "R1": -4.0, "R2": -3.0},
        ],
    )
    receptor_manifest = _write_csv(
        tmp_path / "receptors.csv",
        [
            {"conformer_id": "R1", "receptor_pdbqt": "R1.pdbqt"},
            {"conformer_id": "R2", "receptor_pdbqt": "R2.pdbqt"},
        ],
    )
    run_directory = tmp_path / "run"
    paths = {
        "run_directory": run_directory,
        "selected_receptor_manifest": receptor_manifest,
        "primary_matrix": matrix,
        "problem": run_directory / "problem.json",
        "selection": run_directory / "selection.json",
        "evaluation": run_directory / "evaluation.json",
    }
    config = FullExperimentConfig(
        path=tmp_path / "config.json",
        data_root=tmp_path,
        data={
            "problem": {
                "type": "receptor_subset",
                "mode": "compare",
                "target_size": 1,
                "utility_metric": "bedroc",
                "bedroc_alpha": 20.0,
                "methods": [
                    {"id": "basic_utility"},
                    {"id": "bedroc20_pair_synergy"},
                    {"id": "structure_aware"},
                ],
            },
            "solve": {"backend": "exact"},
            "evaluate": {
                "primary_metric": "bedroc_alpha_20",
                "metrics": ["bedroc_alpha_20", "roc_auc"],
            },
        },
        paths=paths,
        stages=FULL_WORKFLOW_STAGES,
        start_stage="build_problem",
        end_stage="persist",
    )

    built = build_problem_stage(config, matrix)
    assert built["comparison"] is True
    assert built["ready_count"] == 2

    solved = solve_stage(config, paths["problem"])
    assert solved["selection_path"] == paths["selection"]
    selection = json.loads(paths["selection"].read_text(encoding="ascii"))
    assert [record["status"] for record in selection["methods"]] == [
        "ready",
        "ready",
        "unsupported_for_input",
    ]

    evaluated = evaluate_stage(config, paths["selection"])
    assert evaluated["evaluation_path"] == paths["evaluation"]
    assert (run_directory / "comparison.json").is_file()
    assert (run_directory / "methods" / "basic_utility" / "evaluation.json").is_file()
    comparison = json.loads((run_directory / "comparison.json").read_text(encoding="ascii"))
    assert comparison["primary_metric"] == "bedroc_alpha_20"
    ready = [record for record in comparison["methods"] if record["status"] == "ready"]
    assert all("primary_metric_value" in record for record in ready)
    assert all(record["primary_metric_value"] is not None for record in ready)
    assert "roc_auc" in json.loads(
        (run_directory / "methods" / "basic_utility" / "evaluation.json").read_text(
            encoding="ascii"
        )
    )["metrics"]
