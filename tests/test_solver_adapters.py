import json
from pathlib import Path

from qubo_receptor_ensemble.config import load_pipeline_config
from qubo_receptor_ensemble.io import file_sha256
from qubo_receptor_ensemble.pipeline import PipelineRunner
from qubo_receptor_ensemble.solvers import build_problem, solve_problem


def _rows() -> list[dict[str, object]]:
    return [
        {"ligand_id": "A1", "label": "active", "R1": -10.0, "R2": -7.0},
        {"ligand_id": "A2", "label": "active", "R1": -9.0, "R2": -8.0},
        {"ligand_id": "D1", "label": "decoy", "R1": -5.0, "R2": -4.0},
        {"ligand_id": "D2", "label": "decoy", "R1": -4.0, "R2": -3.0},
    ]


def test_exact_adapter_returns_standard_solver_result() -> None:
    problem = build_problem(
        _rows(),
        {
            "type": "receptor_subset",
            "strategy": "qubo",
            "receptor_ids": ["R1", "R2"],
            "target_size": 1,
            "utility_metric": "roc_auc",
            "utility_normalization": "minmax",
            "weights": {"redundancy": 0.0, "count": 0.0, "size": 1.0},
        },
    )

    result = solve_problem(problem, "exact")

    assert result.backend == "exact"
    assert result.subset == ("R1",)
    assert result.objective == result.metadata["objective"]
    assert result.metadata["states_evaluated"] == 4
    assert "linear_coefficients" in result.coefficients


def test_normalized_exact_adapter_wraps_existing_algorithm() -> None:
    problem = build_problem(
        _rows(),
        {
            "type": "receptor_subset",
            "strategy": "normalized_qubo",
            "receptor_ids": ["R1", "R2"],
            "target_size": 1,
            "coverage_fraction": 0.5,
            "utility_metric": "bedroc",
            "size_penalty": 10.0,
            "weights": {
                "active_coverage": 1.0,
                "decoy_exposure": 1.0,
                "active_overlap": 0.0,
                "redundancy": 0.0,
            },
        },
    )

    result = solve_problem(problem, "exact")

    assert result.backend == "exact"
    assert len(result.subset) == 1
    assert result.metadata["states_evaluated"] == 2
    assert result.coefficients["target_size"] == 1


def test_greedy_adapter_returns_the_same_contract() -> None:
    problem = build_problem(
        _rows(),
        {
            "type": "receptor_subset",
            "strategy": "qubo",
            "receptor_ids": ["R1", "R2"],
            "target_size": 1,
            "utility_metric": "roc_auc",
            "utility_normalization": "minmax",
            "weights": {"redundancy": 0.0, "count": 0.0, "size": 1.0},
        },
    )

    result = solve_problem(problem, "greedy")

    assert result.backend == "greedy"
    assert result.subset == ("R1",)
    assert result.metadata["states_evaluated"] == 2


def test_pipeline_real_run_evaluates_allowed_splits_only(tmp_path: Path) -> None:
    matrix = tmp_path / "scores.csv"
    matrix.write_text(
        "ligand_id,label,R1,R2\n"
        "A1,active,-10,-7\n"
        "A2,active,-9,-8\n"
        "D1,decoy,-5,-4\n"
        "D2,decoy,-4,-3\n",
        encoding="utf-8",
    )
    split_manifest = tmp_path / "splits.csv"
    split_manifest.write_text(
        "ligand_id,label,split\n"
        "A1,active,train\n"
        "D1,decoy,train\n"
        "A2,active,validation\n"
        "D2,decoy,test\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "experiment_id": "adapter-e2e",
                "target_id": "TEST",
                "purpose": "adapter test",
                "pipeline": [
                    "prepare",
                    "build_problem",
                    "solve",
                    "evaluate",
                    "persist",
                ],
                "inputs": {
                    "score_source": {
                        "path": "scores.csv",
                        "sha256": file_sha256(matrix),
                    },
                    "split_manifest": {
                        "path": "splits.csv",
                        "sha256": file_sha256(split_manifest),
                    },
                },
                "data_policy": {
                    "allowed_splits": ["train", "validation"],
                    "locked_splits": ["test"],
                    "evaluate_locked_test": False,
                },
                "prepare": {
                    "adapter": "existing_matrix",
                    "source_input": "score_source",
                    "split_input": "split_manifest",
                },
                "problem": {
                    "type": "receptor_subset",
                    "strategy": "qubo",
                    "receptor_ids": ["R1", "R2"],
                    "target_size": 1,
                    "utility_metric": "roc_auc",
                    "utility_normalization": "minmax",
                    "weights": {
                        "redundancy": 0.0,
                        "count": 0.0,
                        "size": 1.0,
                    },
                },
                "solve": {"backend": "exact"},
                "evaluate": {"metrics": ["roc_auc"], "aggregation": "mean_score"},
                "outputs": {"run_directory": "results/adapter-e2e"},
            },
            indent=2,
        ),
        encoding="ascii",
    )

    config = load_pipeline_config(config_path, root=tmp_path)
    manifest = PipelineRunner(config).run()

    assert manifest["status"] == "completed"
    solve = json.loads(
        (config.run_directory / "stages" / "03_solve" / "solve.json").read_text(
            encoding="ascii"
        )
    )
    evaluation = json.loads(
        (config.run_directory / "stages" / "04_evaluate" / "evaluate.json").read_text(
            encoding="ascii"
        )
    )
    assert solve["result"]["subset"] == ["R1"]
    assert set(evaluation["splits"]) == {"train", "validation"}
    assert "test" not in evaluation["splits"]


def test_pipeline_real_partial_run_reconstructs_prior_stages(tmp_path: Path) -> None:
    matrix = tmp_path / "scores.csv"
    matrix.write_text(
        "ligand_id,label,R1,R2\n"
        "A1,active,-10,-7\n"
        "D1,decoy,-5,-4\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "experiment_id": "partial-run",
                "target_id": "TEST",
                "purpose": "partial run test",
                "pipeline": [
                    "prepare",
                    "build_problem",
                    "solve",
                    "evaluate",
                    "persist",
                ],
                "inputs": {
                    "score_source": {
                        "path": "scores.csv",
                        "sha256": file_sha256(matrix),
                    }
                },
                "data_policy": {
                    "allowed_splits": ["train"],
                    "locked_splits": ["test"],
                    "evaluate_locked_test": False,
                },
                "prepare": {"adapter": "existing_matrix"},
                "problem": {
                    "type": "receptor_subset",
                    "strategy": "qubo",
                    "receptor_ids": ["R1", "R2"],
                    "target_size": 1,
                    "weights": {"redundancy": 0.0, "count": 0.0, "size": 1.0},
                },
                "solve": {"backend": "exact"},
                "evaluate": {"metrics": ["roc_auc"]},
                "outputs": {"run_directory": "results/partial-run"},
            },
            indent=2,
        ),
        encoding="ascii",
    )

    config = load_pipeline_config(config_path, root=tmp_path)
    manifest = PipelineRunner(config).run(start_stage="solve", end_stage="solve")

    assert manifest["status"] == "completed"
    assert manifest["stages"]["solve"]["status"] == "completed"
    assert "persist" not in manifest["stages"]
