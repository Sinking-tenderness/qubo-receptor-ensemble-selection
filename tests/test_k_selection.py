import json
from pathlib import Path

from qubo_receptor_ensemble.config import load_pipeline_config
from qubo_receptor_ensemble.io import file_sha256
from qubo_receptor_ensemble.k_selection import (
    KCandidate,
    KSelectionDecision,
    choose_k,
    register_k_selection_policy,
)
from qubo_receptor_ensemble.pipeline import PipelineRunner
from qubo_receptor_ensemble.solvers import SolverResult


def _candidate(k: int, metric: float) -> KCandidate:
    return KCandidate(
        k=k,
        result=SolverResult(
            backend="exact",
            strategy="qubo",
            subset=(f"R{k}",),
            objective=-metric,
            coefficients={},
            metadata={},
        ),
        metrics_by_split={"validation": {"roc_auc": metric}},
    )


def test_custom_k_selection_policy_can_be_registered() -> None:
    class LargestKPolicy:
        name = "largest_k_test_policy"

        def choose(
            self, candidates: list[KCandidate], config: dict[str, object]
        ) -> KSelectionDecision:
            selected = max(candidates, key=lambda candidate: candidate.k)
            return KSelectionDecision(
                policy=self.name,
                selected_k=selected.k,
                candidate_scores={str(candidate.k): float(candidate.k) for candidate in candidates},
                rationale="test policy selects the largest candidate k",
            )

    register_k_selection_policy(LargestKPolicy.name, LargestKPolicy())

    decision = choose_k(
        [_candidate(1, 0.7), _candidate(2, 0.8)],
        {"selector": LargestKPolicy.name, "selection_split": "validation"},
    )

    assert decision.selected_k == 2
    assert decision.policy == LargestKPolicy.name


def test_pipeline_scans_adaptive_k_candidates_and_persists_decision(tmp_path: Path) -> None:
    matrix = tmp_path / "scores.csv"
    matrix.write_text(
        "ligand_id,label,R1,R2\n"
        "A1,active,-10,-7\n"
        "D1,decoy,-5,-4\n"
        "A2,active,-9,-8\n"
        "D2,decoy,-4,-3\n"
        "A3,active,-8,-9\n"
        "D3,decoy,-3,-2\n",
        encoding="utf-8",
    )
    splits = tmp_path / "splits.csv"
    splits.write_text(
        "ligand_id,label,split\n"
        "A1,active,train\n"
        "D1,decoy,train\n"
        "A2,active,validation\n"
        "D2,decoy,validation\n"
        "A3,active,test\n"
        "D3,decoy,test\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "experiment_id": "adaptive-k-e2e",
                "target_id": "TEST",
                "purpose": "adaptive k pipeline test",
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
                        "sha256": file_sha256(splits),
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
                    "k_policy": {
                        "mode": "adaptive",
                        "candidates": [1, 2],
                        "selector": "best_metric",
                        "selection_split": "validation",
                        "selection_metric": "roc_auc",
                    },
                    "weights": {"redundancy": 0.0, "count": 0.0, "size": 1.0},
                },
                "solve": {"backend": "exact"},
                "evaluate": {"metrics": ["roc_auc"], "aggregation": "mean_score"},
                "outputs": {"run_directory": "results/adaptive-k-e2e"},
            },
            indent=2,
        ),
        encoding="ascii",
    )

    config = load_pipeline_config(config_path, root=tmp_path)
    manifest = PipelineRunner(config).run()

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
    assert manifest["status"] == "completed"
    assert solve["k_selection"]["selected_k"] == 1
    assert [item["k"] for item in solve["candidates"]] == [1, 2]
    assert evaluation["k_selection"]["selected_k"] == 1
    assert set(evaluation["candidates"]["1"]["splits"]) == {
        "train",
        "validation",
    }
    assert "test" not in evaluation["candidates"]["1"]["splits"]
