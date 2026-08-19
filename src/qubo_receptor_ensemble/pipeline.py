"""Canonical prepare/build/solve/evaluate/persist experiment pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .artifacts import artifact_record, write_artifact, write_stage_manifest
from .config import (
    PIPELINE_STAGES,
    ResolvedPipelineConfig,
    verify_input_artifacts,
)
from .io import read_csv, write_json
from .k_selection import KCandidate, KSelectionDecision, choose_k
from .screening import ranked_metrics_with_ids
from .solvers import Problem, SolverResult, build_problem, solve_problem


class PipelineNotReadyError(RuntimeError):
    """Backward-compatible name for the old unregistered-solver error."""


class PipelineStateError(RuntimeError):
    """Raised when a real run cannot reconstruct a requested stage boundary."""


@dataclass
class PipelineRunner:
    config: ResolvedPipelineConfig

    def _stage_directory(self, index: int, stage: str) -> Path:
        return self.config.run_directory / "stages" / f"{index:02d}_{stage}"

    def _write_stage(
        self,
        index: int,
        stage: str,
        payload: dict[str, object],
        input_records: dict[str, object],
        status: str,
    ) -> dict[str, object]:
        directory = self._stage_directory(index, stage)
        directory.mkdir(parents=True, exist_ok=True)
        artifact_path = directory / f"{stage}.json"
        output_record = write_artifact(artifact_path, payload)
        manifest_path = directory / "manifest.json"
        manifest = write_stage_manifest(
            manifest_path,
            stage=stage,
            status=status,
            inputs=input_records,
            outputs={stage: output_record},
        )
        return {
            "status": status,
            "manifest": artifact_record(manifest_path),
            "outputs": manifest["outputs"],
        }

    def _prepare_data(self) -> dict[str, object]:
        prepare = self.config.data["prepare"]
        problem_config = self.config.data["problem"]
        policy = self.config.data["data_policy"]
        assert isinstance(prepare, dict)
        assert isinstance(problem_config, dict)
        assert isinstance(policy, dict)

        source_name = str(prepare.get("source_input", "score_source"))
        if source_name not in self.config.inputs:
            if len(self.config.inputs) != 1:
                raise PipelineStateError(
                    "prepare.source_input must identify one declared input"
                )
            source_name = next(iter(self.config.inputs))
        rows = read_csv(self.config.inputs[source_name].path)
        if not rows:
            raise ValueError(f"score source contains no rows: {source_name}")

        split_name = prepare.get("split_input")
        split_by_ligand: dict[str, str] = {}
        if split_name is None:
            split_by_ligand = {str(row["ligand_id"]): "train" for row in rows}
        else:
            split_key = str(split_name)
            if split_key not in self.config.inputs:
                raise PipelineStateError(
                    f"prepare.split_input is not declared: {split_key}"
                )
            split_rows = read_csv(self.config.inputs[split_key].path)
            for row in split_rows:
                ligand_id = str(row.get("ligand_id", ""))
                split = str(row.get("split", ""))
                if not ligand_id or not split:
                    raise ValueError("split manifest requires ligand_id and split")
                if ligand_id in split_by_ligand:
                    raise ValueError(f"duplicate ligand_id in split manifest: {ligand_id}")
                split_by_ligand[ligand_id] = split
                if row.get("label") not in (None, ""):
                    source_row = next(
                        (item for item in rows if item.get("ligand_id") == ligand_id),
                        None,
                    )
                    if source_row is None or str(source_row.get("label")) != str(row["label"]):
                        raise ValueError(f"label differs between matrix and split manifest: {ligand_id}")
            source_ids = {str(row.get("ligand_id", "")) for row in rows}
            if source_ids != set(split_by_ligand):
                raise ValueError("score source and split manifest ligand IDs differ")

        selection_split = str(prepare.get("selection_split", "train"))
        train_rows = [
            {**row, "_split": split_by_ligand[str(row["ligand_id"])]}
            for row in rows
            if split_by_ligand[str(row["ligand_id"])] == selection_split
        ]
        if not train_rows:
            raise ValueError(f"no rows found for selection split: {selection_split}")

        split_rows: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            split = split_by_ligand[str(row["ligand_id"])]
            split_rows.setdefault(split, []).append({**row, "_split": split})
        return {
            "source_input": source_name,
            "selection_split": selection_split,
            "train_rows": train_rows,
            "split_rows": split_rows,
            "allowed_splits": [str(value) for value in policy["allowed_splits"]],
            "locked_splits": [str(value) for value in policy["locked_splits"]],
            "row_count": len(rows),
            "split_counts": {
                split: len(split_rows[split]) for split in sorted(split_rows)
            },
        }

    def _evaluate(
        self, data: dict[str, object], result: SolverResult
    ) -> dict[str, object]:
        evaluate = self.config.data["evaluate"]
        assert isinstance(evaluate, dict)
        split_rows = data["split_rows"]
        assert isinstance(split_rows, dict)
        aggregation = str(evaluate.get("aggregation", "mean_score"))
        if aggregation not in {"mean_score", "min_score"}:
            raise ValueError("evaluate.aggregation must be mean_score or min_score")
        allowed_splits = data["allowed_splits"]
        assert isinstance(allowed_splits, list)

        split_metrics: dict[str, object] = {}
        for split in allowed_splits:
            rows = split_rows.get(split, [])
            if not rows:
                continue
            ranking_data: dict[str, dict[str, object]] = {}
            for row in rows:
                scores = [float(row[receptor_id]) for receptor_id in result.subset]
                score = (
                    0.0
                    if not scores
                    else min(scores)
                    if aggregation == "min_score"
                    else sum(scores) / len(scores)
                )
                ranking_data[str(row["ligand_id"])] = {
                    "label": str(row["label"]),
                    aggregation: score,
                }
            metrics = ranked_metrics_with_ids(ranking_data, aggregation)
            requested = {
                str(metric): metrics[str(metric)]
                for metric in evaluate["metrics"]
                if str(metric) in metrics
            }
            split_metrics[split] = {
                "requested_metrics": requested,
                "all_metrics": metrics,
            }
        return {
            "subset": list(result.subset),
            "aggregation": aggregation,
            "splits": split_metrics,
            "excluded_splits": sorted(
                set(data["locked_splits"]) - set(split_metrics)
            ),
        }

    def _candidate_k_values(self) -> tuple[int, ...]:
        problem_config = self.config.data["problem"]
        assert isinstance(problem_config, dict)
        receptor_ids = problem_config["receptor_ids"]
        assert isinstance(receptor_ids, list)
        policy = problem_config.get("k_policy")
        if policy is None:
            values = [problem_config.get("target_size")]
        else:
            if not isinstance(policy, dict):
                raise PipelineStateError("problem.k_policy must be an object")
            mode = str(policy.get("mode", "adaptive"))
            if mode == "fixed":
                values = [policy.get("value", problem_config.get("target_size"))]
            elif mode == "adaptive":
                candidates = policy.get("candidates")
                if not isinstance(candidates, list) or not candidates:
                    raise PipelineStateError(
                        "adaptive problem.k_policy.candidates must be a non-empty list"
                    )
                values = candidates
            else:
                raise PipelineStateError(
                    "problem.k_policy.mode must be fixed or adaptive"
                )

        normalized: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise PipelineStateError("candidate k values must be integers")
            if not 0 <= value <= len(receptor_ids):
                raise PipelineStateError(
                    "candidate k values must be within the receptor pool"
                )
            if value not in normalized:
                normalized.append(value)
        if not normalized:
            raise PipelineStateError("at least one candidate k is required")
        return tuple(normalized)

    def _build_problems(self, data: dict[str, object]) -> list[Problem]:
        rows = data.get("train_rows")
        if not isinstance(rows, list):
            raise PipelineStateError("build_problem requires prepared training rows")
        problem_config = self.config.data["problem"]
        assert isinstance(problem_config, dict)
        problems: list[Problem] = []
        for k in self._candidate_k_values():
            candidate_config = dict(problem_config)
            candidate_config["target_size"] = k
            problems.append(build_problem(rows, candidate_config))
        return problems

    def _solve_and_select(
        self, data: dict[str, object], problems: list[Problem]
    ) -> dict[str, object]:
        solve_config = self.config.data["solve"]
        assert isinstance(solve_config, dict)
        backend = str(solve_config["backend"])
        results = [solve_problem(problem, backend) for problem in problems]
        evaluations = [self._evaluate(data, result) for result in results]

        problem_config = self.config.data["problem"]
        assert isinstance(problem_config, dict)
        policy = problem_config.get("k_policy")
        if policy is None or (isinstance(policy, dict) and policy.get("mode") == "fixed"):
            if len(results) != 1:
                raise PipelineStateError("fixed k selection must produce one candidate")
            selected_index = 0
            selected_k = int(problems[0].parameters["target_size"])
            decision = KSelectionDecision(
                policy="fixed",
                selected_k=selected_k,
                candidate_scores={},
                rationale="used the configured fixed target_size",
            )
        else:
            if not isinstance(policy, dict):
                raise PipelineStateError("problem.k_policy must be an object")
            candidates = [
                KCandidate(
                    k=int(problem.parameters["target_size"]),
                    result=result,
                    metrics_by_split=evaluation["splits"],
                )
                for problem, result, evaluation in zip(
                    problems, results, evaluations, strict=True
                )
            ]
            decision = choose_k(candidates, policy)
            selected_index = next(
                index
                for index, candidate in enumerate(candidates)
                if candidate.k == decision.selected_k
            )

        return {
            "problems": problems,
            "results": results,
            "evaluations": evaluations,
            "decision": decision,
            "selected_result": results[selected_index],
            "selected_evaluation": evaluations[selected_index],
        }

    def run(
        self,
        *,
        dry_run: bool = False,
        start_stage: str | None = None,
        end_stage: str | None = None,
    ) -> dict[str, object]:
        configured_stages = [str(stage) for stage in self.config.data["pipeline"]]
        start = start_stage or configured_stages[0]
        end = end_stage or configured_stages[-1]
        if start not in configured_stages or end not in configured_stages:
            raise ValueError("stage boundary is not present in the configured pipeline")
        start_index = configured_stages.index(start)
        end_index = configured_stages.index(end)
        if start_index > end_index:
            raise ValueError("start stage must not follow end stage")

        self.config.run_directory.mkdir(parents=True, exist_ok=True)
        input_records = verify_input_artifacts(self.config)
        config_record = artifact_record(self.config.path)
        snapshot_path = self.config.run_directory / "config.snapshot.json"
        write_json(snapshot_path, self.config.data)

        stage_records: dict[str, object] = {}
        selected = configured_stages[start_index : end_index + 1]
        context: dict[str, object] = {}
        if dry_run:
            for stage in selected:
                index = PIPELINE_STAGES.index(stage) + 1
                if stage == "prepare":
                    payload = {
                        "status": "planned",
                        "adapter": self.config.data["prepare"].get(
                            "adapter", "existing_matrix"
                        ),
                        "inputs": input_records,
                    }
                elif stage == "build_problem":
                    payload = {"status": "planned", "problem": self.config.data["problem"]}
                elif stage == "solve":
                    payload = {
                        "status": "planned",
                        "backend": self.config.data["solve"]["backend"],
                        "strategy": self.config.data["problem"]["strategy"],
                    }
                elif stage == "evaluate":
                    payload = {
                        "status": "planned",
                        "evaluation": self.config.data["evaluate"],
                        "data_policy": self.config.data["data_policy"],
                    }
                else:
                    payload = {
                        "status": "planned",
                        "note": "Pipeline persistence is represented by the run manifest.",
                    }
                stage_records[stage] = self._write_stage(
                    index, stage, payload, input_records, "planned"
                )
        else:
            if start_index > 0:
                context["data"] = self._prepare_data()
            if start_index > 1:
                data = context["data"]
                assert isinstance(data, dict)
                context["problems"] = self._build_problems(data)
            if start_index > 2:
                data = context["data"]
                problems = context["problems"]
                assert isinstance(data, dict)
                assert isinstance(problems, list)
                context["selection"] = self._solve_and_select(data, problems)
                selection = context["selection"]
                assert isinstance(selection, dict)
                context["result"] = selection["selected_result"]
                context["evaluation"] = selection["selected_evaluation"]
                context["k_selection"] = selection["decision"]
            if start_index > 3:
                context["evaluation"] = context["evaluation"]
            for stage in selected:
                index = PIPELINE_STAGES.index(stage) + 1
                if stage == "prepare":
                    context["data"] = self._prepare_data()
                    data = context["data"]
                    assert isinstance(data, dict)
                    payload = {
                        "status": "completed",
                        "adapter": self.config.data["prepare"].get(
                            "adapter", "existing_matrix"
                        ),
                        **{
                            key: data[key]
                            for key in (
                                "source_input",
                                "selection_split",
                                "row_count",
                                "split_counts",
                                "allowed_splits",
                                "locked_splits",
                            )
                        },
                    }
                elif stage == "build_problem":
                    data = context.get("data")
                    if not isinstance(data, dict):
                        raise PipelineStateError("build_problem requires prepare data")
                    problems = self._build_problems(data)
                    context["problems"] = problems
                    payload = {
                        "status": "completed",
                        "problem": problems[0].as_dict() if len(problems) == 1 else None,
                        "candidates": [problem.as_dict() for problem in problems],
                    }
                elif stage == "solve":
                    data = context.get("data")
                    problems = context.get("problems")
                    if not isinstance(data, dict) or not isinstance(problems, list):
                        raise PipelineStateError("solve requires a built problem")
                    selection = self._solve_and_select(data, problems)
                    context["selection"] = selection
                    context["result"] = selection["selected_result"]
                    context["evaluation"] = selection["selected_evaluation"]
                    context["k_selection"] = selection["decision"]
                    results = selection["results"]
                    decision = selection["decision"]
                    assert isinstance(results, list)
                    assert isinstance(decision, KSelectionDecision)
                    selected_result = selection["selected_result"]
                    assert isinstance(selected_result, SolverResult)
                    candidate_results = [
                        {
                            "k": int(problem.parameters["target_size"]),
                            **result.as_dict(),
                        }
                        for problem, result in zip(
                            problems, results, strict=True
                        )
                    ]
                    payload = {
                        "status": "completed",
                        "result": selected_result.as_dict(),
                        "selected": selected_result.as_dict(),
                        "candidates": candidate_results,
                        "k_selection": decision.as_dict(),
                    }
                elif stage == "evaluate":
                    selection = context.get("selection")
                    if not isinstance(selection, dict):
                        raise PipelineStateError("evaluate requires solved k candidates")
                    evaluation = selection["selected_evaluation"]
                    decision = selection["decision"]
                    evaluations = selection["evaluations"]
                    problems = selection["problems"]
                    assert isinstance(evaluation, dict)
                    assert isinstance(decision, KSelectionDecision)
                    assert isinstance(evaluations, list)
                    assert isinstance(problems, list)
                    candidate_evaluations = {
                        str(problem.parameters["target_size"]): candidate_evaluation
                        for problem, candidate_evaluation in zip(
                            problems, evaluations, strict=True
                        )
                    }
                    context["evaluation"] = evaluation
                    payload = {
                        "status": "completed",
                        **evaluation,
                        "candidates": candidate_evaluations,
                        "k_selection": decision.as_dict(),
                    }
                else:
                    persisted_result = context.get("result")
                    if isinstance(persisted_result, SolverResult):
                        persisted_result = persisted_result.as_dict()
                    decision = context.get("k_selection")
                    if isinstance(decision, KSelectionDecision):
                        decision = decision.as_dict()
                    payload = {
                        "status": "completed",
                        "result": persisted_result or {},
                        "evaluation": context.get("evaluation", {}),
                        "k_selection": decision or {},
                    }
                stage_records[stage] = self._write_stage(
                    index, stage, payload, input_records, "completed"
                )

        status = "planned" if dry_run else "completed"
        summary = {
            "schema_version": "1.0",
            "experiment_id": self.config.data["experiment_id"],
            "target_id": self.config.data["target_id"],
            "status": status,
            "config": config_record,
            "config_snapshot": artifact_record(snapshot_path),
            "inputs": input_records,
            "stages": stage_records,
            "interpretation_boundary": self.config.data.get(
                "interpretation_boundary", "Pipeline result; review stage-specific boundaries."
            ),
        }
        if not dry_run:
            if "result" in context:
                result = context["result"]
                assert isinstance(result, SolverResult)
                summary["result"] = result.as_dict()
            if "evaluation" in context:
                summary["evaluation"] = context["evaluation"]
            if "k_selection" in context:
                decision = context["k_selection"]
                if isinstance(decision, KSelectionDecision):
                    summary["k_selection"] = decision.as_dict()
        if "persist" in selected:
            summary_path = (
                self.config.run_directory / "stages" / "05_persist" / "summary.json"
            )
        else:
            summary_path = self.config.run_directory / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(summary_path, summary)
        manifest = {
            **summary,
            "summary": artifact_record(summary_path),
        }
        manifest_path = self.config.run_directory / "manifest.json"
        write_json(manifest_path, manifest)
        return manifest


def format_summary(summary: dict[str, object]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True)
