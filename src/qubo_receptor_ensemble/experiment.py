"""Execution engine for the configurable source-to-result experiment."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .docking_adapters import get_docking_adapter
from .full_workflow import (
    FULL_WORKFLOW_STAGES,
    ConfigError,
    FullExperimentConfig,
    front_input_keys,
    select_ism_ligands,
    select_preselected_ligands,
    select_receptor_manifest,
)
from .ligand_selection import (
    select_scaffold_hash_ligands,
    summarize_scaffold_hash_allocation,
)
from .io import file_sha256, read_csv, write_csv, write_json
from .k_selection import KCandidate, KSelectionDecision, choose_k
from .matrix import aggregate_seed_rows, build_matrix, read_score_tables, select_representative_scores
from .raw_preparation import calculate_ligand_box, prepare_raw_receptors
from .screening import ranked_metrics_with_ids
from .solvers import Problem, SolverResult, build_problem, solve_problem


def _as_rooted(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty output: {path}")
    write_csv(path, rows)


def _read_ligand_manifest(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    required = {"ligand_id", "label", "selection_role", "pdbqt_path"}
    if not rows or not required.issubset(rows[0]):
        raise ConfigError(f"prepared ligand manifest requires: {sorted(required)}")
    if len({row["ligand_id"] for row in rows}) != len(rows):
        raise ConfigError("prepared ligand manifest contains duplicate ligand IDs")
    return rows


def _read_receptor_manifest(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    required = {"conformer_id", "receptor_pdbqt"}
    if not rows or not required.issubset(rows[0]):
        raise ConfigError(f"selected receptor manifest requires: {sorted(required)}")
    if len({row["conformer_id"] for row in rows}) != len(rows):
        raise ConfigError("selected receptor manifest contains duplicate receptor IDs")
    return rows


def _prepare_one_ligand(
    row: dict[str, str],
    *,
    index: int,
    root: Path,
    sdf_directory: Path,
    pdbqt_directory: Path,
    meeko_script: Path,
    seed: int,
) -> dict[str, object]:
    from .preparation import build_3d_mol, parse_pdbqt, run_meeko

    ligand_id = row["ligand_id"]
    sdf_path = sdf_directory / f"{ligand_id}.sdf"
    pdbqt_path = pdbqt_directory / f"{ligand_id}.pdbqt"
    molecule, status, message = build_3d_mol(row["smiles"], seed + index)
    if molecule is None:
        raise RuntimeError(f"3D preparation failed for {ligand_id}: {message}")
    molecule.SetProp("_Name", ligand_id)
    molecule.SetProp("ligand_id", ligand_id)
    molecule.SetProp("label", row["label"])
    molecule.SetProp("target_id", row["target_id"])
    from rdkit import Chem

    sdf_directory.mkdir(parents=True, exist_ok=True)
    pdbqt_directory.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(molecule)
    writer.close()
    completed = run_meeko(meeko_script, sdf_path, pdbqt_path)
    details = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if completed.returncode != 0 or not pdbqt_path.is_file():
        raise RuntimeError(f"Meeko failed for {ligand_id}: {details[-500:]}")
    return {
        **row,
        "prep_status": status,
        "prep_message": message,
        "sdf_path": _relative(sdf_path, root),
        "pdbqt_path": _relative(pdbqt_path, root),
        "pdbqt_status": "ok",
        "pdbqt_message": "meeko_ok",
        **parse_pdbqt(pdbqt_path),
    }


def _raw_receptor_sources(paths: dict[str, Path]) -> bool:
    return all(
        key in paths
        for key in ("reference_receptor_pdb", "crystal_ligand", "rcsb_directory")
    )


def _configure_generated_box(config: FullExperimentConfig) -> tuple[Path, dict[str, float]]:
    docking = config.data["docking"]
    assert isinstance(docking, dict)
    policy = docking.get("box", {})
    if not isinstance(policy, dict):
        raise ConfigError("docking.box must be an object")
    if policy.get("method") != "ligand_bounds":
        raise ConfigError(
            "raw preparation requires docking.box.method=ligand_bounds; "
            "a fixed box is not a valid full-workflow input"
        )
    try:
        padding = float(policy["padding"])
        minimum_size = tuple(float(value) for value in policy["minimum_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(
            "docking.box requires numeric padding and a three-value minimum_size"
        ) from exc
    box = calculate_ligand_box(
        config.paths["crystal_ligand"],
        padding=padding,
        minimum_size=minimum_size,
    )
    box_path = config.paths.get(
        "docking_box", config.paths["run_directory"] / "docking_box.json"
    )
    artifact = {
        "status": "ok",
        "method": "ligand_bounds",
        "source_path": str(config.paths["crystal_ligand"]),
        "source_sha256": file_sha256(config.paths["crystal_ligand"]),
        "padding": padding,
        "minimum_size": list(minimum_size),
        "box": box,
    }
    write_json(box_path, artifact)
    docking["box"] = {**policy, **box, "artifact_path": _relative(box_path, config.data_root)}
    return box_path, box


def _prepare_raw_receptor_manifest(
    config: FullExperimentConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    receptor_config = config.data.get("receptor_preparation", {})
    if not isinstance(receptor_config, dict):
        raise ConfigError("receptor_preparation must be an object")
    candidate_ids = receptor_config.get("candidate_ids")
    if candidate_ids is not None and (
        not isinstance(candidate_ids, list) or any(not isinstance(value, str) for value in candidate_ids)
    ):
        raise ConfigError("receptor_preparation.candidate_ids must be a list of strings")
    prepared = prepare_raw_receptors(
        reference_pdb=config.paths["reference_receptor_pdb"],
        rcsb_directory=config.paths["rcsb_directory"],
        output_directory=config.paths["run_directory"],
        receptor_count=int(config.data["selection"]["receptor_count"]),
        reference_chain=str(receptor_config.get("reference_chain", "auto")),
        mobile_chain=str(receptor_config.get("mobile_chain", "auto")),
        minimum_alignment_ca_count=int(
            receptor_config.get("minimum_alignment_ca_count", 50)
        ),
        allow_bad_res=bool(receptor_config.get("allow_bad_res", False)),
        candidate_ids=candidate_ids,
    )
    rows: list[dict[str, object]] = []
    for index, record in enumerate(prepared["selected"]):
        assert isinstance(record, dict)
        alignment = record.get("alignment", {})
        if not isinstance(alignment, dict):
            alignment = {}
        rows.append(
            {
                "conformer_id": str(record["conformer_id"]),
                "target_id": str(config.data["target_id"]),
                "selected_index": index,
                "status": "ok",
                "stage102a_gate_pass": "True",
                "rcsb_id": str(record["rcsb_id"]),
                "source_structure": _relative(Path(str(record["source_structure"])), config.data_root),
                "source_sha256": str(record["source_sha256"]),
                "source_pdb": _relative(Path(str(record["source_pdb"])), config.data_root),
                "receptor_pdb": _relative(Path(str(record["receptor_pdb"])), config.data_root),
                "receptor_pdbqt": _relative(Path(str(record["receptor_pdbqt"])), config.data_root),
                "reference_chain": str(alignment.get("reference_chain", "")),
                "mobile_chain": str(alignment.get("mobile_chain", "")),
                "matched_ca_count": alignment.get("matched_ca_count", ""),
                "rmsd_after_angstrom": alignment.get("rmsd_after_angstrom", ""),
            }
        )
    audit_path = config.paths["run_directory"] / "receptor_preparation_audit.json"
    write_json(audit_path, prepared)
    return rows, {"audit_path": audit_path, "candidate_count": prepared["candidate_count"]}


def prepare_experiment_inputs(
    config: FullExperimentConfig, *, resume: bool = False, overwrite: bool = False
) -> dict[str, object]:
    """Prepare all inputs from raw sources for the exact current experiment run."""
    paths = config.paths
    ligand_manifest_path = paths["prepared_ligand_manifest"]
    receptor_manifest_path = paths["selected_receptor_manifest"]
    box_path = paths.get("docking_box", paths["run_directory"] / "docking_box.json")
    if resume and ligand_manifest_path.is_file() and receptor_manifest_path.is_file() and box_path.is_file():
        ligands = _read_ligand_manifest(ligand_manifest_path)
        receptors = _read_receptor_manifest(receptor_manifest_path)
        if (
            all(_as_rooted(Path(row["pdbqt_path"]), config.data_root).is_file() for row in ligands)
            and all(
                _as_rooted(Path(row["receptor_pdbqt"]), config.data_root).is_file()
                for row in receptors
            )
        ):
            box_artifact = json.loads(box_path.read_text(encoding="ascii"))
            generated_box = box_artifact.get("box")
            if not isinstance(generated_box, dict):
                raise ConfigError(f"generated docking box is invalid: {box_path}")
            docking = config.data["docking"]
            assert isinstance(docking, dict)
            docking["box"] = {**docking.get("box", {}), **generated_box}
            return {
                "ligands": ligands,
                "receptors": receptors,
                "source_ligand_manifest": paths["run_directory"] / "source_ligands.csv",
                "receptor_audit": {
                    "audit_path": paths["run_directory"] / "receptor_preparation_audit.json"
                },
                "docking_box": box_path,
                "resumed": True,
            }
    if not overwrite:
        existing = [
            path
            for path in (ligand_manifest_path, receptor_manifest_path, box_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(f"prepare outputs exist; use --resume: {existing[0]}")

    selection = config.data["selection"]
    assert isinstance(selection, dict)
    ordering = str(selection.get("ordering", "manifest_order"))
    label_counts = {
        str(key): int(value) for key, value in selection["label_counts"].items()
    }
    if ordering == "preselected_manifest":
        selected_ligands = select_preselected_ligands(
            paths["ligand_manifest"],
            target_id=str(config.data["target_id"]),
            label_counts=label_counts,
            ligand_count=int(selection["ligand_count"]),
        )
    elif ordering == "scaffold_hash_allocation":
        allocation_policy = selection.get("allocation", {})
        if not isinstance(allocation_policy, dict):
            raise ConfigError("selection.allocation must be an object")
        selected_ligands = select_scaffold_hash_ligands(
            paths["active_ism"],
            paths["decoy_ism"],
            target_id=str(config.data["target_id"]),
            label_counts=label_counts,
            policy=allocation_policy,
        )
    else:
        selected_ligands = select_ism_ligands(
            paths["active_ism"],
            paths["decoy_ism"],
            target_id=str(config.data["target_id"]),
            label_counts=label_counts,
            ordering=ordering,
            sample_seed=int(selection.get("sample_seed", 0)),
        )
    if len(selected_ligands) != int(selection["ligand_count"]):
        raise ConfigError("selected ligand count differs from configuration")
    source_ligand_manifest = paths["run_directory"] / "source_ligands.csv"
    _write_rows(source_ligand_manifest, selected_ligands)
    allocation_summary_path: Path | None = None
    if ordering == "scaffold_hash_allocation":
        allocation_policy = selection.get("allocation", {})
        assert isinstance(allocation_policy, dict)
        allocation_summary_path = (
            paths["run_directory"] / "source_ligand_allocation_summary.json"
        )
        write_json(
            allocation_summary_path,
            summarize_scaffold_hash_allocation(
                selected_ligands,
                policy=allocation_policy,
            ),
        )
    receptor_audit: dict[str, object] = {}
    if _raw_receptor_sources(paths):
        receptor_rows, receptor_audit = _prepare_raw_receptor_manifest(config)
    else:
        receptor_rows = select_receptor_manifest(
            paths["receptor_manifest"],
            receptor_count=int(selection["receptor_count"]),
        )
        receptor_rows = [
            {**row, "selected_index": index, "target_id": str(config.data["target_id"])}
            for index, row in enumerate(receptor_rows)
        ]
    prepared_root = ligand_manifest_path.parent
    sdf_directory = prepared_root / "ligands_sdf"
    pdbqt_directory = prepared_root / "ligands_pdbqt"
    preparation = config.data.get("preparation", {})
    if not isinstance(preparation, dict):
        raise ConfigError("preparation must be an object")
    from .preparation import find_meeko_script

    meeko_script = find_meeko_script()
    worker_count = max(1, int(preparation.get("workers", 4)))
    base_seed = int(preparation.get("base_seed", 20260818))
    prepared_by_id: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _prepare_one_ligand,
                row,
                index=index,
                root=config.data_root,
                sdf_directory=sdf_directory,
                pdbqt_directory=pdbqt_directory,
                meeko_script=meeko_script,
                seed=base_seed,
            ): row["ligand_id"]
            for index, row in enumerate(selected_ligands)
        }
        for future in as_completed(futures):
            prepared_by_id[futures[future]] = future.result()
    prepared_ligands = [prepared_by_id[row["ligand_id"]] for row in selected_ligands]
    _write_rows(ligand_manifest_path, prepared_ligands)
    _write_rows(receptor_manifest_path, receptor_rows)
    generated_box_path: Path | None = None
    if _raw_receptor_sources(paths):
        generated_box_path, _ = _configure_generated_box(config)
    return {
        "ligands": prepared_ligands,
        "receptors": receptor_rows,
        "source_ligand_manifest": source_ligand_manifest,
        "source_ligand_allocation_summary": allocation_summary_path,
        "receptor_audit": receptor_audit,
        "docking_box": generated_box_path,
        "resumed": False,
    }


def validate_front_inputs(
    config: FullExperimentConfig, *, start_stage: str | None = None
) -> dict[str, object]:
    stage = start_stage or config.start_stage
    if stage not in FULL_WORKFLOW_STAGES:
        raise ConfigError(f"unknown start stage: {stage}")
    requirements = {name: front_input_keys(config.data, name) for name in FULL_WORKFLOW_STAGES}
    records: dict[str, object] = {"stage": stage, "paths": {}}
    path_records = records["paths"]
    assert isinstance(path_records, dict)
    for key in requirements[stage]:
        path = config.paths.get(key)
        if path is None:
            raise ConfigError(f"{stage} requires configured path: {key}")
        if not path.exists():
            raise FileNotFoundError(path)
        path_records[key] = str(path)
    if stage == "dock" and "docking_box" in path_records:
        artifact = json.loads(config.paths["docking_box"].read_text(encoding="ascii"))
        box = artifact.get("box")
        if not isinstance(box, dict):
            raise ConfigError("docking_box artifact must contain an object named box")
        docking = config.data["docking"]
        assert isinstance(docking, dict)
        docking["box"] = {**docking.get("box", {}), **box}
    if stage == "prepare":
        selection = config.data["selection"]
        assert isinstance(selection, dict)
        if str(selection.get("ordering", "manifest_order")) == "preselected_manifest":
            rows = select_preselected_ligands(
                config.paths["ligand_manifest"],
                target_id=str(config.data["target_id"]),
                label_counts={
                    str(key): int(value)
                    for key, value in selection["label_counts"].items()
                },
                ligand_count=int(selection["ligand_count"]),
            )
            path_records["selected_ligand_count"] = len(rows)
    return records


def _score_files(score_directory: Path) -> list[Path]:
    if score_directory.is_file():
        return [score_directory]
    if not score_directory.is_dir():
        raise FileNotFoundError(score_directory)
    files = sorted(path for path in score_directory.rglob("*.csv") if "checkpoint" not in path.name)
    if not files:
        raise FileNotFoundError(f"no score tables found in {score_directory}")
    return files


def aggregate_score_tables(
    *,
    score_directory: Path,
    ligand_manifest: Path,
    receptor_count: int,
    seed_count: int,
    matrices_directory: Path,
) -> dict[str, object]:
    ligand_rows = _read_ligand_manifest(ligand_manifest)
    ligand_by_id = {row["ligand_id"]: row for row in ligand_rows}
    files = _score_files(score_directory)
    raw_rows = read_score_tables(files)
    rows_by_seed: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        seed = str(row.get("seed", ""))
        if not seed:
            raise ValueError("score table rows require seed")
        rows_by_seed[seed].append(row)
    if len(rows_by_seed) != seed_count:
        raise ValueError(f"expected {seed_count} seed tables, got {len(rows_by_seed)}")
    seed_groups: list[tuple[str, list[dict[str, str]]]] = []
    for seed, rows in sorted(rows_by_seed.items()):
        representatives = select_representative_scores(rows, "pose_rank_1")
        normalized = [
            {
                **row,
                "representative_score": row["representative_score"],
                "representative_method": "pose_rank_1",
                "selection_role": ligand_by_id[str(row["ligand_id"])] ["selection_role"],
            }
            for row in representatives
        ]
        seed_groups.append((seed, [{key: str(value) for key, value in row.items()} for row in normalized]))
    combined = aggregate_seed_rows(
        seed_groups,
        ligand_by_id,
        receptor_count,
        "pose_rank_1",
    )
    primary = build_matrix(combined, "median_representative_score")
    sensitivity = build_matrix(combined, "minimum_representative_score")
    matrices_directory.mkdir(parents=True, exist_ok=True)
    aggregated_path = matrices_directory / "aggregated_long.csv"
    primary_path = matrices_directory / "primary_median_matrix.csv"
    sensitivity_path = matrices_directory / "sensitivity_minimum_matrix.csv"
    summary_path = matrices_directory / "summary.json"
    _write_rows(aggregated_path, combined)
    _write_rows(primary_path, primary)
    _write_rows(sensitivity_path, sensitivity)
    summary = {
        "status": "ok",
        "seed_count": len(seed_groups),
        "receptor_count": receptor_count,
        "ligand_count": len(ligand_rows),
        "pair_count": len(combined),
        "engine": sorted({str(row.get("engine", "unknown")) for row in raw_rows}),
        "outputs": {
            "aggregated_long": str(aggregated_path),
            "primary_matrix": str(primary_path),
            "sensitivity_matrix": str(sensitivity_path),
        },
    }
    write_json(summary_path, summary)
    return {
        "summary": summary,
        "aggregated_long": aggregated_path,
        "primary_matrix": primary_path,
        "sensitivity_matrix": sensitivity_path,
        "summary_path": summary_path,
    }


def _load_problem_payload(config: FullExperimentConfig, matrix_path: Path) -> dict[str, object]:
    rows = read_csv(matrix_path)
    receptor_manifest = _read_receptor_manifest(config.paths["selected_receptor_manifest"])
    problem_config = dict(config.data["problem"])
    problem_config["receptor_ids"] = [row["conformer_id"] for row in receptor_manifest]
    return {"matrix_path": str(matrix_path), "rows": rows, "problem_config": problem_config}


def build_problem_stage(config: FullExperimentConfig, matrix_path: Path) -> dict[str, object]:
    payload = _load_problem_payload(config, matrix_path)
    rows = payload["rows"]
    problem_config = payload["problem_config"]
    assert isinstance(rows, list) and isinstance(problem_config, dict)
    problem = build_problem(rows, problem_config)
    payload["problem"] = problem.as_dict()
    write_json(config.paths["problem"], payload)
    return {"problem_path": config.paths["problem"], "problem": problem}


def solve_stage(config: FullExperimentConfig, problem_path: Path) -> dict[str, object]:
    payload = json.loads(problem_path.read_text(encoding="ascii"))
    problem_config = payload["problem_config"]
    matrix_path = Path(str(payload["matrix_path"]))
    rows = read_csv(matrix_path)
    problem = build_problem(rows, problem_config)
    backend = str(config.data["solve"]["backend"])
    result = solve_problem(problem, backend)
    output = {
        "status": "ok",
        "matrix_path": str(matrix_path),
        "problem_path": str(problem_path),
        "problem_config": problem_config,
        "result": result.as_dict(),
    }
    write_json(config.paths["selection"], output)
    return {"selection_path": config.paths["selection"], "result": result}


def evaluate_stage(config: FullExperimentConfig, selection_path: Path) -> dict[str, object]:
    payload = json.loads(selection_path.read_text(encoding="ascii"))
    result = payload["result"]
    subset = [str(value) for value in result["subset"]]
    rows = read_csv(Path(str(payload["matrix_path"])))
    evaluate = config.data["evaluate"]
    assert isinstance(evaluate, dict)
    aggregation = str(evaluate.get("aggregation", "mean_score"))
    ranking_data: dict[str, dict[str, object]] = {}
    for row in rows:
        scores = [float(row[receptor]) for receptor in subset]
        score = min(scores) if aggregation == "min_score" else sum(scores) / len(scores)
        ranking_data[row["ligand_id"]] = {"label": row["label"], aggregation: score}
    metrics = ranked_metrics_with_ids(ranking_data, aggregation)
    output = {
        "status": "ok",
        "subset": subset,
        "aggregation": aggregation,
        "metrics": {str(metric): metrics[str(metric)] for metric in evaluate["metrics"] if str(metric) in metrics},
        "all_metrics": metrics,
        "locked_test_rows_read": 0,
    }
    write_json(config.paths["evaluation"], output)
    return {"evaluation_path": config.paths["evaluation"], "evaluation": output}


class FullExperimentRunner:
    def __init__(self, config: FullExperimentConfig) -> None:
        self.config = config

    def run(
        self,
        *,
        dry_run: bool = False,
        start_stage: str | None = None,
        end_stage: str | None = None,
        resume: bool = False,
        overwrite: bool = False,
    ) -> dict[str, object]:
        start = start_stage or self.config.start_stage
        end = end_stage or self.config.end_stage
        if start not in FULL_WORKFLOW_STAGES or end not in FULL_WORKFLOW_STAGES:
            raise ConfigError("stage boundary is not in the canonical workflow")
        if FULL_WORKFLOW_STAGES.index(start) > FULL_WORKFLOW_STAGES.index(end):
            raise ConfigError("start stage must not follow end stage")
        selected = FULL_WORKFLOW_STAGES[
            FULL_WORKFLOW_STAGES.index(start) : FULL_WORKFLOW_STAGES.index(end) + 1
        ]
        if dry_run:
            return {
                "status": "planned",
                "experiment_id": self.config.data["experiment_id"],
                "target_id": self.config.data["target_id"],
                "workflow_mode": self.config.workflow_mode,
                "start_stage": start,
                "end_stage": end,
                "stages": list(selected),
                "redock": self.config.data["docking"]["redock"],
                "engine": self.config.data["docking"]["engine"],
                "paths": {key: str(value) for key, value in self.config.paths.items()},
            }
        if start != "prepare":
            validate_front_inputs(self.config, start_stage=start)
        self.config.paths["run_directory"].mkdir(parents=True, exist_ok=True)
        write_json(self.config.paths["run_directory"] / "config.snapshot.json", self.config.data)
        stage_records: dict[str, object] = {}
        context: dict[str, object] = {}
        for stage in selected:
            if stage == "prepare":
                context["prepared"] = prepare_experiment_inputs(
                    self.config, resume=resume, overwrite=overwrite
                )
                write_json(self.config.paths["run_directory"] / "config.snapshot.json", self.config.data)
                prepared_context = context["prepared"]
                assert isinstance(prepared_context, dict)
                stage_records[stage] = {
                    "status": "completed",
                    "prepared_ligand_manifest": str(self.config.paths["prepared_ligand_manifest"]),
                    "selected_receptor_manifest": str(self.config.paths["selected_receptor_manifest"]),
                    "source_ligand_manifest": str(prepared_context["source_ligand_manifest"])
                    if isinstance(prepared_context.get("source_ligand_manifest"), Path)
                    else "",
                    "receptor_preparation_audit": str(
                        prepared_context.get("receptor_audit", {}).get("audit_path", "")
                    )
                    if isinstance(prepared_context.get("receptor_audit"), dict)
                    else "",
                    "docking_box": str(prepared_context["docking_box"])
                    if isinstance(prepared_context.get("docking_box"), Path)
                    else str(self.config.paths.get("docking_box", "")),
                }
            elif stage == "dock":
                prepared = context.get("prepared")
                if not isinstance(prepared, dict):
                    prepared = {
                        "ligands": _read_ligand_manifest(self.config.paths["prepared_ligand_manifest"]),
                        "receptors": _read_receptor_manifest(self.config.paths["selected_receptor_manifest"]),
                    }
                stage_records[stage] = self._dock(prepared, resume=resume)
            elif stage == "aggregate":
                aggregated = aggregate_score_tables(
                    score_directory=self.config.paths["score_tables"],
                    ligand_manifest=self.config.paths["prepared_ligand_manifest"],
                    receptor_count=int(self.config.data["selection"]["receptor_count"]),
                    seed_count=len(self.config.data["docking"]["seeds"]),
                    matrices_directory=self.config.paths["matrices"],
                )
                context["primary_matrix"] = aggregated["primary_matrix"]
                stage_records[stage] = {"status": "completed", **{key: str(value) for key, value in aggregated.items() if isinstance(value, Path)}}
            elif stage == "build_problem":
                matrix_path = context.get("primary_matrix", self.config.paths["primary_matrix"])
                assert isinstance(matrix_path, Path)
                built = build_problem_stage(self.config, matrix_path)
                context["problem_path"] = built["problem_path"]
                stage_records[stage] = {"status": "completed", "problem": str(built["problem_path"])}
            elif stage == "solve":
                problem_path = context.get("problem_path", self.config.paths["problem"])
                assert isinstance(problem_path, Path)
                solved = solve_stage(self.config, problem_path)
                context["selection_path"] = solved["selection_path"]
                stage_records[stage] = {"status": "completed", "selection": str(solved["selection_path"])}
            elif stage == "evaluate":
                selection_path = context.get("selection_path", self.config.paths["selection"])
                assert isinstance(selection_path, Path)
                evaluated = evaluate_stage(self.config, selection_path)
                context["evaluation_path"] = evaluated["evaluation_path"]
                stage_records[stage] = {"status": "completed", "evaluation": str(evaluated["evaluation_path"])}
            elif stage == "persist":
                evaluation_path = context.get("evaluation_path", self.config.paths["evaluation"])
                assert isinstance(evaluation_path, Path)
                evaluation = json.loads(evaluation_path.read_text(encoding="ascii"))
                summary = {
                    "status": "completed",
                    "experiment_id": self.config.data["experiment_id"],
                    "target_id": self.config.data["target_id"],
                    "workflow_mode": self.config.workflow_mode,
                    "redock": self.config.data["docking"]["redock"],
                    "engine": self.config.data["docking"]["engine"],
                    "selection": self.config.data["selection"],
                    "start_stage": start,
                    "end_stage": end,
                    "evaluation": evaluation,
                    "stages": stage_records,
                }
                summary_path = self.config.paths["run_directory"] / "summary.json"
                write_json(summary_path, summary)
                stage_records[stage] = {"status": "completed", "summary": str(summary_path)}
        summary = {
            "status": "completed",
            "experiment_id": self.config.data["experiment_id"],
            "target_id": self.config.data["target_id"],
            "start_stage": start,
            "end_stage": end,
            "stages": stage_records,
        }
        write_json(self.config.paths["run_directory"] / "manifest.json", summary)
        return summary

    def _dock(self, prepared: dict[str, object], *, resume: bool) -> dict[str, object]:
        ligands = prepared["ligands"]
        receptors = prepared["receptors"]
        assert isinstance(ligands, list) and isinstance(receptors, list)
        docking = self.config.data["docking"]
        assert isinstance(docking, dict)
        score_directory = self.config.paths["score_tables"]
        score_directory.mkdir(parents=True, exist_ok=True)
        if docking.get("redock", True) is False:
            _score_files(score_directory)
            return {"status": "replay", "score_tables": str(score_directory)}
        adapter = get_docking_adapter(self.config.data)
        completed = 0
        for seed in docking["seeds"]:
            for receptor in receptors:
                receptor_path = _as_rooted(Path(receptor["receptor_pdbqt"]), self.config.data_root)
                receptor_id = str(receptor["conformer_id"])
                output_directory = score_directory / f"seed_{seed}" / receptor_id
                score_table = score_directory / f"seed_{seed}__{receptor_id}.csv"
                adapter.run_batch(
                    target_id=str(self.config.data["target_id"]),
                    receptor_id=receptor_id,
                    receptor_path=receptor_path,
                    ligands=ligands,
                    seed=int(seed),
                    output_dir=output_directory,
                    score_table=score_table,
                    config=self.config.data,
                    root=self.config.data_root,
                    resume=resume,
                )
                completed += 1
        return {
            "status": "completed",
            "engine": adapter.name,
            "seed_count": len(docking["seeds"]),
            "receptor_count": len(receptors),
            "batch_count": completed,
            "score_tables": str(score_directory),
        }
