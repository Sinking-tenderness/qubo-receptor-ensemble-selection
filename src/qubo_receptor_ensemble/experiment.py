"""Execution engine for the configurable source-to-result experiment."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping

from .docking_adapters import get_docking_adapter
from .adaptive_cardinality import estimate_adaptive_cardinality
from .full_workflow import (
    FULL_WORKFLOW_STAGES,
    ConfigError,
    FullExperimentConfig,
    front_input_keys,
    select_ism_ligands,
    select_manual_ligands,
    select_manual_receptors,
    select_preselected_ligands,
    select_receptor_manifest,
)
from .ligand_selection import (
    select_scaffold_hash_ligands,
    summarize_scaffold_hash_allocation,
)
from .io import file_sha256, read_csv, write_csv, write_json
from .k_selection import KCandidate, KSelectionDecision, choose_k
from .methods import resolve_method_requests
from .matrix import aggregate_seed_rows, build_matrix, read_score_tables, select_representative_scores
from .raw_preparation import calculate_ligand_box, prepare_raw_receptors
from .screening import ranked_metrics_with_ids
from .solvers import Problem, ProblemError, SolverResult, build_problem, solve_problem


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


def _resolve_manual_receptor_rows(config: FullExperimentConfig) -> list[dict[str, str]]:
    selection = config.data["selection"]
    assert isinstance(selection, dict)
    policy = selection.get("receptor_selection")
    if not isinstance(policy, dict) or policy.get("mode") != "manual":
        raise ConfigError("manual receptor selection is not configured")
    rows = select_manual_receptors(
        policy.get("receptors"),
        receptor_count=int(selection["receptor_count"]),
    )
    prepared_by_rcsb: dict[str, dict[str, object]] = {}
    if rows and all("rcsb_id" in row and "receptor_pdbqt" not in row for row in rows):
        audit_path = config.paths.get("receptor_preparation_audit")
        if audit_path is None:
            audit_path = config.paths["run_directory"] / "receptor_preparation_audit.json"
        if not audit_path.is_file():
            raise FileNotFoundError(audit_path)
        audit = json.loads(audit_path.read_text(encoding="ascii"))
        selected = audit.get("selected", [])
        if not isinstance(selected, list):
            raise ConfigError("receptor preparation audit selected must be a list")
        prepared_by_rcsb = {
            str(record["rcsb_id"]).upper(): record
            for record in selected
            if isinstance(record, dict) and record.get("rcsb_id")
        }
    resolved: list[dict[str, str]] = []
    for row in rows:
        normalized = dict(row)
        if "receptor_pdbqt" not in normalized:
            rcsb_id = normalized.get("rcsb_id", "").upper()
            record = prepared_by_rcsb.get(rcsb_id)
            if record is None or not record.get("receptor_pdbqt"):
                raise ConfigError(
                    f"receptor preparation audit is missing configured RCSB ID: {rcsb_id}"
                )
            normalized["receptor_pdbqt"] = str(record["receptor_pdbqt"])
        path = _as_rooted(Path(normalized["receptor_pdbqt"]), config.data_root)
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_sha256 = normalized.get("receptor_pdbqt_sha256", "").strip()
        if expected_sha256:
            actual_sha256 = file_sha256(path)
            if actual_sha256.upper() != expected_sha256.upper():
                raise ConfigError(
                    f"manual receptor PDBQT SHA-256 mismatch for {normalized['conformer_id']}"
                )
        normalized["receptor_pdbqt"] = _relative(path, config.data_root)
        resolved.append(normalized)
    return resolved


def _manual_receptor_selection_enabled(config: FullExperimentConfig) -> bool:
    selection = config.data.get("selection")
    if not isinstance(selection, dict):
        return False
    policy = selection.get("receptor_selection")
    return isinstance(policy, dict) and policy.get("mode") == "manual"


def _configured_receptor_rows(config: FullExperimentConfig) -> list[dict[str, str]]:
    if _manual_receptor_selection_enabled(config):
        return _resolve_manual_receptor_rows(config)
    return _read_receptor_manifest(config.paths["selected_receptor_manifest"])


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
    from .preparation import (
        build_3d_mol,
        macrocycle_closure_atom_types,
        parse_pdbqt,
        run_meeko,
    )

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
    if pdbqt_path.exists():
        pdbqt_path.unlink()
    flexible = run_meeko(
        meeko_script,
        sdf_path,
        pdbqt_path,
        rigid_macrocycles=False,
    )
    flexible_audit: dict[str, object] = {}
    flexible_valid = flexible.returncode == 0 and pdbqt_path.is_file()
    if flexible_valid:
        flexible_audit = parse_pdbqt(pdbqt_path)
        flexible_valid = (
            int(flexible_audit["pdbqt_atom_count"]) > 0
            and str(flexible_audit["torsdof"]) != ""
        )
    flexible_pseudoatoms = (
        macrocycle_closure_atom_types(pdbqt_path)
        if flexible_valid
        else []
    )
    preparation_variant = "meeko_flexible"
    pdbqt_message = "meeko_ok"
    if not flexible_valid or flexible_pseudoatoms:
        if pdbqt_path.exists():
            pdbqt_path.unlink()
        rigid = run_meeko(
            meeko_script,
            sdf_path,
            pdbqt_path,
            rigid_macrocycles=True,
        )
        if rigid.returncode != 0 or not pdbqt_path.is_file():
            details = "\n".join(
                part.strip()
                for part in (
                    flexible.stdout,
                    flexible.stderr,
                    rigid.stdout,
                    rigid.stderr,
                )
                if part.strip()
            )
            raise RuntimeError(f"Meeko failed for {ligand_id}: {details[-500:]}")
        preparation_variant = "meeko_rigid_macrocycles"
        pdbqt_message = (
            "meeko_rigid_after_closure_pseudoatom_detection"
            if flexible_pseudoatoms
            else "meeko_rigid_after_flexible_failure"
        )
    remaining_pseudoatoms = macrocycle_closure_atom_types(pdbqt_path)
    if remaining_pseudoatoms:
        raise ValueError(
            f"closure pseudoatoms remain for {ligand_id}: {remaining_pseudoatoms}"
        )
    pdbqt_audit = parse_pdbqt(pdbqt_path)
    if (
        int(pdbqt_audit["pdbqt_atom_count"]) <= 0
        or str(pdbqt_audit["torsdof"]) == ""
    ):
        raise ValueError(f"invalid PDBQT for {ligand_id}")
    return {
        **row,
        "prep_status": status,
        "prep_message": message,
        "sdf_path": _relative(sdf_path, root),
        "pdbqt_path": _relative(pdbqt_path, root),
        "pdbqt_status": "ok",
        "pdbqt_message": pdbqt_message,
        "preparation_variant": preparation_variant,
        **pdbqt_audit,
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
    selection = config.data["selection"]
    assert isinstance(selection, dict)
    manual_policy = selection.get("receptor_selection")
    configured_receptors: list[dict[str, str]] | None = None
    if isinstance(manual_policy, dict) and manual_policy.get("mode") == "manual":
        configured_receptors = select_manual_receptors(
            manual_policy.get("receptors"),
            receptor_count=int(selection["receptor_count"]),
        )
        candidate_ids = [row["rcsb_id"] for row in configured_receptors]
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
    selected_records = prepared["selected"]
    assert isinstance(selected_records, list)
    records_by_rcsb = {
        str(record["rcsb_id"]).upper(): record
        for record in selected_records
        if isinstance(record, dict)
    }
    ordered_records = selected_records
    if configured_receptors is not None:
        try:
            ordered_records = [records_by_rcsb[row["rcsb_id"]] for row in configured_receptors]
        except KeyError as exc:
            raise ConfigError(f"raw receptor preparation did not return configured RCSB ID: {exc}") from exc
    rows: list[dict[str, object]] = []
    for index, record in enumerate(ordered_records):
        assert isinstance(record, dict)
        alignment = record.get("alignment", {})
        if not isinstance(alignment, dict):
            alignment = {}
        configured = configured_receptors[index] if configured_receptors is not None else {}
        rows.append(
            {
                "conformer_id": str(configured.get("conformer_id", record["conformer_id"])),
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
    if configured_receptors is not None:
        prepared["configured_receptors"] = configured_receptors
    write_json(audit_path, prepared)
    return rows, {"audit_path": audit_path, "candidate_count": prepared["candidate_count"]}


def prepare_experiment_inputs(
    config: FullExperimentConfig, *, resume: bool = False, overwrite: bool = False
) -> dict[str, object]:
    """Prepare all inputs from raw sources for the exact current experiment run."""
    paths = config.paths
    ligand_manifest_path = paths["prepared_ligand_manifest"]
    manual_receptors = _manual_receptor_selection_enabled(config)
    receptor_manifest_path = (
        None if manual_receptors else paths.get("selected_receptor_manifest")
    )
    docking = config.data["docking"]
    assert isinstance(docking, dict)
    box_policy = docking.get("box", {})
    assert isinstance(box_policy, dict)
    generates_box = _raw_receptor_sources(paths) and box_policy.get("method") == "ligand_bounds"
    box_path = paths.get("docking_box", paths["run_directory"] / "docking_box.json")
    resume_paths = [ligand_manifest_path]
    if generates_box:
        resume_paths.append(box_path)
    if receptor_manifest_path is not None:
        resume_paths.insert(1, receptor_manifest_path)
    if resume and all(path.is_file() for path in resume_paths):
        ligands = _read_ligand_manifest(ligand_manifest_path)
        receptors = _configured_receptor_rows(config)
        if (
            all(_as_rooted(Path(row["pdbqt_path"]), config.data_root).is_file() for row in ligands)
            and all(
                _as_rooted(Path(row["receptor_pdbqt"]), config.data_root).is_file()
                for row in receptors
            )
        ):
            if generates_box:
                box_artifact = json.loads(box_path.read_text(encoding="ascii"))
                generated_box = box_artifact.get("box")
                if not isinstance(generated_box, dict):
                    raise ConfigError(f"generated docking box is invalid: {box_path}")
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
        existing = [path for path in resume_paths if path.exists()]
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
    elif ordering == "manual_ids":
        selected_ligands = select_manual_ligands(
            paths["active_ism"],
            paths["decoy_ism"],
            selection.get("ligand_ids"),
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
    if manual_receptors and _raw_receptor_sources(paths):
        receptor_rows, receptor_audit = _prepare_raw_receptor_manifest(config)
    elif manual_receptors:
        receptor_rows = _resolve_manual_receptor_rows(config)
    elif _raw_receptor_sources(paths):
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
    if receptor_manifest_path is not None:
        _write_rows(receptor_manifest_path, receptor_rows)
    generated_box_path: Path | None = None
    if _raw_receptor_sources(paths) and generates_box:
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
        ordering = str(selection.get("ordering", "manifest_order"))
        if ordering == "preselected_manifest":
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
        elif ordering == "manual_ids":
            rows = select_manual_ligands(
                config.paths["active_ism"],
                config.paths["decoy_ism"],
                selection.get("ligand_ids"),
                target_id=str(config.data["target_id"]),
                label_counts={
                    str(key): int(value)
                    for key, value in selection["label_counts"].items()
                },
                ligand_count=int(selection["ligand_count"]),
            )
            path_records["selected_ligand_count"] = len(rows)
        receptor_selection = selection.get("receptor_selection")
        if (
            isinstance(receptor_selection, dict)
            and receptor_selection.get("mode") == "manual"
        ):
            if _raw_receptor_sources(config.paths):
                rows = select_manual_receptors(
                    receptor_selection.get("receptors"),
                    receptor_count=int(selection["receptor_count"]),
                )
            else:
                rows = _resolve_manual_receptor_rows(config)
            path_records["selected_receptor_count"] = len(rows)
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
    receptor_rows = _configured_receptor_rows(config)
    problem_config = dict(config.data["problem"])
    problem_config["receptor_ids"] = [row["conformer_id"] for row in receptor_rows]
    k_policy = problem_config.get("k_policy")
    if isinstance(k_policy, dict) and k_policy.get("mode") == "adaptive":
        ligand_manifest = _read_ligand_manifest(config.paths["prepared_ligand_manifest"])
        metadata_by_ligand = {row["ligand_id"]: row for row in ligand_manifest}
        matrix_ids = {str(row.get("ligand_id", "")) for row in rows}
        manifest_ids = set(metadata_by_ligand)
        if matrix_ids != manifest_ids:
            missing = sorted(manifest_ids - matrix_ids)
            extra = sorted(matrix_ids - manifest_ids)
            raise ConfigError(
                "adaptive cardinality requires matrix and ligand manifest IDs to match; "
                f"missing_in_matrix={missing}, extra_in_matrix={extra}"
            )
        scaffold_field = str(k_policy.get("scaffold_field", "scaffold_smiles"))
        source_metadata_by_ligand: dict[str, dict[str, str]] = {}
        source_manifest_path = config.paths.get("source_ligand_manifest")
        if source_manifest_path is not None and source_manifest_path.is_file():
            source_metadata_by_ligand = {
                str(row.get("ligand_id", "")): row
                for row in read_csv(source_manifest_path)
                if str(row.get("ligand_id", "")).strip()
            }
        needs_scaffold = any(
            not str(source_metadata_by_ligand.get(ligand_id, {}).get(scaffold_field, "")).strip()
            for ligand_id in manifest_ids
        )
        selection = config.data.get("selection")
        active_ism = config.paths.get("active_ism")
        decoy_ism = config.paths.get("decoy_ism")
        if (
            needs_scaffold
            and isinstance(selection, dict)
            and str(selection.get("ordering", "")) == "manual_ids"
            and active_ism is not None
            and decoy_ism is not None
            and active_ism.is_file()
            and decoy_ism.is_file()
        ):
            label_counts = selection.get("label_counts", {})
            if not isinstance(label_counts, dict):
                raise ConfigError("selection.label_counts must be an object")
            raw_rows = select_manual_ligands(
                active_ism,
                decoy_ism,
                selection.get("ligand_ids"),
                target_id=str(config.data["target_id"]),
                label_counts={str(key): int(value) for key, value in label_counts.items()},
                ligand_count=int(selection["ligand_count"]),
            )
            source_metadata_by_ligand.update(
                {row["ligand_id"]: row for row in raw_rows if row.get("ligand_id")}
            )
        enriched_rows: list[dict[str, object]] = []
        for row in rows:
            ligand_id = str(row.get("ligand_id", ""))
            manifest_row = metadata_by_ligand[ligand_id]
            if str(row.get("label", "")) != str(manifest_row.get("label", "")):
                raise ConfigError(
                    f"adaptive cardinality label mismatch for ligand_id={ligand_id}"
                )
            enriched = {**row, **manifest_row}
            if not str(enriched.get(scaffold_field, "")).strip():
                source_row = source_metadata_by_ligand.get(ligand_id, {})
                source_scaffold = str(source_row.get(scaffold_field, "")).strip()
                if source_scaffold:
                    enriched[scaffold_field] = source_scaffold
            enriched_rows.append(enriched)
        rows = enriched_rows
    return {"matrix_path": str(matrix_path), "rows": rows, "problem_config": problem_config}


def resolve_problem_requests(problem_config: dict[str, object]) -> list[dict[str, object]]:
    """Public workflow helper for expanding single or comparison method config."""
    return resolve_method_requests(problem_config)


def _method_directory(config: FullExperimentConfig, method_id: str) -> Path:
    path = config.paths["run_directory"] / "methods" / method_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_problem_stage(
    config: FullExperimentConfig,
    matrix_path: Path,
    *,
    progress: Callable[[str, Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    payload = _load_problem_payload(config, matrix_path)
    rows = payload["rows"]
    problem_config = payload["problem_config"]
    assert isinstance(rows, list) and isinstance(problem_config, dict)
    evaluate_config = config.data.get("evaluate", {})
    aggregation = (
        str(evaluate_config.get("aggregation", "mean_score"))
        if isinstance(evaluate_config, dict)
        else "mean_score"
    )
    k_policy = problem_config.get("k_policy")
    adaptive_decision: dict[str, object] | None = None
    if isinstance(k_policy, dict) and k_policy.get("mode") == "adaptive":
        if str(problem_config.get("mode", "single")) == "compare" or "methods" in problem_config:
            raise ConfigError(
                "adaptive cardinality is not supported with comparison problems"
            )
        try:
            decision = estimate_adaptive_cardinality(
                rows,
                [str(value) for value in problem_config["receptor_ids"]],
                problem_config=problem_config,
                solver_backend=str(config.data["solve"]["backend"]),
                candidate_ks=(
                    [int(value) for value in k_policy["candidates"]]
                    if "candidates" in k_policy
                    else None
                ),
                scaffold_field=str(k_policy.get("scaffold_field", "scaffold_smiles")),
                inner_fold_count=int(k_policy.get("inner_fold_count", 3)),
                bootstrap_iterations=int(k_policy.get("bootstrap_iterations", 1000)),
                lower_quantile=float(k_policy.get("lower_quantile", 0.05)),
                minimum_effect=float(k_policy.get("minimum_effect", 0.0)),
                required_probability=float(k_policy.get("required_probability", 0.9)),
                cost_per_receptor=float(k_policy.get("cost_per_receptor", 0.0)),
                selection_tie_tolerance=float(
                    k_policy.get("selection_tie_tolerance", 0.0)
                ),
                require_rescue_contrast=bool(
                    k_policy.get("require_rescue_contrast", False)
                ),
                rescue_fractions=[
                    float(value)
                    for value in k_policy.get("rescue_fractions", [0.01, 0.05])
                ],
                bedroc_alpha=float(problem_config.get("bedroc_alpha", 20.0)),
                random_seed=int(k_policy.get("random_seed", 0)),
                progress=progress,
                aggregation=aggregation,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"adaptive cardinality selection failed: {exc}") from exc
        problem_config["target_size"] = decision.selected_k
        adaptive_decision = decision.as_dict()
        write_json(
            config.paths["run_directory"] / "adaptive_cardinality.json",
            adaptive_decision,
        )
    if str(problem_config.get("mode", "single")) == "compare" or "methods" in problem_config:
        requests = resolve_problem_requests(problem_config)
        receptor_rows = _configured_receptor_rows(config)
        receptor_ids = [row["conformer_id"] for row in receptor_rows]
        records: list[dict[str, object]] = []
        capabilities: list[dict[str, object]] = []
        for request in requests:
            request = {**request, "receptor_ids": receptor_ids}
            method_id = str(request["method_id"])
            method_directory = _method_directory(config, method_id)
            method_problem_path = method_directory / "problem.json"
            try:
                problem = build_problem(rows, request)
            except (ProblemError, ValueError) as exc:
                record = {
                    "method_id": method_id,
                    "status": "unsupported_for_input",
                    "error": str(exc),
                }
                capabilities.append(record)
                records.append(record)
                continue
            method_payload = {
                "mode": "single",
                "matrix_path": str(matrix_path),
                "rows": rows,
                "problem_config": request,
                "problem": problem.as_dict(),
            }
            write_json(method_problem_path, method_payload)
            record = {
                "method_id": method_id,
                "status": "ready",
                "problem_path": str(method_problem_path),
            }
            capabilities.append({**record, "formulation_kind": problem.formulation.get("method", {}).get("formulation_kind", "qubo") if isinstance(problem.formulation.get("method", {}), dict) else "qubo"})
            records.append(record)
        index = {
            "mode": "compare",
            "matrix_path": str(matrix_path),
            "methods": records,
            "primary_metric": str(
                config.data.get("evaluate", {}).get(
                    "primary_metric", "bedroc_alpha_20"
                )
            ),
        }
        write_json(config.paths["run_directory"] / "method_capabilities.json", {"methods": capabilities})
        write_json(config.paths["problem"], index)
        return {
            "problem_path": config.paths["problem"],
            "comparison": True,
            "method_count": len(records),
            "ready_count": sum(record["status"] == "ready" for record in records),
        }
    problem = build_problem(rows, problem_config)
    payload["problem"] = problem.as_dict()
    if adaptive_decision is not None:
        payload["adaptive_cardinality"] = adaptive_decision
    write_json(config.paths["problem"], payload)
    result: dict[str, object] = {
        "problem_path": config.paths["problem"],
        "problem": problem,
    }
    if adaptive_decision is not None:
        result["adaptive_cardinality"] = adaptive_decision
    return result


def solve_stage(config: FullExperimentConfig, problem_path: Path) -> dict[str, object]:
    payload = json.loads(problem_path.read_text(encoding="ascii"))
    if payload.get("mode") == "compare":
        records = payload.get("methods", [])
        if not isinstance(records, list):
            raise ProblemError("comparison problem index requires methods")
        solved_records: list[dict[str, object]] = []
        for record in records:
            if not isinstance(record, dict):
                raise ProblemError("comparison method record must be an object")
            if record.get("status") != "ready":
                solved_records.append(dict(record))
                continue
            method_problem_path = Path(str(record["problem_path"]))
            method_payload = json.loads(method_problem_path.read_text(encoding="ascii"))
            problem = build_problem(
                read_csv(Path(str(method_payload["matrix_path"]))),
                dict(method_payload["problem_config"]),
            )
            backend = str(config.data["solve"]["backend"])
            result = solve_problem(problem, backend)
            selection_path = method_problem_path.parent / "selection.json"
            write_json(
                selection_path,
                {
                    "status": "ok",
                    "matrix_path": method_payload["matrix_path"],
                    "problem_path": str(method_problem_path),
                    "problem_config": method_payload["problem_config"],
                    "result": result.as_dict(),
                },
            )
            solved_records.append(
                {
                    **record,
                    "selection_path": str(selection_path),
                    "result": result.as_dict(),
                }
            )
        output = {"mode": "compare", "methods": solved_records}
        write_json(config.paths["selection"], output)
        return {"selection_path": config.paths["selection"], "results": solved_records}
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
    if isinstance(payload.get("adaptive_cardinality"), dict):
        output["adaptive_cardinality"] = payload["adaptive_cardinality"]
    write_json(config.paths["selection"], output)
    return {"selection_path": config.paths["selection"], "result": result}


def evaluate_stage(config: FullExperimentConfig, selection_path: Path) -> dict[str, object]:
    payload = json.loads(selection_path.read_text(encoding="ascii"))
    if payload.get("mode") == "compare":
        records = payload.get("methods", [])
        if not isinstance(records, list):
            raise ProblemError("comparison selection index requires methods")
        evaluated_records: list[dict[str, object]] = []
        for record in records:
            if not isinstance(record, dict):
                raise ProblemError("comparison selection record must be an object")
            if record.get("status") != "ready":
                evaluated_records.append(dict(record))
                continue
            method_selection_path = Path(str(record["selection_path"]))
            method_payload = json.loads(method_selection_path.read_text(encoding="ascii"))
            method_evaluation_path = method_selection_path.parent / "evaluation.json"
            evaluated = _evaluate_selection_payload(
                config,
                method_payload,
                method_evaluation_path,
            )
            evaluated_records.append(
                {
                    **record,
                    "evaluation_path": str(method_evaluation_path),
                    "primary_metric_value": evaluated["primary_metric_value"],
                }
            )
            write_json(
                method_selection_path.parent / "summary.json",
                {
                    "method_id": record["method_id"],
                    "evaluation": evaluated["evaluation"],
                },
            )
        output = {
            "mode": "compare",
            "primary_metric": str(
                config.data.get("evaluate", {}).get(
                    "primary_metric", "bedroc_alpha_20"
                )
            ),
            "methods": evaluated_records,
        }
        comparison_path = config.paths["run_directory"] / "comparison.json"
        write_json(comparison_path, output)
        write_json(config.paths["evaluation"], output)
        return {"evaluation_path": config.paths["evaluation"], "comparison_path": comparison_path}
    return _evaluate_selection_payload(config, payload, config.paths["evaluation"])


def _evaluate_selection_payload(
    config: FullExperimentConfig,
    payload: dict[str, object],
    evaluation_path: Path,
) -> dict[str, object]:
    result = payload["result"]
    subset = [str(value) for value in result["subset"]]
    rows = read_csv(Path(str(payload["matrix_path"])))
    evaluate = config.data["evaluate"]
    assert isinstance(evaluate, dict)
    aggregation = str(evaluate.get("aggregation", "mean_score"))
    problem_config = payload.get("problem_config", {})
    if not isinstance(problem_config, dict):
        problem_config = {}
    bedroc_alpha = float(problem_config.get("bedroc_alpha", 20.0))
    ranking_data: dict[str, dict[str, object]] = {}
    for row in rows:
        scores = [float(row[receptor]) for receptor in subset]
        score = min(scores) if aggregation == "min_score" else sum(scores) / len(scores)
        ranking_data[row["ligand_id"]] = {"label": row["label"], aggregation: score}
    metrics = ranked_metrics_with_ids(
        ranking_data,
        aggregation,
        bedroc_alpha=bedroc_alpha,
    )
    primary_metric = str(
        evaluate.get("primary_metric", f"bedroc_alpha_{bedroc_alpha:g}")
    )
    selected_metrics = {
        str(metric): metrics[str(metric)]
        for metric in evaluate["metrics"]
        if str(metric) in metrics
    }
    if primary_metric in metrics:
        selected_metrics.setdefault(primary_metric, metrics[primary_metric])
    output = {
        "status": "ok",
        "subset": subset,
        "aggregation": aggregation,
        "primary_metric": primary_metric,
        "primary_metric_value": metrics.get(primary_metric),
        "metrics": selected_metrics,
        "all_metrics": metrics,
        "locked_test_rows_read": 0,
    }
    if isinstance(payload.get("adaptive_cardinality"), dict):
        output["adaptive_cardinality"] = payload["adaptive_cardinality"]
    write_json(evaluation_path, output)
    return {
        "evaluation_path": evaluation_path,
        "evaluation": output,
        "primary_metric_value": output["primary_metric_value"],
    }


class FullExperimentRunner:
    def __init__(self, config: FullExperimentConfig) -> None:
        self.config = config

    @staticmethod
    def _print_adaptive_progress(event: str, payload: Mapping[str, object]) -> None:
        if event == "adaptive_started":
            print(
                f"[adaptive] metric={payload['metric']} "
                f"aggregation={payload['aggregation']} "
                f"candidates={payload['candidates']}",
                flush=True,
            )
        elif event == "inner_fold_started":
            print(
                f"[adaptive] inner_fold={payload['fold']}/{payload['fold_count']}",
                flush=True,
            )
        elif event == "candidate_completed":
            print(
                f"[adaptive] inner_fold={payload['fold']}/{payload['fold_count']} "
                f"candidate_k={payload['candidate_k']} completed",
                flush=True,
            )
        elif event == "transition_evaluated":
            print(
                f"[adaptive] transition={payload['from_k']}->{payload['to_k']} "
                f"metric={payload['metric']} "
                f"aggregation={payload['aggregation']} "
                f"marginal_state={payload['marginal_state']} "
                f"candidate_passed={str(payload['candidate_passed']).lower()}",
                flush=True,
            )
        elif event == "adaptive_completed":
            print(
                f"[adaptive] metric={payload['metric']} "
                f"aggregation={payload['aggregation']} "
                f"selected_k={payload['selected_k']}",
                flush=True,
            )

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
        for stage_index, stage in enumerate(
            selected, start=FULL_WORKFLOW_STAGES.index(start) + 1
        ):
            print(
                f"[stage {stage_index}/{len(FULL_WORKFLOW_STAGES)}] {stage} started",
                flush=True,
            )
            if stage == "prepare":
                context["prepared"] = prepare_experiment_inputs(
                    self.config, resume=resume, overwrite=overwrite
                )
                write_json(self.config.paths["run_directory"] / "config.snapshot.json", self.config.data)
                prepared_context = context["prepared"]
                assert isinstance(prepared_context, dict)
                prepare_record: dict[str, object] = {
                    "status": "completed",
                    "prepared_ligand_manifest": str(self.config.paths["prepared_ligand_manifest"]),
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
                if (
                    not _manual_receptor_selection_enabled(self.config)
                    and "selected_receptor_manifest" in self.config.paths
                ):
                    prepare_record["selected_receptor_manifest"] = str(
                        self.config.paths["selected_receptor_manifest"]
                    )
                stage_records[stage] = prepare_record
            elif stage == "dock":
                prepared = context.get("prepared")
                if not isinstance(prepared, dict):
                    prepared = {
                        "ligands": _read_ligand_manifest(
                            self.config.paths["prepared_ligand_manifest"]
                        ),
                        "receptors": _configured_receptor_rows(self.config),
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
                built = build_problem_stage(
                    self.config,
                    matrix_path,
                    progress=self._print_adaptive_progress,
                )
                context["problem_path"] = built["problem_path"]
                stage_records[stage] = {
                    "status": "completed",
                    "problem": str(built["problem_path"]),
                    **(
                        {"adaptive_cardinality": built["adaptive_cardinality"]}
                        if isinstance(built.get("adaptive_cardinality"), dict)
                        else {}
                    ),
                }
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
                adaptive_path = self.config.paths["run_directory"] / "adaptive_cardinality.json"
                if adaptive_path.is_file():
                    summary["adaptive_cardinality"] = json.loads(
                        adaptive_path.read_text(encoding="ascii")
                    )
                summary_path = self.config.paths["run_directory"] / "summary.json"
                write_json(summary_path, summary)
                stage_records[stage] = {"status": "completed", "summary": str(summary_path)}
            print(
                f"[stage {stage_index}/{len(FULL_WORKFLOW_STAGES)}] {stage} completed",
                flush=True,
            )
        summary = {
            "status": "completed",
            "experiment_id": self.config.data["experiment_id"],
            "target_id": self.config.data["target_id"],
            "start_stage": start,
            "end_stage": end,
            "stages": stage_records,
        }
        adaptive_path = self.config.paths["run_directory"] / "adaptive_cardinality.json"
        if adaptive_path.is_file():
            summary["adaptive_cardinality"] = json.loads(
                adaptive_path.read_text(encoding="ascii")
            )
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
