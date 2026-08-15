"""Run resumable Stage102A EGFR and FA10 Phase A Uni-Dock matrices."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.experimental.unidock import run_stage09_mk14_train696_production as common
from scripts.experimental.unidock import run_unidock_batch_targeted as targeted


def target_inputs(root: Path, config: dict[str, Any], target: str) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    spec = config["phase_a_development_expansion"]["targets"][target]
    receptor_path = root / f"data/processed/stage102a_{target.lower()}_passing_receptor_manifest.csv"
    ligand_path = root / f"data/processed/stage102a_{target.lower()}_phase_a_pdbqt_manifest.csv"
    receptors = common.read_csv(receptor_path)
    ligands = common.read_csv(ligand_path)
    if len(receptors) != int(spec["passing_receptor_count"]):
        raise ValueError(f"{target} receptor count differs")
    if len(ligands) != 600 or Counter(row["label"] for row in ligands) != Counter({"active": 120, "decoy": 480}):
        raise ValueError(f"{target} ligand manifest differs")
    for rows, path_key, hash_key, id_key in (
        (receptors, "receptor_pdbqt", "receptor_pdbqt_sha256", "conformer_id"),
        (ligands, "pdbqt_path", "pdbqt_sha256", "ligand_id"),
    ):
        for row in rows:
            path = common.rooted_path(root, row[path_key])
            if not path.is_file() or common.file_sha256(path) != row[hash_key].upper():
                raise ValueError(f"{target} input differs: {row[id_key]}")
    protocol = {
        "executable": "unidock",
        "required_package_version": "1.1.3",
        "profile_id": "enhanced",
        "scoring": "vina",
        "exhaustiveness": 1024,
        "max_step": 80,
        "refine_step": 5,
        "num_modes": 1,
        "energy_range": 3,
        "verbosity": 1,
        "cuda_visible_devices": "0",
        "maximum_absolute_score_kcal_per_mol": 1000.0,
        "box": spec["box"],
    }
    return receptors, ligands, protocol


def collect(root: Path, config_path: Path, target: str, receptors: list[dict[str, str]], ligands: list[dict[str, str]], protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    run_directory = root / f"results/runs/stage102a_{target.lower()}_phase_a_production"
    config_sha = common.file_sha256(config_path)
    ligand_ids = {row["ligand_id"] for row in ligands}
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for seed in (("seed0", 20260821), ("seed1", 20260822), ("seed2", 20260823)):
        seed_id, base_seed = seed
        for receptor in receptors:
            receptor_id = receptor["conformer_id"]
            paths = common.batch_paths(run_directory, seed_id, receptor_id)
            signature = common.protocol_signature(config_sha, seed_id, base_seed, receptor, ligands, protocol)
            value = common.checkpoint(root, paths, signature, ligand_ids)
            if value is None:
                missing.append({"seed_id": seed_id, "receptor_id": receptor_id})
            else:
                batch_rows, summary = value
                rows.extend(batch_rows)
                summaries.append(summary)
    return rows, summaries, missing


def finalize(root: Path, config_path: Path, target: str, receptors: list[dict[str, str]], ligands: list[dict[str, str]], protocol: dict[str, Any], executed: int, resumed: int, elapsed: float) -> dict[str, Any]:
    run_directory = root / f"results/runs/stage102a_{target.lower()}_phase_a_production"
    rows, summaries, missing = collect(root, config_path, target, receptors, ligands, protocol)
    progress = {
        "schema_version": "1.0",
        "status": "stage102a_phase_a_matrix_ok" if not missing else "stage102a_phase_a_partial_ok",
        "target_id": target,
        "completed_batch_count": len(summaries),
        "missing_batch_count": len(missing),
        "missing_batches": missing,
        "completed_pair_count": len(rows),
        "expected_pair_count": len(receptors) * len(ligands) * 3,
        "executed_batches_this_invocation": executed,
        "resumed_batches_this_invocation": resumed,
        "invocation_elapsed_seconds": elapsed,
    }
    common.write_json(run_directory / "progress.json", progress)
    if missing:
        print(json.dumps(progress, indent=2, sort_keys=True))
        return progress
    if len(rows) != len(receptors) * len(ligands) * 3:
        raise ValueError(f"{target} complete pair count differs")
    for row in rows:
        score = float(row["gpu_score"])
        if not math.isfinite(score) or abs(score) > 1000.0:
            raise ValueError(f"{target} invalid score: {row['ligand_id']}")
    seed_order = {f"seed{index}": index for index in range(3)}
    receptor_order = {row["conformer_id"]: index for index, row in enumerate(receptors)}
    ligand_order = {row["ligand_id"]: index for index, row in enumerate(ligands)}
    rows.sort(key=lambda row: (seed_order[str(row["seed_id"])], receptor_order[str(row["receptor_id"])], ligand_order[str(row["ligand_id"])]))
    common.write_csv(run_directory / "scores.csv", rows)
    batch_rows = [{
        "seed_id": summary["seed_id"],
        "base_seed": summary["base_seed"],
        "receptor_id": summary["receptor_id"],
        "ligand_count": summary["ligand_count"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "score_minimum": summary["score_minimum"],
        "score_maximum": summary["score_maximum"],
        "known_warning_event_count": summary["warning_adjudication"]["known_warning_event_count"],
        "unresolved_warning_event_count": summary["warning_adjudication"]["unresolved_warning_event_count"],
        "pose_integrity_failure_count": summary["pose_integrity_audit"]["failure_count"],
        "signature": summary["signature"],
        "status": summary["status"],
    } for summary in summaries]
    batch_rows.sort(key=lambda row: (seed_order[str(row["seed_id"])], receptor_order[str(row["receptor_id"])]))
    common.write_csv(run_directory / "batch_runs.csv", batch_rows)
    receptor_ids = [row["conformer_id"] for row in receptors]
    common.write_csv(run_directory / "primary_median_score_matrix.csv", common.matrix_rows(rows, ligands, receptor_ids, "median"))
    common.write_csv(run_directory / "sensitivity_minimum_score_matrix.csv", common.matrix_rows(rows, ligands, receptor_ids, "minimum"))
    result = {
        **progress,
        "status": "stage102a_phase_a_matrix_ok",
        "receptor_count": len(receptors),
        "ligand_count": len(ligands),
        "seed_count": 3,
        "unresolved_warning_event_count": 0,
        "pose_integrity_failure_count": 0,
        "data_boundary": {"fresh_validation_rows_read": 0, "locked_test_rows_read": 0},
        "interpretation": "Development-expansion score generation only; no adaptive rule or hardware claim follows until Stage102 analysis passes.",
    }
    common.write_json(run_directory / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_target(root: Path, config_path: Path, target: str, unidock: str | None, audit_only: bool, resume: bool, finalize_only: bool) -> dict[str, Any]:
    config = common.read_json(config_path)
    receptors, ligands, protocol = target_inputs(root, config, target)
    if audit_only:
        result = {"schema_version": "1.0", "status": "audit_only_ok", "target_id": target, "receptor_count": len(receptors), "ligand_count": len(ligands), "selected_pair_count": len(receptors) * len(ligands) * 3, "operation": "input audit only; no docking started"}
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    executed = resumed_count = 0
    started = time.perf_counter()
    if not finalize_only:
        executable = common.executable_evidence(unidock or "unidock", "1.1.3")
        run_directory = root / f"results/runs/stage102a_{target.lower()}_phase_a_production"
        config_sha = common.file_sha256(config_path)
        ligand_ids = {row["ligand_id"] for row in ligands}
        for seed_id, base_seed in (("seed0", 20260821), ("seed1", 20260822), ("seed2", 20260823)):
            for receptor in receptors:
                receptor_id = receptor["conformer_id"]
                paths = common.batch_paths(run_directory, seed_id, receptor_id)
                paths["directory"].mkdir(parents=True, exist_ok=True)
                signature = common.protocol_signature(config_sha, seed_id, base_seed, receptor, ligands, protocol)
                value = common.checkpoint(root, paths, signature, ligand_ids) if resume else None
                if value is not None:
                    resumed_count += 1
                    print(f"resume ok: {target}/{seed_id}/{receptor_id}", flush=True)
                    continue
                print(f"running: {target}/{seed_id}/{receptor_id}", flush=True)
                rows, summary = targeted.run_batch(root, paths, str(executable["resolved_executable"]), receptor, ligands, protocol, seed_id, base_seed, signature)
                rows, pose_audit = common.audit_batch_poses(root, ligands, rows)
                warning = common.classify_warning_log(paths["log"], pose_audit)
                summary["pose_integrity_audit"] = pose_audit
                summary["warning_adjudication"] = warning
                summary["status"] = "ok" if int(pose_audit["failure_count"]) == 0 and int(warning["unresolved_warning_event_count"]) == 0 else "technical_integrity_failed"
                common.write_csv(paths["scores"], rows)
                summary["scores_sha256"] = common.file_sha256(paths["scores"])
                common.write_json(paths["summary"], summary)
                if summary["status"] != "ok":
                    raise ValueError(f"{target} technical integrity failed: {seed_id}/{receptor_id}")
                executed += 1
                print(f"completed: {target}/{seed_id}/{receptor_id} in {float(summary['elapsed_seconds']):.3f} s", flush=True)
    return finalize(root, config_path, target, receptors, ligands, protocol, executed, resumed_count, time.perf_counter() - started)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage102_prospective_marginal_learning.json"))
    parser.add_argument("--target", action="append", choices=("EGFR", "FA10"))
    parser.add_argument("--unidock", default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = (root / args.config).resolve()
    targets = args.target or ["EGFR", "FA10"]
    results = [run_target(root, config_path, target, args.unidock, args.audit_only, args.resume, args.finalize_only) for target in targets]
    allowed = {"audit_only_ok", "stage102a_phase_a_partial_ok", "stage102a_phase_a_matrix_ok"}
    return 0 if all(result["status"] in allowed for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
