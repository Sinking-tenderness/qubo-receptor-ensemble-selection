"""Resume Stage 11 under the finite-score guard Amendment 01."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

try:
    from . import run_stage11_mk14_fresh_validation_confirmation as production
except ImportError:
    import run_stage11_mk14_fresh_validation_confirmation as production


ORIGINAL_SCORE_GUARD = 100.0
AMENDED_SCORE_GUARD = 1000.0
ORIGINAL_RUN_BATCH = production.run_batch
ORIGINAL_FINALIZE = production.finalize
ORIGINAL_COLLECT_BATCHES = production.stage09.collect_batches
AMENDMENT_DESCRIPTOR: dict[str, object] = {}


def validate_amendment(
    root: Path,
    config_path: Path,
    config: dict[str, object],
    amendment_path: Path,
) -> dict[str, object]:
    amendment = production.read_json(amendment_path)
    if amendment.get("status") != "stage11_score_guard_amendment01_frozen":
        raise ValueError("Stage 11 Amendment 01 is not frozen")
    source = dict(amendment["source_config"])
    if production.relative_path(root, config_path) != str(source["path"]):
        raise ValueError("Stage 11 Amendment 01 source config path differs")
    if production.file_sha256(config_path) != str(source["sha256"]).upper():
        raise ValueError("Stage 11 Amendment 01 source config hash differs")
    protocol = dict(config["unidock"])
    if float(protocol["maximum_absolute_score_kcal_per_mol"]) != float(
        amendment["original_score_guard_kcal_per_mol"]
    ):
        raise ValueError("Stage 11 original score guard differs")
    if float(amendment["original_score_guard_kcal_per_mol"]) != ORIGINAL_SCORE_GUARD:
        raise ValueError("Stage 11 Amendment 01 original guard differs")
    if float(amendment["amended_score_guard_kcal_per_mol"]) != AMENDED_SCORE_GUARD:
        raise ValueError("Stage 11 Amendment 01 amended guard differs")
    return amendment


def amended_run_batch(
    root: Path,
    paths: dict[str, Path],
    executable: str,
    receptor: dict[str, str],
    ligands: list[dict[str, str]],
    protocol: dict[str, object],
    seed_id: str,
    base_seed: int,
    signature: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if float(protocol["maximum_absolute_score_kcal_per_mol"]) != ORIGINAL_SCORE_GUARD:
        raise ValueError("Stage 11 recovery received a non-original signature protocol")
    execution_protocol = copy.deepcopy(protocol)
    execution_protocol["maximum_absolute_score_kcal_per_mol"] = AMENDED_SCORE_GUARD
    rows, summary = ORIGINAL_RUN_BATCH(
        root,
        paths,
        executable,
        receptor,
        ligands,
        execution_protocol,
        seed_id,
        base_seed,
        signature,
    )
    outlier_ids = [
        str(row["ligand_id"])
        for row in rows
        if abs(float(row["gpu_score"])) > ORIGINAL_SCORE_GUARD
    ]
    for row in rows:
        row["score_guard_amendment_id"] = AMENDMENT_DESCRIPTOR["amendment_id"]
        row["score_outlier_over_original_guard"] = (
            abs(float(row["gpu_score"])) > ORIGINAL_SCORE_GUARD
        )
    summary["technical_amendment"] = AMENDMENT_DESCRIPTOR
    summary["checkpoint_signature_score_guard_kcal_per_mol"] = ORIGINAL_SCORE_GUARD
    summary["execution_score_guard_kcal_per_mol"] = AMENDED_SCORE_GUARD
    summary["score_outlier_over_original_guard_count"] = len(outlier_ids)
    summary["score_outlier_over_original_guard_ligand_ids"] = outlier_ids
    return rows, summary


def collect_with_original_signature_protocol(
    root: Path,
    config: dict[str, object],
    receptors: list[dict[str, str]],
    ligands: list[dict[str, str]],
    config_sha256: str,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, str]],
]:
    signature_config = copy.deepcopy(config)
    signature_config["unidock"][
        "maximum_absolute_score_kcal_per_mol"
    ] = ORIGINAL_SCORE_GUARD
    return ORIGINAL_COLLECT_BATCHES(
        root,
        signature_config,
        receptors,
        ligands,
        config_sha256,
    )


def amended_finalize(
    root: Path,
    config_path: Path,
    config: dict[str, object],
    receptors: list[dict[str, str]],
    ligands: list[dict[str, str]],
    input_audit: dict[str, object],
    executable_info: dict[str, object] | None,
    executed_batches: int,
    resumed_batches: int,
    invocation_elapsed: float,
    selected_seed_ids: list[str],
    selected_receptor_ids: list[str],
) -> dict[str, object]:
    if float(
        config["unidock"]["maximum_absolute_score_kcal_per_mol"]
    ) != ORIGINAL_SCORE_GUARD:
        raise ValueError("Stage 11 recovery finalizer received a modified config")
    execution_config = copy.deepcopy(config)
    execution_config["unidock"][
        "maximum_absolute_score_kcal_per_mol"
    ] = AMENDED_SCORE_GUARD
    previous_collect = production.stage09.collect_batches
    production.stage09.collect_batches = collect_with_original_signature_protocol
    try:
        result = ORIGINAL_FINALIZE(
            root,
            config_path,
            execution_config,
            receptors,
            ligands,
            input_audit,
            executable_info,
            executed_batches,
            resumed_batches,
            invocation_elapsed,
            selected_seed_ids,
            selected_receptor_ids,
        )
    finally:
        production.stage09.collect_batches = previous_collect

    result["technical_amendment"] = AMENDMENT_DESCRIPTOR
    result["checkpoint_signature_score_guard_kcal_per_mol"] = ORIGINAL_SCORE_GUARD
    result["execution_score_guard_kcal_per_mol"] = AMENDED_SCORE_GUARD
    outputs = dict(config["outputs"])
    output_key = (
        "summary_json"
        if result.get("status") == "stage11_fresh_validation_unidock_matrix_ok"
        else "progress_json"
    )
    production.write_json(
        production.rooted_path(root, str(outputs[output_key])), result
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run(
    config_path: Path,
    amendment_path: Path,
    root: Path,
    unidock: str | None,
    audit_only: bool,
    resume: bool,
    seed_ids: list[str] | None,
    receptor_ids: list[str] | None,
    finalize_only: bool,
) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    amendment_path = amendment_path.resolve()
    config = production.read_json(config_path)
    amendment = validate_amendment(root, config_path, config, amendment_path)
    global AMENDMENT_DESCRIPTOR
    AMENDMENT_DESCRIPTOR = {
        "amendment_id": amendment["amendment_id"],
        "path": production.relative_path(root, amendment_path),
        "sha256": production.file_sha256(amendment_path),
        "policy": amendment["score_policy"],
    }
    previous_run_batch = production.run_batch
    previous_finalize = production.finalize
    production.run_batch = amended_run_batch
    production.finalize = amended_finalize
    try:
        return production.run(
            config_path,
            root,
            unidock,
            audit_only,
            resume,
            seed_ids,
            receptor_ids,
            finalize_only,
        )
    finally:
        production.run_batch = previous_run_batch
        production.finalize = previous_finalize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--unidock", default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed-id", action="append", default=None)
    parser.add_argument("--receptor-id", action="append", default=None)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only and args.finalize_only:
        parser.error("--audit-only and --finalize-only are mutually exclusive")
    result = run(
        args.config,
        args.amendment,
        args.root,
        args.unidock,
        args.audit_only,
        args.resume,
        args.seed_id,
        args.receptor_id,
        args.finalize_only,
    )
    return 0 if result["status"] in {
        "audit_only_ok",
        "stage11_partial_ok",
        "stage11_fresh_validation_unidock_matrix_ok",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
