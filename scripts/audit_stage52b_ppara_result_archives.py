"""Audit downloaded Stage 52b PPARA core and diagnostic result trees."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def checked_descriptor(root: Path, descriptor: dict[str, Any]) -> Path:
    path = root / str(descriptor["path"])
    if not path.is_file() or sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"output descriptor differs: {path}")
    return path


def matrix_values(path: Path) -> tuple[list[str], dict[tuple[str, str], float]]:
    rows = read_csv(path)
    receptors = [key for key in rows[0] if key not in {"ligand_id", "label", "selection_role"}]
    values = {
        (row["ligand_id"], receptor): float(row[receptor])
        for row in rows
        for receptor in receptors
    }
    return receptors, values


def verify_pose_hash(item: tuple[Path, str]) -> bool:
    path, expected = item
    return path.is_file() and sha256(path) == expected.upper()


def run(
    core: Path,
    diagnostics: Path,
    core_archive: Path,
    diagnostics_archive: Path,
    output: Path,
) -> dict[str, Any]:
    core = core.resolve()
    diagnostics = diagnostics.resolve()
    run_rel = Path("results/runs/stage52b_ppara_train374_unidock113_production")
    summary = read_json(core / run_rel / "summary.json")
    remote_audit = read_json(
        core / "data/stage52b_ppara_train374_unidock113_production_audit.json"
    )
    if summary["status"] != "stage52b_ppara_train374_unidock_matrix_ok":
        raise ValueError("Stage 52b summary did not pass")
    if remote_audit["status"] != (
        "independent_stage52b_ppara_train374_unidock_matrix_audit_ok"
    ):
        raise ValueError("Stage 52b remote independent audit did not pass")

    outputs = summary["outputs"]
    scores_path = checked_descriptor(core, outputs["scores_csv"])
    batches_path = checked_descriptor(core, outputs["batch_runs_csv"])
    median_path = checked_descriptor(core, outputs["median_matrix_csv"])
    minimum_path = checked_descriptor(core, outputs["minimum_matrix_csv"])
    checked_descriptor(core, outputs["progress_json"])
    rows = read_csv(scores_path)
    batches = read_csv(batches_path)
    if len(rows) != 22440 or len(batches) != 60:
        raise ValueError("Stage 52b result dimensions differ")
    keys = {(row["seed_id"], row["receptor_id"], row["ligand_id"]) for row in rows}
    if len(keys) != 22440:
        raise ValueError("Stage 52b result keys are not unique")
    if Counter(row["seed_id"] for row in rows) != Counter(
        {"seed0": 7480, "seed1": 7480, "seed2": 7480}
    ):
        raise ValueError("Stage 52b seed dimensions differ")
    receptor_ids = list(summary["input_audit"]["receptor_ids"])
    if set(row["receptor_id"] for row in rows) != set(receptor_ids):
        raise ValueError("Stage 52b receptor identities differ")
    if Counter(row["label"] for row in rows) != Counter(
        {"active": 11220, "decoy": 11220}
    ):
        raise ValueError("Stage 52b score labels differ")
    if any(
        row["status"] != "ok"
        or row["pose_integrity_status"] != "ok"
        or int(row["pose_count"]) != 1
        or not math.isfinite(float(row["gpu_score"]))
        for row in rows
    ):
        raise ValueError("Stage 52b score-row technical gate differs")
    if any(
        row["status"] != "ok"
        or int(row["unresolved_warning_event_count"]) != 0
        or int(row["pose_integrity_failure_count"]) != 0
        for row in batches
    ):
        raise ValueError("Stage 52b batch technical gate differs")

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["ligand_id"], row["receptor_id"])].append(float(row["gpu_score"]))
    if len(grouped) != 7480 or any(len(values) != 3 for values in grouped.values()):
        raise ValueError("Stage 52b seed aggregation grid differs")
    median_receptors, observed_median = matrix_values(median_path)
    minimum_receptors, observed_minimum = matrix_values(minimum_path)
    if median_receptors != receptor_ids or minimum_receptors != receptor_ids:
        raise ValueError("Stage 52b matrix receptor order differs")
    for key, values in grouped.items():
        ordered = sorted(values)
        if abs(observed_median[key] - ordered[1]) > 1e-12:
            raise ValueError(f"Stage 52b median differs: {key}")
        if abs(observed_minimum[key] - ordered[0]) > 1e-12:
            raise ValueError(f"Stage 52b minimum differs: {key}")

    pose_items = [
        (
            diagnostics / row["output_pose_path"],
            row["output_pose_sha256"],
        )
        for row in rows
    ]
    with ThreadPoolExecutor(max_workers=16) as executor:
        pose_hash_matches = list(executor.map(verify_pose_hash, pose_items, chunksize=64))
    pose_hash_mismatch_count = pose_hash_matches.count(False)
    if pose_hash_mismatch_count:
        raise ValueError(f"Stage 52b pose hash mismatches: {pose_hash_mismatch_count}")

    observed_target_ids = sorted({row["target_id"] for row in rows})
    expected_target_id = "PPARA"
    target_id_metadata_defect = observed_target_ids != [expected_target_id]
    if observed_target_ids != ["MK14"]:
        raise ValueError("Stage 52b target-id defect is not the known helper defect")
    result = {
        "schema_version": "1.0",
        "status": "stage52b_ppara_downloaded_result_audit_ok_with_metadata_amendment_required",
        "source_archives": {
            "core": {
                "filename": core_archive.name,
                "sha256": sha256(core_archive),
                "size_bytes": core_archive.stat().st_size,
            },
            "diagnostics": {
                "filename": diagnostics_archive.name,
                "sha256": sha256(diagnostics_archive),
                "size_bytes": diagnostics_archive.stat().st_size,
            },
        },
        "technical_integrity": {
            "batch_count": len(batches),
            "pair_count": len(rows),
            "unique_key_count": len(keys),
            "pose_file_count": len(pose_items),
            "pose_hash_mismatch_count": 0,
            "unresolved_warning_event_count": 0,
            "pose_integrity_failure_count": 0,
            "median_matrix_exact": True,
            "minimum_matrix_exact": True,
        },
        "runtime": {
            "batch_elapsed_seconds": summary["batch_elapsed_seconds"],
            "current_invocation_elapsed_seconds": summary[
                "current_invocation_elapsed_seconds"
            ],
        },
        "metadata_adjudication": {
            "target_id_metadata_defect": target_id_metadata_defect,
            "observed_target_ids": observed_target_ids,
            "expected_target_id": expected_target_id,
            "root_cause": (
                "run_unidock_gpu_equivalence.run_batch emitted the historical "
                "hard-coded MK14 target_id for a PPARA run"
            ),
            "score_or_pose_effect": False,
            "gpu_redocking_required": False,
            "required_action": (
                "create a non-destructive derived score table changing only target_id "
                "from MK14 to PPARA before downstream target-grouped analysis"
            ),
        },
        "data_boundary": summary["data_boundary"],
        "decision": {
            "stage52c_metadata_amendment_authorized": True,
            "stage53_train_only_method_comparison_authorized_after_amendment": True,
            "stage51_confirmatory_status_changed": False,
            "fresh_validation_authorized": False,
            "quantum_hardware_authorized": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--core-archive", type=Path, required=True)
    parser.add_argument("--diagnostics-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.core,
        args.diagnostics,
        args.core_archive,
        args.diagnostics_archive,
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
