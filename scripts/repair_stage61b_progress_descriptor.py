"""Repair the stale Stage61b progress descriptor after successful finalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.experimental.unidock import run_stage61b_ppard_remaining144_production as runner


EXPECTED_CONFIG_SHA256 = "644A4A1B42FA4526A4A297FB775F37519AB4CEF9E6BF14C19D5DC1EFB1764019"


def repair(root: Path, config_path: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path if config_path.is_absolute() else root / config_path
    config = runner.common.read_json(config_path)
    config_sha256 = runner.common.file_sha256(config_path)
    if config_sha256 != EXPECTED_CONFIG_SHA256:
        raise ValueError("Stage61b config hash is not the approved executed config")
    outputs = dict(config["outputs"])
    summary_path = runner.common.rooted_path(root, str(outputs["summary_json"]))
    progress_path = runner.common.rooted_path(root, str(outputs["progress_json"]))
    summary = runner.common.read_json(summary_path)
    progress = runner.common.read_json(progress_path)
    if summary.get("status") != "stage61b_ppard_remaining144_unidock_matrix_ok":
        raise ValueError("Stage61b matrix did not complete before metadata repair")
    if str(summary["config"]["sha256"]).upper() != config_sha256:
        raise ValueError("Stage61b summary config hash differs")
    if int(summary["batch_count"]) != 87 or int(summary["pair_count"]) != 12528:
        raise ValueError("Stage61b completed dimensions differ")
    if int(summary["unresolved_warning_event_count"]) != 0 or int(
        summary["pose_integrity_failure_count"]
    ) != 0:
        raise ValueError("Stage61b technical gate did not pass")
    if progress.get("status") != "stage61b_production_complete":
        raise ValueError("Stage61b progress status is not the finalized status")
    source_outputs = dict(summary["outputs"])
    for key in (
        "scores_csv", "batch_runs_csv", "median_matrix_csv", "minimum_matrix_csv"
    ):
        runner.common.verified_path(root, dict(source_outputs[key]))

    before_summary_sha256 = runner.common.file_sha256(summary_path)
    before = dict(source_outputs["progress_json"])
    after = runner.common.output_descriptor(root, progress_path)
    if before["path"] != after["path"]:
        raise ValueError("Stage61b progress descriptor path differs")
    changed_fields = [
        key for key in ("sha256", "size_bytes") if before.get(key) != after.get(key)
    ]
    summary["outputs"]["progress_json"] = after
    runner.common.write_json(summary_path, summary)
    amendment = {
        "schema_version": "1.0",
        "amendment_id": "stage61b-ppard-progress-descriptor-amendment01-v1",
        "status": "stage61b_progress_descriptor_amendment01_ok",
        "operation": "metadata-only repair of the stale progress.json output descriptor",
        "config": {
            "path": runner.common.relative_path(root, config_path),
            "sha256": config_sha256,
        },
        "summary": {
            "path": runner.common.relative_path(root, summary_path),
            "sha256_before": before_summary_sha256,
            "sha256_after": runner.common.file_sha256(summary_path),
        },
        "progress_descriptor_before": before,
        "progress_descriptor_after": after,
        "changed_descriptor_fields": changed_fields,
        "docking_score_fields_changed": 0,
        "pose_files_changed": 0,
        "batch_files_changed": 0,
        "docking_jobs_reexecuted": 0,
        "verified_batch_count": 87,
        "verified_pair_count": 12528,
        "data_boundary": {
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "quantum_hardware_jobs": 0,
        },
        "next_step": "rerun the unchanged independent Stage61b matrix auditor",
    }
    output = output if output.is_absolute() else root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    runner.common.write_json(output, amendment)
    print(json.dumps(amendment, indent=2, sort_keys=True))
    return amendment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage61b_ppard_remaining144_unidock113_production.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage61b_ppard_progress_descriptor_amendment01.json"),
    )
    args = parser.parse_args()
    repair(args.root, args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
