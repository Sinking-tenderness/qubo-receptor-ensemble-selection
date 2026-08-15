"""Repair the Stage58b progress descriptor after successful production."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.experimental.unidock import audit_stage09_mk14_train696_production as prior
from scripts.experimental.unidock import run_stage58b_ppard_pilot96_production as runner


FROZEN_CONFIG_SHA256 = "8A30F6762577C0AA25D325B418AD29855F4881DD6D9C5C854D76A9A9F10C7D76"
FROZEN_ADAPTER_SHA256 = "CEA229B53A052F2ADED432941C2CE848863E19748376D663E99D141FD3B40105"


def run(config_path: Path, root: Path) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    if runner.common.file_sha256(config_path) != FROZEN_CONFIG_SHA256:
        raise ValueError("Stage58b recovery requires the executed frozen config")
    config = runner.common.read_json(config_path)
    adapter = dict(config["implementation"])["production_adapter"]
    adapter_path = runner.common.rooted_path(root, str(adapter["path"]))
    if (
        str(adapter["sha256"]).upper() != FROZEN_ADAPTER_SHA256
        or runner.common.file_sha256(adapter_path) != FROZEN_ADAPTER_SHA256
    ):
        raise ValueError("Stage58b recovery requires the executed frozen adapter")

    outputs = dict(config["outputs"])
    summary_path = runner.common.rooted_path(root, str(outputs["summary_json"]))
    progress_path = runner.common.rooted_path(root, str(outputs["progress_json"]))
    summary = runner.common.read_json(summary_path)
    progress = runner.common.read_json(progress_path)
    if summary.get("status") != "stage58b_ppard_pilot96_unidock_matrix_ok":
        raise ValueError("Stage58b docking did not reach the PPARD completion status")
    if progress.get("status") != "stage58b_production_complete":
        raise ValueError("Stage58b progress ledger is not complete")
    if (
        int(summary["batch_count"]) != 87
        or int(summary["pair_count"]) != 8352
        or int(progress["completed_batch_count"]) != 87
        or int(progress["completed_pair_count"]) != 8352
        or int(progress["missing_batch_count"]) != 0
    ):
        raise ValueError("Stage58b completion dimensions differ")
    if int(summary["unresolved_warning_event_count"]) != 0 or int(
        summary["pose_integrity_failure_count"]
    ) != 0:
        raise ValueError("Stage58b technical gate did not pass")
    if str(summary["config"]["sha256"]).upper() != FROZEN_CONFIG_SHA256:
        raise ValueError("Stage58b summary references a different config")

    source_outputs = dict(summary["outputs"])
    for key in ("scores_csv", "batch_runs_csv", "median_matrix_csv", "minimum_matrix_csv"):
        prior.checked_output(root, dict(source_outputs[key]))
    previous = dict(source_outputs["progress_json"])
    current = runner.common.output_descriptor(root, progress_path)
    if str(previous["path"]) != str(current["path"]):
        raise ValueError("Stage58b progress path differs")

    repaired = str(previous["sha256"]).upper() != str(current["sha256"]).upper()
    summary["outputs"]["progress_json"] = current
    summary["packaging_recovery_amendment"] = {
        "amendment_id": "stage58b-ppard-packaging-recovery-amendment01-v1",
        "scope": "summary progress descriptor only; docking scores, poses, protocol, and batch ledgers unchanged",
        "reason": "the PPARD status rewrite changed progress.json after its original descriptor was recorded",
        "previous_progress_sha256": str(previous["sha256"]).upper(),
        "current_progress_sha256": str(current["sha256"]).upper(),
        "descriptor_changed": repaired,
    }
    runner.common.write_json(summary_path, summary)
    result = {
        "schema_version": "1.0",
        "status": "stage58b_packaging_metadata_repaired",
        "config_sha256": FROZEN_CONFIG_SHA256,
        "completed_batch_count": 87,
        "completed_pair_count": 8352,
        "docking_batches_rerun": 0,
        "docking_scores_changed": 0,
        "pose_files_changed": 0,
        "summary_progress_descriptor_changed": repaired,
        "previous_progress_sha256": str(previous["sha256"]).upper(),
        "current_progress_sha256": str(current["sha256"]).upper(),
        "summary": runner.common.output_descriptor(root, summary_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage58b_ppard_pilot96_unidock113_production.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
