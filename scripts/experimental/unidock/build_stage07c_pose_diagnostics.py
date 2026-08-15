"""Build the Stage 07c warning-focused pose diagnostics archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from .run_stage07c_unidock_warning_adjudication import (
        file_sha256,
        read_csv,
        read_json,
        rooted_path,
        validate_config,
        verify_implementation,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from run_stage07c_unidock_warning_adjudication import (
        file_sha256,
        read_csv,
        read_json,
        rooted_path,
        validate_config,
        verify_implementation,
    )


CONFIG = "configs/stage07c_mk14_unidock113_warning_adjudication.json"


def relative_files(root: Path, directory: Path) -> list[str]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return [
        str(path.relative_to(root)).replace("\\", "/")
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def diagnostic_paths(
    root: Path, config: dict[str, object]
) -> tuple[list[str], list[dict[str, object]]]:
    outputs = dict(config["outputs"])
    run_directory = rooted_path(root, str(outputs["run_directory"]))
    batch_path = rooted_path(root, str(outputs["batch_runs_csv"]))
    batch_rows = read_csv(batch_path)
    affected = [
        row
        for row in batch_rows
        if int(row["known_warning_event_count"]) > 0
        or int(row["unresolved_warning_event_count"]) > 0
        or int(row["pose_integrity_failure_count"]) > 0
    ]
    fixed = [
        CONFIG,
        str(outputs["batch_runs_csv"]),
        str(outputs["run_summary_json"]),
        str(outputs["evaluation_result_json"]),
        str(outputs["evaluation_report_md"]),
        str(outputs["replay_comparison_csv"]),
    ]
    paths = [path for path in fixed if rooted_path(root, path).is_file()]
    affected_metadata: list[dict[str, object]] = []
    for row in affected:
        batch_directory = (
            run_directory
            / row["run_role"]
            / "batches"
            / row["run_id"]
            / row["receptor_id"]
        )
        paths.extend(relative_files(root, batch_directory))
        affected_metadata.append(
            {
                "run_role": row["run_role"],
                "run_id": row["run_id"],
                "receptor_id": row["receptor_id"],
                "known_warning_event_count": int(
                    row["known_warning_event_count"]
                ),
                "unresolved_warning_event_count": int(
                    row["unresolved_warning_event_count"]
                ),
                "pose_integrity_failure_count": int(
                    row["pose_integrity_failure_count"]
                ),
            }
        )
    receptor_manifest = str(config["inputs"]["receptor_manifest"]["path"])
    ligand_manifest = str(config["inputs"]["ligand_manifest"]["path"])
    paths.extend((receptor_manifest, ligand_manifest))
    if affected:
        paths.extend(
            manifest_paths(root, receptor_manifest, "receptor_pdbqt")
        )
        paths.extend(manifest_paths(root, ligand_manifest, "pdbqt_path"))
    environment_directory = run_directory / "environment"
    if environment_directory.is_dir():
        paths.extend(relative_files(root, environment_directory))
    return sorted(set(path.replace("\\", "/") for path in paths)), affected_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = read_json(root / CONFIG)
    validate_config(config)
    verify_implementation(config, "pose_diagnostics_builder", Path(__file__))
    paths, affected = diagnostic_paths(root, config)
    result = write_bundle(root, args.output, paths)
    result.update(
        {
            "operation": "Stage 07c warning-focused pose diagnostics bundle",
            "status": "findings_archived" if affected else "no_findings",
            "affected_batch_count": len(affected),
            "affected_batches": affected,
            "fresh_validation_rows": 0,
            "test_rows": 0,
            "config_sha256": file_sha256(root / CONFIG),
        }
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
