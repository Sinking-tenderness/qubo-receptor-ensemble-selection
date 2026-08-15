"""Package independently audited Stage61b results after metadata recovery."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from scripts.experimental.unidock import run_stage61b_ppard_remaining144_production as runner


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


def files_under(path: Path) -> list[Path]:
    return sorted(value for value in path.rglob("*") if value.is_file())


def write_archive(root: Path, output: Path, paths: list[Path]) -> dict[str, Any]:
    root = root.resolve()
    unique = sorted({path.resolve() for path in paths}, key=lambda value: value.as_posix())
    for path in unique:
        if not path.is_file() or root not in path.parents:
            raise ValueError(f"invalid Stage61b package path: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in unique:
                    name = path.relative_to(root).as_posix()
                    info = archive.gettarinfo(str(path), arcname=name)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    return {
        "path": str(output),
        "sha256": sha256(output),
        "size_bytes": output.stat().st_size,
        "file_count": len(unique),
    }


def run(root: Path, output_root: Path) -> dict[str, Any]:
    root = root.resolve()
    output_root = output_root.resolve()
    config_path = root / "configs/stage61b_ppard_remaining144_unidock113_production.json"
    config = runner.common.read_json(config_path)
    outputs = dict(config["outputs"])
    summary_path = runner.common.rooted_path(root, str(outputs["summary_json"]))
    audit_path = runner.common.rooted_path(root, str(outputs["audit_json"]))
    amendment_path = root / "data/stage61b_ppard_progress_descriptor_amendment01.json"
    summary = runner.common.read_json(summary_path)
    audit = runner.common.read_json(audit_path)
    amendment = runner.common.read_json(amendment_path)
    if summary.get("status") != "stage61b_ppard_remaining144_unidock_matrix_ok":
        raise ValueError("Stage61b summary did not pass")
    if audit.get("status") != "independent_stage61b_ppard_remaining144_unidock_matrix_audit_ok":
        raise ValueError("Stage61b independent audit did not pass")
    if amendment.get("status") != "stage61b_progress_descriptor_amendment01_ok":
        raise ValueError("Stage61b metadata amendment did not pass")
    for descriptor in dict(summary["outputs"]).values():
        runner.common.verified_path(root, dict(descriptor))
    if int(audit["pair_count"]) != 12528 or int(audit["batch_count"]) != 87:
        raise ValueError("Stage61b audit dimensions differ")
    if int(audit["unresolved_warning_event_count"]) != 0 or int(
        audit["pose_integrity_failure_count"]
    ) != 0:
        raise ValueError("Stage61b audit technical gate differs")

    run_directory = runner.common.rooted_path(root, str(outputs["run_directory"]))
    fixed = [
        config_path,
        summary_path,
        audit_path,
        amendment_path,
        root / "data/processed/stage58b_ppard_stage57_passing29_receptor_manifest.csv",
        root / "data/stage58b_ppard_stage57_passing29_receptor_manifest_summary.json",
        root / "data/stage60_ppard_transferred_qubo_freeze_result.json",
        root / "data/stage60_ppard_transferred_qubo_freeze_audit.json",
        root / "data/stage60_ppard_transferred_qubo_model_record.json",
        root / "data/stage61a_ppard_remaining144_unidock_input_audit.json",
        root / "data/processed/stage61a_ppard_remaining144_unidock_pdbqt_manifest.csv",
        root / "data/stage61a_ppard_remaining144_unidock_input_summary.json",
    ]
    core = fixed + [
        runner.common.rooted_path(root, str(outputs[key]))
        for key in (
            "progress_json", "scores_csv", "batch_runs_csv",
            "median_matrix_csv", "minimum_matrix_csv",
        )
    ]
    environment = run_directory / "environment"
    if environment.is_dir():
        core.extend(files_under(environment))
    for path in files_under(run_directory / "batches"):
        if path.name in {"scores.csv", "batch_summary.json", "unidock.log"}:
            core.append(path)
    diagnostic = fixed + files_under(run_directory)
    core_output = output_root / "stage61b_ppard_remaining144_unidock113_production_core_v1.tar.gz"
    diagnostic_output = output_root / "stage61b_ppard_remaining144_unidock113_production_diagnostics_v1.tar.gz"
    result = {
        "schema_version": "1.0",
        "status": "stage61b_recovered_result_packaging_ok",
        "core": write_archive(root, core_output, core),
        "diagnostics": write_archive(root, diagnostic_output, diagnostic),
        "batch_count": 87,
        "pair_count": 12528,
        "docking_jobs_reexecuted": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("/root/autodl-tmp"))
    args = parser.parse_args()
    run(args.root, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
