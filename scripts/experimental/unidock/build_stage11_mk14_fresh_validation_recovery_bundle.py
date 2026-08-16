"""Merge the Stage 11 input and failure archives into one recovery bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from scripts.prepare_receptor import file_sha256
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from scripts.prepare_receptor import file_sha256


INPUT_ARCHIVE_SHA256 = (
    "F7CA1C8EAD79AAA072196FADF51ED65BEB2D3506B7E4AA3C40719DB68AD070FB"
)
FAILURE_ARCHIVE_SHA256 = (
    "43F9D117ED63BCFEC57184E501EB7E2ED2F090CAC3E13E6910918CE05EECA8DB"
)
SOURCE_CONFIG_SHA256 = (
    "1011825B132B3C254B36CA4A004BA4BE5DFA6BC613F386BD6437818BBEF2EBCC"
)
CONFIG = "configs/stage11_mk14_fresh_validation_unidock113_confirmation.json"
AMENDMENT = "configs/stage11_mk14_fresh_validation_score_guard_amendment01.json"
RECOVERY_FILES = (
    AMENDMENT,
    "scripts/experimental/unidock/run_stage11_mk14_fresh_validation_recovery_amendment01.py",
    "scripts/experimental/unidock/run_stage11_mk14_fresh_validation_recovery_remote.sh",
    "scripts/experimental/unidock/build_stage11_mk14_fresh_validation_recovery_bundle.py",
    "reports/stage-11/mk14_fresh_validation_score_guard_amendment01_recovery.md",
)


def safe_member_path(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe recovery archive path: {name}")
    return path.as_posix()


def merge_archive(archive_path: Path, staging: Path) -> int:
    count = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            name = safe_member_path(member.name)
            if name == "bundle_manifest.sha256" or member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"unsupported recovery archive member: {name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"recovery archive member is unreadable: {name}")
            destination = staging.joinpath(*PurePosixPath(name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            count += 1
    return count


def validate_staging(staging: Path) -> dict[str, object]:
    config_path = staging / CONFIG
    if file_sha256(config_path) != SOURCE_CONFIG_SHA256:
        raise ValueError("recovery staging source config differs")
    amendment = json.loads((staging / AMENDMENT).read_text(encoding="ascii"))
    runner = dict(amendment["recovery_runner"])
    if file_sha256(staging / str(runner["path"])) != str(
        runner["sha256"]
    ).upper():
        raise ValueError("recovery staging runner identity differs")
    batch_summaries = list(
        staging.glob(
            "results/runs/stage11_mk14_fresh_validation_unidock113_confirmation/"
            "batches/*/*/batch_summary.json"
        )
    )
    score_ledgers = list(
        staging.glob(
            "results/runs/stage11_mk14_fresh_validation_unidock113_confirmation/"
            "batches/*/*/scores.csv"
        )
    )
    poses = list(
        staging.glob(
            "results/runs/stage11_mk14_fresh_validation_unidock113_confirmation/"
            "batches/*/*/poses/*_out.pdbqt"
        )
    )
    logs = list(
        staging.glob(
            "results/runs/stage11_mk14_fresh_validation_unidock113_confirmation/"
            "batches/*/*/unidock.log"
        )
    )
    if len(batch_summaries) != 8 or len(score_ledgers) != 8:
        raise ValueError("recovery staging complete checkpoint count differs")
    if len(poses) != 14184 or len(logs) != 9:
        raise ValueError("recovery staging diagnostic pose count differs")
    forbidden_outputs = (
        "results/runs/stage11_mk14_fresh_validation_unidock113_confirmation/summary.json",
        "data/stage11_mk14_fresh_validation_unidock113_confirmation_audit.json",
        "data/stage11_mk14_fresh_validation_unidock113_confirmation_result.json",
    )
    if any((staging / value).exists() for value in forbidden_outputs):
        raise ValueError("recovery staging unexpectedly contains a final result")
    return {
        "complete_checkpoint_count": len(batch_summaries),
        "complete_checkpoint_pose_count": len(batch_summaries) * 1576,
        "archived_pose_count": len(poses),
        "archived_unidock_log_count": len(logs),
        "interrupted_complete_pose_batch_count": len(logs) - len(batch_summaries),
    }


def build(
    root: Path,
    input_archive: Path,
    failure_archive: Path,
    output: Path,
) -> dict[str, object]:
    root = root.resolve()
    input_archive = input_archive.resolve()
    failure_archive = failure_archive.resolve()
    output = output.resolve()
    if file_sha256(input_archive) != INPUT_ARCHIVE_SHA256:
        raise ValueError("Stage 11 source input archive identity differs")
    if file_sha256(failure_archive) != FAILURE_ARCHIVE_SHA256:
        raise ValueError("Stage 11 failure archive identity differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="stage11_recovery_", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        input_file_count = merge_archive(input_archive, staging)
        failure_file_count = merge_archive(failure_archive, staging)
        for relative in RECOVERY_FILES:
            source = root / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        audit = validate_staging(staging)
        relative_paths = sorted(
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file()
        )
        result = write_bundle(staging, output, relative_paths)
    result.update(
        {
            "operation": "Stage 11 Amendment 01 one-file recovery bundle",
            "source_input_archive": {
                "path": input_archive.as_posix(),
                "sha256": INPUT_ARCHIVE_SHA256,
                "merged_file_count": input_file_count,
            },
            "source_failure_archive": {
                "path": failure_archive.as_posix(),
                "sha256": FAILURE_ARCHIVE_SHA256,
                "merged_file_count": failure_file_count,
            },
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "amendment_id": "stage11-mk14-fresh-validation-score-guard-amendment01",
            "original_score_guard_kcal_per_mol": 100.0,
            "amended_score_guard_kcal_per_mol": 1000.0,
            "checkpoint_audit": audit,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--input-archive", type=Path, required=True)
    parser.add_argument("--failure-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.root,
        args.input_archive,
        args.failure_archive,
        args.output,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
