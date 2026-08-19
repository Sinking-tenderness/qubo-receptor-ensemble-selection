"""Stable artifact and stage-manifest helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .io import file_sha256, write_json


def artifact_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.resolve().as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def write_artifact(path: Path, value: object) -> dict[str, object]:
    write_json(path, value)
    return artifact_record(path)


def write_stage_manifest(
    path: Path,
    *,
    stage: str,
    status: str,
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
    details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "stage": stage,
        "status": status,
        "inputs": dict(inputs),
        "outputs": dict(outputs),
    }
    if details:
        manifest["details"] = dict(details)
    write_json(path, manifest)
    return manifest
