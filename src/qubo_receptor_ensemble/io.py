"""Shared file, JSON, CSV, and hashing helpers.

These are the canonical implementations consolidated from duplicated
definitions that previously lived in many ``scripts/*.py`` modules.
Behavior matches the dominant variant used across the repository; scripts
whose local variant was semantically identical now import from here.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Return the uppercase SHA-256 hex digest of a file, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, object]:
    """Read a JSON object from ``path``; reject non-object roots."""
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    """Write ``value`` as sorted, indented ASCII JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV table as a list of rows, tolerating a UTF-8 BOM."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows as CSV with ordered union field names; refuse empty input."""
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_filename(text: str) -> str:
    """Sanitize arbitrary text into a filesystem-safe name."""
    keep = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)
