"""Independently audit the Stage58c PPARD target-id amendment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fingerprint(row: dict[str, str]) -> str:
    payload = json.dumps(
        {key: value for key, value in row.items() if key != "target_id"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def checked(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage58c output identity differs: {path}")
    return path


def run(result_path: Path, root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    result = read_json(result_path.resolve())
    if result.get("status") != "stage58c_ppard_target_id_amendment_ok":
        raise ValueError("Stage58c amendment did not pass")
    config_path = checked(root, result["config"])
    config = read_json(config_path)
    auditor = root / config["implementation"]["independent_auditor"]["path"]
    if auditor.resolve() != Path(__file__).resolve() or sha256(auditor) != config[
        "implementation"
    ]["independent_auditor"]["sha256"]:
        raise ValueError("Stage58c auditor identity differs")
    source_path = checked(root, result["source"])
    amended_path = checked(root, result["output"])
    source = read_csv(source_path)
    amended = read_csv(amended_path)
    if len(source) != 8352 or len(amended) != 8352:
        raise ValueError("Stage58c audit row count differs")
    if {row["target_id"] for row in source} != {"MK14"} or {
        row["target_id"] for row in amended
    } != {"PPARD"}:
        raise ValueError("Stage58c target-id transition differs")
    if [fingerprint(row) for row in source] != [fingerprint(row) for row in amended]:
        raise ValueError("Stage58c changed a non-target field")
    changed = sum(
        sum(left[key] != right[key] for key in left)
        for left, right in zip(source, amended, strict=True)
    )
    if changed != 8352:
        raise ValueError("Stage58c changed-field count differs")
    audit = {
        "schema_version": "1.0",
        "status": "stage58c_ppard_target_id_amendment_independent_audit_ok",
        "source_result": descriptor(root, result_path.resolve()),
        "source_scores": descriptor(root, source_path),
        "corrected_scores": descriptor(root, amended_path),
        "row_count": 8352,
        "changed_field_count": 8352,
        "changed_columns": ["target_id"],
        "target_id_before": ["MK14"],
        "target_id_after": ["PPARD"],
        "all_non_target_fields_exact": True,
        "docking_scores_changed": 0,
        "pose_fields_changed": 0,
        "gpu_redocking_required": False,
        "stage59_preregistered_pilot_gate_authorized": True,
        "data_boundary": result["data_boundary"],
        "interpretation_boundary": result["interpretation_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.result, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
