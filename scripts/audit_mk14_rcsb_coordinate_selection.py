"""Independently audit a MAPK14 max-min coordinate-selection result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def reconstruct_maxmin(
    eligible_ids: list[str],
    existing_ids: list[str],
    distance_by_pair: dict[tuple[str, str], float],
    count: int,
) -> list[dict[str, object]]:
    selected = list(existing_ids)
    remaining = sorted(set(eligible_ids) - set(existing_ids))
    result: list[dict[str, object]] = []
    for rank in range(1, count + 1):
        scored: list[tuple[float, str]] = []
        for candidate in remaining:
            distances = [
                distance_by_pair[tuple(sorted((candidate, selected_id)))]
                for selected_id in selected
            ]
            scored.append((min(distances), candidate))
        if not scored:
            raise ValueError("too few candidates during independent max-min audit")
        minimum_distance, chosen = sorted(
            scored, key=lambda value: (-value[0], value[1])
        )[0]
        result.append(
            {
                "selection_rank": rank,
                "conformer_id": chosen,
                "minimum_standardized_distance_to_selected_pool": minimum_distance,
            }
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return result


def checked_record(record: dict[str, object]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(record["sha256"]).upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return path


def run_audit(config_path: Path) -> dict[str, object]:
    config = read_json(config_path)
    inputs = config.get("inputs")
    expected = config.get("expected")
    if not isinstance(inputs, dict) or not isinstance(expected, dict):
        raise ValueError("audit config is missing inputs or expected values")
    paths = {
        key: checked_record(record)
        for key, record in inputs.items()
        if isinstance(record, dict)
    }
    required = {"selection_summary", "eligible_pool", "pairwise_distances", "selected_manifest"}
    if set(paths) != required:
        raise ValueError("audit inputs differ from the required set")

    source_summary = read_json(paths["selection_summary"])
    if source_summary.get("status") != "expanded8_structural_selection_ok":
        raise ValueError("source coordinate selection did not pass")
    boundary = source_summary.get("data_boundary")
    if not isinstance(boundary, dict) or any(int(value) != 0 for value in boundary.values()):
        raise ValueError("source selection crossed a frozen data boundary")

    eligible_rows = read_csv(paths["eligible_pool"])
    eligible_ids = sorted(row["conformer_id"] for row in eligible_rows)
    distance_rows = read_csv(paths["pairwise_distances"])
    distance_by_pair = {
        tuple(sorted((row["conformer_id_a"], row["conformer_id_b"]))): float(
            row["standardized_pocket_distance"]
        )
        for row in distance_rows
    }
    selected_rows = read_csv(paths["selected_manifest"])
    existing_rows = [row for row in selected_rows if row["pool_role"] == "existing_seed"]
    new_rows = [
        row for row in selected_rows if row["pool_role"] == "new_maxmin_addition"
    ]
    existing_rows.sort(key=lambda row: int(row["selection_rank"]))
    new_rows.sort(key=lambda row: int(row["selection_rank"]))
    existing_ids = [row["conformer_id"] for row in existing_rows]
    expected_existing = [str(value) for value in expected["existing_receptor_ids"]]
    if existing_ids != expected_existing:
        raise ValueError("existing receptor seed order differs")

    reconstructed = reconstruct_maxmin(
        eligible_ids,
        existing_ids,
        distance_by_pair,
        int(expected["new_receptor_count"]),
    )
    observed_ids = [row["conformer_id"] for row in new_rows]
    reconstructed_ids = [str(row["conformer_id"]) for row in reconstructed]
    if observed_ids != reconstructed_ids:
        raise ValueError("independent max-min receptor IDs differ")
    for observed, rebuilt in zip(new_rows, reconstructed):
        difference = abs(
            float(observed["minimum_standardized_distance_to_selected_pool"])
            - float(rebuilt["minimum_standardized_distance_to_selected_pool"])
        )
        if difference > 1e-12:
            raise ValueError("independent max-min distance differs")

    if len(eligible_rows) != int(expected["eligible_pool_count"]):
        raise ValueError("eligible pool count differs")
    if len(selected_rows) != int(expected["final_selected_pool_count"]):
        raise ValueError("final selected pool count differs")
    source_output_hashes = source_summary.get("outputs")
    if not isinstance(source_output_hashes, dict):
        raise ValueError("source output hashes are missing")
    for key, input_key in {
        "eligible_pool_manifest_csv": "eligible_pool",
        "pairwise_distances_csv": "pairwise_distances",
        "selected_expansion_manifest_csv": "selected_manifest",
    }.items():
        record = source_output_hashes.get(key)
        if not isinstance(record, dict) or str(record.get("sha256", "")).upper() != file_sha256(
            paths[input_key]
        ):
            raise ValueError(f"source summary output hash differs: {key}")

    output_path = Path(str(config["output_json"]))
    result = {
        "schema_version": "1.0",
        "audit_id": config["audit_id"],
        "status": "independent_coordinate_selection_audit_ok",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_summary": {
            "path": paths["selection_summary"].as_posix(),
            "sha256": file_sha256(paths["selection_summary"]),
        },
        "eligible_pool_count": len(eligible_rows),
        "pairwise_distance_count": len(distance_rows),
        "existing_receptor_ids": existing_ids,
        "independently_reconstructed_new_receptors": reconstructed,
        "final_selected_receptor_ids": existing_ids + reconstructed_ids,
        "data_boundary": boundary,
        "interpretation_boundary": "This audit verifies file identity, frozen data boundaries, and deterministic max-min reconstruction only. It does not validate docking or enrichment.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_audit(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
