"""Independently audit the frozen Stage 08 MAPK14 16-conformer pool."""

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


def checked_record(record: dict[str, object]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(record["sha256"]).upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return path


def independent_maxmin(
    eligible_ids: list[str],
    seed_ids: list[str],
    distance_rows: list[dict[str, str]],
    count: int,
) -> list[dict[str, object]]:
    distances: dict[frozenset[str], float] = {}
    for row in distance_rows:
        pair = frozenset((row["conformer_id_a"], row["conformer_id_b"]))
        if len(pair) != 2 or pair in distances:
            raise ValueError("pairwise distance identity is invalid")
        distances[pair] = float(row["standardized_pocket_distance"])
    expected_pairs = len(eligible_ids) * (len(eligible_ids) - 1) // 2
    if len(distances) != expected_pairs:
        raise ValueError("pairwise distance table is incomplete")

    chosen = list(seed_ids)
    available = sorted(set(eligible_ids).difference(chosen))
    result: list[dict[str, object]] = []
    for rank in range(1, count + 1):
        scored = {
            candidate: min(
                distances[frozenset((candidate, existing))] for existing in chosen
            )
            for candidate in available
        }
        winner = min(scored, key=lambda candidate: (-scored[candidate], candidate))
        result.append(
            {
                "selection_rank": rank,
                "conformer_id": winner,
                "minimum_standardized_distance_to_selected_pool": scored[winner],
            }
        )
        chosen.append(winner)
        available.remove(winner)
    return result


def run_audit(config_path: Path) -> dict[str, object]:
    config = read_json(config_path)
    inputs = config.get("inputs")
    expected = config.get("expected")
    if not isinstance(inputs, dict) or not isinstance(expected, dict):
        raise ValueError("Stage 08 audit config is incomplete")
    paths = {
        key: checked_record(value)
        for key, value in inputs.items()
        if isinstance(value, dict)
    }
    required = {
        "selection_summary",
        "selection_manifest",
        "eligible_pool",
        "pairwise_distances",
        "expanded8_selection",
    }
    if set(paths) != required:
        raise ValueError("Stage 08 audit inputs differ")

    summary = read_json(paths["selection_summary"])
    if summary.get("status") != "expanded16_structural_selection_ok":
        raise ValueError("Stage 08 structural selection did not pass")
    boundary = summary.get("data_boundary")
    if not isinstance(boundary, dict) or any(int(value) != 0 for value in boundary.values()):
        raise ValueError("Stage 08 structural selection crossed a data boundary")

    eligible_ids = sorted(
        row["conformer_id"] for row in read_csv(paths["eligible_pool"])
    )
    seed_ids = [str(value) for value in expected["seed_receptor_ids"]]
    rebuilt = independent_maxmin(
        eligible_ids,
        seed_ids,
        read_csv(paths["pairwise_distances"]),
        int(expected["maxmin_addition_count"]),
    )

    manifest = read_csv(paths["selection_manifest"])
    if len(manifest) != int(expected["final_receptor_count"]):
        raise ValueError("Stage 08 final receptor count differs")
    seed_rows = sorted(
        (row for row in manifest if row["pool_role"] == "existing_seed"),
        key=lambda row: int(row["selection_rank"]),
    )
    addition_rows = sorted(
        (row for row in manifest if row["pool_role"] == "new_maxmin_addition"),
        key=lambda row: int(row["selection_rank"]),
    )
    if [row["conformer_id"] for row in seed_rows] != seed_ids:
        raise ValueError("Stage 08 seed receptor order differs")
    if [row["conformer_id"] for row in addition_rows] != [
        str(row["conformer_id"]) for row in rebuilt
    ]:
        raise ValueError("independent Stage 08 receptor IDs differ")
    for observed, expected_row in zip(addition_rows, rebuilt):
        delta = abs(
            float(observed["minimum_standardized_distance_to_selected_pool"])
            - float(expected_row["minimum_standardized_distance_to_selected_pool"])
        )
        if delta > 1e-12:
            raise ValueError("independent Stage 08 max-min distance differs")

    old_additions = sorted(
        (
            row
            for row in read_csv(paths["expanded8_selection"])
            if row["pool_role"] == "new_maxmin_addition"
        ),
        key=lambda row: int(row["selection_rank"]),
    )
    prefix_count = int(expected["reproduced_prefix_count"])
    if [row["conformer_id"] for row in addition_rows[:prefix_count]] != [
        row["conformer_id"] for row in old_additions
    ]:
        raise ValueError("independent audit did not reproduce expanded-eight prefix")

    outputs = summary.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Stage 08 source output hashes are missing")
    descriptor = outputs.get("selection_manifest_csv")
    if not isinstance(descriptor, dict) or str(descriptor.get("sha256", "")).upper() != file_sha256(
        paths["selection_manifest"]
    ):
        raise ValueError("Stage 08 source manifest hash differs")

    result = {
        "schema_version": "1.0",
        "audit_id": config["audit_id"],
        "status": "independent_expanded16_structural_audit_ok",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "eligible_pool_count": len(eligible_ids),
        "pairwise_distance_count": len(read_csv(paths["pairwise_distances"])),
        "seed_receptor_ids": seed_ids,
        "independently_reconstructed_additions": rebuilt,
        "reproduced_expanded8_prefix_count": prefix_count,
        "new_expanded16_receptor_ids": [
            str(row["conformer_id"]) for row in rebuilt[prefix_count:]
        ],
        "final_receptor_ids": [row["conformer_id"] for row in manifest],
        "data_boundary": {str(key): int(value) for key, value in boundary.items()},
        "next_gate": "prepare and cognate-redock the eight new expanded16 receptors",
        "decision_boundary": "This audit establishes deterministic structural selection only; it does not establish docking quality, enrichment, receptor complementarity, QUBO benefit, or quantum advantage.",
    }
    output_path = Path(str(config["output_json"]))
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
