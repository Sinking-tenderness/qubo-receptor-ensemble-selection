"""Freeze a label-independent 16-conformer MAPK14 structural pool."""

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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty selection manifest")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )


def checked_record(record: dict[str, object]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(record["sha256"]).upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return path


def load_complete_distances(
    rows: list[dict[str, str]], eligible_ids: list[str]
) -> dict[tuple[str, str], float]:
    eligible = set(eligible_ids)
    distances: dict[tuple[str, str], float] = {}
    for row in rows:
        first = row["conformer_id_a"]
        second = row["conformer_id_b"]
        if first == second or first not in eligible or second not in eligible:
            raise ValueError("distance table contains an invalid receptor pair")
        pair = tuple(sorted((first, second)))
        if pair in distances:
            raise ValueError(f"duplicate structural-distance pair: {pair}")
        value = float(row["standardized_pocket_distance"])
        if value < 0:
            raise ValueError("structural distance cannot be negative")
        distances[pair] = value
    expected_count = len(eligible_ids) * (len(eligible_ids) - 1) // 2
    if len(distances) != expected_count:
        raise ValueError(
            f"structural-distance table is incomplete: {len(distances)} != {expected_count}"
        )
    return distances


def deterministic_maxmin(
    eligible_ids: list[str],
    seed_ids: list[str],
    distances: dict[tuple[str, str], float],
    count: int,
) -> list[dict[str, object]]:
    if len(seed_ids) != len(set(seed_ids)):
        raise ValueError("seed receptor IDs are not unique")
    if not set(seed_ids).issubset(eligible_ids):
        raise ValueError("a seed receptor is absent from the eligible pool")
    selected = list(seed_ids)
    remaining = sorted(set(eligible_ids) - set(seed_ids))
    additions: list[dict[str, object]] = []
    for rank in range(1, count + 1):
        if not remaining:
            raise ValueError("too few eligible candidates for requested expansion")
        candidates = []
        for candidate in remaining:
            minimum = min(
                distances[tuple(sorted((candidate, chosen)))] for chosen in selected
            )
            candidates.append((minimum, candidate))
        minimum_distance, chosen = sorted(
            candidates, key=lambda item: (-item[0], item[1])
        )[0]
        additions.append(
            {
                "selection_rank": rank,
                "conformer_id": chosen,
                "minimum_standardized_distance_to_selected_pool": minimum_distance,
            }
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return additions


def validate_zero_boundary(summary: dict[str, object]) -> dict[str, int]:
    boundary = summary.get("data_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("source structural summary has no data boundary")
    normalized = {str(key): int(value) for key, value in boundary.items()}
    if any(normalized.values()):
        raise ValueError("source structural selection crossed a frozen data boundary")
    return normalized


def run_selection(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = read_json(config_path)
    required_keys = {
        "schema_version",
        "experiment_id",
        "purpose",
        "data_boundary",
        "inputs",
        "selection",
        "expected",
        "outputs",
        "decision_boundary",
    }
    if set(config) != required_keys:
        raise ValueError("Stage 08 structural-selection config keys differ")

    inputs = config["inputs"]
    if not isinstance(inputs, dict):
        raise ValueError("inputs must be an object")
    paths = {
        key: checked_record(record)
        for key, record in inputs.items()
        if isinstance(record, dict)
    }
    required_inputs = {
        "source_selection_summary",
        "source_selection_audit",
        "eligible_pool",
        "candidate_audit",
        "pairwise_distances",
        "expanded8_selection",
        "historical_preparation_failure",
    }
    if set(paths) != required_inputs:
        raise ValueError("Stage 08 structural-selection inputs differ")

    source_summary = read_json(paths["source_selection_summary"])
    source_audit = read_json(paths["source_selection_audit"])
    if source_summary.get("status") != "expanded8_structural_selection_ok":
        raise ValueError("source structural selection did not pass")
    if source_audit.get("status") != "independent_coordinate_selection_audit_ok":
        raise ValueError("source independent structural audit did not pass")
    source_boundary = validate_zero_boundary(source_summary)

    configured_boundary = config["data_boundary"]
    if not isinstance(configured_boundary, dict) or any(
        int(value) != 0 for value in configured_boundary.values()
    ):
        raise ValueError("Stage 08 structural selection must have a zero-use boundary")

    eligible_rows = read_csv(paths["eligible_pool"])
    eligible_by_id = {row["conformer_id"]: row for row in eligible_rows}
    if len(eligible_by_id) != len(eligible_rows):
        raise ValueError("eligible pool contains duplicate receptor IDs")
    eligible_ids = sorted(eligible_by_id)

    candidate_audit = {
        row["conformer_id"]: row for row in read_csv(paths["candidate_audit"])
    }
    for receptor_id in eligible_ids:
        row = candidate_audit.get(receptor_id)
        if row is None or row["status"] != "coordinate_eligible":
            raise ValueError(f"eligible receptor lacks a passing audit row: {receptor_id}")

    historical_failure = read_json(paths["historical_preparation_failure"])
    excluded_failure_id = str(historical_failure["failed_conformer_id"])
    failed_audit = candidate_audit.get(excluded_failure_id)
    if excluded_failure_id in eligible_by_id:
        raise ValueError("historically failed receptor remains coordinate eligible")
    if failed_audit is None or "incomplete_standard_amino_acid_residue" not in str(
        failed_audit["exclusion_reasons"]
    ):
        raise ValueError("historical preparation failure is not excluded by the v3 gate")

    expected = config["expected"]
    selection = config["selection"]
    if not isinstance(expected, dict) or not isinstance(selection, dict):
        raise ValueError("selection and expected settings must be objects")
    if len(eligible_ids) != int(expected["eligible_pool_count"]):
        raise ValueError("eligible receptor count differs")
    seed_ids = [str(value) for value in selection["seed_receptor_ids"]]
    addition_count = int(selection["new_receptor_count"])
    if len(seed_ids) + addition_count != int(expected["final_receptor_count"]):
        raise ValueError("final receptor count is inconsistent")

    distances = load_complete_distances(
        read_csv(paths["pairwise_distances"]), eligible_ids
    )
    additions = deterministic_maxmin(
        eligible_ids, seed_ids, distances, addition_count
    )

    expanded8_rows = read_csv(paths["expanded8_selection"])
    old_additions = sorted(
        (row for row in expanded8_rows if row["pool_role"] == "new_maxmin_addition"),
        key=lambda row: int(row["selection_rank"]),
    )
    prefix_count = int(expected["required_reproduced_prefix_count"])
    if len(old_additions) != prefix_count:
        raise ValueError("source expanded-eight prefix count differs")
    for old, new in zip(old_additions, additions[:prefix_count]):
        if old["conformer_id"] != new["conformer_id"]:
            raise ValueError("expanded-sixteen selection does not reproduce old prefix")
        difference = abs(
            float(old["minimum_standardized_distance_to_selected_pool"])
            - float(new["minimum_standardized_distance_to_selected_pool"])
        )
        if difference > 1e-12:
            raise ValueError("reproduced max-min prefix distance differs")

    output_settings = config["outputs"]
    if not isinstance(output_settings, dict):
        raise ValueError("outputs must be an object")
    manifest_path = Path(str(output_settings["selection_manifest_csv"]))
    summary_path = Path(str(output_settings["summary_json"]))
    if not overwrite and (manifest_path.exists() or summary_path.exists()):
        raise FileExistsError("Stage 08 structural outputs already exist")

    rows: list[dict[str, object]] = []
    for rank, receptor_id in enumerate(seed_ids, start=1):
        rows.append(
            {
                "pool_role": "existing_seed",
                "selection_rank": rank,
                "final_pool_rank": rank,
                **eligible_by_id[receptor_id],
                "minimum_standardized_distance_to_selected_pool": "",
                "prefix_status": "frozen_seed",
            }
        )
    for addition in additions:
        rank = int(addition["selection_rank"])
        receptor_id = str(addition["conformer_id"])
        rows.append(
            {
                "pool_role": "new_maxmin_addition",
                "selection_rank": rank,
                "final_pool_rank": len(seed_ids) + rank,
                **eligible_by_id[receptor_id],
                "minimum_standardized_distance_to_selected_pool": addition[
                    "minimum_standardized_distance_to_selected_pool"
                ],
                "prefix_status": (
                    "reproduced_expanded8_addition"
                    if rank <= prefix_count
                    else "new_expanded16_addition"
                ),
            }
        )
    write_csv(manifest_path, rows)

    new_extension = additions[prefix_count:]
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "expanded16_structural_selection_ok",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "selection_rule": selection["selection_rule"],
        "counts": {
            "eligible_pool_count": len(eligible_ids),
            "pairwise_distance_count": len(distances),
            "seed_receptor_count": len(seed_ids),
            "maxmin_addition_count": len(additions),
            "reproduced_expanded8_addition_count": prefix_count,
            "new_expanded16_addition_count": len(new_extension),
            "final_receptor_count": len(rows),
        },
        "seed_receptor_ids": seed_ids,
        "maxmin_additions": additions,
        "reproduced_expanded8_additions": additions[:prefix_count],
        "new_expanded16_additions": new_extension,
        "final_receptor_ids": [str(row["conformer_id"]) for row in rows],
        "historical_preparation_failure_exclusion": {
            "conformer_id": excluded_failure_id,
            "source_status": historical_failure["status"],
            "v3_exclusion_reasons": failed_audit["exclusion_reasons"],
        },
        "source_data_boundary": source_boundary,
        "data_boundary": {
            str(key): int(value) for key, value in configured_boundary.items()
        },
        "outputs": {
            "selection_manifest_csv": {
                "path": manifest_path.as_posix(),
                "sha256": file_sha256(manifest_path),
            }
        },
        "next_gate": "independent structural reconstruction, then prepare and cognate-redock only the eight new additions before production docking",
        "decision_boundary": config["decision_boundary"],
    }
    write_json(summary_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_selection(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
