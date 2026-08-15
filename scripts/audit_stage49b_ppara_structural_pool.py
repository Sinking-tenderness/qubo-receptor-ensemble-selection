"""Independently audit the Stage49b PPARA structural-pool result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def recompute_maxmin_order(
    ids: set[str], reference_id: str, distances: dict[tuple[str, str], float], count: int
) -> list[str]:
    selected = [reference_id]
    remaining = sorted(ids - {reference_id})
    while len(selected) < count:
        candidates = []
        for candidate in remaining:
            minimum = min(
                distances[tuple(sorted((candidate, chosen)))] for chosen in selected
            )
            candidates.append((minimum, candidate))
        _, chosen = min(candidates, key=lambda value: (-value[0], value[1]))
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def audit(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    summary_path = root / "data/stage49b_ppara_structural_selection_summary.json"
    summary = read_json(summary_path)
    if summary.get("status") != "stage49b_ppara_structural_pool_ok":
        raise ValueError("Stage49b structural-pool result is incomplete")

    config_path = root / summary["config"]["path"]
    if sha256(config_path) != summary["config"]["sha256"]:
        raise ValueError("Stage49b config identity mismatch")
    config = read_json(config_path)
    implementation = root / config["implementation"]["path"]
    if sha256(implementation) != config["implementation"]["sha256"]:
        raise ValueError("Stage49b implementation identity mismatch")
    for record in config["inputs"].values():
        path = root / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Stage49b input identity mismatch: {record['path']}")
    for record in summary["artifacts"].values():
        path = root / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Stage49b artifact identity mismatch: {record['path']}")

    artifacts = summary["artifacts"]
    coordinate_rows = read_csv(root / artifacts["coordinate_audit_csv"]["path"])
    eligible_rows = read_csv(root / artifacts["eligible_pool_manifest_csv"]["path"])
    feature_rows = read_csv(root / artifacts["feature_matrix_csv"]["path"])
    distance_rows = read_csv(root / artifacts["pairwise_distances_csv"]["path"])
    selected_rows = read_csv(root / artifacts["selected_redocking_manifest_csv"]["path"])

    counts = summary["counts"]
    eligible_ids = {
        row["conformer_id"]
        for row in coordinate_rows
        if row["status"] == "coordinate_eligible"
    }
    excluded_ids = {
        row["conformer_id"]
        for row in coordinate_rows
        if row["status"] == "coordinate_excluded"
    }
    if len(coordinate_rows) != 75 or len(eligible_ids) != 66 or len(excluded_ids) != 9:
        raise ValueError("Stage49b coordinate counts differ")
    if len(eligible_rows) != 66 or {row["conformer_id"] for row in eligible_rows} != eligible_ids:
        raise ValueError("Stage49b eligible manifest differs from the coordinate audit")
    if len(feature_rows) != 66 or {row["conformer_id"] for row in feature_rows} != eligible_ids:
        raise ValueError("Stage49b feature matrix has the wrong row identities")
    if len(feature_rows[0]) - 1 != 1122:
        raise ValueError("Stage49b feature count differs")
    expected_pair_count = math.comb(len(eligible_ids), 2)
    if len(distance_rows) != expected_pair_count or expected_pair_count != 2145:
        raise ValueError("Stage49b pairwise-distance count differs")

    distances: dict[tuple[str, str], float] = {}
    for row in distance_rows:
        pair = tuple(sorted((row["conformer_id_a"], row["conformer_id_b"])))
        if pair in distances or pair[0] == pair[1]:
            raise ValueError("Stage49b distance matrix has duplicate or diagonal rows")
        distances[pair] = float(row["standardized_pocket_distance"])
    if set(distances) != {
        tuple(sorted((first, second)))
        for index, first in enumerate(sorted(eligible_ids))
        for second in sorted(eligible_ids)[index + 1 :]
    }:
        raise ValueError("Stage49b distance matrix is incomplete")

    selected_ids = [row["conformer_id"] for row in selected_rows]
    if len(selected_ids) != 64 or len(set(selected_ids)) != 64:
        raise ValueError("Stage49b selected-pool cardinality differs")
    if not set(selected_ids).issubset(eligible_ids):
        raise ValueError("Stage49b selected pool contains an ineligible structure")
    expected_order = recompute_maxmin_order(
        eligible_ids, "PPARA_2P54_reference", distances, 64
    )
    if selected_ids != expected_order:
        raise ValueError("Stage49b deterministic max-min order differs")
    if [int(row["selection_rank"]) for row in selected_rows] != list(range(1, 65)):
        raise ValueError("Stage49b selection ranks are not contiguous")

    reason_counts: dict[str, int] = {}
    for row in coordinate_rows:
        for reason in filter(None, row["exclusion_reasons"].split(";")):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if dict(sorted(reason_counts.items())) != summary["coordinate_exclusion_reason_counts"]:
        raise ValueError("Stage49b exclusion-reason counts differ")
    if any(int(value) != 0 for value in summary["data_boundary"].values()):
        raise ValueError("Stage49b crossed a protected evidence boundary")
    if counts["minimum_coordinate_eligible_count"] != 32 or counts["selected_receptor_count"] != 64:
        raise ValueError("Stage49b frozen selection gates differ")

    result = {
        "schema_version": "1.0",
        "status": "stage49b_ppara_structural_pool_independent_audit_ok",
        "audited_result": {
            "path": summary_path.relative_to(root).as_posix(),
            "sha256": sha256(summary_path),
        },
        "coordinate_counts": {"audited": 75, "eligible": 66, "excluded": 9},
        "selected_receptor_count": 64,
        "raw_feature_count": 1122,
        "pairwise_distance_count": 2145,
        "deterministic_maxmin_order_reproduced": True,
        "reference_is_selection_rank_one": selected_ids[0] == "PPARA_2P54_reference",
        "artifact_identities_ok": True,
        "evidence_boundary_ok": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage49b_ppara_structural_selection_independent_audit.json"),
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.root / args.output
    audit(args.root, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
