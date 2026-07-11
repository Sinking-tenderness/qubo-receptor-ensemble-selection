"""Extend an existing ligand-by-receptor matrix with raw docking tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RAW_REQUIRED = {"target_id", "receptor_id", "ligand_id", "label", "pose_rank", "docking_score", "status"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_raw_tables(paths: list[Path]) -> tuple[list[dict[str, object]], set[str]]:
    output: list[dict[str, object]] = []
    receptor_ids: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for path in paths:
        rows = read_csv(path)
        missing = RAW_REQUIRED.difference(rows[0] if rows else set())
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in rows:
            key = (row["ligand_id"], row["receptor_id"])
            grouped.setdefault(key, []).append(row)
        for (ligand_id, receptor_id), group in grouped.items():
            if (ligand_id, receptor_id) in seen:
                raise ValueError(f"duplicate ligand/receptor pair: {key}")
            seen.add((ligand_id, receptor_id))
            receptor_ids.add(receptor_id)
            ok = [row for row in group if row["status"] == "ok" and row["docking_score"] != ""]
            if not ok:
                raise ValueError(f"failed docking pair cannot extend matrix: {key}")
            rank_one = [row for row in ok if row["pose_rank"] == "1"]
            selected = rank_one[0] if rank_one else min(ok, key=lambda row: float(row["docking_score"]))
            score = float(selected["docking_score"])
            best = min(ok, key=lambda row: float(row["docking_score"]))
            output.append({
                "target_id": selected["target_id"], "ligand_id": ligand_id,
                "label": selected["label"], "receptor_id": receptor_id,
                "representative_score": score, "representative_method": "pose_rank_1",
                "status": "ok", "pose_count": len(ok),
                "best_pose_rank": best["pose_rank"],
                "best_docking_score": float(best["docking_score"]),
                "ranking_score": -score,
            })
    return output, receptor_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-matrix", type=Path, required=True)
    parser.add_argument("--new-score-table", type=Path, nargs="+", required=True)
    parser.add_argument("--long-output", type=Path, required=True)
    parser.add_argument("--matrix-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    base_rows = read_csv(args.base_matrix)
    if not base_rows:
        raise ValueError("base matrix is empty")
    base_fields = set(base_rows[0])
    base_receptors = sorted(base_fields - {"target_id", "ligand_id", "label"})
    if not base_receptors:
        raise ValueError("base matrix has no receptor columns")
    ligand_ids = {row["ligand_id"] for row in base_rows}
    labels = {row["ligand_id"]: row["label"] for row in base_rows}
    new_long, new_receptors = parse_raw_tables(args.new_score_table)
    if new_receptors & set(base_receptors):
        raise ValueError("new receptor IDs overlap base matrix columns")
    if {row["ligand_id"] for row in new_long} != ligand_ids:
        raise ValueError("new score tables do not contain exactly the base ligand IDs")
    if any(labels[row["ligand_id"]] != row["label"] for row in new_long):
        raise ValueError("ligand labels differ between base matrix and new tables")

    new_by_pair = {(row["ligand_id"], row["receptor_id"]): row for row in new_long}
    matrix_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    for base in base_rows:
        matrix = dict(base)
        target_id = base["target_id"]
        for receptor_id in sorted(new_receptors):
            item = new_by_pair[(base["ligand_id"], receptor_id)]
            matrix[receptor_id] = item["representative_score"]
        matrix_rows.append(matrix)
        for receptor_id in base_receptors:
            score = float(base[receptor_id])
            long_rows.append({
                "target_id": target_id, "ligand_id": base["ligand_id"],
                "label": base["label"], "receptor_id": receptor_id,
                "representative_score": score, "representative_method": "base_matrix",
                "status": "ok", "pose_count": "", "best_pose_rank": "",
                "best_docking_score": score, "ranking_score": -score,
            })
        long_rows.extend(
            row for row in new_long if row["ligand_id"] == base["ligand_id"]
        )

    summary = {
        "ligand_count": len(matrix_rows),
        "receptor_count": len(base_receptors) + len(new_receptors),
        "base_receptor_ids": base_receptors,
        "added_receptor_ids": sorted(new_receptors),
        "failed_pairs": 0,
    }
    write_csv(args.long_output, long_rows)
    write_csv(args.matrix_output, matrix_rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
