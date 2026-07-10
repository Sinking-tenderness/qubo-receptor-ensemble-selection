"""Build ligand-by-receptor docking score matrices from long docking tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REQUIRED_COLUMNS = {
    "target_id",
    "receptor_id",
    "ligand_id",
    "label",
    "pose_rank",
    "docking_score",
    "status",
}


def validate_columns(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ValueError("score table has no header")
    missing = REQUIRED_COLUMNS.difference(fieldnames)
    if missing:
        raise ValueError(f"score table is missing required columns: {sorted(missing)}")


def read_score_tables(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            validate_columns(reader.fieldnames)
            rows.extend(reader)
    if not rows:
        raise ValueError("no score rows were read")
    return rows


def parse_pose_rank(value: str) -> int | None:
    if value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def select_representative_scores(rows: list[dict[str, str]], representative: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    labels_by_ligand: dict[str, str] = {}
    target_by_ligand: dict[str, str] = {}

    for row in rows:
        ligand_id = row["ligand_id"]
        receptor_id = row["receptor_id"]
        labels_by_ligand.setdefault(ligand_id, row["label"])
        target_by_ligand.setdefault(ligand_id, row["target_id"])
        grouped.setdefault((ligand_id, receptor_id), []).append(row)

    output: list[dict[str, object]] = []
    for (ligand_id, receptor_id), group_rows in sorted(grouped.items()):
        ok_rows = [row for row in group_rows if row["status"] == "ok" and row["docking_score"] != ""]
        if not ok_rows:
            output.append(
                {
                    "target_id": target_by_ligand.get(ligand_id, ""),
                    "ligand_id": ligand_id,
                    "label": labels_by_ligand.get(ligand_id, ""),
                    "receptor_id": receptor_id,
                    "representative_score": "",
                    "representative_method": representative,
                    "status": "failed",
                    "pose_count": 0,
                    "best_pose_rank": "",
                    "best_docking_score": "",
                    "ranking_score": "",
                }
            )
            continue

        scored_rows = [
            {
                **row,
                "_pose_rank": parse_pose_rank(row["pose_rank"]),
                "_score": float(row["docking_score"]),
            }
            for row in ok_rows
        ]
        best_row = min(scored_rows, key=lambda row: float(row["_score"]))
        rank1_rows = [row for row in scored_rows if row["_pose_rank"] == 1]

        if representative == "pose_rank_1":
            selected_row = rank1_rows[0] if rank1_rows else best_row
            representative_score = float(selected_row["_score"])
        elif representative == "min_score":
            selected_row = best_row
            representative_score = float(best_row["_score"])
        else:
            raise ValueError(f"unsupported representative method: {representative}")

        output.append(
            {
                "target_id": target_by_ligand.get(ligand_id, ""),
                "ligand_id": ligand_id,
                "label": labels_by_ligand.get(ligand_id, ""),
                "receptor_id": receptor_id,
                "representative_score": representative_score,
                "representative_method": representative,
                "status": "ok",
                "pose_count": len(ok_rows),
                "best_pose_rank": selected_row["_pose_rank"],
                "best_docking_score": best_row["_score"],
                "ranking_score": -representative_score,
            }
        )
    return output


def build_wide_matrix(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ligand_meta: dict[str, dict[str, object]] = {}
    receptor_ids = sorted({str(row["receptor_id"]) for row in long_rows})
    for row in long_rows:
        ligand_id = str(row["ligand_id"])
        ligand_meta.setdefault(
            ligand_id,
            {
                "target_id": row["target_id"],
                "ligand_id": ligand_id,
                "label": row["label"],
            },
        )
        value = row["representative_score"] if row["status"] == "ok" else ""
        ligand_meta[ligand_id][str(row["receptor_id"])] = value

    matrix_rows: list[dict[str, object]] = []
    for ligand_id in sorted(ligand_meta):
        row = ligand_meta[ligand_id]
        for receptor_id in receptor_ids:
            row.setdefault(receptor_id, "")
        matrix_rows.append(row)
    return matrix_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(long_rows: list[dict[str, object]], matrix_rows: list[dict[str, object]]) -> dict[str, object]:
    receptor_ids = sorted({str(row["receptor_id"]) for row in long_rows})
    labels: dict[str, int] = {}
    failure_count = 0
    for row in long_rows:
        labels[str(row["label"])] = labels.get(str(row["label"]), 0) + 1
        if row["status"] != "ok":
            failure_count += 1

    missing_by_receptor: dict[str, int] = {}
    for receptor_id in receptor_ids:
        missing_by_receptor[receptor_id] = sum(1 for row in matrix_rows if row.get(receptor_id, "") == "")

    return {
        "ligand_count": len(matrix_rows),
        "receptor_count": len(receptor_ids),
        "receptor_ids": receptor_ids,
        "long_row_count": len(long_rows),
        "label_counts_in_long_rows": labels,
        "failed_ligand_receptor_pairs": failure_count,
        "missing_scores_by_receptor": missing_by_receptor,
        "score_direction": "lower representative_score is better for Vina; ranking_score = -representative_score is higher-is-better",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-table", type=Path, nargs="+", required=True)
    parser.add_argument("--long-output", type=Path, required=True)
    parser.add_argument("--matrix-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--representative",
        choices=["pose_rank_1", "min_score"],
        default="pose_rank_1",
        help="How to choose one score per ligand-receptor pair.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw_rows = read_score_tables(args.score_table)
    long_rows = select_representative_scores(raw_rows, args.representative)
    matrix_rows = build_wide_matrix(long_rows)
    summary = build_summary(long_rows, matrix_rows)

    write_csv(args.long_output, long_rows)
    write_csv(args.matrix_output, matrix_rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"long_output={args.long_output}")
    print(f"matrix_output={args.matrix_output}")
    print(f"summary_output={args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
