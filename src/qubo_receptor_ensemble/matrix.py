"""Docking score matrix construction.

Consolidated from ``scripts/build_score_matrix.py`` and
``scripts/aggregate_seed_replicates.py``; behavior is identical to the
originals.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
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


def audit_ligand_manifest(
    rows: list[dict[str, str]],
    expected_count: int,
    expected_role_label_counts: dict[str, int],
    allowed_roles: set[str],
) -> dict[str, dict[str, str]]:
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} ligands, got {len(rows)}")
    by_id = {row["ligand_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("ligand manifest contains duplicate ligand IDs")
    observed = Counter(
        f"{row.get('selection_role', '')}:{row['label']}" for row in rows
    )
    if dict(observed) != expected_role_label_counts:
        raise ValueError(
            f"role/label counts differ: expected {expected_role_label_counts}, "
            f"got {dict(observed)}"
        )
    unexpected_roles = sorted(
        {row.get("selection_role", "") for row in rows}.difference(allowed_roles)
    )
    if unexpected_roles:
        raise ValueError(f"prohibited ligand roles found: {unexpected_roles}")
    if any(row.get("split", "") == "test" for row in rows):
        raise ValueError("locked test rows are prohibited")
    return by_id


def aggregate_seed_rows(
    seed_groups: list[tuple[str, list[dict[str, str]]]],
    ligand_by_id: dict[str, dict[str, str]],
    expected_receptor_count: int,
    representative_method: str,
) -> list[dict[str, object]]:
    if len(seed_groups) < 2:
        raise ValueError("at least two seed replicates are required")
    ligand_ids = set(ligand_by_id)
    expected_pairs = len(ligand_ids) * expected_receptor_count
    scores_by_seed: dict[str, dict[tuple[str, str], float]] = {}
    reference_keys: set[tuple[str, str]] | None = None

    for seed_id, rows in seed_groups:
        if seed_id in scores_by_seed:
            raise ValueError(f"duplicate seed ID: {seed_id}")
        if len(rows) != expected_pairs:
            raise ValueError(
                f"seed {seed_id} expected {expected_pairs} rows, got {len(rows)}"
            )
        seed_scores: dict[tuple[str, str], float] = {}
        receptors_by_ligand: dict[str, set[str]] = {
            ligand_id: set() for ligand_id in ligand_ids
        }
        for row in rows:
            ligand_id = row["ligand_id"]
            receptor_id = row["receptor_id"]
            key = (ligand_id, receptor_id)
            if ligand_id not in ligand_by_id:
                raise ValueError(f"seed {seed_id} contains unknown ligand: {ligand_id}")
            if key in seed_scores:
                raise ValueError(f"seed {seed_id} contains duplicate pair: {key}")
            ligand = ligand_by_id[ligand_id]
            if row.get("label") != ligand["label"]:
                raise ValueError(f"seed {seed_id} label differs for {ligand_id}")
            if row.get("status") != "ok":
                raise ValueError(f"seed {seed_id} failed pair: {key}")
            if row.get("representative_method") != representative_method:
                raise ValueError(f"seed {seed_id} representative method differs")
            try:
                score = float(row["representative_score"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"seed {seed_id} has invalid score: {key}") from exc
            if not math.isfinite(score):
                raise ValueError(f"seed {seed_id} has non-finite score: {key}")
            seed_scores[key] = score
            receptors_by_ligand[ligand_id].add(receptor_id)
        if any(len(values) != expected_receptor_count for values in receptors_by_ligand.values()):
            raise ValueError(f"seed {seed_id} receptor coverage differs by ligand")
        keys = set(seed_scores)
        if reference_keys is not None and keys != reference_keys:
            raise ValueError(f"seed {seed_id} pair identities differ")
        reference_keys = keys
        scores_by_seed[seed_id] = seed_scores

    seed_ids = [seed_id for seed_id, _ in seed_groups]
    assert reference_keys is not None
    output: list[dict[str, object]] = []
    for ligand_id, receptor_id in sorted(reference_keys):
        values = [scores_by_seed[seed_id][(ligand_id, receptor_id)] for seed_id in seed_ids]
        ligand = ligand_by_id[ligand_id]
        output.append(
            {
                "target_id": ligand.get("target_id", ""),
                "ligand_id": ligand_id,
                "label": ligand["label"],
                "selection_role": ligand["selection_role"],
                "receptor_id": receptor_id,
                "seed_count": len(values),
                **{
                    f"{seed_id}_representative_score": value
                    for seed_id, value in zip(seed_ids, values)
                },
                "median_representative_score": statistics.median(values),
                "minimum_representative_score": min(values),
                "maximum_representative_score": max(values),
                "seed_score_range": max(values) - min(values),
                "primary_ranking_score": -statistics.median(values),
                "sensitivity_ranking_score": -min(values),
                "representative_method": representative_method,
                "status": "ok",
            }
        )
    return output


def build_matrix(
    rows: list[dict[str, object]], score_field: str
) -> list[dict[str, object]]:
    receptor_ids = sorted({str(row["receptor_id"]) for row in rows})
    by_ligand: dict[str, dict[str, object]] = {}
    for row in rows:
        ligand_id = str(row["ligand_id"])
        matrix_row = by_ligand.setdefault(
            ligand_id,
            {
                "target_id": row["target_id"],
                "ligand_id": ligand_id,
                "label": row["label"],
                "selection_role": row["selection_role"],
            },
        )
        matrix_row[str(row["receptor_id"])] = row[score_field]
    output = [by_ligand[ligand_id] for ligand_id in sorted(by_ligand)]
    for row in output:
        for receptor_id in receptor_ids:
            if receptor_id not in row:
                raise ValueError(f"matrix is missing receptor {receptor_id}")
    return output


def load_config(path: Path) -> dict[str, object]:
    """Load and validate an aggregation config JSON (aggregate_seed_replicates)."""
    config = json.loads(path.read_text(encoding="ascii"))
    required = {
        "schema_version",
        "experiment_id",
        "purpose",
        "ligand_manifest",
        "seed_runs",
        "expected",
        "aggregation",
        "outputs",
        "interpretation_boundary",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"aggregation config is missing keys: {sorted(missing)}")
    return config
