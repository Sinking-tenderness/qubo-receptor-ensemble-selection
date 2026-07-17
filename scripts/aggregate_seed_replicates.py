"""Audit and aggregate paired docking seed replicates into score matrices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    ligand_spec = config["ligand_manifest"]
    expected = config["expected"]
    aggregation = config["aggregation"]
    outputs = config["outputs"]
    assert isinstance(ligand_spec, dict)
    assert isinstance(expected, dict)
    assert isinstance(aggregation, dict)
    assert isinstance(outputs, dict)

    ligand_path = Path(str(ligand_spec["path"]))
    if not ligand_path.is_file() or file_sha256(ligand_path) != str(
        ligand_spec["sha256"]
    ).upper():
        raise ValueError("ligand manifest is missing or its hash differs")
    expected_role_counts = {
        str(key): int(value)
        for key, value in expected["role_label_counts"].items()
    }
    ligand_by_id = audit_ligand_manifest(
        read_csv(ligand_path),
        int(expected["ligand_count"]),
        expected_role_counts,
        set(str(role) for role in expected["allowed_selection_roles"]),
    )

    seed_groups: list[tuple[str, list[dict[str, str]]]] = []
    seed_evidence: list[dict[str, object]] = []
    for run in config["seed_runs"]:
        seed_id = str(run["seed_id"])
        run_config_path = Path(str(run["config_path"]))
        summary_path = Path(str(run["summary_path"]))
        representative_path = Path(str(run["representative_scores_path"]))
        if not run_config_path.is_file() or file_sha256(run_config_path) != str(
            run["config_sha256"]
        ).upper():
            raise ValueError(f"seed {seed_id} config is missing or its hash differs")
        if not summary_path.is_file() or not representative_path.is_file():
            raise FileNotFoundError(f"seed {seed_id} outputs are incomplete")
        summary = json.loads(summary_path.read_text(encoding="ascii"))
        if summary.get("status") not in {"ok", "ok_with_search_warning"}:
            raise ValueError(f"seed {seed_id} run did not pass")
        if int(summary["docking_parameters"]["base_seed"]) != int(run["base_seed"]):
            raise ValueError(f"seed {seed_id} base seed differs")
        if int(summary["failed_receptor_ligand_pairs"]) != 0:
            raise ValueError(f"seed {seed_id} contains failed pairs")
        if int(summary["observed_receptor_ligand_pairs"]) != int(
            expected["receptor_ligand_pairs_per_seed"]
        ):
            raise ValueError(f"seed {seed_id} pair count differs")
        summary_output = summary["outputs"]["representative_long_csv"]
        if Path(str(summary_output["path"])).as_posix() != representative_path.as_posix():
            raise ValueError(f"seed {seed_id} representative path differs")
        representative_hash = file_sha256(representative_path)
        if representative_hash != str(summary_output["sha256"]).upper():
            raise ValueError(f"seed {seed_id} representative hash differs")
        seed_groups.append((seed_id, read_csv(representative_path)))
        seed_evidence.append(
            {
                "seed_id": seed_id,
                "base_seed": int(run["base_seed"]),
                "summary_path": summary_path.as_posix(),
                "summary_sha256": file_sha256(summary_path),
                "representative_scores_path": representative_path.as_posix(),
                "representative_scores_sha256": representative_hash,
                "search_quality_warning_count": int(
                    summary["search_quality_warning_count"]
                ),
            }
        )

    if len(seed_groups) != int(expected["seed_count"]):
        raise ValueError("seed count differs")
    combined = aggregate_seed_rows(
        seed_groups,
        ligand_by_id,
        int(expected["receptor_count"]),
        str(aggregation["representative_method"]),
    )
    primary_matrix = build_matrix(combined, "median_representative_score")
    sensitivity_matrix = build_matrix(combined, "minimum_representative_score")
    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    materialized = [
        output_paths["aggregated_long_csv"],
        output_paths["primary_median_matrix_csv"],
        output_paths["sensitivity_minimum_matrix_csv"],
        output_paths["summary_json"],
    ]
    if not args.overwrite and any(path.exists() for path in materialized):
        raise FileExistsError("aggregation outputs exist; use --overwrite")
    write_csv(output_paths["aggregated_long_csv"], combined)
    write_csv(output_paths["primary_median_matrix_csv"], primary_matrix)
    write_csv(output_paths["sensitivity_minimum_matrix_csv"], sensitivity_matrix)
    ranges = [float(row["seed_score_range"]) for row in combined]
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "ligand_count": len(ligand_by_id),
        "receptor_count": int(expected["receptor_count"]),
        "seed_count": len(seed_groups),
        "aggregated_pair_count": len(combined),
        "locked_test_manifest_rows": 0,
        "aggregation": aggregation,
        "seed_evidence": seed_evidence,
        "seed_score_range_kcal_per_mol": {
            "median": statistics.median(ranges),
            "maximum": max(ranges),
        },
        "outputs": {
            key: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for key, path in output_paths.items()
            if key != "summary_json"
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    output_paths["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["summary_json"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
