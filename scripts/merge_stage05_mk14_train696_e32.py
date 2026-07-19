"""Audit and merge reused and newly computed MAPK14 train e32 aggregates."""

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


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def verify_file(spec: dict[str, object], label: str) -> Path:
    path = Path(str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = str(spec.get("sha256", "")).upper()
    if expected_hash and file_sha256(path) != expected_hash:
        raise ValueError(f"{label} SHA-256 differs")
    return path


def rows_by_id(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    output = {row["ligand_id"]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"{label} contains duplicate ligand IDs")
    return output


def audit_manifest_partition(
    full_rows: list[dict[str, str]],
    reused_rows: list[dict[str, str]],
    new_rows: list[dict[str, str]],
    manifest_specs: dict[str, dict[str, object]],
    expected: dict[str, object],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    counts = {
        "full": int(expected["full_ligand_count"]),
        "reused": int(expected["reused_ligand_count"]),
        "new": int(expected["new_ligand_count"]),
    }
    observed_rows = {"full": full_rows, "reused": reused_rows, "new": new_rows}
    for name, rows in observed_rows.items():
        if len(rows) != counts[name]:
            raise ValueError(f"{name} manifest expected {counts[name]} rows, got {len(rows)}")
        expected_role = str(manifest_specs[name]["selection_role"])
        roles = {row.get("selection_role", "") for row in rows}
        if roles != {expected_role}:
            raise ValueError(f"{name} manifest roles differ: {sorted(roles)}")
        if {row.get("split", "") for row in rows} != {"train"}:
            raise ValueError(f"{name} manifest is not train-only")
        if any(row.get("pdbqt_status") != "ok" for row in rows):
            raise ValueError(f"{name} manifest contains a failed PDBQT row")

    full_by_id = rows_by_id(full_rows, "full manifest")
    reused_by_id = rows_by_id(reused_rows, "reused manifest")
    new_by_id = rows_by_id(new_rows, "new manifest")
    reused_ids = set(reused_by_id)
    new_ids = set(new_by_id)
    if reused_ids.intersection(new_ids):
        raise ValueError("reused and new ligand IDs overlap")
    if reused_ids.union(new_ids) != set(full_by_id):
        raise ValueError("reused and new ligand IDs do not form the full panel")

    expected_labels = {
        str(key): int(value)
        for key, value in dict(expected["full_label_counts"]).items()
    }
    if dict(Counter(row["label"] for row in full_rows)) != expected_labels:
        raise ValueError("full manifest label counts differ")

    ignored_fields = {"selection_role"}
    for source_name, source in (("reused", reused_by_id), ("new", new_by_id)):
        for ligand_id, source_row in source.items():
            full_row = full_by_id[ligand_id]
            common = set(source_row).intersection(full_row).difference(ignored_fields)
            differing = sorted(
                field for field in common if source_row[field] != full_row[field]
            )
            if differing:
                raise ValueError(
                    f"{source_name} manifest differs from full panel for "
                    f"{ligand_id}: {differing}"
                )
    return full_by_id, reused_by_id, new_by_id


def validate_seed_protocol(
    aggregation_config: dict[str, object],
    summary: dict[str, object],
    source_manifest_spec: dict[str, object],
    protocol: dict[str, object],
    expected_ligand_count: int,
    expected_label_counts: dict[str, int],
) -> None:
    aggregate_manifest = aggregation_config["ligand_manifest"]
    if aggregate_manifest["path"] != source_manifest_spec["path"]:
        raise ValueError("aggregation ligand manifest path differs")
    if aggregate_manifest["sha256"].upper() != source_manifest_spec["sha256"].upper():
        raise ValueError("aggregation ligand manifest SHA-256 differs")
    aggregate_expected = aggregation_config["expected"]
    expected_role = str(source_manifest_spec["selection_role"])
    expected_role_counts = {
        f"{expected_role}:{label}": count
        for label, count in expected_label_counts.items()
    }
    if int(aggregate_expected["ligand_count"]) != expected_ligand_count:
        raise ValueError("aggregation expected ligand count differs")
    if int(aggregate_expected["receptor_count"]) != int(protocol["receptor_count"]):
        raise ValueError("aggregation expected receptor count differs")
    if int(aggregate_expected["seed_count"]) != len(protocol["base_seeds"]):
        raise ValueError("aggregation expected seed count differs")
    if int(aggregate_expected["receptor_ligand_pairs_per_seed"]) != (
        expected_ligand_count * int(protocol["receptor_count"])
    ):
        raise ValueError("aggregation expected pair count differs")
    if set(aggregate_expected["allowed_selection_roles"]) != {expected_role}:
        raise ValueError("aggregation allowed roles differ")
    observed_role_counts = {
        str(key): int(value)
        for key, value in aggregate_expected["role_label_counts"].items()
    }
    if observed_role_counts != expected_role_counts:
        raise ValueError("aggregation role/label counts differ")
    if aggregation_config["aggregation"]["representative_method"] != protocol[
        "representative_method"
    ]:
        raise ValueError("aggregation representative method differs")
    if summary["aggregation"]["representative_method"] != protocol[
        "representative_method"
    ]:
        raise ValueError("summary representative method differs")
    if int(summary.get("locked_test_manifest_rows", -1)) != 0:
        raise ValueError("aggregation summary contains locked test rows")

    seed_runs = list(aggregation_config["seed_runs"])
    evidence = {str(item["seed_id"]): item for item in summary["seed_evidence"]}
    expected_seeds = [int(value) for value in protocol["base_seeds"]]
    if len(seed_runs) != len(expected_seeds) or len(evidence) != len(expected_seeds):
        raise ValueError("seed count differs from the frozen protocol")

    required_inputs = {
        "receptor_manifest": protocol["receptor_manifest"],
        "ligand_manifest": source_manifest_spec,
        "vina_executable": protocol["vina_executable"],
        "vina_config": protocol["vina_config"],
        "parallel_runner": protocol["parallel_runner"],
        "score_matrix_module": protocol["score_matrix_module"],
    }
    for index, run in enumerate(seed_runs):
        seed_id = str(run["seed_id"])
        if seed_id not in evidence:
            raise ValueError(f"missing seed evidence: {seed_id}")
        seed_config_path = Path(str(run["config_path"]))
        if not seed_config_path.is_file():
            raise FileNotFoundError(seed_config_path)
        if file_sha256(seed_config_path) != str(run["config_sha256"]).upper():
            raise ValueError(f"{seed_id} config SHA-256 differs")
        seed_config = load_json(seed_config_path)
        if int(run["base_seed"]) != expected_seeds[index]:
            raise ValueError(f"{seed_id} aggregation base seed differs")
        if int(seed_config["docking"]["base_seed"]) != expected_seeds[index]:
            raise ValueError(f"{seed_id} benchmark base seed differs")
        if seed_config["docking"]["representative_method"] != protocol["representative_method"]:
            raise ValueError(f"{seed_id} representative method differs")
        if int(seed_config["expected_receptor_count"]) != int(protocol["receptor_count"]):
            raise ValueError(f"{seed_id} receptor count differs")
        if int(seed_config["expected_ligand_count"]) != expected_ligand_count:
            raise ValueError(f"{seed_id} ligand count differs")
        observed_labels = {
            str(key): int(value)
            for key, value in seed_config["expected_label_counts"].items()
        }
        if observed_labels != expected_label_counts:
            raise ValueError(f"{seed_id} label counts differ")
        for key, required in required_inputs.items():
            if seed_config["inputs"][key] != required["path"]:
                raise ValueError(f"{seed_id} {key} path differs")
            if seed_config["input_sha256"][key].upper() != required["sha256"].upper():
                raise ValueError(f"{seed_id} {key} SHA-256 differs")

        item = evidence[seed_id]
        summary_path = Path(str(run["summary_path"]))
        representative_path = Path(str(run["representative_scores_path"]))
        if item["summary_path"] != summary_path.as_posix():
            raise ValueError(f"{seed_id} summary path differs")
        if item["representative_scores_path"] != representative_path.as_posix():
            raise ValueError(f"{seed_id} representative path differs")
        if file_sha256(summary_path) != str(item["summary_sha256"]).upper():
            raise ValueError(f"{seed_id} summary evidence SHA-256 differs")
        if file_sha256(representative_path) != str(
            item["representative_scores_sha256"]
        ).upper():
            raise ValueError(f"{seed_id} representative evidence SHA-256 differs")
        seed_summary = load_json(summary_path)
        if seed_summary.get("status") not in {"ok", "ok_with_search_warning"}:
            raise ValueError(f"{seed_id} benchmark did not pass")
        if int(seed_summary["docking_parameters"]["base_seed"]) != expected_seeds[index]:
            raise ValueError(f"{seed_id} summary base seed differs")
        expected_pairs = expected_ligand_count * int(protocol["receptor_count"])
        if int(seed_summary["observed_receptor_ligand_pairs"]) != expected_pairs:
            raise ValueError(f"{seed_id} summary pair count differs")
        if int(seed_summary["failed_receptor_ligand_pairs"]) != 0:
            raise ValueError(f"{seed_id} summary contains failed pairs")
        summary_output = seed_summary["outputs"]["representative_long_csv"]
        if summary_output["path"] != representative_path.as_posix():
            raise ValueError(f"{seed_id} summary representative path differs")
        if summary_output["sha256"].upper() != file_sha256(representative_path):
            raise ValueError(f"{seed_id} summary representative SHA-256 differs")


def validate_matrix(
    path: Path,
    source_rows: list[dict[str, str]],
    ligand_by_id: dict[str, dict[str, str]],
    receptor_ids: list[str],
    score_field: str,
    source_role: str,
) -> None:
    matrix_rows = read_csv(path)
    matrix_by_id = rows_by_id(matrix_rows, f"matrix {path}")
    if set(matrix_by_id) != set(ligand_by_id):
        raise ValueError(f"matrix ligand identities differ: {path}")
    scores = {
        (row["ligand_id"], row["receptor_id"]): float(row[score_field])
        for row in source_rows
    }
    for ligand_id, row in matrix_by_id.items():
        if row["label"] != ligand_by_id[ligand_id]["label"]:
            raise ValueError(f"matrix label differs for {ligand_id}")
        if row["selection_role"] != source_role:
            raise ValueError(f"matrix role differs for {ligand_id}")
        for receptor_id in receptor_ids:
            try:
                observed = float(row[receptor_id])
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"matrix score is missing or invalid for {ligand_id}, {receptor_id}"
                ) from exc
            expected_score = scores[(ligand_id, receptor_id)]
            if not math.isclose(observed, expected_score, abs_tol=1e-9):
                raise ValueError(f"matrix score differs for {ligand_id}, {receptor_id}")


def validate_aggregate_source(
    source_name: str,
    source_spec: dict[str, object],
    manifest_spec: dict[str, object],
    ligand_by_id: dict[str, dict[str, str]],
    protocol: dict[str, object],
    receptor_ids: list[str],
    expected_ligand_count: int,
) -> tuple[list[dict[str, str]], dict[str, object], dict[str, str]]:
    aggregation_config_path = verify_file(
        source_spec["aggregation_config"], f"{source_name} aggregation config"
    )
    aggregation_config = load_json(aggregation_config_path)
    summary_path = verify_file(source_spec["summary"], f"{source_name} summary")
    summary = load_json(summary_path)
    if summary.get("status") != "ok":
        raise ValueError(f"{source_name} aggregation did not pass")
    if summary["config"]["path"] != aggregation_config_path.as_posix():
        raise ValueError(f"{source_name} aggregation config path differs")
    if summary["config"]["sha256"].upper() != file_sha256(aggregation_config_path):
        raise ValueError(f"{source_name} aggregation config hash differs")
    if int(summary["ligand_count"]) != expected_ligand_count:
        raise ValueError(f"{source_name} ligand count differs")
    if int(summary["receptor_count"]) != len(receptor_ids):
        raise ValueError(f"{source_name} receptor count differs")
    if int(summary["seed_count"]) != len(protocol["base_seeds"]):
        raise ValueError(f"{source_name} seed count differs")

    expected_labels = dict(Counter(row["label"] for row in ligand_by_id.values()))
    validate_seed_protocol(
        aggregation_config,
        summary,
        manifest_spec,
        {**protocol, "receptor_count": len(receptor_ids)},
        expected_ligand_count,
        expected_labels,
    )

    output_paths: dict[str, Path] = {}
    output_hashes: dict[str, str] = {}
    for key in (
        "aggregated_long_csv",
        "primary_median_matrix_csv",
        "sensitivity_minimum_matrix_csv",
    ):
        summary_output = summary["outputs"][key]
        expected_path = Path(str(aggregation_config["outputs"][key]))
        if summary_output["path"] != expected_path.as_posix():
            raise ValueError(f"{source_name} {key} path differs")
        if not expected_path.is_file():
            raise FileNotFoundError(expected_path)
        actual_hash = file_sha256(expected_path)
        if actual_hash != str(summary_output["sha256"]).upper():
            raise ValueError(f"{source_name} {key} SHA-256 differs")
        output_paths[key] = expected_path
        output_hashes[key] = actual_hash

    rows = read_csv(output_paths["aggregated_long_csv"])
    expected_pairs = expected_ligand_count * len(receptor_ids)
    if len(rows) != expected_pairs or int(summary["aggregated_pair_count"]) != expected_pairs:
        raise ValueError(f"{source_name} pair count differs")
    pairs: set[tuple[str, str]] = set()
    source_role = str(manifest_spec["selection_role"])
    seed_fields = [f"seed{index}_representative_score" for index in range(3)]
    for row in rows:
        ligand_id = row["ligand_id"]
        receptor_id = row["receptor_id"]
        pair = (ligand_id, receptor_id)
        if pair in pairs:
            raise ValueError(f"{source_name} contains a duplicate pair: {pair}")
        pairs.add(pair)
        if ligand_id not in ligand_by_id or receptor_id not in receptor_ids:
            raise ValueError(f"{source_name} contains an unknown pair: {pair}")
        ligand = ligand_by_id[ligand_id]
        if row["label"] != ligand["label"] or row["selection_role"] != source_role:
            raise ValueError(f"{source_name} metadata differs for {pair}")
        if row["status"] != "ok" or row["representative_method"] != protocol["representative_method"]:
            raise ValueError(f"{source_name} pair did not pass: {pair}")
        if int(row["seed_count"]) != 3:
            raise ValueError(f"{source_name} seed count differs for {pair}")
        values = [float(row[field]) for field in seed_fields]
        expected_values = {
            "median_representative_score": statistics.median(values),
            "minimum_representative_score": min(values),
            "maximum_representative_score": max(values),
            "seed_score_range": max(values) - min(values),
            "primary_ranking_score": -statistics.median(values),
            "sensitivity_ranking_score": -min(values),
        }
        for field, expected_value in expected_values.items():
            observed = float(row[field])
            if not math.isfinite(observed) or not math.isclose(
                observed, expected_value, abs_tol=1e-9
            ):
                raise ValueError(f"{source_name} {field} differs for {pair}")

    expected_pairs_set = {
        (ligand_id, receptor_id)
        for ligand_id in ligand_by_id
        for receptor_id in receptor_ids
    }
    if pairs != expected_pairs_set:
        raise ValueError(f"{source_name} pair coverage differs")
    validate_matrix(
        output_paths["primary_median_matrix_csv"],
        rows,
        ligand_by_id,
        receptor_ids,
        "median_representative_score",
        source_role,
    )
    validate_matrix(
        output_paths["sensitivity_minimum_matrix_csv"],
        rows,
        ligand_by_id,
        receptor_ids,
        "minimum_representative_score",
        source_role,
    )
    evidence = {
        "aggregation_config_sha256": file_sha256(aggregation_config_path),
        "summary_sha256": file_sha256(summary_path),
        **output_hashes,
    }
    return rows, summary, evidence


def build_matrix(
    rows: list[dict[str, object]], receptor_ids: list[str], score_field: str
) -> list[dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for row in rows:
        ligand_id = str(row["ligand_id"])
        matrix_row = output.setdefault(
            ligand_id,
            {
                "target_id": row["target_id"],
                "ligand_id": ligand_id,
                "label": row["label"],
                "selection_role": row["selection_role"],
            },
        )
        matrix_row[str(row["receptor_id"])] = row[score_field]
    matrix = [output[ligand_id] for ligand_id in sorted(output)]
    for row in matrix:
        if any(receptor_id not in row for receptor_id in receptor_ids):
            raise ValueError(f"merged matrix coverage differs for {row['ligand_id']}")
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    required = {
        "schema_version",
        "experiment_id",
        "purpose",
        "authorization",
        "protocol",
        "manifests",
        "sources",
        "expected",
        "outputs",
        "interpretation_boundary",
    }
    if missing := required.difference(config):
        raise ValueError(f"merge config is missing keys: {sorted(missing)}")

    verify_file(config["authorization"], "authorization")
    protocol = config["protocol"]
    manifests = config["manifests"]
    expected = config["expected"]
    outputs = config["outputs"]
    assert isinstance(protocol, dict)
    assert isinstance(manifests, dict)
    assert isinstance(expected, dict)
    assert isinstance(outputs, dict)
    for name in (
        "receptor_manifest",
        "vina_executable",
        "vina_config",
        "parallel_runner",
        "score_matrix_module",
    ):
        verify_file(protocol[name], f"protocol {name}")

    receptor_rows = read_csv(Path(str(protocol["receptor_manifest"]["path"])))
    receptor_ids = [row["conformer_id"] for row in receptor_rows]
    if len(receptor_ids) != int(expected["receptor_count"]):
        raise ValueError("receptor count differs")
    if len(set(receptor_ids)) != len(receptor_ids):
        raise ValueError("receptor manifest contains duplicate IDs")
    if any(row.get("status") != "ok" for row in receptor_rows):
        raise ValueError("receptor manifest contains a failed row")

    manifest_rows = {
        name: read_csv(verify_file(spec, f"{name} manifest"))
        for name, spec in manifests.items()
    }
    full_by_id, reused_by_id, new_by_id = audit_manifest_partition(
        manifest_rows["full"],
        manifest_rows["reused"],
        manifest_rows["new"],
        manifests,
        expected,
    )

    reused_rows, reused_summary, reused_evidence = validate_aggregate_source(
        "reused",
        config["sources"]["reused"],
        manifests["reused"],
        reused_by_id,
        protocol,
        receptor_ids,
        int(expected["reused_ligand_count"]),
    )
    new_rows, new_summary, new_evidence = validate_aggregate_source(
        "new",
        config["sources"]["new"],
        manifests["new"],
        new_by_id,
        protocol,
        receptor_ids,
        int(expected["new_ligand_count"]),
    )

    full_role = str(manifests["full"]["selection_role"])
    combined: list[dict[str, object]] = []
    for row in reused_rows + new_rows:
        combined.append({**row, "selection_role": full_role})
    combined.sort(key=lambda row: (str(row["ligand_id"]), str(row["receptor_id"])))
    if len(combined) != int(expected["aggregated_pair_count"]):
        raise ValueError("merged pair count differs")
    if {str(row["ligand_id"]) for row in combined} != set(full_by_id):
        raise ValueError("merged ligand identities differ")

    primary_matrix = build_matrix(
        combined, receptor_ids, "median_representative_score"
    )
    sensitivity_matrix = build_matrix(
        combined, receptor_ids, "minimum_representative_score"
    )
    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    materialized = list(output_paths.values())
    if not args.overwrite and any(path.exists() for path in materialized):
        raise FileExistsError("merge outputs exist; use --overwrite")
    write_csv(output_paths["aggregated_long_csv"], combined)
    write_csv(output_paths["primary_median_matrix_csv"], primary_matrix)
    write_csv(output_paths["sensitivity_minimum_matrix_csv"], sensitivity_matrix)

    ranges = [float(row["seed_score_range"]) for row in combined]
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "authorization": {
            "path": config["authorization"]["path"],
            "sha256": file_sha256(Path(str(config["authorization"]["path"]))),
        },
        "ligand_count": len(full_by_id),
        "label_counts": dict(Counter(row["label"] for row in full_by_id.values())),
        "receptor_count": len(receptor_ids),
        "receptor_ids": receptor_ids,
        "seed_count": len(protocol["base_seeds"]),
        "aggregated_pair_count": len(combined),
        "evidence_cells": {
            "reused_e32": int(expected["reused_seed_cells"]),
            "new_e32": int(expected["new_seed_cells"]),
            "complete_e32": int(expected["complete_seed_cells"]),
            "diagnostic_e64_used": 0,
        },
        "source_evidence": {
            "reused": {
                "ligand_count": int(reused_summary["ligand_count"]),
                "selection_role_before_normalization": manifests["reused"]["selection_role"],
                **reused_evidence,
            },
            "new": {
                "ligand_count": int(new_summary["ligand_count"]),
                "selection_role": manifests["new"]["selection_role"],
                **new_evidence,
            },
        },
        "role_normalization": {
            "from": manifests["reused"]["selection_role"],
            "to": full_role,
            "ligand_count": len(reused_by_id),
            "performed_after_identity_hash_and_protocol_validation": True,
        },
        "locked_validation_manifest_rows": 0,
        "locked_test_manifest_rows": 0,
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
