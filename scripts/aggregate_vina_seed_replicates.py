"""Aggregate complete Vina seed-replicate matrices with stability auditing."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

try:
    from .prepare_receptor import file_sha256
except ImportError:
    from prepare_receptor import file_sha256


REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "purpose",
    "source_runs",
    "expected_receptor_count",
    "expected_ligand_count",
    "expected_label_counts",
    "aggregation",
    "outputs",
    "interpretation_boundary",
}
REQUIRED_SOURCE_KEYS = {
    "replicate_id",
    "config_path",
    "config_sha256",
    "expected_experiment_id",
    "expected_base_seed",
}
REQUIRED_OUTPUT_KEYS = {
    "run_directory",
    "aggregate_long_csv",
    "minimum_score_matrix_csv",
    "median_score_matrix_csv",
    "seed_stability_warnings_csv",
    "summary_json",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
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


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(config, dict):
        raise ValueError("seed aggregation config must be a JSON object")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"seed aggregation config is missing keys: {', '.join(missing)}")
    source_runs = config["source_runs"]
    label_counts = config["expected_label_counts"]
    aggregation = config["aggregation"]
    outputs = config["outputs"]
    if (
        not isinstance(source_runs, list)
        or len(source_runs) < 2
        or any(
            not isinstance(source, dict) or not REQUIRED_SOURCE_KEYS.issubset(source)
            for source in source_runs
        )
    ):
        raise ValueError("source_runs must contain at least two complete source objects")
    replicate_ids = [str(source["replicate_id"]) for source in source_runs]
    if len(replicate_ids) != len(set(replicate_ids)):
        raise ValueError("replicate IDs must be unique")
    base_seeds = [int(source["expected_base_seed"]) for source in source_runs]
    if any(seed <= 0 for seed in base_seeds) or len(base_seeds) != len(set(base_seeds)):
        raise ValueError("source base seeds must be unique and positive")
    if not isinstance(label_counts, dict) or not label_counts:
        raise ValueError("expected_label_counts must be a non-empty object")
    if any(int(value) <= 0 for value in label_counts.values()):
        raise ValueError("expected label counts must be positive")
    if int(config["expected_receptor_count"]) <= 0 or int(config["expected_ligand_count"]) <= 0:
        raise ValueError("expected receptor and ligand counts must be positive")
    if not isinstance(aggregation, dict):
        raise ValueError("aggregation must be a JSON object")
    if aggregation.get("primary_method") != "minimum_score":
        raise ValueError("primary_method must be minimum_score")
    if aggregation.get("sensitivity_method") != "median_score":
        raise ValueError("sensitivity_method must be median_score")
    for key in (
        "maximum_seed_range_kcal_per_mol",
        "maximum_minimum_median_delta_kcal_per_mol",
    ):
        if float(aggregation.get(key, 0.0)) <= 0.0:
            raise ValueError(f"{key} must be positive")
    if int(aggregation.get("minimum_favorable_replicates", 0)) <= 0:
        raise ValueError("minimum_favorable_replicates must be positive")
    if not isinstance(aggregation.get("flag_nonnegative_minimum_score"), bool):
        raise ValueError("flag_nonnegative_minimum_score must be boolean")
    if not isinstance(outputs, dict) or not REQUIRED_OUTPUT_KEYS.issubset(outputs):
        raise ValueError("outputs is missing one or more required paths")
    return config


def load_source_run(
    source: dict[str, object], expected_pairs: int
) -> dict[str, object]:
    config_path = Path(str(source["config_path"]))
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config_hash = file_sha256(config_path)
    if config_hash != str(source["config_sha256"]).upper():
        raise ValueError(f"source config SHA-256 differs: {config_path}")
    run_config = json.loads(config_path.read_text(encoding="ascii"))
    if run_config.get("experiment_id") != source["expected_experiment_id"]:
        raise ValueError(f"source experiment ID differs: {config_path}")
    docking = run_config.get("docking", {})
    if int(docking.get("base_seed", 0)) != int(source["expected_base_seed"]):
        raise ValueError(f"source base seed differs: {config_path}")
    if docking.get("representative_method") != "pose_rank_1":
        raise ValueError("source representative method must be pose_rank_1")
    outputs = run_config.get("outputs", {})
    summary_path = Path(str(outputs.get("summary_json", "")))
    representative_path = Path(str(outputs.get("representative_long_csv", "")))
    if not summary_path.is_file() or not representative_path.is_file():
        raise FileNotFoundError(
            summary_path if not summary_path.is_file() else representative_path
        )
    summary = json.loads(summary_path.read_text(encoding="ascii"))
    if summary.get("experiment_id") != source["expected_experiment_id"]:
        raise ValueError(f"source summary experiment ID differs: {summary_path}")
    if summary.get("status") not in {"ok", "ok_with_search_warning"}:
        raise ValueError(f"source run did not complete: {summary_path}")
    if int(summary.get("successful_receptor_ligand_pairs", 0)) != expected_pairs:
        raise ValueError(f"source pair count differs: {summary_path}")
    summary_config = summary.get("config", {})
    if summary_config.get("sha256") != config_hash:
        raise ValueError(f"source summary config hash differs: {summary_path}")
    output_record = summary.get("outputs", {}).get("representative_long_csv", {})
    representative_hash = file_sha256(representative_path)
    if output_record.get("sha256") != representative_hash:
        raise ValueError(f"source representative hash differs: {representative_path}")
    rows = read_csv(representative_path)
    if len(rows) != expected_pairs:
        raise ValueError(f"source representative row count differs: {representative_path}")
    return {
        "replicate_id": str(source["replicate_id"]),
        "base_seed": int(source["expected_base_seed"]),
        "config_path": config_path.as_posix(),
        "config_sha256": config_hash,
        "summary_path": summary_path.as_posix(),
        "summary_sha256": file_sha256(summary_path),
        "representative_path": representative_path.as_posix(),
        "representative_sha256": representative_hash,
        "input_hashes": {
            key: value.get("sha256")
            for key, value in summary.get("inputs", {}).items()
        },
        "rows": rows,
    }


def validate_source_rows(
    sources: list[dict[str, object]],
    expected_ligand_count: int,
    expected_receptor_count: int,
    expected_label_counts: dict[str, object],
) -> list[tuple[str, str]]:
    expected_keys: set[tuple[str, str]] | None = None
    metadata: dict[tuple[str, str], tuple[str, str]] = {}
    for source in sources:
        rows = source["rows"]
        assert isinstance(rows, list)
        keys: set[tuple[str, str]] = set()
        for row in rows:
            key = (row["ligand_id"], row["receptor_id"])
            if key in keys:
                raise ValueError(f"duplicate source receptor-ligand pair: {key}")
            if row.get("status") != "ok" or row.get("representative_score") == "":
                raise ValueError(f"source pair did not pass: {key}")
            keys.add(key)
            pair_metadata = (row["target_id"], row["label"])
            if key in metadata and metadata[key] != pair_metadata:
                raise ValueError(f"source metadata differs: {key}")
            metadata[key] = pair_metadata
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError("source receptor-ligand key sets differ")
    assert expected_keys is not None
    ligand_ids = {key[0] for key in expected_keys}
    receptor_ids = {key[1] for key in expected_keys}
    if len(ligand_ids) != expected_ligand_count:
        raise ValueError("source ligand count differs")
    if len(receptor_ids) != expected_receptor_count:
        raise ValueError("source receptor count differs")
    labels: dict[str, str] = {}
    for key, (_, label) in metadata.items():
        labels.setdefault(key[0], label)
        if labels[key[0]] != label:
            raise ValueError(f"ligand label differs across receptors: {key[0]}")
    observed_labels: dict[str, int] = {}
    for label in labels.values():
        observed_labels[label] = observed_labels.get(label, 0) + 1
    normalized_expected = {key: int(value) for key, value in expected_label_counts.items()}
    if observed_labels != normalized_expected:
        raise ValueError(
            f"source label counts differ: expected {normalized_expected}, got {observed_labels}"
        )
    return sorted(expected_keys)


def aggregate_replicates(
    sources: list[dict[str, object]],
    keys: list[tuple[str, str]],
    aggregation: dict[str, object],
) -> list[dict[str, object]]:
    lookups: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    for source in sources:
        rows = source["rows"]
        assert isinstance(rows, list)
        lookups[str(source["replicate_id"])] = {
            (row["ligand_id"], row["receptor_id"]): row for row in rows
        }
    output: list[dict[str, object]] = []
    for key in keys:
        replicate_rows = [
            (source, lookups[str(source["replicate_id"])][key]) for source in sources
        ]
        scores = [float(row["representative_score"]) for _, row in replicate_rows]
        minimum_score = min(scores)
        median_score = statistics.median(scores)
        maximum_score = max(scores)
        seed_range = maximum_score - minimum_score
        minimum_median_delta = median_score - minimum_score
        favorable_count = sum(score < 0.0 for score in scores)
        best_index = scores.index(minimum_score)
        reasons: list[str] = []
        if seed_range > float(aggregation["maximum_seed_range_kcal_per_mol"]):
            reasons.append("seed_range_exceeded")
        if minimum_median_delta > float(
            aggregation["maximum_minimum_median_delta_kcal_per_mol"]
        ):
            reasons.append("minimum_median_delta_exceeded")
        if favorable_count < int(aggregation["minimum_favorable_replicates"]):
            reasons.append("insufficient_favorable_replicates")
        if bool(aggregation["flag_nonnegative_minimum_score"]) and minimum_score >= 0.0:
            reasons.append("nonnegative_minimum_score")
        first_row = replicate_rows[0][1]
        output_row: dict[str, object] = {
            "target_id": first_row["target_id"],
            "ligand_id": key[0],
            "label": first_row["label"],
            "receptor_id": key[1],
        }
        for source, row in replicate_rows:
            replicate_id = str(source["replicate_id"])
            output_row[f"score_{replicate_id}"] = float(row["representative_score"])
            output_row[f"base_seed_{replicate_id}"] = int(source["base_seed"])
        output_row.update(
            {
                "minimum_score": minimum_score,
                "median_score": median_score,
                "maximum_score": maximum_score,
                "seed_range": round(seed_range, 6),
                "minimum_median_delta": round(minimum_median_delta, 6),
                "favorable_replicate_count": favorable_count,
                "replicate_count": len(scores),
                "best_replicate_id": sources[best_index]["replicate_id"],
                "primary_score": minimum_score,
                "primary_method": aggregation["primary_method"],
                "seed_stability_warning": bool(reasons),
                "seed_stability_warning_reasons": ";".join(reasons),
            }
        )
        output.append(output_row)
    return output


def build_wide_matrix(
    rows: list[dict[str, object]], score_field: str
) -> list[dict[str, object]]:
    receptor_ids = sorted({str(row["receptor_id"]) for row in rows})
    by_ligand: dict[str, dict[str, object]] = {}
    for row in rows:
        ligand_id = str(row["ligand_id"])
        output = by_ligand.setdefault(
            ligand_id,
            {
                "target_id": row["target_id"],
                "ligand_id": ligand_id,
                "label": row["label"],
            },
        )
        output[str(row["receptor_id"])] = row[score_field]
    for output in by_ligand.values():
        for receptor_id in receptor_ids:
            output.setdefault(receptor_id, "")
    return [by_ligand[ligand_id] for ligand_id in sorted(by_ligand)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    source_specs = config["source_runs"]
    expected_label_counts = config["expected_label_counts"]
    aggregation = config["aggregation"]
    outputs = config["outputs"]
    assert isinstance(source_specs, list)
    assert isinstance(expected_label_counts, dict)
    assert isinstance(aggregation, dict)
    assert isinstance(outputs, dict)

    expected_receptors = int(config["expected_receptor_count"])
    expected_ligands = int(config["expected_ligand_count"])
    expected_pairs = expected_receptors * expected_ligands
    sources = [load_source_run(source, expected_pairs) for source in source_specs]
    input_hash_sets = [source["input_hashes"] for source in sources]
    if any(hashes != input_hash_sets[0] for hashes in input_hash_sets[1:]):
        raise ValueError("source run input hashes differ")
    keys = validate_source_rows(
        sources,
        expected_ligands,
        expected_receptors,
        expected_label_counts,
    )

    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    core_outputs = [path for key, path in output_paths.items() if key != "run_directory"]
    existing = [path for path in core_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("seed aggregate outputs exist; use --overwrite after review")
    if args.overwrite:
        for path in existing:
            path.unlink()
    output_paths["run_directory"].mkdir(parents=True, exist_ok=True)

    aggregate_rows = aggregate_replicates(sources, keys, aggregation)
    minimum_matrix = build_wide_matrix(aggregate_rows, "minimum_score")
    median_matrix = build_wide_matrix(aggregate_rows, "median_score")
    warnings = [row for row in aggregate_rows if row["seed_stability_warning"]]
    write_csv(output_paths["aggregate_long_csv"], aggregate_rows)
    write_csv(output_paths["minimum_score_matrix_csv"], minimum_matrix)
    write_csv(output_paths["median_score_matrix_csv"], median_matrix)
    if warnings:
        write_csv(output_paths["seed_stability_warnings_csv"], warnings)
    elif output_paths["seed_stability_warnings_csv"].exists():
        output_paths["seed_stability_warnings_csv"].unlink()

    reason_names = (
        "seed_range_exceeded",
        "minimum_median_delta_exceeded",
        "insufficient_favorable_replicates",
        "nonnegative_minimum_score",
    )
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok_with_seed_stability_warning" if warnings else "ok",
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "source_runs": [
            {
                key: source[key]
                for key in (
                    "replicate_id",
                    "base_seed",
                    "config_path",
                    "config_sha256",
                    "summary_path",
                    "summary_sha256",
                    "representative_path",
                    "representative_sha256",
                )
            }
            for source in sources
        ],
        "receptor_count": expected_receptors,
        "ligand_count": expected_ligands,
        "label_counts": {
            key: int(value) for key, value in expected_label_counts.items()
        },
        "receptor_ligand_pair_count": len(aggregate_rows),
        "replicate_count": len(sources),
        "aggregation": aggregation,
        "seed_stability_warning_count": len(warnings),
        "seed_stability_warning_reason_counts": {
            reason: sum(
                reason in str(row["seed_stability_warning_reasons"]).split(";")
                for row in warnings
            )
            for reason in reason_names
        },
        "score_ranges_kcal_per_mol": {
            "minimum_matrix": {
                "minimum": min(float(row["minimum_score"]) for row in aggregate_rows),
                "maximum": max(float(row["minimum_score"]) for row in aggregate_rows),
            },
            "median_matrix": {
                "minimum": min(float(row["median_score"]) for row in aggregate_rows),
                "maximum": max(float(row["median_score"]) for row in aggregate_rows),
            },
        },
        "outputs": {
            key: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for key, path in output_paths.items()
            if key not in {"run_directory", "summary_json"} and path.is_file()
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    output_paths["summary_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
