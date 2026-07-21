"""Audit the complete MAPK14 train-696 e32 matrix and retain seed uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

try:
    from .aggregate_seed_replicates import file_sha256, read_csv
    from .audit_stage05_development_matrix import validate_matrix
except ImportError:
    from aggregate_seed_replicates import file_sha256, read_csv
    from audit_stage05_development_matrix import validate_matrix


FLAG_FIELDS = (
    "ligand_id",
    "label",
    "receptor_id",
    "flag_reasons",
    "minimum_seed_score",
    "median_seed_score",
    "maximum_seed_score",
    "consensus_replicate_count",
    "median_minus_minimum",
    "maximum_minus_minimum",
)


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def require_hash(path: Path, expected: object, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(expected).upper():
        raise ValueError(f"{name} SHA-256 differs")


def write_flags(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FLAG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def audit_rows(
    rows: list[dict[str, str]],
    ligand_by_id: dict[str, dict[str, str]],
    receptor_ids: list[str],
    seed_ids: list[str],
    expected_role: str,
    consensus_delta: float,
    minimum_consensus: int,
    range_threshold: float,
    extreme_range_threshold: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    expected_pairs = len(ligand_by_id) * len(receptor_ids)
    if len(rows) != expected_pairs:
        raise ValueError(f"expected {expected_pairs} aggregate pairs, got {len(rows)}")
    expected_pairs_set = {
        (ligand_id, receptor_id)
        for ligand_id in ligand_by_id
        for receptor_id in receptor_ids
    }
    observed_pairs: set[tuple[str, str]] = set()
    flags: list[dict[str, object]] = []
    ranges: list[float] = []
    median_minimum_deltas: list[float] = []
    nonnegative_medians = 0
    failed_consensus = 0
    raw_range_diagnostics = 0
    extreme_range_diagnostics = 0
    label_range_counts: Counter[str] = Counter()
    receptor_range_counts: Counter[str] = Counter()

    for row in rows:
        key = (row["ligand_id"], row["receptor_id"])
        if key in observed_pairs:
            raise ValueError(f"duplicate aggregate pair: {key}")
        observed_pairs.add(key)
        if key not in expected_pairs_set:
            raise ValueError(f"unknown aggregate pair: {key}")
        ligand = ligand_by_id[key[0]]
        if (
            row.get("status") != "ok"
            or row.get("selection_role") != expected_role
            or row.get("label") != ligand["label"]
            or int(row.get("seed_count", 0)) != len(seed_ids)
        ):
            raise ValueError(f"invalid aggregate metadata: {key}")
        values = [float(row[f"{seed_id}_representative_score"]) for seed_id in seed_ids]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"non-finite seed score: {key}")
        minimum = min(values)
        median = statistics.median(values)
        maximum = max(values)
        raw_range = maximum - minimum
        median_minimum = median - minimum
        consensus_count = sum(value <= minimum + consensus_delta for value in values)
        stored = {
            "minimum_representative_score": minimum,
            "median_representative_score": median,
            "maximum_representative_score": maximum,
            "seed_score_range": raw_range,
        }
        for field, expected_value in stored.items():
            if not math.isclose(
                float(row[field]), expected_value, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(f"stored aggregate differs: {key} / {field}")

        reasons: list[str] = []
        if median >= 0.0:
            reasons.append("nonnegative_median_score")
            nonnegative_medians += 1
        if consensus_count < minimum_consensus:
            reasons.append("minimum_score_consensus_not_reached")
            failed_consensus += 1
        if raw_range > range_threshold:
            reasons.append("raw_seed_range_diagnostic")
            raw_range_diagnostics += 1
            label_range_counts[ligand["label"]] += 1
            receptor_range_counts[key[1]] += 1
        if raw_range > extreme_range_threshold:
            reasons.append("extreme_raw_seed_range_diagnostic")
            extreme_range_diagnostics += 1
        ranges.append(raw_range)
        median_minimum_deltas.append(median_minimum)
        if reasons:
            flags.append(
                {
                    "ligand_id": key[0],
                    "label": ligand["label"],
                    "receptor_id": key[1],
                    "flag_reasons": ";".join(reasons),
                    "minimum_seed_score": minimum,
                    "median_seed_score": median,
                    "maximum_seed_score": maximum,
                    "consensus_replicate_count": consensus_count,
                    "median_minus_minimum": median_minimum,
                    "maximum_minus_minimum": raw_range,
                }
            )

    if observed_pairs != expected_pairs_set:
        raise ValueError("aggregate pair coverage differs")
    return flags, {
        "aggregated_pair_count": len(rows),
        "ligand_count": len(ligand_by_id),
        "receptor_count": len(receptor_ids),
        "seed_count": len(seed_ids),
        "nonnegative_median_score_pair_count": nonnegative_medians,
        "minimum_score_consensus_diagnostic_pair_count": failed_consensus,
        "raw_range_diagnostic_pair_count": raw_range_diagnostics,
        "extreme_raw_range_diagnostic_pair_count": extreme_range_diagnostics,
        "uncertain_pair_union_count": len(flags),
        "raw_range_diagnostic_label_counts": dict(label_range_counts),
        "raw_range_diagnostic_receptor_counts": dict(receptor_range_counts),
        "median_minus_minimum_kcal_per_mol": {
            "median": statistics.median(median_minimum_deltas),
            "maximum": max(median_minimum_deltas),
        },
        "raw_seed_range_kcal_per_mol": {
            "median": statistics.median(ranges),
            "maximum": max(ranges),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    expected = config["expected"]
    policy = config["uncertainty_policy"]
    outputs = config["outputs"]
    assert isinstance(inputs, dict)
    assert isinstance(hashes, dict)
    assert isinstance(expected, dict)
    assert isinstance(policy, dict)
    assert isinstance(outputs, dict)
    if set(inputs) != set(hashes):
        raise ValueError("input paths and hashes differ")
    paths = {key: Path(str(value)) for key, value in inputs.items()}
    for key, path in paths.items():
        require_hash(path, hashes[key], key)
    authorization = config["authorization"]
    assert isinstance(authorization, dict)
    require_hash(
        Path(str(authorization["path"])), authorization["sha256"], "authorization"
    )
    archive = Path(str(config["source_archive"]["path"]))
    require_hash(archive, config["source_archive"]["sha256"], "source archive")

    output_summary = Path(str(outputs["summary_json"]))
    output_flags = Path(str(outputs["uncertain_pairs_csv"]))
    if not args.overwrite and (output_summary.exists() or output_flags.exists()):
        raise FileExistsError("audit outputs exist; use --overwrite")

    inherited = load_json(paths["inherited_uncertainty_preregistration"])
    e64 = load_json(paths["inherited_e64_audit"])
    if (
        inherited["protocol_amendment"]["authorized_matrix"]
        != "the unchanged uniform three-seed e32 matrices"
        or e64.get("status")
        != "independent_e64_audit_ok_uniform_e64_not_supported"
        or e64.get("complete_uniform_e64_recomputation_supported") is not False
        or int(e64.get("e32_matrix_cells_replaced", -1)) != 0
    ):
        raise ValueError("inherited uncertainty policy evidence differs")

    merged = load_json(paths["merged_summary"])
    if (
        merged.get("status") != "ok"
        or int(merged.get("ligand_count", 0)) != int(expected["ligand_count"])
        or int(merged.get("receptor_count", 0)) != int(expected["receptor_count"])
        or int(merged.get("seed_count", 0)) != len(expected["seed_ids"])
        or int(merged.get("aggregated_pair_count", 0))
        != int(expected["aggregated_pair_count"])
        or int(merged.get("locked_validation_manifest_rows", -1)) != 0
        or int(merged.get("locked_test_manifest_rows", -1)) != 0
        or int(merged["evidence_cells"].get("diagnostic_e64_used", -1)) != 0
    ):
        raise ValueError("merged summary differs from the frozen matrix")
    expected_output_keys = {
        "aggregated_long_csv": "aggregated_seed_scores",
        "primary_median_matrix_csv": "primary_median_matrix",
        "sensitivity_minimum_matrix_csv": "sensitivity_minimum_matrix",
    }
    for summary_key, input_key in expected_output_keys.items():
        evidence = merged["outputs"][summary_key]
        if (
            Path(str(evidence["path"])).as_posix() != paths[input_key].as_posix()
            or str(evidence["sha256"]).upper() != file_sha256(paths[input_key])
        ):
            raise ValueError(f"merged output evidence differs: {summary_key}")

    new_aggregate = load_json(paths["new_aggregate_summary"])
    if new_aggregate.get("status") != "ok" or int(
        new_aggregate.get("locked_test_manifest_rows", -1)
    ) != 0:
        raise ValueError("new-ligand aggregation did not pass")
    warning_count = 0
    for item in new_aggregate["seed_evidence"]:
        summary_path = Path(str(item["summary_path"]))
        scores_path = Path(str(item["representative_scores_path"]))
        require_hash(summary_path, item["summary_sha256"], f"{item['seed_id']} summary")
        require_hash(
            scores_path,
            item["representative_scores_sha256"],
            f"{item['seed_id']} scores",
        )
        seed_summary = load_json(summary_path)
        if seed_summary.get("status") not in {"ok", "ok_with_search_warning"}:
            raise ValueError(f"{item['seed_id']} execution failed")
        if int(seed_summary.get("failed_receptor_ligand_pairs", -1)) != int(
            expected["failed_pair_count"]
        ):
            raise ValueError(f"{item['seed_id']} failed pair count differs")
        warning_count += int(seed_summary.get("search_quality_warning_count", 0))

    ligand_rows = read_csv(paths["full_ligand_manifest"])
    if len(ligand_rows) != int(expected["ligand_count"]):
        raise ValueError("ligand count differs")
    ligand_by_id = {row["ligand_id"]: row for row in ligand_rows}
    if len(ligand_by_id) != len(ligand_rows):
        raise ValueError("ligand manifest contains duplicate IDs")
    if Counter(row["label"] for row in ligand_rows) != Counter(
        {key: int(value) for key, value in expected["label_counts"].items()}
    ):
        raise ValueError("ligand label counts differ")
    if any(
        row.get("selection_role") != expected["selection_role"]
        or row.get("split") != expected["split"]
        for row in ligand_rows
    ):
        raise ValueError("ligand manifest is not train-only")

    receptor_rows = read_csv(paths["receptor_manifest"])
    receptor_ids = [row["conformer_id"] for row in receptor_rows]
    if (
        len(receptor_ids) != int(expected["receptor_count"])
        or len(set(receptor_ids)) != len(receptor_ids)
        or any(row.get("status") != "ok" for row in receptor_rows)
    ):
        raise ValueError("receptor manifest differs")
    aggregate_rows = read_csv(paths["aggregated_seed_scores"])
    flags, audit = audit_rows(
        aggregate_rows,
        ligand_by_id,
        receptor_ids,
        [str(value) for value in expected["seed_ids"]],
        str(expected["selection_role"]),
        float(policy["minimum_score_consensus_delta_kcal_per_mol"]),
        int(policy["minimum_consensus_replicates"]),
        float(policy["raw_range_diagnostic_threshold_kcal_per_mol"]),
        float(policy["extreme_raw_range_diagnostic_threshold_kcal_per_mol"]),
    )
    validate_matrix(
        read_csv(paths["primary_median_matrix"]),
        aggregate_rows,
        receptor_ids,
        "median_representative_score",
    )
    validate_matrix(
        read_csv(paths["sensitivity_minimum_matrix"]),
        aggregate_rows,
        receptor_ids,
        "minimum_representative_score",
    )
    passed = (
        audit["nonnegative_median_score_pair_count"]
        == int(policy["required_nonnegative_median_pair_count"])
        and int(merged["evidence_cells"]["complete_e32"])
        == int(expected["complete_seed_cell_count"])
        and int(policy["required_cell_replacement_count"]) == 0
        and int(policy["required_e64_matrix_score_count"]) == 0
    )
    write_flags(output_flags, flags)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": (
            "matrix_admission_passed_with_retained_seed_uncertainty"
            if passed
            else "matrix_admission_rejected"
        ),
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "source_archive": {"path": archive.as_posix(), "sha256": file_sha256(archive)},
        "merged_summary": {
            "path": paths["merged_summary"].as_posix(),
            "sha256": file_sha256(paths["merged_summary"]),
        },
        "thresholds": policy,
        "audit": {**audit, "execution_search_quality_warning_count": warning_count},
        "matrix_cells_replaced": 0,
        "e64_scores_used_in_matrix": 0,
        "qubo_fitted": False,
        "enrichment_metrics_calculated": False,
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "outputs": {
            "uncertain_pairs_csv": {
                "path": output_flags.as_posix(),
                "sha256": file_sha256(output_flags),
            }
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
