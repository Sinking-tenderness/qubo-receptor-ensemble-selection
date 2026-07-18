"""Audit a complete MAPK14 e32 matrix with a two-of-three consensus gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

try:
    from .aggregate_seed_replicates import file_sha256, read_csv
    from .audit_stage05_development_matrix import validate_matrix
    from .audit_stage05_expanded_train_matrix import require_hash, verify_seed_evidence
except ImportError:
    from aggregate_seed_replicates import file_sha256, read_csv
    from audit_stage05_development_matrix import validate_matrix
    from audit_stage05_expanded_train_matrix import require_hash, verify_seed_evidence


FLAG_FIELDS = (
    "ligand_id",
    "receptor_id",
    "selection_role",
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


def write_flags(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FLAG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def audit_consensus_rows(
    rows: list[dict[str, str]],
    seed_ids: list[str],
    expected_ligands: int,
    expected_receptors: int,
    consensus_delta: float,
    minimum_consensus_replicates: int,
    maximum_median_minus_minimum: float,
    maximum_nonnegative_median_pairs: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    expected_pairs = expected_ligands * expected_receptors
    if len(rows) != expected_pairs:
        raise ValueError(f"expected {expected_pairs} aggregate pairs, got {len(rows)}")
    seen: set[tuple[str, str]] = set()
    ligand_receptors: dict[str, set[str]] = {}
    flags = []
    consensus_counts = []
    median_minimum_deltas = []
    raw_ranges = []
    nonnegative_medians = 0
    failed_consensus = 0
    excessive_median_minimum = 0
    for row in rows:
        key = (row["ligand_id"], row["receptor_id"])
        if key in seen:
            raise ValueError(f"duplicate aggregate pair: {key}")
        seen.add(key)
        if row.get("status") != "ok":
            raise ValueError(f"failed aggregate pair: {key}")
        if row.get("selection_role") != "development_train":
            raise ValueError(f"prohibited aggregate role: {key}")
        ligand_receptors.setdefault(row["ligand_id"], set()).add(row["receptor_id"])
        values = [float(row[f"{seed_id}_representative_score"]) for seed_id in seed_ids]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite aggregate score: {key}")
        minimum = min(values)
        median = statistics.median(values)
        maximum = max(values)
        consensus_count = sum(value <= minimum + consensus_delta for value in values)
        median_minimum = median - minimum
        raw_range = maximum - minimum
        stored = {
            "minimum_representative_score": minimum,
            "median_representative_score": median,
            "maximum_representative_score": maximum,
            "seed_score_range": raw_range,
        }
        for field, expected in stored.items():
            if not math.isclose(
                float(row[field]), expected, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(f"stored aggregate differs: {key} / {field}")
        reasons = []
        if median >= 0.0:
            reasons.append("nonnegative_median_score")
            nonnegative_medians += 1
        if consensus_count < minimum_consensus_replicates:
            reasons.append("insufficient_consensus_replicates")
            failed_consensus += 1
        if median_minimum > maximum_median_minus_minimum:
            reasons.append("median_minus_minimum_exceeded")
            excessive_median_minimum += 1
        consensus_counts.append(consensus_count)
        median_minimum_deltas.append(median_minimum)
        raw_ranges.append(raw_range)
        if reasons:
            flags.append(
                {
                    "ligand_id": key[0],
                    "receptor_id": key[1],
                    "selection_role": row["selection_role"],
                    "flag_reasons": ";".join(reasons),
                    "minimum_seed_score": minimum,
                    "median_seed_score": median,
                    "maximum_seed_score": maximum,
                    "consensus_replicate_count": consensus_count,
                    "median_minus_minimum": median_minimum,
                    "maximum_minus_minimum": raw_range,
                }
            )
    if len(ligand_receptors) != expected_ligands or any(
        len(receptors) != expected_receptors for receptors in ligand_receptors.values()
    ):
        raise ValueError("aggregate receptor coverage differs")
    passed = (
        nonnegative_medians <= maximum_nonnegative_median_pairs
        and failed_consensus == 0
        and excessive_median_minimum == 0
    )
    return flags, {
        "aggregated_pair_count": len(rows),
        "ligand_count": len(ligand_receptors),
        "receptor_count": expected_receptors,
        "seed_count": len(seed_ids),
        "nonnegative_median_score_pair_count": nonnegative_medians,
        "insufficient_consensus_pair_count": failed_consensus,
        "excessive_median_minus_minimum_pair_count": excessive_median_minimum,
        "flagged_pair_count": len(flags),
        "minimum_consensus_replicate_count": min(consensus_counts),
        "median_minus_minimum_kcal_per_mol": {
            "median": statistics.median(median_minimum_deltas),
            "maximum": max(median_minimum_deltas),
        },
        "raw_seed_range_kcal_per_mol": {
            "median": statistics.median(raw_ranges),
            "maximum": max(raw_ranges),
        },
        "admission_passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--aggregation-summary", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--flagged-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in (args.output_summary, args.flagged_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists; use --overwrite: {path}")

    amendment = load_json(args.amendment)
    original_spec = amendment["original_preregistration"]
    trigger = amendment["trigger"]
    protocol = amendment["uniform_e32_recomputation"]
    admission = amendment["e32_matrix_admission"]
    assert isinstance(original_spec, dict)
    assert isinstance(trigger, dict)
    assert isinstance(protocol, dict)
    assert isinstance(admission, dict)
    original_path = Path(str(original_spec["path"]))
    diagnostic_audit_path = Path(str(trigger["diagnostic_audit_path"]))
    require_hash(original_path, original_spec["sha256"], "original preregistration")
    require_hash(
        diagnostic_audit_path,
        trigger["diagnostic_audit_sha256"],
        "diagnostic audit",
    )
    original = load_json(original_path)
    frozen = original["frozen_inputs"]
    assert isinstance(frozen, dict)
    receptor_spec = frozen["receptor_manifest"]
    ligand_spec = frozen["train_ligand_manifest"]
    assert isinstance(receptor_spec, dict)
    assert isinstance(ligand_spec, dict)
    receptor_path = Path(str(receptor_spec["path"]))
    ligand_path = Path(str(ligand_spec["path"]))
    require_hash(receptor_path, receptor_spec["sha256"], "receptor manifest")
    require_hash(ligand_path, ligand_spec["sha256"], "ligand manifest")
    receptor_rows = read_csv(receptor_path)
    ligand_rows = read_csv(ligand_path)
    expected_receptors = int(protocol["receptor_count"])
    expected_ligands = int(protocol["ligand_count"])
    expected_pairs = expected_receptors * expected_ligands
    if len(receptor_rows) != expected_receptors or len(ligand_rows) != expected_ligands:
        raise ValueError("frozen manifest row count differs")
    if any(row.get("status") != "ok" for row in receptor_rows):
        raise ValueError("receptor manifest contains a failed receptor")
    if any(row.get("selection_role") != "development_train" for row in ligand_rows):
        raise ValueError("ligand manifest is not train-only")

    aggregate_summary = load_json(args.aggregation_summary)
    if aggregate_summary.get("status") != "ok":
        raise ValueError("e32 aggregation did not pass")
    if int(aggregate_summary.get("locked_test_manifest_rows", -1)) != 0:
        raise ValueError("e32 aggregation contains locked test rows")
    if int(aggregate_summary.get("aggregated_pair_count", -1)) != expected_pairs:
        raise ValueError("e32 aggregate pair count differs")
    evidence = aggregate_summary.get("seed_evidence")
    if not isinstance(evidence, list):
        raise ValueError("e32 aggregation has no seed evidence")
    seed_ids = verify_seed_evidence(
        evidence,
        int(admission["required_seed_count"]),
        expected_pairs,
        expected_ligands,
        expected_receptors,
        int(admission["required_failed_pairs_per_seed"]),
    )
    if [int(item["base_seed"]) for item in evidence] != [
        int(value) for value in protocol["paired_base_seeds"]
    ]:
        raise ValueError("e32 aggregate base seeds differ")

    output_specs = aggregate_summary.get("outputs")
    if not isinstance(output_specs, dict):
        raise ValueError("e32 aggregation has no output evidence")
    aggregate_paths = {}
    for key in (
        "aggregated_long_csv",
        "primary_median_matrix_csv",
        "sensitivity_minimum_matrix_csv",
    ):
        spec = output_specs[key]
        if not isinstance(spec, dict):
            raise ValueError(f"invalid e32 output evidence: {key}")
        path = Path(str(spec["path"]))
        require_hash(path, spec["sha256"], key)
        aggregate_paths[key] = path

    rows = read_csv(aggregate_paths["aggregated_long_csv"])
    flags, audit = audit_consensus_rows(
        rows,
        seed_ids,
        expected_ligands,
        expected_receptors,
        float(admission["maximum_consensus_delta_kcal_per_mol"]),
        int(admission["minimum_consensus_replicates_per_pair"]),
        float(admission["maximum_allowed_median_minus_minimum_kcal_per_mol"]),
        int(admission["maximum_allowed_nonnegative_median_score_pairs"]),
    )
    receptor_ids = [row["conformer_id"] for row in receptor_rows]
    validate_matrix(
        read_csv(aggregate_paths["primary_median_matrix_csv"]),
        rows,
        receptor_ids,
        "median_representative_score",
    )
    validate_matrix(
        read_csv(aggregate_paths["sensitivity_minimum_matrix_csv"]),
        rows,
        receptor_ids,
        "minimum_representative_score",
    )
    write_flags(args.flagged_output, flags)
    result = {
        "schema_version": "1.0",
        "status": (
            "e32_matrix_admission_passed"
            if audit["admission_passed"]
            else "e32_matrix_admission_rejected"
        ),
        "amendment": {"path": args.amendment.as_posix(), "sha256": file_sha256(args.amendment)},
        "aggregation_summary": {
            "path": args.aggregation_summary.as_posix(),
            "sha256": file_sha256(args.aggregation_summary),
        },
        "thresholds": admission,
        "audit": audit,
        "original_e16_matrix_cells_reused": 0,
        "e32_cells_selectively_replaced": 0,
        "qubo_fitted": False,
        "enrichment_metrics_calculated": False,
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "outputs": {
            "flagged_pairs_csv": {
                "path": args.flagged_output.as_posix(),
                "sha256": file_sha256(args.flagged_output),
            }
        },
        "interpretation_note": amendment["interpretation_boundary"],
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if audit["admission_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
