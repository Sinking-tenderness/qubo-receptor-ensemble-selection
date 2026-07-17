"""Audit MAPK14 seed aggregation before any enrichment metric is calculated."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

try:
    from .aggregate_seed_replicates import file_sha256, read_csv
except ImportError:
    from aggregate_seed_replicates import file_sha256, read_csv


FLAG_FIELDS = (
    "ligand_id",
    "receptor_id",
    "selection_role",
    "flag_reasons",
    "minimum_seed_score",
    "median_seed_score",
    "maximum_seed_score",
    "seed_score_range",
)


def write_flags(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FLAG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def close(first: float, second: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(first, second, rel_tol=0.0, abs_tol=tolerance)


def audit_aggregated_rows(
    rows: list[dict[str, str]],
    seed_ids: list[str],
    expected_ligand_count: int,
    expected_receptor_count: int,
    allowed_roles: set[str],
    maximum_nonnegative_pairs: int,
    maximum_seed_range: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    expected_pairs = expected_ligand_count * expected_receptor_count
    if len(rows) != expected_pairs:
        raise ValueError(f"expected {expected_pairs} aggregated pairs, got {len(rows)}")
    if len(seed_ids) < 2 or len(seed_ids) != len(set(seed_ids)):
        raise ValueError("seed IDs must contain at least two unique values")

    seen: set[tuple[str, str]] = set()
    receptors_by_ligand: dict[str, set[str]] = {}
    ligand_metadata: dict[str, tuple[str, str]] = {}
    flags: list[dict[str, object]] = []
    ranges: list[float] = []
    nonnegative_pairs = 0
    excessive_range_pairs = 0

    for row in rows:
        ligand_id = row["ligand_id"]
        receptor_id = row["receptor_id"]
        role = row.get("selection_role", "")
        key = (ligand_id, receptor_id)
        if key in seen:
            raise ValueError(f"duplicate aggregated pair: {key}")
        seen.add(key)
        if row.get("status") != "ok":
            raise ValueError(f"failed aggregated pair: {key}")
        if role not in allowed_roles:
            raise ValueError(f"prohibited selection role for {ligand_id}: {role}")
        metadata = (row.get("label", ""), role)
        if ligand_id in ligand_metadata and ligand_metadata[ligand_id] != metadata:
            raise ValueError(f"ligand metadata differs across receptors: {ligand_id}")
        ligand_metadata[ligand_id] = metadata
        receptors_by_ligand.setdefault(ligand_id, set()).add(receptor_id)

        values: list[float] = []
        for seed_id in seed_ids:
            field = f"{seed_id}_representative_score"
            try:
                score = float(row[field])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"invalid seed score for {key}: {field}") from exc
            if not math.isfinite(score):
                raise ValueError(f"non-finite seed score for {key}: {field}")
            values.append(score)

        observed_minimum = min(values)
        observed_median = statistics.median(values)
        observed_maximum = max(values)
        observed_range = observed_maximum - observed_minimum
        expected_values = {
            "minimum_representative_score": observed_minimum,
            "median_representative_score": observed_median,
            "maximum_representative_score": observed_maximum,
            "seed_score_range": observed_range,
        }
        for field, expected_value in expected_values.items():
            try:
                stored_value = float(row[field])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"invalid aggregate value for {key}: {field}") from exc
            if not close(stored_value, expected_value):
                raise ValueError(f"aggregate value differs for {key}: {field}")

        reasons: list[str] = []
        if any(value >= 0.0 for value in values):
            reasons.append("nonnegative_seed_score")
            nonnegative_pairs += 1
        if observed_range > maximum_seed_range:
            reasons.append("seed_score_range_exceeded")
            excessive_range_pairs += 1
        ranges.append(observed_range)
        if reasons:
            flags.append(
                {
                    "ligand_id": ligand_id,
                    "receptor_id": receptor_id,
                    "selection_role": role,
                    "flag_reasons": ";".join(reasons),
                    "minimum_seed_score": observed_minimum,
                    "median_seed_score": observed_median,
                    "maximum_seed_score": observed_maximum,
                    "seed_score_range": observed_range,
                }
            )

    if len(ligand_metadata) != expected_ligand_count:
        raise ValueError(
            f"expected {expected_ligand_count} ligands, got {len(ligand_metadata)}"
        )
    if any(
        len(receptors) != expected_receptor_count
        for receptors in receptors_by_ligand.values()
    ):
        raise ValueError("receptor coverage differs by ligand")
    admission_passed = (
        nonnegative_pairs <= maximum_nonnegative_pairs
        and excessive_range_pairs == 0
    )
    return flags, {
        "aggregated_pair_count": len(rows),
        "ligand_count": len(ligand_metadata),
        "receptor_count": expected_receptor_count,
        "seed_count": len(seed_ids),
        "nonnegative_seed_score_pair_count": nonnegative_pairs,
        "excessive_seed_range_pair_count": excessive_range_pairs,
        "flagged_pair_count": len(flags),
        "seed_score_range_kcal_per_mol": {
            "median": statistics.median(ranges),
            "maximum": max(ranges),
        },
        "admission_passed": admission_passed,
    }


def validate_matrix(
    matrix_rows: list[dict[str, str]],
    aggregated_rows: list[dict[str, str]],
    receptor_ids: list[str],
    score_field: str,
) -> None:
    expected_by_ligand: dict[str, dict[str, float]] = {}
    for row in aggregated_rows:
        expected_by_ligand.setdefault(row["ligand_id"], {})[row["receptor_id"]] = float(
            row[score_field]
        )
    observed_ids = [row["ligand_id"] for row in matrix_rows]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("score matrix contains duplicate ligand IDs")
    if set(observed_ids) != set(expected_by_ligand):
        raise ValueError("score matrix ligand IDs differ from aggregated rows")
    for row in matrix_rows:
        ligand_id = row["ligand_id"]
        for receptor_id in receptor_ids:
            try:
                observed = float(row[receptor_id])
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"invalid matrix score: {ligand_id}/{receptor_id}"
                ) from exc
            expected = expected_by_ligand[ligand_id].get(receptor_id)
            if expected is None or not close(observed, expected):
                raise ValueError(f"matrix score differs: {ligand_id}/{receptor_id}")


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--aggregation-summary", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--flagged-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in (args.output_summary, args.flagged_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists; use --overwrite: {path}")

    preregistration = load_json(args.preregistration)
    summary = load_json(args.aggregation_summary)
    admission = preregistration["matrix_admission"]
    data_roles = preregistration["data_roles"]
    frozen_inputs = preregistration["frozen_inputs"]
    assert isinstance(admission, dict)
    assert isinstance(data_roles, dict)
    assert isinstance(frozen_inputs, dict)
    if summary.get("status") != "ok":
        raise ValueError("seed aggregation did not pass")
    if int(summary.get("locked_test_manifest_rows", -1)) != 0:
        raise ValueError("aggregation contains locked test rows")

    expected_seed_count = int(admission["required_seed_count"])
    expected_pairs = int(admission["required_pairs_per_seed"])
    if int(summary.get("seed_count", 0)) != expected_seed_count:
        raise ValueError("aggregation seed count differs")
    if int(summary.get("aggregated_pair_count", 0)) != expected_pairs:
        raise ValueError("aggregation pair count differs")
    seed_evidence = summary.get("seed_evidence")
    if not isinstance(seed_evidence, list) or len(seed_evidence) != expected_seed_count:
        raise ValueError("aggregation seed evidence differs")
    seed_ids = [str(row["seed_id"]) for row in seed_evidence]

    aggregation_config = frozen_inputs["seed_aggregation_config"]
    ligand_manifest = frozen_inputs["ligand_manifest"]
    assert isinstance(aggregation_config, dict)
    assert isinstance(ligand_manifest, dict)
    for name, spec in (
        ("seed aggregation config", aggregation_config),
        ("ligand manifest", ligand_manifest),
    ):
        path = Path(str(spec["path"]))
        if not path.is_file() or file_sha256(path) != str(spec["sha256"]).upper():
            raise ValueError(f"{name} is missing or its hash differs")

    output_specs = summary.get("outputs")
    if not isinstance(output_specs, dict):
        raise ValueError("aggregation summary has no outputs")
    required_outputs = {
        "aggregated_long_csv",
        "primary_median_matrix_csv",
        "sensitivity_minimum_matrix_csv",
    }
    if not required_outputs.issubset(output_specs):
        raise ValueError("aggregation summary is missing output records")
    paths: dict[str, Path] = {}
    for key in required_outputs:
        spec = output_specs[key]
        if not isinstance(spec, dict):
            raise ValueError(f"invalid aggregation output record: {key}")
        path = Path(str(spec["path"]))
        if not path.is_file() or file_sha256(path) != str(spec["sha256"]).upper():
            raise ValueError(f"aggregation output is missing or its hash differs: {key}")
        paths[key] = path

    allowed_roles = {
        str(data_roles[role]["selection_role"])
        for role in ("train", "validation")
    }
    expected_ligands = int(summary["ligand_count"])
    expected_receptors = int(summary["receptor_count"])
    aggregated_rows = read_csv(paths["aggregated_long_csv"])
    flags, audit = audit_aggregated_rows(
        aggregated_rows,
        seed_ids,
        expected_ligands,
        expected_receptors,
        allowed_roles,
        int(admission["maximum_allowed_nonnegative_score_pairs"]),
        float(admission["maximum_allowed_seed_score_range_kcal_per_mol"]),
    )
    receptor_ids = [str(value) for value in preregistration["receptor_ids"]]
    if len(receptor_ids) != expected_receptors:
        raise ValueError("preregistered receptor count differs")
    validate_matrix(
        read_csv(paths["primary_median_matrix_csv"]),
        aggregated_rows,
        receptor_ids,
        "median_representative_score",
    )
    validate_matrix(
        read_csv(paths["sensitivity_minimum_matrix_csv"]),
        aggregated_rows,
        receptor_ids,
        "minimum_representative_score",
    )

    write_flags(args.flagged_output, flags)
    result = {
        "schema_version": "1.0",
        "status": "matrix_admission_passed" if audit["admission_passed"] else "matrix_admission_rejected",
        "preregistration": {
            "path": args.preregistration.as_posix(),
            "sha256": file_sha256(args.preregistration),
        },
        "aggregation_summary": {
            "path": args.aggregation_summary.as_posix(),
            "sha256": file_sha256(args.aggregation_summary),
        },
        "thresholds": admission,
        "audit": audit,
        "outputs": {
            "flagged_pairs_csv": {
                "path": args.flagged_output.as_posix(),
                "sha256": file_sha256(args.flagged_output),
            }
        },
        "enrichment_metrics_calculated": False,
        "test_evaluated": False,
        "interpretation_note": (
            "This label-blind matrix admission audit checks execution and seed "
            "stability only. It does not establish enrichment, receptor "
            "complementarity, QUBO benefit, or biological activity."
        ),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if audit["admission_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
