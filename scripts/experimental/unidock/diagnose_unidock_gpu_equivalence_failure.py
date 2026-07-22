"""Diagnose the frozen train-only MAPK14 Uni-Dock equivalence failure."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

try:
    from .audit_unidock_gpu_equivalence import (
        best_receptor_agreement,
        group_metrics,
        pearson,
        quantile,
        spearman,
    )
    from .run_unidock_gpu_equivalence import (
        MACROCYCLE_CLOSURE_ATOM_TYPE,
        file_sha256,
        read_csv,
        read_json,
        relative_path,
        rooted_path,
        validate_inputs,
        write_json,
    )
except ImportError:
    from audit_unidock_gpu_equivalence import (
        best_receptor_agreement,
        group_metrics,
        pearson,
        quantile,
        spearman,
    )
    from run_unidock_gpu_equivalence import (
        MACROCYCLE_CLOSURE_ATOM_TYPE,
        file_sha256,
        read_csv,
        read_json,
        relative_path,
        rooted_path,
        validate_inputs,
        write_json,
    )


def verified_descriptor(root: Path, descriptor: dict[str, object]) -> Path:
    path = rooted_path(root, str(descriptor["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = file_sha256(path)
    expected = str(descriptor["sha256"]).upper()
    if observed != expected:
        raise ValueError(f"SHA-256 differs for {path}: {observed}")
    return path


def comparison_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("comparison metrics require at least one row")
    groups = group_metrics(rows)
    cpu = [float(row["cpu_vina_e32_score"]) for row in rows]
    gpu = [float(row["gpu_unidock_detail_score"]) for row in rows]
    deltas = [float(row["score_delta_gpu_minus_cpu"]) for row in rows]
    absolute = [abs(value) for value in deltas]
    label_values: dict[str, list[float]] = {}
    for row, delta in zip(rows, deltas):
        label_values.setdefault(str(row["label"]), []).append(delta)
    label_means = {
        label: statistics.fmean(values)
        for label, values in sorted(label_values.items())
    }
    active_decoy_gap = None
    if {"active", "decoy"}.issubset(label_means):
        active_decoy_gap = abs(
            label_means["active"] - label_means["decoy"]
        )
    lowest_spearman_groups = sorted(
        groups,
        key=lambda row: (
            float(row["spearman"]),
            str(row["seed_id"]),
            str(row["receptor_id"]),
        ),
    )[:5]
    return {
        "pair_count": len(rows),
        "ligand_count": len({str(row["ligand_id"]) for row in rows}),
        "label_pair_counts": dict(
            sorted(Counter(str(row["label"]) for row in rows).items())
        ),
        "overall_pearson": pearson(cpu, gpu),
        "overall_spearman": spearman(cpu, gpu),
        "mean_signed_delta_kcal_per_mol": statistics.fmean(deltas),
        "median_absolute_delta_kcal_per_mol": statistics.median(absolute),
        "p95_absolute_delta_kcal_per_mol": quantile(absolute, 0.95),
        "maximum_absolute_delta_kcal_per_mol": max(absolute),
        "minimum_group_spearman": min(
            float(row["spearman"]) for row in groups
        ),
        "median_group_spearman": statistics.median(
            float(row["spearman"]) for row in groups
        ),
        "median_group_top5pct_overlap": statistics.median(
            float(row["top5pct_overlap"]) for row in groups
        ),
        "median_group_top10pct_overlap": statistics.median(
            float(row["top10pct_overlap"]) for row in groups
        ),
        "lowest_spearman_groups": [
            {
                "seed_id": row["seed_id"],
                "receptor_id": row["receptor_id"],
                "spearman": row["spearman"],
                "p95_absolute_delta_kcal_per_mol": row[
                    "p95_absolute_delta_kcal_per_mol"
                ],
                "top5pct_overlap": row["top5pct_overlap"],
            }
            for row in lowest_spearman_groups
        ],
        "mean_signed_delta_by_label_kcal_per_mol": label_means,
        "active_decoy_mean_delta_gap_kcal_per_mol": active_decoy_gap,
        "best_receptor_identity": best_receptor_agreement(rows),
    }


def manifest_pseudoatom_summary(
    path: Path, manifest_id: str
) -> dict[str, object]:
    rows = read_csv(path)
    affected = [
        row
        for row in rows
        if any(
            MACROCYCLE_CLOSURE_ATOM_TYPE.fullmatch(atom_type)
            for atom_type in row.get("pdbqt_atom_types", "").split(";")
            if atom_type
        )
    ]
    return {
        "manifest_id": manifest_id,
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "ligand_count": len(rows),
        "macrocycle_closure_pseudoatom_ligand_count": len(affected),
        "affected_label_counts": dict(
            sorted(Counter(row["label"] for row in affected).items())
        ),
    }


def verify_run_integrity(
    root: Path,
    source_config_path: Path,
    gpu_summary_path: Path,
    equivalence_summary_path: Path,
) -> dict[str, object]:
    gpu_summary = read_json(gpu_summary_path)
    equivalence_summary = read_json(equivalence_summary_path)
    checked = 0

    def check(path_value: str, expected_hash: str) -> None:
        nonlocal checked
        path = rooted_path(root, path_value)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if actual != expected_hash.upper():
            raise ValueError(f"embedded SHA-256 differs for {path}: {actual}")
        checked += 1

    for descriptor in gpu_summary["outputs"].values():
        check(str(descriptor["path"]), str(descriptor["sha256"]))
    for descriptor in equivalence_summary["outputs"].values():
        check(str(descriptor["path"]), str(descriptor["sha256"]))
    check(
        relative_path(root, source_config_path),
        str(gpu_summary["config"]["sha256"]),
    )
    check(
        relative_path(root, source_config_path),
        str(equivalence_summary["config"]["sha256"]),
    )

    run_directory = gpu_summary_path.parent
    batch_summaries = sorted(run_directory.glob("batches/*/*/batch_summary.json"))
    if len(batch_summaries) != int(gpu_summary["batch_count"]):
        raise ValueError("archived batch-summary count differs")
    batch_index = {
        (row["seed_id"], row["receptor_id"]): row
        for row in read_csv(run_directory / "gpu_batch_runs.csv")
    }
    index_matches = 0
    for path in batch_summaries:
        summary = read_json(path)
        check(str(summary["log_path"]), str(summary["log_sha256"]))
        check(str(summary["scores_path"]), str(summary["scores_sha256"]))
        key = (str(summary["seed_id"]), str(summary["receptor_id"]))
        indexed = batch_index.get(key)
        if indexed is None:
            raise ValueError(f"batch is absent from gpu_batch_runs.csv: {key}")
        if (
            indexed["log_sha256"] != summary["log_sha256"]
            or indexed["scores_sha256"] != summary["scores_sha256"]
            or indexed["status"] != summary["status"]
        ):
            raise ValueError(f"batch index differs from batch summary: {key}")
        index_matches += 1
    return {
        "status": "verified",
        "embedded_hash_check_count": checked,
        "batch_summary_count": len(batch_summaries),
        "batch_index_match_count": index_matches,
    }


def log_warning_counts(run_directory: Path) -> dict[str, int]:
    counts = {
        "omp_num_threads_invalid_value": 0,
        "add_to_output_container": 0,
        "coordinate_size_mismatch": 0,
    }
    for path in sorted(run_directory.glob("batches/*/*/unidock.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        counts["omp_num_threads_invalid_value"] += text.count(
            "Invalid value for environment variable OMP_NUM_THREADS"
        )
        counts["add_to_output_container"] += text.count(
            "WARNING: in add_to_output_container"
        )
        counts["coordinate_size_mismatch"] += text.count("t.coords.size()=")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    diagnostic_config_path = args.config.resolve()
    diagnostic_config = read_json(diagnostic_config_path)
    inputs = diagnostic_config["inputs"]
    source_config_path = verified_descriptor(
        root, inputs["source_equivalence_config"]
    )
    gpu_summary_path = verified_descriptor(root, inputs["gpu_run_summary"])
    pairwise_path = verified_descriptor(root, inputs["equivalence_pairwise"])
    equivalence_summary_path = verified_descriptor(
        root, inputs["equivalence_summary"]
    )
    source_config = read_json(source_config_path)
    _, _, input_audit = validate_inputs(root, source_config)
    macrocycle_ligands = input_audit[
        "macrocycle_closure_pseudoatom_ligands"
    ]
    macrocycle_ids = {
        str(row["ligand_id"]) for row in macrocycle_ligands
    }
    if not macrocycle_ids:
        raise ValueError("no macrocycle closure pseudoatom ligand was detected")

    pair_rows = [dict(row) for row in read_csv(pairwise_path)]
    for row in pair_rows:
        for key in (
            "cpu_vina_e32_score",
            "gpu_unidock_detail_score",
            "score_delta_gpu_minus_cpu",
            "absolute_score_delta",
        ):
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f"non-finite pairwise value: {key}")
            row[key] = value
    affected = [
        row for row in pair_rows if str(row["ligand_id"]) in macrocycle_ids
    ]
    unaffected = [
        row for row in pair_rows if str(row["ligand_id"]) not in macrocycle_ids
    ]
    expected_affected = (
        len(macrocycle_ids)
        * int(source_config["expected"]["receptor_count"])
        * int(source_config["expected"]["seed_count"])
    )
    if len(affected) != expected_affected:
        raise ValueError("macrocycle-affected pair count differs")

    affected_absolute = [float(row["absolute_score_delta"]) for row in affected]
    large_delta = float(diagnostic_config["diagnostic_thresholds"]["large_absolute_delta_kcal_per_mol"])
    large_delta_rows = [
        row
        for row in pair_rows
        if float(row["absolute_score_delta"]) > large_delta
    ]
    formal_summary = read_json(equivalence_summary_path)
    gpu_summary = read_json(gpu_summary_path)
    run_directory = gpu_summary_path.parent
    archive = diagnostic_config["source_archive"]
    archive_verified = False
    if args.archive is not None:
        if not args.archive.is_file():
            raise FileNotFoundError(args.archive)
        archive_hash = file_sha256(args.archive)
        if archive_hash != str(archive["sha256"]).upper():
            raise ValueError(f"source archive SHA-256 differs: {archive_hash}")
        archive_verified = True

    prevalence = []
    for descriptor in diagnostic_config["prevalence_manifests"]:
        path = verified_descriptor(root, descriptor)
        item = manifest_pseudoatom_summary(path, str(descriptor["manifest_id"]))
        item["path"] = relative_path(root, path)
        prevalence.append(item)

    integrity = verify_run_integrity(
        root,
        source_config_path,
        gpu_summary_path,
        equivalence_summary_path,
    )
    output: dict[str, object] = {
        "schema_version": "1.0",
        "experiment_id": diagnostic_config["experiment_id"],
        "status": "macrocycle_input_incompatibility_confirmed",
        "operation": "post-gate consumed-train failure diagnosis; the frozen equivalence gate is not recalculated or changed",
        "diagnostic_config": {
            "path": relative_path(root, diagnostic_config_path),
            "sha256": file_sha256(diagnostic_config_path),
        },
        "source_archive": {
            "filename": archive["filename"],
            "sha256": archive["sha256"],
            "verified_from_local_archive": archive_verified,
        },
        "integrity": integrity,
        "gpu_execution": {
            "status": gpu_summary["status"],
            "completed_pair_count": gpu_summary["gpu_pair_count"],
            "batch_count": gpu_summary["batch_count"],
            "elapsed_seconds": gpu_summary["gpu_batch_elapsed_seconds_total"],
            "pairs_per_second": gpu_summary["gpu_pairs_per_second"],
            "speedup_vs_recorded_32vcpu": formal_summary["throughput"]["speedup_vs_recorded_32vcpu"],
        },
        "formal_gate": {
            "status": formal_summary["status"],
            "all_gate_checks_passed": formal_summary[
                "all_gate_checks_passed"
            ],
            "gate_checks": formal_summary["gate_checks"],
            "decision_unchanged": True,
            "unidock_admitted_as_cpu_vina_replacement": False,
        },
        "macrocycle_failure": {
            "ligand_count": len(macrocycle_ids),
            "label_counts": dict(
                sorted(Counter(str(row["label"]) for row in macrocycle_ligands).items())
            ),
            "ligands": macrocycle_ligands,
            "affected_pair_count": len(affected),
            "minimum_absolute_delta_kcal_per_mol": min(affected_absolute),
            "median_absolute_delta_kcal_per_mol": statistics.median(
                affected_absolute
            ),
            "maximum_absolute_delta_kcal_per_mol": max(affected_absolute),
            "affected_pairs_above_large_delta_threshold": sum(
                value > large_delta for value in affected_absolute
            ),
            "large_delta_threshold_kcal_per_mol": large_delta,
            "all_large_delta_pairs_are_macrocycle_pseudoatom_ligands": {
                str(row["ligand_id"]) for row in large_delta_rows
            }.issubset(macrocycle_ids),
            "gpu_score_range_kcal_per_mol": {
                "minimum": min(
                    float(row["gpu_unidock_detail_score"]) for row in affected
                ),
                "maximum": max(
                    float(row["gpu_unidock_detail_score"]) for row in affected
                ),
            },
            "log_warning_counts": log_warning_counts(run_directory),
            "interpretation": "The four Train-160 ligands containing Meeko CG0/G0 flexible-macrocycle closure pseudoatoms account for all 60 score deltas above 10 kcal/mol. Their Uni-Dock scores are nonphysical and indicate an engine/input compatibility failure, not binding evidence.",
        },
        "diagnostic_nonmacrocycle_subset": {
            **comparison_metrics(unaffected),
            "excluded_ligand_ids": sorted(macrocycle_ids),
            "formal_gate_replacement": False,
            "interpretation": "This label-aware exclusion is diagnostic only. It cannot replace the frozen gate or authorize validation because the excluded Train-160 rows are all decoys and the minimum group Spearman still misses the frozen threshold.",
        },
        "pseudoatom_prevalence": prevalence,
        "data_boundary": {
            "source_split": "consumed development train",
            "validation_docking_scores_read": False,
            "test_rows_read": False,
            "enrichment_metrics_calculated": False,
        },
        "decision": {
            "immediate": "Do not run Train-696 or fresh validation with the failed Uni-Dock protocol.",
            "next_train_only_diagnostic": "Re-prepare the affected consumed-train macrocycles with Meeko --rigid_macrocycles and separately test a higher-search Uni-Dock profile on the nonmacrocycle groups that missed rank equivalence.",
            "fallback": "Retain official AutoDock Vina 1.2.7 CPU evidence if the revised train-only GPU protocol cannot pass a newly preregistered full equivalence gate.",
        },
    }
    output_path = rooted_path(root, str(diagnostic_config["output"]["path"]))
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"diagnostic output exists: {output_path}")
    write_json(output_path, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
