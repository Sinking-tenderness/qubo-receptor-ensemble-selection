"""Diagnose every frozen MAPK14 uncertainty-QUBO candidate after gate failure."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
    from .prepare_receptor import file_sha256
    from .run_stage05_mk14_uncertainty_qubo_gate import (
        MATRIX_IDS,
        SEED_IDS,
        audit_inputs,
        checked_input_paths,
        fit_qubo,
        load_config,
        make_context,
        qubo_candidate_configs,
        score_records,
        write_csv,
        write_json,
    )
    from .run_stage05_mk14_method_gate import make_frozen_group_folds
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids
    from prepare_receptor import file_sha256
    from run_stage05_mk14_uncertainty_qubo_gate import (
        MATRIX_IDS,
        SEED_IDS,
        audit_inputs,
        checked_input_paths,
        fit_qubo,
        load_config,
        make_context,
        qubo_candidate_configs,
        score_records,
        write_csv,
        write_json,
    )
    from run_stage05_mk14_method_gate import make_frozen_group_folds


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def load_diagnostic_config(path: Path) -> dict[str, object]:
    config = read_json(path)
    required = {
        "schema_version",
        "experiment_id",
        "purpose",
        "inputs",
        "input_sha256",
        "expected",
        "diagnostic_classification",
        "outputs",
        "interpretation_boundary",
    }
    if set(config) != required:
        raise ValueError("diagnostic config keys differ")
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    assert isinstance(inputs, dict)
    assert isinstance(hashes, dict)
    if set(inputs) != set(hashes):
        raise ValueError("diagnostic input paths and hashes differ")
    for key, value in inputs.items():
        path_value = Path(str(value))
        if not path_value.is_file():
            raise FileNotFoundError(path_value)
        if file_sha256(path_value) != str(hashes[key]).upper():
            raise ValueError(f"diagnostic input SHA-256 differs: {key}")
    return config


def add_records(
    destination: dict[str, dict[str, dict[str, object]]],
    matrix_id: str,
    records: dict[str, dict[str, object]],
) -> None:
    overlap = set(destination[matrix_id]) & set(records)
    if overlap:
        raise ValueError(f"duplicate candidate OOF records: {matrix_id}")
    destination[matrix_id].update(records)


def candidate_diagnostics(
    candidate: dict[str, object],
    contexts: list[dict[str, object]],
    full_context: dict[str, object],
    receptor_ids: list[str],
    model: dict[str, object],
) -> dict[str, object]:
    qubo_records = {matrix_id: {} for matrix_id in MATRIX_IDS}
    linear_records = {matrix_id: {} for matrix_id in MATRIX_IDS}
    fold_subsets: list[list[str]] = []
    fold_linear_subsets: list[list[str]] = []
    fold_jaccards: list[float] = []
    for context in contexts:
        subset, details = fit_qubo(candidate, context, receptor_ids, model)
        linear_subset = tuple(details["matched_linear_subset"])
        fold_subsets.append(list(subset))
        fold_linear_subsets.append(list(linear_subset))
        fold_jaccards.append(float(details["seed_pairwise_jaccard"]["mean"]))
        for matrix_id in MATRIX_IDS:
            rows = context["matrices"][matrix_id]["validation"]
            add_records(
                qubo_records,
                matrix_id,
                score_records(
                    rows, subset, str(candidate["aggregation"])
                ),
            )
            add_records(
                linear_records,
                matrix_id,
                score_records(
                    rows, linear_subset, str(candidate["aggregation"])
                ),
            )
    qubo_metrics = {
        matrix_id: ranked_metrics_with_ids(records)
        for matrix_id, records in qubo_records.items()
    }
    linear_metrics = {
        matrix_id: ranked_metrics_with_ids(records)
        for matrix_id, records in linear_records.items()
    }
    seed_deltas = {
        seed: float(qubo_metrics[seed]["bedroc_alpha_20"])
        - float(linear_metrics[seed]["bedroc_alpha_20"])
        for seed in SEED_IDS
    }
    primary_delta = float(qubo_metrics["primary"]["bedroc_alpha_20"]) - float(
        linear_metrics["primary"]["bedroc_alpha_20"]
    )
    mean_seed_delta = statistics.fmean(seed_deltas.values())
    worst_seed_delta = min(seed_deltas.values())
    final_subset, final_details = fit_qubo(
        candidate, full_context, receptor_ids, model
    )
    final_linear_subset = tuple(final_details["matched_linear_subset"])
    checks = {
        "primary_nonworse": primary_delta >= 0.0,
        "mean_seed_nonworse": mean_seed_delta >= 0.0,
        "every_seed_nonworse": worst_seed_delta >= 0.0,
        "full_subset_differs_from_linear": final_subset
        != final_linear_subset,
        "fold_seed_jaccard": statistics.fmean(fold_jaccards) >= 0.5,
        "final_seed_jaccard": float(
            final_details["seed_pairwise_jaccard"]["mean"]
        )
        >= 0.5,
    }
    return {
        "family": candidate["family"],
        "target_size": candidate["target_size"],
        "aggregation": candidate["aggregation"],
        "weights": json.dumps(candidate["weights"], sort_keys=True),
        "fold_subsets": json.dumps(fold_subsets),
        "fold_linear_subsets": json.dumps(fold_linear_subsets),
        "fold_subset_difference_count": sum(
            subset != linear
            for subset, linear in zip(fold_subsets, fold_linear_subsets)
        ),
        "final_subset": "+".join(final_subset),
        "final_linear_subset": "+".join(final_linear_subset),
        "fold_seed_fit_mean_pairwise_jaccard": statistics.fmean(
            fold_jaccards
        ),
        "final_seed_fit_mean_pairwise_jaccard": final_details[
            "seed_pairwise_jaccard"
        ]["mean"],
        "primary_qubo_bedroc": qubo_metrics["primary"]["bedroc_alpha_20"],
        "primary_linear_bedroc": linear_metrics["primary"]["bedroc_alpha_20"],
        "primary_bedroc_delta": primary_delta,
        **{
            f"{seed}_qubo_bedroc": qubo_metrics[seed]["bedroc_alpha_20"]
            for seed in SEED_IDS
        },
        **{
            f"{seed}_linear_bedroc": linear_metrics[seed]["bedroc_alpha_20"]
            for seed in SEED_IDS
        },
        **{f"{seed}_bedroc_delta": value for seed, value in seed_deltas.items()},
        "mean_seed_bedroc_delta": mean_seed_delta,
        "worst_seed_bedroc_delta": worst_seed_delta,
        **checks,
        "all_diagnostic_checks": all(checks.values()),
    }


def diagnostic_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        -float(row["worst_seed_bedroc_delta"]),
        -float(row["primary_bedroc_delta"]),
        -float(row["mean_seed_bedroc_delta"]),
        -float(row["primary_qubo_bedroc"]),
        int(row["target_size"]),
        str(row["family"]),
        str(row["aggregation"]),
        str(row["weights"]),
    )


def compact_candidate(row: dict[str, object]) -> dict[str, object]:
    keys = (
        "family",
        "target_size",
        "aggregation",
        "weights",
        "final_subset",
        "final_linear_subset",
        "fold_subset_difference_count",
        "primary_qubo_bedroc",
        "primary_linear_bedroc",
        "primary_bedroc_delta",
        "mean_seed_bedroc_delta",
        "worst_seed_bedroc_delta",
        "fold_seed_fit_mean_pairwise_jaccard",
        "final_seed_fit_mean_pairwise_jaccard",
        "all_diagnostic_checks",
    )
    return {key: row[key] for key in keys}


def run_diagnostic(
    config_path: Path, overwrite: bool = False
) -> dict[str, object]:
    config = load_diagnostic_config(config_path)
    expected = config["expected"]
    outputs = config["outputs"]
    assert isinstance(expected, dict)
    assert isinstance(outputs, dict)
    prereg_path = Path(str(config["inputs"]["preregistration"]))
    gate_result = read_json(Path(str(config["inputs"]["tracked_gate_result"])))
    gate_summary = read_json(Path(str(config["inputs"]["gate_summary"])))
    if (
        gate_result.get("status") != expected["failed_gate_status"]
        or gate_summary.get("status") != expected["failed_gate_status"]
        or int(gate_result["data_boundary"]["validation_rows_read"]) != 0
        or int(gate_result["data_boundary"]["test_rows_read"]) != 0
    ):
        raise ValueError("the frozen failed gate evidence changed")
    prereg = load_config(prereg_path)
    paths = checked_input_paths(prereg)
    audited = audit_inputs(prereg, paths)
    receptor_ids = [str(value) for value in prereg["receptor_ids"]]
    model = prereg["model"]
    cv = prereg["cross_validation"]
    assert isinstance(model, dict)
    assert isinstance(cv, dict)
    manifest_rows = audited["manifest_rows"]
    matrices = audited["matrices"]
    assert isinstance(manifest_rows, list)
    assert isinstance(matrices, dict)
    if len(manifest_rows) != int(expected["ligand_count"]):
        raise ValueError("diagnostic train ligand count differs")
    matrices_by_id = {
        matrix_id: {str(row["ligand_id"]): row for row in rows}
        for matrix_id, rows in matrices.items()
    }
    ligand_ids = {row["ligand_id"] for row in manifest_rows}
    assignments = make_frozen_group_folds(
        manifest_rows,
        int(cv["outer_fold_count"]),
        int(cv["fold_seed"]),
    )
    contexts = []
    for fold in range(int(expected["fold_count"])):
        validation_ids = {
            ligand_id
            for ligand_id, assigned in assignments.items()
            if assigned == fold
        }
        contexts.append(
            make_context(
                ligand_ids - validation_ids,
                validation_ids,
                matrices_by_id,
                receptor_ids,
                model,
            )
        )
    full_context = make_context(
        ligand_ids, set(), matrices_by_id, receptor_ids, model
    )
    candidates = qubo_candidate_configs(model)
    if len(candidates) != int(expected["candidate_count"]):
        raise ValueError("diagnostic candidate count differs")

    rows = [
        candidate_diagnostics(
            candidate, contexts, full_context, receptor_ids, model
        )
        for candidate in candidates
    ]
    viable = [row for row in rows if row["all_diagnostic_checks"]]
    best = min(rows, key=diagnostic_sort_key)
    status = (
        "posthoc_candidate_family_contains_viable_train_candidate"
        if viable
        else "posthoc_candidate_family_has_no_viable_train_candidate"
    )
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": status,
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "implementation": {
            "path": f"scripts/{Path(__file__).name}",
            "sha256": file_sha256(Path(__file__)),
        },
        "failed_gate": {
            "status": gate_result["status"],
            "tracked_result_sha256": file_sha256(
                Path(str(config["inputs"]["tracked_gate_result"]))
            ),
            "source_summary_sha256": file_sha256(
                Path(str(config["inputs"]["gate_summary"]))
            ),
            "retroactively_changed": False,
        },
        "audit": {
            "candidate_count": len(rows),
            "ligand_count": len(ligand_ids),
            "fold_count": len(contexts),
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "e32_cells_replaced": 0,
            "e64_scores_used_in_matrix": 0,
        },
        "counts": {
            "primary_nonworse": sum(row["primary_nonworse"] for row in rows),
            "mean_seed_nonworse": sum(
                row["mean_seed_nonworse"] for row in rows
            ),
            "every_seed_nonworse": sum(
                row["every_seed_nonworse"] for row in rows
            ),
            "full_subset_differs_from_linear": sum(
                row["full_subset_differs_from_linear"] for row in rows
            ),
            "passes_all_diagnostic_checks": len(viable),
        },
        "best_by_worst_seed_delta": compact_candidate(best),
        "top_viable_candidates": [
            compact_candidate(row)
            for row in sorted(viable, key=diagnostic_sort_key)[:10]
        ],
        "next_action": (
            "A viable frozen-grid train candidate exists, but this post hoc scan cannot promote it. Design and preregister a nested delta-aware selector before any validation access."
            if viable
            else "No frozen-grid candidate beats its own linear part robustly. Redesign the pairwise objective before another train-only gate."
        ),
        "interpretation_note": config["interpretation_boundary"],
    }
    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    existing = [
        path
        for key, path in output_paths.items()
        if key != "run_directory" and path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError("diagnostic outputs exist; use --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()
    output_paths["run_directory"].mkdir(parents=True, exist_ok=True)
    write_csv(output_paths["candidate_diagnostics_csv"], rows)
    summary["outputs"] = {
        "candidate_diagnostics_csv": {
            "path": output_paths["candidate_diagnostics_csv"].as_posix(),
            "sha256": file_sha256(output_paths["candidate_diagnostics_csv"]),
        }
    }
    write_json(output_paths["summary_json"], summary)
    write_json(output_paths["tracked_summary_json"], summary)
    print(
        json.dumps(
            {
                "status": status,
                "counts": summary["counts"],
                "best_by_worst_seed_delta": summary[
                    "best_by_worst_seed_delta"
                ],
                "validation_rows_read": 0,
                "test_rows_read": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_diagnostic(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
