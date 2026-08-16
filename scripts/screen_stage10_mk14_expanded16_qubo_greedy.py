"""Screen frozen QUBO objectives for greedy failures on Stage 09 Train-696."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import read_json, write_json  # noqa: F401 (deduped)
import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.normalized_receptor_qubo import build_coefficients, coefficient_energy
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage05_mk14_uncertainty_qubo_gate import (
    MATRIX_IDS,
    SEED_IDS,
    make_context,
    metrics_for_context,
    pair_synergy_terms_for_aggregation,
    pair_utility_terms_for_aggregation,
    robust_metric_summary,
)


TOLERANCE = 1e-12




def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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




def rooted(root: Path, value: str) -> Path:
    path = (root / value.replace("\\", "/")).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path leaves repository root: {value}") from error
    return path


def verified(root: Path, descriptor: dict[str, object]) -> Path:
    path = rooted(root, str(descriptor["path"]))
    if not path.is_file() or file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"Stage 10 input identity differs: {path}")
    return path


def fixed_cardinality_exact(
    coefficients: dict[str, object], receptor_ids: list[str], target_size: int
) -> tuple[tuple[str, ...], float]:
    return min(
        (
            (subset, coefficient_energy(subset, coefficients))
            for subset in itertools.combinations(sorted(receptor_ids), target_size)
        ),
        key=lambda item: (item[1], item[0]),
    )


def fixed_cardinality_greedy(
    coefficients: dict[str, object], receptor_ids: list[str], target_size: int
) -> tuple[tuple[str, ...], float, list[dict[str, object]]]:
    selected: tuple[str, ...] = ()
    path: list[dict[str, object]] = []
    while len(selected) < target_size:
        candidates = [
            tuple(sorted((*selected, receptor_id)))
            for receptor_id in receptor_ids
            if receptor_id not in selected
        ]
        selected = min(
            candidates,
            key=lambda subset: (coefficient_energy(subset, coefficients), subset),
        )
        path.append(
            {
                "step": len(selected),
                "subset": list(selected),
                "energy": coefficient_energy(selected, coefficients),
            }
        )
    return selected, coefficient_energy(selected, coefficients), path


def metric_quality(metrics: dict[str, float]) -> tuple[float, ...]:
    return (
        float(metrics["worst_seed_bedroc"]),
        float(metrics["primary_bedroc"]),
        float(metrics["mean_seed_bedroc"]),
        float(metrics["primary_pr_auc"]),
        float(metrics["primary_roc_auc"]),
    )


def metric_key(metrics: dict[str, float], subset: tuple[str, ...]) -> tuple[object, ...]:
    return (*(-value for value in metric_quality(metrics)), subset)


def metric_greedy_path(
    receptor_ids: list[str],
    maximum_size: int,
    evaluate: Callable[[tuple[str, ...]], dict[str, float]],
) -> dict[int, tuple[str, ...]]:
    selected: tuple[str, ...] = ()
    output: dict[int, tuple[str, ...]] = {}
    while len(selected) < maximum_size:
        candidates = [
            tuple(sorted((*selected, receptor_id)))
            for receptor_id in receptor_ids
            if receptor_id not in selected
        ]
        selected = min(candidates, key=lambda subset: metric_key(evaluate(subset), subset))
        output[len(selected)] = selected
    return output


def metric_exact(
    receptor_ids: list[str],
    target_size: int,
    evaluate: Callable[[tuple[str, ...]], dict[str, float]],
) -> tuple[str, ...]:
    return min(
        itertools.combinations(sorted(receptor_ids), target_size),
        key=lambda subset: metric_key(evaluate(subset), subset),
    )


def matrix_map(
    rows: list[dict[str, str]], receptor_ids: list[str]
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for row in rows:
        ligand_id = row["ligand_id"]
        if ligand_id in output:
            raise ValueError(f"duplicate matrix ligand: {ligand_id}")
        output[ligand_id] = {
            "ligand_id": ligand_id,
            "label": row["label"],
            **{receptor_id: float(row[receptor_id]) for receptor_id in receptor_ids},
        }
    return output


def build_matrices(
    primary_rows: list[dict[str, str]],
    sensitivity_rows: list[dict[str, str]],
    score_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    receptor_ids: list[str],
) -> dict[str, dict[str, dict[str, object]]]:
    manifest_by_id = {row["ligand_id"]: row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise ValueError("Stage 10 ligand manifest contains duplicate IDs")
    matrices = {
        "primary": matrix_map(primary_rows, receptor_ids),
        "sensitivity": matrix_map(sensitivity_rows, receptor_ids),
    }
    for seed_id in SEED_IDS:
        matrices[seed_id] = {
            ligand_id: {"ligand_id": ligand_id, "label": row["label"]}
            for ligand_id, row in manifest_by_id.items()
        }
    seen: set[tuple[str, str, str]] = set()
    for row in score_rows:
        seed_id = row["seed_id"]
        if seed_id not in SEED_IDS:
            raise ValueError(f"unexpected Stage 10 seed: {seed_id}")
        ligand_id = row["ligand_id"]
        receptor_id = row["receptor_id"]
        key = (seed_id, ligand_id, receptor_id)
        if key in seen:
            raise ValueError(f"duplicate Stage 10 score key: {key}")
        seen.add(key)
        matrices[seed_id][ligand_id][receptor_id] = float(row["gpu_score"])
    expected_keys = len(SEED_IDS) * len(manifest_rows) * len(receptor_ids)
    if len(seen) != expected_keys:
        raise ValueError("Stage 10 seed score grid is incomplete")
    ligand_ids = set(manifest_by_id)
    for matrix_id in MATRIX_IDS:
        if set(matrices[matrix_id]) != ligand_ids:
            raise ValueError(f"Stage 10 matrix ligand set differs: {matrix_id}")
        for row in matrices[matrix_id].values():
            if any(receptor_id not in row for receptor_id in receptor_ids):
                raise ValueError(f"Stage 10 matrix receptor grid differs: {matrix_id}")
    return matrices


def terms_for_family(
    context: dict[str, object], family: str, source: str, aggregation: str
) -> dict[str, object]:
    terms = context["terms"] if source == "primary" else context["seed_terms"][source]
    if family == "pair_utility_qubo":
        return pair_utility_terms_for_aggregation(terms, aggregation)
    if family == "pair_synergy_qubo":
        return pair_synergy_terms_for_aggregation(terms, aggregation)
    if family != "coverage_qubo":
        raise ValueError(f"unknown Stage 10 objective family: {family}")
    return terms


def metric_fields(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def delta_fields(
    prefix: str, left: dict[str, float], right: dict[str, float]
) -> dict[str, float]:
    return {
        f"{prefix}_{key}_delta": float(left[key]) - float(right[key])
        for key in left
    }


def summarize_trials(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    families = sorted({str(row["objective_family"]) for row in rows})
    for family in families:
        selected = [row for row in rows if row["objective_family"] == family]
        failures = [row for row in selected if bool(row["strict_objective_failure"])]
        heldout = [row for row in failures if row["context_id"] != "full_train"]
        better_qgreedy = [
            row
            for row in heldout
            if float(row["holdout_exact_vs_qubo_greedy_primary_bedroc_delta"]) > TOLERANCE
            and float(row["holdout_exact_vs_qubo_greedy_mean_seed_bedroc_delta"])
            > TOLERANCE
            and float(row["holdout_exact_vs_qubo_greedy_worst_seed_bedroc_delta"])
            > TOLERANCE
        ]
        better_direct = [
            row
            for row in heldout
            if float(row["holdout_exact_vs_direct_greedy_primary_bedroc_delta"])
            > TOLERANCE
            and float(row["holdout_exact_vs_direct_greedy_mean_seed_bedroc_delta"])
            > TOLERANCE
            and float(row["holdout_exact_vs_direct_greedy_worst_seed_bedroc_delta"])
            > TOLERANCE
        ]
        output.append(
            {
                "objective_family": family,
                "trial_count": len(selected),
                "strict_failure_count": len(failures),
                "strict_failure_rate": len(failures) / len(selected),
                "maximum_objective_regret": max(
                    (float(row["objective_regret"]) for row in failures), default=0.0
                ),
                "heldout_failure_count": len(heldout),
                "heldout_exact_better_than_qubo_greedy_all_robust_count": len(
                    better_qgreedy
                ),
                "heldout_exact_better_than_direct_greedy_all_robust_count": len(
                    better_direct
                ),
            }
        )
    return output


def write_report(path: Path, result: dict[str, object]) -> None:
    lines = [
        "# Stage 10 MAPK14 Expanded16 QUBO-Greedy Screen",
        "",
        "## Scope",
        "",
        "This is a Train-696 diagnostic using only the audited Stage 09 Uni-Dock matrix.",
        "Frozen objective definitions and weights were transferred without retuning.",
        "No validation or test rows were read.",
        "",
        "## Objective Results",
        "",
        "| Objective | Trials | Greedy failures | Max regret | Held-out exact better than QUBO greedy | Held-out exact better than direct greedy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["objective_summary"]:
        lines.append(
            "| {objective_family} | {trial_count} | {strict_failure_count} | "
            "{maximum_objective_regret:.6f} | "
            "{heldout_exact_better_than_qubo_greedy_all_robust_count} | "
            "{heldout_exact_better_than_direct_greedy_all_robust_count} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(result["interpretation_note"]),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = dict(config["implementation"])
    script = verified(root, implementation)
    if script.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 10 implementation path differs")
    inputs = dict(config["inputs"])
    paths = {key: verified(root, dict(value)) for key, value in inputs.items()}
    stage09_summary = read_json(paths["stage09_summary"])
    stage09_audit = read_json(paths["stage09_audit"])
    if stage09_summary.get("status") != "stage09_train696_unidock_matrix_ok":
        raise ValueError("Stage 09 matrix did not pass")
    if stage09_audit.get("status") != "independent_stage09_train696_unidock_matrix_audit_ok":
        raise ValueError("Stage 09 independent audit did not pass")
    if any(int(value) != 0 for value in dict(stage09_audit["data_boundary"]).values()):
        raise ValueError("Stage 09 audit crossed a data boundary")

    receptor_rows = read_csv(paths["receptor_manifest"])
    ligand_rows = read_csv(paths["ligand_manifest"])
    receptor_ids = [row["conformer_id"] for row in receptor_rows]
    expected = dict(config["expected"])
    if len(receptor_ids) != int(expected["receptor_count"]):
        raise ValueError("Stage 10 receptor count differs")
    if len(ligand_rows) != int(expected["ligand_count"]):
        raise ValueError("Stage 10 ligand count differs")
    if Counter(row["label"] for row in ligand_rows) != Counter(
        {key: int(value) for key, value in dict(expected["label_counts"]).items()}
    ):
        raise ValueError("Stage 10 ligand labels differ")
    if {row["split"] for row in ligand_rows} != {"train"}:
        raise ValueError("Stage 10 observed a non-train ligand")
    if {row["selection_role"] for row in ligand_rows} != {
        "development_train_expanded"
    }:
        raise ValueError("Stage 10 selection role differs")

    objective_specs = dict(config["objective_specs"])
    for family, spec_value in objective_specs.items():
        spec = dict(spec_value)
        source_result = read_json(verified(root, dict(spec["source_result"])))
        if source_result.get("status") != spec["required_status"]:
            raise ValueError(f"Stage 10 source objective status differs: {family}")
        selected = dict(source_result["selected_qubo"])
        frozen = dict(spec["frozen_candidate"])
        for key in ("family", "target_size", "aggregation"):
            if selected[key] != frozen[key]:
                raise ValueError(f"Stage 10 frozen objective differs: {family} {key}")
        if {key: float(value) for key, value in selected["weights"].items()} != {
            key: float(value) for key, value in frozen["weights"].items()
        }:
            raise ValueError(f"Stage 10 frozen weights differ: {family}")

    matrices = build_matrices(
        read_csv(paths["primary_matrix"]),
        read_csv(paths["sensitivity_matrix"]),
        read_csv(paths["seed_scores"]),
        ligand_rows,
        receptor_ids,
    )
    screen = dict(config["screen"])
    fold_assignments = make_frozen_group_folds(
        ligand_rows, int(screen["outer_fold_count"]), int(screen["fold_seed"])
    )
    all_ids = {row["ligand_id"] for row in ligand_rows}
    contexts: dict[str, dict[str, object]] = {
        "full_train": make_context(
            all_ids, set(), matrices, receptor_ids, dict(config["model"])
        )
    }
    for fold in range(int(screen["outer_fold_count"])):
        validation_ids = {
            ligand_id
            for ligand_id, assignment in fold_assignments.items()
            if assignment == fold
        }
        contexts[f"outer_fold_{fold}"] = make_context(
            all_ids - validation_ids,
            validation_ids,
            matrices,
            receptor_ids,
            dict(config["model"]),
        )

    metric_cache: dict[
        tuple[str, str, str, tuple[str, ...]], dict[str, float]
    ] = {}

    def evaluate(
        context_id: str, split: str, aggregation: str, subset: tuple[str, ...]
    ) -> dict[str, float]:
        key = (context_id, split, aggregation, subset)
        if key not in metric_cache:
            metrics = metrics_for_context(
                contexts[context_id], subset, aggregation, split
            )
            metric_cache[key] = robust_metric_summary(metrics)
        return metric_cache[key]

    target_sizes = [int(value) for value in screen["target_sizes"]]
    direct_greedy: dict[tuple[str, str, int], tuple[str, ...]] = {}
    direct_exact: dict[tuple[str, str, int], tuple[str, ...]] = {}
    for context_id in contexts:
        for aggregation in screen["direct_metric_aggregations"]:
            path = metric_greedy_path(
                receptor_ids,
                max(target_sizes),
                lambda subset, c=context_id, a=aggregation: evaluate(
                    c, "train", str(a), subset
                ),
            )
            for size in target_sizes:
                direct_greedy[(context_id, str(aggregation), size)] = path[size]
            for size in screen["direct_metric_exact_target_sizes"]:
                direct_exact[(context_id, str(aggregation), int(size))] = metric_exact(
                    receptor_ids,
                    int(size),
                    lambda subset, c=context_id, a=aggregation: evaluate(
                        c, "train", str(a), subset
                    ),
                )

    trial_rows: list[dict[str, object]] = []
    model = dict(config["model"])
    for context_id, context in contexts.items():
        for family, spec_value in objective_specs.items():
            frozen = dict(dict(spec_value)["frozen_candidate"])
            aggregation = str(frozen["aggregation"])
            weights = {
                key: float(value) for key, value in dict(frozen["weights"]).items()
            }
            for source in screen["coefficient_sources"]:
                terms = terms_for_family(context, family, str(source), aggregation)
                for target_size in target_sizes:
                    coefficients = build_coefficients(
                        terms,
                        receptor_ids,
                        target_size,
                        weights,
                        float(model["size_penalty"]),
                    )
                    exact, exact_energy = fixed_cardinality_exact(
                        coefficients, receptor_ids, target_size
                    )
                    greedy, greedy_energy, greedy_path = fixed_cardinality_greedy(
                        coefficients, receptor_ids, target_size
                    )
                    direct = direct_greedy[(context_id, aggregation, target_size)]
                    train_exact = evaluate(context_id, "train", aggregation, exact)
                    train_greedy = evaluate(context_id, "train", aggregation, greedy)
                    train_direct = evaluate(context_id, "train", aggregation, direct)
                    row: dict[str, object] = {
                        "context_id": context_id,
                        "objective_family": family,
                        "coefficient_source": source,
                        "target_size": target_size,
                        "aggregation": aggregation,
                        "exact_subset": "+".join(exact),
                        "qubo_greedy_subset": "+".join(greedy),
                        "direct_metric_greedy_subset": "+".join(direct),
                        "subset_differs": exact != greedy,
                        "exact_objective": exact_energy,
                        "qubo_greedy_objective": greedy_energy,
                        "objective_regret": greedy_energy - exact_energy,
                        "strict_objective_failure": greedy_energy - exact_energy
                        > TOLERANCE,
                        "qubo_greedy_path": json.dumps(greedy_path, sort_keys=True),
                        **metric_fields("train_exact", train_exact),
                        **metric_fields("train_qubo_greedy", train_greedy),
                        **metric_fields("train_direct_greedy", train_direct),
                        **delta_fields(
                            "train_exact_vs_qubo_greedy", train_exact, train_greedy
                        ),
                        **delta_fields(
                            "train_exact_vs_direct_greedy", train_exact, train_direct
                        ),
                    }
                    if context_id != "full_train":
                        holdout_exact = evaluate(
                            context_id, "validation", aggregation, exact
                        )
                        holdout_greedy = evaluate(
                            context_id, "validation", aggregation, greedy
                        )
                        holdout_direct = evaluate(
                            context_id, "validation", aggregation, direct
                        )
                        row.update(
                            {
                                **metric_fields("holdout_exact", holdout_exact),
                                **metric_fields(
                                    "holdout_qubo_greedy", holdout_greedy
                                ),
                                **metric_fields(
                                    "holdout_direct_greedy", holdout_direct
                                ),
                                **delta_fields(
                                    "holdout_exact_vs_qubo_greedy",
                                    holdout_exact,
                                    holdout_greedy,
                                ),
                                **delta_fields(
                                    "holdout_exact_vs_direct_greedy",
                                    holdout_exact,
                                    holdout_direct,
                                ),
                            }
                        )
                    trial_rows.append(row)

    direct_rows: list[dict[str, object]] = []
    for (context_id, aggregation, size), exact in direct_exact.items():
        greedy = direct_greedy[(context_id, aggregation, size)]
        train_exact = evaluate(context_id, "train", aggregation, exact)
        train_greedy = evaluate(context_id, "train", aggregation, greedy)
        row: dict[str, object] = {
            "context_id": context_id,
            "aggregation": aggregation,
            "target_size": size,
            "exact_subset": "+".join(exact),
            "greedy_subset": "+".join(greedy),
            "subset_differs": exact != greedy,
            **metric_fields("train_exact", train_exact),
            **metric_fields("train_greedy", train_greedy),
            **delta_fields("train_exact_vs_greedy", train_exact, train_greedy),
        }
        if context_id != "full_train":
            holdout_exact = evaluate(context_id, "validation", aggregation, exact)
            holdout_greedy = evaluate(context_id, "validation", aggregation, greedy)
            row.update(
                {
                    **metric_fields("holdout_exact", holdout_exact),
                    **metric_fields("holdout_greedy", holdout_greedy),
                    **delta_fields(
                        "holdout_exact_vs_greedy", holdout_exact, holdout_greedy
                    ),
                }
            )
        direct_rows.append(row)

    outputs = dict(config["outputs"])
    trials_path = rooted(root, str(outputs["qubo_trials_csv"]))
    direct_path = rooted(root, str(outputs["direct_metric_trials_csv"]))
    summary_path = rooted(root, str(outputs["summary_json"]))
    report_path = rooted(root, str(outputs["report_md"]))
    if not overwrite and any(
        path.exists() for path in (trials_path, direct_path, summary_path, report_path)
    ):
        raise FileExistsError("Stage 10 outputs exist; pass --overwrite")
    write_csv(trials_path, trial_rows)
    write_csv(direct_path, direct_rows)
    objective_summary = summarize_trials(trial_rows)
    promising = [
        row
        for row in trial_rows
        if row["context_id"] != "full_train"
        and bool(row["strict_objective_failure"])
        and float(row["holdout_exact_vs_qubo_greedy_primary_bedroc_delta"])
        > TOLERANCE
        and float(row["holdout_exact_vs_qubo_greedy_mean_seed_bedroc_delta"])
        > TOLERANCE
        and float(row["holdout_exact_vs_qubo_greedy_worst_seed_bedroc_delta"])
        > TOLERANCE
    ]
    promising.sort(
        key=lambda row: (
            -float(row["holdout_exact_vs_qubo_greedy_primary_bedroc_delta"]),
            str(row["objective_family"]),
            str(row["context_id"]),
        )
    )
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage10_expanded16_qubo_greedy_screen_complete",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "input_dimensions": {
            "receptor_count": len(receptor_ids),
            "ligand_count": len(ligand_rows),
            "seed_count": len(SEED_IDS),
        },
        "objective_trial_count": len(trial_rows),
        "direct_metric_trial_count": len(direct_rows),
        "objective_summary": objective_summary,
        "strict_objective_failure_count": sum(
            bool(row["strict_objective_failure"]) for row in trial_rows
        ),
        "promising_heldout_case_count": len(promising),
        "top_promising_cases": promising[:20],
        "direct_metric_greedy_failure_count": sum(
            bool(row["subset_differs"]) for row in direct_rows
        ),
        "data_boundary": {
            "validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "outputs": {
            "qubo_trials_csv": {
                "path": trials_path.relative_to(root).as_posix(),
                "sha256": file_sha256(trials_path),
            },
            "direct_metric_trials_csv": {
                "path": direct_path.relative_to(root).as_posix(),
                "sha256": file_sha256(direct_path),
            },
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    write_json(summary_path, result)
    write_report(report_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
