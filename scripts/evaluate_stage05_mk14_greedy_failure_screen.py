"""Screen train-only MAPK14 objectives for strict greedy local optima."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Callable

try:
    from .normalized_receptor_qubo import (
        build_coefficients,
        coefficient_energy,
    )
    from .prepare_receptor import file_sha256
    from .run_receptor_selection_validation_gate import write_csv
    from .run_stage05_mk14_method_gate import make_frozen_group_folds
    from .run_stage05_mk14_uncertainty_qubo_gate import (
        SEED_IDS,
        audit_inputs,
        checked_input_paths,
        load_config as load_gate_config,
        make_context,
        metrics_for_context,
        pair_synergy_terms_for_aggregation,
        pair_utility_terms_for_aggregation,
        robust_metric_summary,
    )
except ImportError:
    from normalized_receptor_qubo import (
        build_coefficients,
        coefficient_energy,
    )
    from prepare_receptor import file_sha256
    from run_receptor_selection_validation_gate import write_csv
    from run_stage05_mk14_method_gate import make_frozen_group_folds
    from run_stage05_mk14_uncertainty_qubo_gate import (
        SEED_IDS,
        audit_inputs,
        checked_input_paths,
        load_config as load_gate_config,
        make_context,
        metrics_for_context,
        pair_synergy_terms_for_aggregation,
        pair_utility_terms_for_aggregation,
        robust_metric_summary,
    )


OBJECTIVE_ORDER = (
    "coverage_qubo",
    "pair_utility_qubo",
    "pair_synergy_qubo",
)
COEFFICIENT_SOURCES = ("primary", *SEED_IDS)
TOLERANCE = 1e-12


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def verified_path(entry: dict[str, object]) -> Path:
    path = Path(str(entry["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(entry["sha256"]).upper():
        raise ValueError(f"input SHA-256 differs: {path}")
    return path


def load_screen_config(path: Path) -> dict[str, object]:
    config = read_json(path)
    required = {
        "schema_version",
        "authorization_id",
        "implementation",
        "source_gate_config",
        "objective_results",
        "screen",
        "outputs",
        "interpretation_boundary",
    }
    if set(config) != required:
        raise ValueError("greedy-screen config keys differ")
    implementation = dict(config["implementation"])
    if Path(str(implementation["path"])).resolve() != Path(__file__).resolve():
        raise ValueError("implementation path differs from config")
    if file_sha256(Path(__file__)) != str(implementation["sha256"]).upper():
        raise ValueError("implementation SHA-256 differs from config")
    screen = dict(config["screen"])
    if tuple(screen["coefficient_sources"]) != COEFFICIENT_SOURCES:
        raise ValueError("coefficient sources differ from the frozen screen")
    target_sizes = [int(value) for value in screen["target_sizes"]]
    if target_sizes != list(range(2, 8)):
        raise ValueError("target sizes must be 2 through 7")
    if int(screen["subpool_target_size"]) != 3:
        raise ValueError("subpool target size must remain three")
    if [int(value) for value in screen["subpool_sizes"]] != list(range(4, 9)):
        raise ValueError("subpool sizes must be 4 through 8")
    if int(screen["outer_fold_count"]) != 4:
        raise ValueError("outer fold count must remain four")
    return config


def ensure_output_boundary(outputs: dict[str, Path], overwrite: bool) -> None:
    existing = [
        path
        for key, path in outputs.items()
        if key != "run_directory" and path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError("greedy-screen outputs exist; use --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()
    outputs["run_directory"].mkdir(parents=True, exist_ok=True)


def load_objective_specs(
    config: dict[str, object],
) -> dict[str, dict[str, object]]:
    entries = dict(config["objective_results"])
    if set(entries) != set(OBJECTIVE_ORDER):
        raise ValueError("objective result set differs")
    specs: dict[str, dict[str, object]] = {}
    for family in OBJECTIVE_ORDER:
        entry = dict(entries[family])
        result = read_json(verified_path(entry))
        if result.get("status") != entry["expected_status"]:
            raise ValueError(f"objective result status differs: {family}")
        if int(result.get("validation_rows_read", -1)) != 0:
            raise ValueError(f"objective result read validation rows: {family}")
        if int(result.get("test_rows_read", -1)) != 0:
            raise ValueError(f"objective result read test rows: {family}")
        selected = dict(result["selected_qubo"])
        if selected.get("family") != family:
            raise ValueError(f"selected objective family differs: {family}")
        if int(selected["target_size"]) != 3:
            raise ValueError(f"selected objective budget differs: {family}")
        if selected.get("aggregation") != "min_score":
            raise ValueError(f"selected aggregation differs: {family}")
        specs[family] = {
            "family": family,
            "aggregation": str(selected["aggregation"]),
            "weights": {
                key: float(value)
                for key, value in dict(selected["weights"]).items()
            },
            "source_result": verified_path(entry).as_posix(),
        }
    return specs


def fixed_cardinality_exact(
    coefficients: dict[str, object],
    receptor_pool: tuple[str, ...],
    target_size: int,
) -> tuple[tuple[str, ...], float]:
    if not 1 <= target_size <= len(receptor_pool):
        raise ValueError("target size is outside the available receptor pool")
    return min(
        (
            (subset, coefficient_energy(subset, coefficients))
            for subset in itertools.combinations(sorted(receptor_pool), target_size)
        ),
        key=lambda item: (item[1], item[0]),
    )


def fixed_cardinality_greedy(
    coefficients: dict[str, object],
    receptor_pool: tuple[str, ...],
    target_size: int,
) -> tuple[tuple[str, ...], float, list[dict[str, object]]]:
    if not 1 <= target_size <= len(receptor_pool):
        raise ValueError("target size is outside the available receptor pool")
    selected: tuple[str, ...] = ()
    path: list[dict[str, object]] = []
    while len(selected) < target_size:
        candidates = [
            tuple(sorted((*selected, receptor_id)))
            for receptor_id in receptor_pool
            if receptor_id not in selected
        ]
        selected = min(
            candidates,
            key=lambda subset: (
                coefficient_energy(subset, coefficients),
                subset,
            ),
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


def metric_selection_key(
    metrics: dict[str, float], subset: tuple[str, ...]
) -> tuple[object, ...]:
    return (*(-value for value in metric_quality(metrics)), subset)


def metric_exact(
    receptor_pool: tuple[str, ...],
    target_size: int,
    evaluate: Callable[[tuple[str, ...]], dict[str, float]],
) -> tuple[tuple[str, ...], dict[str, float]]:
    subset = min(
        itertools.combinations(sorted(receptor_pool), target_size),
        key=lambda value: metric_selection_key(evaluate(value), value),
    )
    return subset, evaluate(subset)


def metric_greedy(
    receptor_pool: tuple[str, ...],
    target_size: int,
    evaluate: Callable[[tuple[str, ...]], dict[str, float]],
) -> tuple[tuple[str, ...], dict[str, float]]:
    selected: tuple[str, ...] = ()
    while len(selected) < target_size:
        candidates = [
            tuple(sorted((*selected, receptor_id)))
            for receptor_id in receptor_pool
            if receptor_id not in selected
        ]
        selected = min(
            candidates,
            key=lambda value: metric_selection_key(evaluate(value), value),
        )
    return selected, evaluate(selected)


def strict_metric_failure(
    exact: dict[str, float], greedy: dict[str, float]
) -> bool:
    return any(
        not math.isclose(left, right, abs_tol=TOLERANCE)
        for left, right in zip(metric_quality(exact), metric_quality(greedy))
    )


def terms_for_objective(
    context: dict[str, object],
    family: str,
    source: str,
    aggregation: str,
) -> dict[str, object]:
    terms = (
        context["terms"]
        if source == "primary"
        else context["seed_terms"][source]
    )
    if family == "pair_utility_qubo":
        return pair_utility_terms_for_aggregation(terms, aggregation)
    if family == "pair_synergy_qubo":
        return pair_synergy_terms_for_aggregation(terms, aggregation)
    if family != "coverage_qubo":
        raise ValueError(f"unsupported objective family: {family}")
    return terms


def objective_coefficients(
    context: dict[str, object],
    spec: dict[str, object],
    source: str,
    receptor_ids: list[str],
    target_size: int,
    model: dict[str, object],
) -> dict[str, object]:
    return build_coefficients(
        terms_for_objective(
            context,
            str(spec["family"]),
            source,
            str(spec["aggregation"]),
        ),
        receptor_ids,
        target_size,
        dict(spec["weights"]),
        float(model["size_penalty"]),
    )


def metric_delta_fields(
    exact: dict[str, float], greedy: dict[str, float], prefix: str
) -> dict[str, float]:
    return {
        f"{prefix}_primary_bedroc_delta": float(exact["primary_bedroc"])
        - float(greedy["primary_bedroc"]),
        f"{prefix}_mean_seed_bedroc_delta": float(exact["mean_seed_bedroc"])
        - float(greedy["mean_seed_bedroc"]),
        f"{prefix}_worst_seed_bedroc_delta": float(exact["worst_seed_bedroc"])
        - float(greedy["worst_seed_bedroc"]),
    }


def empty_holdout_fields() -> dict[str, object]:
    return {
        "holdout_primary_bedroc_delta": None,
        "holdout_mean_seed_bedroc_delta": None,
        "holdout_worst_seed_bedroc_delta": None,
    }


def trial_summary(
    rows: list[dict[str, object]], group_keys: tuple[str, ...]
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    summaries: list[dict[str, object]] = []
    for key, values in sorted(groups.items()):
        failures = [row for row in values if bool(row["strict_failure"])]
        heldout = [
            row
            for row in failures
            if row["holdout_primary_bedroc_delta"] is not None
        ]
        regrets = [
            float(row["objective_regret"])
            for row in failures
            if row["objective_regret"] is not None
        ]
        deltas = [
            float(row["holdout_primary_bedroc_delta"]) for row in heldout
        ]
        summaries.append(
            {
                **dict(zip(group_keys, key)),
                "trial_count": len(values),
                "strict_failure_count": len(failures),
                "strict_failure_rate": len(failures) / len(values),
                "maximum_objective_regret": max(regrets) if regrets else None,
                "mean_objective_regret_among_failures": (
                    statistics.fmean(regrets) if regrets else None
                ),
                "heldout_failure_case_count": len(heldout),
                "heldout_primary_bedroc_exact_better_count": sum(
                    value > TOLERANCE for value in deltas
                ),
                "heldout_primary_bedroc_exact_better_fraction": (
                    sum(value > TOLERANCE for value in deltas) / len(deltas)
                    if deltas
                    else None
                ),
                "mean_holdout_primary_bedroc_delta": (
                    statistics.fmean(deltas) if deltas else None
                ),
            }
        )
    return summaries


def write_report(
    path: Path,
    full_summary: list[dict[str, object]],
    current_summary: list[dict[str, object]],
    growth_summary: list[dict[str, object]],
    direct_full_summary: list[dict[str, object]],
    direct_growth_summary: list[dict[str, object]],
    top_cases: list[dict[str, object]],
    overall: dict[str, object],
) -> None:
    lines = [
        "# MAPK14 Train-only Greedy Failure Screen",
        "",
        "## Scope",
        "",
        "This post hoc diagnostic uses Train-696 only. It compares deterministic",
        "forward greedy selection with exact fixed-cardinality enumeration. No fresh",
        "validation or locked-test rows were read. Exact enumeration is an optimization",
        "reference; it is not evidence that a quantum device found the solution.",
        "",
        "## Full eight-receptor QUBO objective screen",
        "",
        "| Objective | Trials | Strict failures | Rate | Max regret | Held-out exact-better fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in full_summary:
        fraction = row["heldout_primary_bedroc_exact_better_fraction"]
        lines.append(
            "| {objective_family} | {trial_count} | {strict_failure_count} | "
            "{strict_failure_rate:.3f} | {regret} | {fraction} |".format(
                **row,
                regret=(
                    "NA"
                    if row["maximum_objective_regret"] is None
                    else f"{float(row['maximum_objective_regret']):.6f}"
                ),
                fraction=(
                    "NA" if fraction is None else f"{float(fraction):.3f}"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Frozen primary-source budget-three check",
            "",
            "| Objective | Contexts | Strict failures |",
            "|---|---:|---:|",
        ]
    )
    for row in current_summary:
        lines.append(
            f"| {row['objective_family']} | {row['trial_count']} | "
            f"{row['strict_failure_count']} |"
        )
    lines.extend(
        [
            "",
            "## Receptor-pool growth screen at budget three",
            "",
            "The full eight-receptor coefficient normalization is held fixed while the",
            "available variable pool is restricted. This isolates search behavior from",
            "changes in coefficient estimation.",
            "",
            "| Objective | Pool size | Trials | Strict failures | Rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in growth_summary:
        lines.append(
            f"| {row['objective_family']} | {row['pool_size']} | "
            f"{row['trial_count']} | {row['strict_failure_count']} | "
            f"{float(row['strict_failure_rate']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Direct robust-BEDROC selection screen",
            "",
            "This second check optimizes the existing robust metric hierarchy directly,",
            "rather than optimizing a fitted QUBO surrogate.",
            "",
            "| Aggregation | Full-pool contexts | Strict failures | Rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in direct_full_summary:
        lines.append(
            f"| {row['aggregation']} | {row['trial_count']} | "
            f"{row['strict_failure_count']} | "
            f"{float(row['strict_failure_rate']):.3f} |"
        )
    lines.extend(
        [
            "",
            "| Aggregation | Pool size | Trials | Strict failures | Rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in direct_growth_summary:
        lines.append(
            f"| {row['aggregation']} | {row['pool_size']} | "
            f"{row['trial_count']} | {row['strict_failure_count']} | "
            f"{float(row['strict_failure_rate']):.3f} |"
        )
    lines.extend(["", "## Largest full-pool objective regrets", ""])
    for row in top_cases:
        lines.append(
            "- {objective_family}, {context_id}, {coefficient_source}, k={target_size}: "
            "regret={objective_regret:.6f}; greedy={greedy_subset}; exact={exact_subset}".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Strict greedy local optima observed: `{str(overall['greedy_local_optima_observed']).lower()}`.",
            f"Frozen pair-synergy primary-source budget-three failures: "
            f"`{overall['frozen_pair_synergy_primary_budget3_failure_count']}`.",
            f"Across outer-fold QUBO failure cases, exact selection improved held-out "
            f"primary BEDROC in `{overall['outer_fold_failure_exact_better_holdout_primary_count']}` "
            f"of `{overall['outer_fold_qubo_failure_count']}` cases "
            f"(`{float(overall['outer_fold_failure_exact_better_holdout_primary_fraction']):.3f}`).",
            "",
            "A positive QUBO objective regret proves only that forward greedy missed the",
            "exact optimum of the fitted quadratic objective. Better held-out BEDROC must",
            "be demonstrated separately, and neither result establishes quantum advantage.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = load_screen_config(config_path)
    outputs = {
        key: Path(str(value)) for key, value in dict(config["outputs"]).items()
    }
    ensure_output_boundary(outputs, overwrite)

    gate_config_path = verified_path(dict(config["source_gate_config"]))
    gate_config = load_gate_config(gate_config_path)
    audited = audit_inputs(gate_config, checked_input_paths(gate_config))
    receptor_ids = [str(value) for value in gate_config["receptor_ids"]]
    if len(receptor_ids) != 8:
        raise ValueError("the source receptor pool must contain eight receptors")
    if int(gate_config["expected"]["validation_rows"]) != 0:
        raise ValueError("source gate exposes validation rows")
    if int(gate_config["expected"]["test_rows"]) != 0:
        raise ValueError("source gate exposes test rows")

    manifest_rows = audited["manifest_rows"]
    raw_matrices = audited["matrices"]
    matrices = {
        matrix_id: {
            str(row["ligand_id"]): row for row in matrix_rows
        }
        for matrix_id, matrix_rows in raw_matrices.items()
    }
    ligand_ids = set(matrices["primary"])
    screen = dict(config["screen"])
    assignments = make_frozen_group_folds(
        manifest_rows,
        int(screen["outer_fold_count"]),
        int(screen["fold_seed"]),
    )
    contexts: dict[str, dict[str, object]] = {
        "full_train": make_context(
            ligand_ids,
            set(),
            matrices,
            receptor_ids,
            gate_config["model"],
        )
    }
    for fold in range(int(screen["outer_fold_count"])):
        holdout_ids = {
            ligand_id
            for ligand_id, assigned in assignments.items()
            if assigned == fold
        }
        contexts[f"outer_fold_{fold}"] = make_context(
            ligand_ids - holdout_ids,
            holdout_ids,
            matrices,
            receptor_ids,
            gate_config["model"],
        )

    specs = load_objective_specs(config)
    metric_cache: dict[
        tuple[str, str, str, tuple[str, ...]], dict[str, float]
    ] = {}

    def cached_metrics(
        context_id: str,
        split: str,
        aggregation: str,
        subset: tuple[str, ...],
    ) -> dict[str, float]:
        key = (context_id, split, aggregation, tuple(sorted(subset)))
        if key not in metric_cache:
            metric_cache[key] = robust_metric_summary(
                metrics_for_context(
                    contexts[context_id],
                    tuple(sorted(subset)),
                    aggregation,
                    split,
                )
            )
        return metric_cache[key]

    receptor_pool = tuple(sorted(receptor_ids))
    full_rows: list[dict[str, object]] = []
    for family in OBJECTIVE_ORDER:
        spec = specs[family]
        aggregation = str(spec["aggregation"])
        for context_id, context in contexts.items():
            has_holdout = context_id != "full_train"
            for source in COEFFICIENT_SOURCES:
                for target_size in screen["target_sizes"]:
                    size = int(target_size)
                    coefficients = objective_coefficients(
                        context,
                        spec,
                        source,
                        receptor_ids,
                        size,
                        gate_config["model"],
                    )
                    exact, exact_energy = fixed_cardinality_exact(
                        coefficients, receptor_pool, size
                    )
                    greedy, greedy_energy, greedy_path = fixed_cardinality_greedy(
                        coefficients, receptor_pool, size
                    )
                    regret = greedy_energy - exact_energy
                    train_exact = cached_metrics(
                        context_id, "train", aggregation, exact
                    )
                    train_greedy = cached_metrics(
                        context_id, "train", aggregation, greedy
                    )
                    row: dict[str, object] = {
                        "comparison_type": "qubo_objective",
                        "objective_family": family,
                        "context_id": context_id,
                        "coefficient_source": source,
                        "aggregation": aggregation,
                        "pool_size": len(receptor_pool),
                        "available_pool": "+".join(receptor_pool),
                        "target_size": size,
                        "exact_subset": "+".join(exact),
                        "greedy_subset": "+".join(greedy),
                        "subset_differs": exact != greedy,
                        "strict_failure": regret > TOLERANCE,
                        "exact_objective": exact_energy,
                        "greedy_objective": greedy_energy,
                        "objective_regret": regret,
                        "greedy_path": json.dumps(greedy_path, sort_keys=True),
                        **metric_delta_fields(
                            train_exact, train_greedy, "train"
                        ),
                    }
                    if has_holdout:
                        row.update(
                            metric_delta_fields(
                                cached_metrics(
                                    context_id,
                                    "validation",
                                    aggregation,
                                    exact,
                                ),
                                cached_metrics(
                                    context_id,
                                    "validation",
                                    aggregation,
                                    greedy,
                                ),
                                "holdout",
                            )
                        )
                    else:
                        row.update(empty_holdout_fields())
                    full_rows.append(row)

    for context_id in contexts:
        has_holdout = context_id != "full_train"
        for aggregation in screen["direct_metric_aggregations"]:
            for target_size in screen["direct_metric_target_sizes"]:
                size = int(target_size)

                def evaluate(subset: tuple[str, ...]) -> dict[str, float]:
                    return cached_metrics(
                        context_id, "train", str(aggregation), subset
                    )

                exact, train_exact = metric_exact(
                    receptor_pool, size, evaluate
                )
                greedy, train_greedy = metric_greedy(
                    receptor_pool, size, evaluate
                )
                row = {
                    "comparison_type": "direct_robust_bedroc",
                    "objective_family": "direct_robust_bedroc",
                    "context_id": context_id,
                    "coefficient_source": "not_applicable",
                    "aggregation": str(aggregation),
                    "pool_size": len(receptor_pool),
                    "available_pool": "+".join(receptor_pool),
                    "target_size": size,
                    "exact_subset": "+".join(exact),
                    "greedy_subset": "+".join(greedy),
                    "subset_differs": exact != greedy,
                    "strict_failure": strict_metric_failure(
                        train_exact, train_greedy
                    ),
                    "exact_objective": None,
                    "greedy_objective": None,
                    "objective_regret": None,
                    "greedy_path": None,
                    **metric_delta_fields(train_exact, train_greedy, "train"),
                }
                if has_holdout:
                    row.update(
                        metric_delta_fields(
                            cached_metrics(
                                context_id,
                                "validation",
                                str(aggregation),
                                exact,
                            ),
                            cached_metrics(
                                context_id,
                                "validation",
                                str(aggregation),
                                greedy,
                            ),
                            "holdout",
                        )
                    )
                else:
                    row.update(empty_holdout_fields())
                full_rows.append(row)

    subpool_rows: list[dict[str, object]] = []
    subpool_target = int(screen["subpool_target_size"])
    for family in OBJECTIVE_ORDER:
        spec = specs[family]
        aggregation = str(spec["aggregation"])
        for context_id, context in contexts.items():
            has_holdout = context_id != "full_train"
            for source in COEFFICIENT_SOURCES:
                coefficients = objective_coefficients(
                    context,
                    spec,
                    source,
                    receptor_ids,
                    subpool_target,
                    gate_config["model"],
                )
                for pool_size in screen["subpool_sizes"]:
                    for available in itertools.combinations(
                        receptor_pool, int(pool_size)
                    ):
                        exact, exact_energy = fixed_cardinality_exact(
                            coefficients, available, subpool_target
                        )
                        greedy, greedy_energy, _ = fixed_cardinality_greedy(
                            coefficients, available, subpool_target
                        )
                        regret = greedy_energy - exact_energy
                        row = {
                            "comparison_type": "qubo_objective",
                            "objective_family": family,
                            "context_id": context_id,
                            "coefficient_source": source,
                            "aggregation": aggregation,
                            "pool_size": int(pool_size),
                            "available_pool": "+".join(available),
                            "target_size": subpool_target,
                            "exact_subset": "+".join(exact),
                            "greedy_subset": "+".join(greedy),
                            "subset_differs": exact != greedy,
                            "strict_failure": regret > TOLERANCE,
                            "exact_objective": exact_energy,
                            "greedy_objective": greedy_energy,
                            "objective_regret": regret,
                            **metric_delta_fields(
                                cached_metrics(
                                    context_id, "train", aggregation, exact
                                ),
                                cached_metrics(
                                    context_id, "train", aggregation, greedy
                                ),
                                "train",
                            ),
                        }
                        if has_holdout and regret > TOLERANCE:
                            row.update(
                                metric_delta_fields(
                                    cached_metrics(
                                        context_id,
                                        "validation",
                                        aggregation,
                                        exact,
                                    ),
                                    cached_metrics(
                                        context_id,
                                        "validation",
                                        aggregation,
                                        greedy,
                                    ),
                                    "holdout",
                                )
                            )
                        else:
                            row.update(empty_holdout_fields())
                        subpool_rows.append(row)

    for context_id in contexts:
        has_holdout = context_id != "full_train"
        for aggregation in screen["direct_metric_aggregations"]:

            def evaluate(subset: tuple[str, ...]) -> dict[str, float]:
                return cached_metrics(
                    context_id, "train", str(aggregation), subset
                )

            for pool_size in screen["subpool_sizes"]:
                for available in itertools.combinations(
                    receptor_pool, int(pool_size)
                ):
                    exact, train_exact = metric_exact(
                        available, subpool_target, evaluate
                    )
                    greedy, train_greedy = metric_greedy(
                        available, subpool_target, evaluate
                    )
                    strict = strict_metric_failure(train_exact, train_greedy)
                    row = {
                        "comparison_type": "direct_robust_bedroc",
                        "objective_family": "direct_robust_bedroc",
                        "context_id": context_id,
                        "coefficient_source": "not_applicable",
                        "aggregation": str(aggregation),
                        "pool_size": int(pool_size),
                        "available_pool": "+".join(available),
                        "target_size": subpool_target,
                        "exact_subset": "+".join(exact),
                        "greedy_subset": "+".join(greedy),
                        "subset_differs": exact != greedy,
                        "strict_failure": strict,
                        "exact_objective": None,
                        "greedy_objective": None,
                        "objective_regret": None,
                        **metric_delta_fields(
                            train_exact, train_greedy, "train"
                        ),
                    }
                    if has_holdout and strict:
                        row.update(
                            metric_delta_fields(
                                cached_metrics(
                                    context_id,
                                    "validation",
                                    str(aggregation),
                                    exact,
                                ),
                                cached_metrics(
                                    context_id,
                                    "validation",
                                    str(aggregation),
                                    greedy,
                                ),
                                "holdout",
                            )
                        )
                    else:
                        row.update(empty_holdout_fields())
                    subpool_rows.append(row)

    qubo_full_rows = [
        row for row in full_rows if row["comparison_type"] == "qubo_objective"
    ]
    full_summary = trial_summary(qubo_full_rows, ("objective_family",))
    current_rows = [
        row
        for row in qubo_full_rows
        if row["coefficient_source"] == "primary"
        and int(row["target_size"]) == 3
    ]
    current_summary = trial_summary(current_rows, ("objective_family",))
    growth_summary = trial_summary(
        [
            row
            for row in subpool_rows
            if row["comparison_type"] == "qubo_objective"
        ],
        ("objective_family", "pool_size"),
    )
    direct_full_summary = trial_summary(
        [
            row
            for row in full_rows
            if row["comparison_type"] == "direct_robust_bedroc"
        ],
        ("aggregation",),
    )
    direct_growth_summary = trial_summary(
        [
            row
            for row in subpool_rows
            if row["comparison_type"] == "direct_robust_bedroc"
        ],
        ("aggregation", "pool_size"),
    )
    top_cases = sorted(
        [row for row in qubo_full_rows if bool(row["strict_failure"])],
        key=lambda row: -float(row["objective_regret"]),
    )[:10]
    frozen_pair_rows = [
        row
        for row in current_rows
        if row["objective_family"] == "pair_synergy_qubo"
    ]
    outer_qubo_failures = [
        row
        for row in qubo_full_rows
        if bool(row["strict_failure"])
        and row["context_id"] != "full_train"
    ]
    overall = {
        "greedy_local_optima_observed": any(
            bool(row["strict_failure"]) for row in qubo_full_rows
        ),
        "full_pool_qubo_trial_count": len(qubo_full_rows),
        "full_pool_qubo_strict_failure_count": sum(
            bool(row["strict_failure"]) for row in qubo_full_rows
        ),
        "frozen_pair_synergy_primary_budget3_trial_count": len(
            frozen_pair_rows
        ),
        "frozen_pair_synergy_primary_budget3_failure_count": sum(
            bool(row["strict_failure"]) for row in frozen_pair_rows
        ),
        "outer_fold_qubo_failure_count": len(outer_qubo_failures),
        "outer_fold_failure_exact_better_holdout_primary_count": sum(
            float(row["holdout_primary_bedroc_delta"]) > TOLERANCE
            for row in outer_qubo_failures
        ),
        "outer_fold_failure_exact_better_holdout_primary_fraction": (
            sum(
                float(row["holdout_primary_bedroc_delta"]) > TOLERANCE
                for row in outer_qubo_failures
            )
            / len(outer_qubo_failures)
            if outer_qubo_failures
            else None
        ),
        "validation_rows_read": 0,
        "test_rows_read": 0,
    }

    write_csv(outputs["full_pool_trials_csv"], full_rows)
    write_csv(outputs["subpool_trials_csv"], subpool_rows)
    write_csv(outputs["subpool_summary_csv"], growth_summary + direct_growth_summary)
    write_report(
        outputs["report_md"],
        full_summary,
        current_summary,
        growth_summary,
        direct_full_summary,
        direct_growth_summary,
        top_cases,
        overall,
    )
    result = {
        "schema_version": "1.0",
        "authorization_id": config["authorization_id"],
        "status": "posthoc_train_only_greedy_failure_screen_complete",
        "implementation": {
            "path": Path(__file__).as_posix(),
            "sha256": file_sha256(Path(__file__)),
        },
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "data_boundary": {
            "train_ligand_count": len(ligand_ids),
            "receptor_count": len(receptor_ids),
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "evidence_class": "posthoc_train_only_diagnostic",
        },
        "objective_specs": specs,
        "overall": overall,
        "full_pool_qubo_summary": full_summary,
        "frozen_primary_budget3_summary": current_summary,
        "direct_metric_full_pool_summary": direct_full_summary,
        "subpool_qubo_summary": growth_summary,
        "subpool_direct_metric_summary": direct_growth_summary,
        "largest_full_pool_regret_cases": top_cases,
        "outputs": {
            key: {
                "path": path.as_posix(),
                "sha256": file_sha256(path),
            }
            for key, path in outputs.items()
            if key != "run_directory" and key != "result_json"
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    outputs["result_json"].parent.mkdir(parents=True, exist_ok=True)
    write_json(outputs["result_json"], result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "overall": overall,
                "full_pool_qubo_summary": full_summary,
                "frozen_primary_budget3_summary": current_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
