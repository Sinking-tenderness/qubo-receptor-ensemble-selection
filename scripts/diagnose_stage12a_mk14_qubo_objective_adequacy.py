"""Diagnose MAPK14 QUBO objective adequacy using Stage 09 train rows only."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

try:
    from .evaluate_virtual_screening import bedroc
    from .normalized_receptor_qubo import build_coefficients, coefficient_energy
    from .prepare_receptor import file_sha256
    from .run_stage05_mk14_method_gate import make_frozen_group_folds
    from .run_stage05_mk14_uncertainty_qubo_gate import make_context
    from .screen_stage10_mk14_expanded16_qubo_greedy import (
        build_matrices,
        read_csv,
        read_json,
        terms_for_family,
    )
except ImportError:
    from evaluate_virtual_screening import bedroc
    from normalized_receptor_qubo import build_coefficients, coefficient_energy
    from prepare_receptor import file_sha256
    from run_stage05_mk14_method_gate import make_frozen_group_folds
    from run_stage05_mk14_uncertainty_qubo_gate import make_context
    from screen_stage10_mk14_expanded16_qubo_greedy import (
        build_matrices,
        read_csv,
        read_json,
        terms_for_family,
    )


SEED_IDS = ("seed0", "seed1", "seed2")
MATRIX_IDS = ("primary", "sensitivity", *SEED_IDS)


def rooted(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def verified(root: Path, descriptor: dict[str, object]) -> Path:
    path = rooted(root, str(descriptor["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"input SHA-256 differs: {path}")
    return path


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_subset(value: str) -> tuple[str, ...]:
    subset = tuple(sorted(item for item in value.split("+") if item))
    if not subset:
        raise ValueError("empty receptor subset")
    return subset


def design_matrix(
    subsets: list[tuple[str, ...]],
    receptor_ids: list[str],
    include_quadratic: bool,
) -> tuple[np.ndarray, list[str]]:
    receptor_index = {receptor_id: index for index, receptor_id in enumerate(receptor_ids)}
    pairs = list(itertools.combinations(receptor_ids, 2)) if include_quadratic else []
    names = [f"x::{receptor_id}" for receptor_id in receptor_ids]
    names.extend(f"xx::{first}__{second}" for first, second in pairs)
    matrix = np.zeros((len(subsets), len(names)), dtype=float)
    pair_offset = len(receptor_ids)
    pair_index = {pair: pair_offset + index for index, pair in enumerate(pairs)}
    for row_index, subset in enumerate(subsets):
        selected = set(subset)
        for receptor_id in subset:
            matrix[row_index, receptor_index[receptor_id]] = 1.0
        for pair, column_index in pair_index.items():
            if pair[0] in selected and pair[1] in selected:
                matrix[row_index, column_index] = 1.0
    return matrix, names


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else 0.0


def choose_highest(values: np.ndarray, subsets: list[tuple[str, ...]]) -> int:
    return min(
        range(len(subsets)),
        key=lambda index: (-float(values[index]), subsets[index]),
    )


def descending_rank(values: np.ndarray, selected_index: int) -> int:
    selected = float(values[selected_index])
    return 1 + sum(float(value) > selected for value in values)


def mean_jaccard(subsets: list[tuple[str, ...]]) -> float:
    values = []
    for first, second in itertools.combinations(subsets, 2):
        left = set(first)
        right = set(second)
        values.append(len(left & right) / len(left | right))
    return statistics.fmean(values) if values else 1.0


def subset_bedroc(rows: list[dict[str, object]], subset: tuple[str, ...]) -> float:
    ranked = sorted(
        rows,
        key=lambda row: (
            min(float(row[receptor_id]) for receptor_id in subset),
            str(row["ligand_id"]),
        ),
    )
    return float(
        bedroc(
            [
                {"binary_label": int(row["label"] == "active")}
                for row in ranked
            ],
            20.0,
        )
    )


def score_subsets(
    context: dict[str, object],
    subsets: list[tuple[str, ...]],
    split: str,
) -> dict[str, np.ndarray]:
    matrices = dict(context["matrices"])
    if not list(dict(matrices["primary"])[split]):
        raise ValueError(f"empty context split: {split}")
    values = {
        matrix_id: np.array(
            [
                subset_bedroc(list(dict(matrices[matrix_id])[split]), subset)
                for subset in subsets
            ],
            dtype=float,
        )
        for matrix_id in MATRIX_IDS
    }
    seed_values = np.vstack([values[seed_id] for seed_id in SEED_IDS])
    values["mean_seed"] = np.mean(seed_values, axis=0)
    values["worst_seed"] = np.min(seed_values, axis=0)
    values["robust_composite"] = (
        values["primary"] + values["mean_seed"] + values["worst_seed"]
    ) / 3.0
    return values


def subset_metrics(values: dict[str, np.ndarray], index: int) -> dict[str, float]:
    return {
        key: float(values[key][index])
        for key in (
            "primary",
            "sensitivity",
            "mean_seed",
            "worst_seed",
            "robust_composite",
        )
    }


def with_prefix(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def fit_surrogate(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> tuple[Ridge, np.ndarray]:
    model = Ridge(alpha=alpha, fit_intercept=True, solver="svd")
    model.fit(features, target)
    return model, np.asarray(model.predict(features), dtype=float)


def build_explicit_qubo(
    model: Ridge,
    feature_names: list[str],
    receptor_ids: list[str],
    target_size: int,
    cardinality_penalty: float,
) -> dict[str, object]:
    surrogate_linear: dict[str, float] = {}
    surrogate_quadratic: dict[str, float] = {}
    for name, coefficient in zip(feature_names, model.coef_):
        if name.startswith("x::"):
            surrogate_linear[name.removeprefix("x::")] = float(coefficient)
        elif name.startswith("xx::"):
            surrogate_quadratic[name.removeprefix("xx::")] = float(coefficient)
        else:
            raise ValueError(f"unknown surrogate feature: {name}")
    linear = {
        receptor_id: -surrogate_linear[receptor_id]
        + cardinality_penalty * (1 - 2 * target_size)
        for receptor_id in receptor_ids
    }
    quadratic = {
        f"{first}__{second}": -surrogate_quadratic[f"{first}__{second}"]
        + 2.0 * cardinality_penalty
        for first, second in itertools.combinations(receptor_ids, 2)
    }
    return {
        "convention": (
            "Q(x)=constant+sum_i linear[i]*x_i+"
            "sum_i<j quadratic[i__j]*x_i*x_j"
        ),
        "constant": -float(model.intercept_)
        + cardinality_penalty * target_size**2,
        "linear": linear,
        "quadratic": quadratic,
        "target_size": target_size,
        "cardinality_penalty": cardinality_penalty,
        "surrogate": {
            "prediction_convention": (
                "predicted_utility=intercept+sum_i h_i*x_i+"
                "sum_i<j J_ij*x_i*x_j"
            ),
            "intercept": float(model.intercept_),
            "linear": surrogate_linear,
            "quadratic": surrogate_quadratic,
        },
    }


def qubo_energy(subset: tuple[str, ...], qubo: dict[str, object]) -> float:
    selected = set(subset)
    value = float(qubo["constant"])
    linear = dict(qubo["linear"])
    quadratic = dict(qubo["quadratic"])
    value += sum(float(linear[receptor_id]) for receptor_id in selected)
    value += sum(
        float(coefficient)
        for key, coefficient in quadratic.items()
        if set(key.split("__", 1)).issubset(selected)
    )
    return value


def exact_all_cardinalities(
    receptor_ids: list[str], qubo: dict[str, object]
) -> tuple[tuple[str, ...], float]:
    best_subset: tuple[str, ...] = ()
    best_energy = math.inf
    for size in range(len(receptor_ids) + 1):
        for subset in itertools.combinations(receptor_ids, size):
            canonical = tuple(sorted(subset))
            energy = qubo_energy(canonical, qubo)
            if (energy, canonical) < (best_energy, best_subset):
                best_subset = canonical
                best_energy = energy
    return best_subset, best_energy


def aggregate_surrogate_trials(
    trial_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in trial_rows:
        grouped[(str(row["model_family"]), float(row["alpha"]))].append(row)
    output: list[dict[str, object]] = []
    for (family, alpha), rows in sorted(grouped.items()):
        selected_subsets = [parse_subset(str(row["selected_subset"])) for row in rows]
        output.append(
            {
                "model_family": family,
                "alpha": alpha,
                "fold_count": len(rows),
                "mean_holdout_robust_composite": statistics.fmean(
                    float(row["holdout_robust_composite"]) for row in rows
                ),
                "worst_holdout_robust_composite": min(
                    float(row["holdout_robust_composite"]) for row in rows
                ),
                "mean_holdout_primary_bedroc": statistics.fmean(
                    float(row["holdout_primary"]) for row in rows
                ),
                "mean_holdout_worst_seed_bedroc": statistics.fmean(
                    float(row["holdout_worst_seed"]) for row in rows
                ),
                "mean_holdout_rank_spearman": statistics.fmean(
                    float(row["holdout_rank_spearman"]) for row in rows
                ),
                "mean_delta_vs_v1": statistics.fmean(
                    float(row["holdout_delta_vs_v1"]) for row in rows
                ),
                "worst_delta_vs_v1": min(
                    float(row["holdout_delta_vs_v1"]) for row in rows
                ),
                "positive_fold_count_vs_v1": sum(
                    float(row["holdout_delta_vs_v1"]) > 0.0 for row in rows
                ),
                "mean_delta_vs_direct_greedy": statistics.fmean(
                    float(row["holdout_delta_vs_direct_greedy"]) for row in rows
                ),
                "worst_delta_vs_direct_greedy": min(
                    float(row["holdout_delta_vs_direct_greedy"]) for row in rows
                ),
                "positive_fold_count_vs_direct_greedy": sum(
                    float(row["holdout_delta_vs_direct_greedy"]) > 0.0
                    for row in rows
                ),
                "mean_pairwise_jaccard": mean_jaccard(selected_subsets),
                "selected_subsets": ["+".join(value) for value in selected_subsets],
            }
        )
    return output


def write_report(path: Path, result: dict[str, object]) -> None:
    diagnosis = dict(result["v1_diagnosis"])
    selection = dict(result["surrogate_selection"])
    candidate = dict(result["provisional_v2_candidate"])
    lines = [
        "# Stage 12A MAPK14 QUBO Objective Adequacy Diagnostic",
        "",
        "## Scope",
        "",
        "This is a post hoc development diagnostic over all 560 three-receptor subsets.",
        "It reads only Stage 09 Train-696 rows and never reads Stage 11 validation scores.",
        "The provisional v2 candidate requires independent protein-target validation.",
        "",
        "## Frozen v1 diagnosis",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Mean train rank correlation, -energy vs robust BEDROC | {diagnosis['mean_train_rank_spearman']:.6f} |",
        f"| Mean holdout rank correlation, -energy vs robust BEDROC | {diagnosis['mean_holdout_rank_spearman']:.6f} |",
        f"| Mean train-to-holdout subset rank correlation | {diagnosis['mean_train_holdout_rank_spearman']:.6f} |",
        f"| Mean v1 holdout delta vs direct greedy | {diagnosis['mean_v1_delta_vs_direct_greedy']:+.6f} |",
        "",
        "## Surrogate comparison",
        "",
        "| Model | Alpha | Holdout composite | Holdout rank rho | Delta vs v1 | Delta vs direct greedy | Fold wins vs direct |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("best_additive", "best_quadratic"):
        value = dict(selection[key])
        lines.append(
            f"| {value['model_family']} | {value['alpha']:g} | "
            f"{value['mean_holdout_robust_composite']:.6f} | "
            f"{value['mean_holdout_rank_spearman']:.6f} | "
            f"{value['mean_delta_vs_v1']:+.6f} | "
            f"{value['mean_delta_vs_direct_greedy']:+.6f} | "
            f"{value['positive_fold_count_vs_direct_greedy']}/4 |"
        )
    lines.extend(
        [
            "",
            "## Provisional v2",
            "",
            f"- Selected alpha: `{candidate['alpha']}`",
            f"- Full-train subset: `{candidate['selected_subset']}`",
            f"- Exact penalized-QUBO subset: `{candidate['exact_qubo_subset']}`",
            f"- Development status: `{result['status']}`",
            "",
            "## Interpretation",
            "",
            f"- The best quadratic surrogate improves mean holdout composite over the "
            f"best additive surrogate by {selection['quadratic_mean_holdout_gain_over_additive']:+.6f}, "
            "so pair terms contain real signal.",
            f"- It changes mean holdout composite versus frozen v1 by "
            f"{selection['best_quadratic']['mean_delta_vs_v1']:+.6f} and wins "
            f"{selection['best_quadratic']['positive_fold_count_vs_v1']}/4 folds.",
            "- The full-train v2 returns the same three-receptor subset already tested "
            "in Stage 11, so it creates no new MAPK14 confirmatory candidate.",
            "- Retain v1 for the external-target pilot; do not spend more docking on a "
            "retuned MAPK14 objective.",
            "",
            "## Decision boundary",
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
    implementation_path = verified(root, dict(config["implementation"]))
    if implementation_path.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 12A implementation path differs")

    inputs = dict(config["inputs"])
    stage10_config_path = verified(root, dict(inputs["stage10_config"]))
    stage10_result_path = verified(root, dict(inputs["stage10_result"]))
    qubo_trials_path = verified(root, dict(inputs["stage10_qubo_trials"]))
    direct_trials_path = verified(root, dict(inputs["stage10_direct_trials"]))
    stage10_config = read_json(stage10_config_path)
    stage10_result = read_json(stage10_result_path)
    if stage10_result.get("status") != "stage10_expanded16_qubo_greedy_screen_complete":
        raise ValueError("Stage 10 source screen did not complete")
    if any(int(value) != 0 for value in dict(stage10_result["data_boundary"]).values()):
        raise ValueError("Stage 10 source crossed a data boundary")
    if str(stage10_result["config"]["sha256"]).upper() != file_sha256(
        stage10_config_path
    ):
        raise ValueError("Stage 10 config identity differs")
    source_outputs = dict(stage10_result["outputs"])
    if file_sha256(qubo_trials_path) != str(
        source_outputs["qubo_trials_csv"]["sha256"]
    ).upper() or file_sha256(direct_trials_path) != str(
        source_outputs["direct_metric_trials_csv"]["sha256"]
    ).upper():
        raise ValueError("Stage 10 trial identity differs")

    stage09_inputs = dict(stage10_config["inputs"])
    stage09_paths = {
        key: verified(root, dict(value)) for key, value in stage09_inputs.items()
    }
    stage09_summary = read_json(stage09_paths["stage09_summary"])
    stage09_audit = read_json(stage09_paths["stage09_audit"])
    if stage09_summary.get("status") != "stage09_train696_unidock_matrix_ok":
        raise ValueError("Stage 09 matrix did not complete")
    if stage09_audit.get("status") != "independent_stage09_train696_unidock_matrix_audit_ok":
        raise ValueError("Stage 09 matrix audit did not pass")

    receptor_rows = read_csv(stage09_paths["receptor_manifest"])
    ligand_rows = read_csv(stage09_paths["ligand_manifest"])
    receptor_ids = [row["conformer_id"] for row in receptor_rows]
    expected = dict(config["expected"])
    if len(receptor_ids) != int(expected["receptor_count"]):
        raise ValueError("Stage 12A receptor count differs")
    if len(ligand_rows) != int(expected["ligand_count"]):
        raise ValueError("Stage 12A ligand count differs")
    if Counter(row["label"] for row in ligand_rows) != Counter(
        {key: int(value) for key, value in dict(expected["label_counts"]).items()}
    ):
        raise ValueError("Stage 12A label counts differ")
    if {row["split"] for row in ligand_rows} != {"train"}:
        raise ValueError("Stage 12A observed a non-train row")

    matrices = build_matrices(
        read_csv(stage09_paths["primary_matrix"]),
        read_csv(stage09_paths["sensitivity_matrix"]),
        read_csv(stage09_paths["seed_scores"]),
        ligand_rows,
        receptor_ids,
    )
    screen = dict(stage10_config["screen"])
    fold_assignments = make_frozen_group_folds(
        ligand_rows, int(screen["outer_fold_count"]), int(screen["fold_seed"])
    )
    all_ids = {row["ligand_id"] for row in ligand_rows}
    contexts: dict[str, dict[str, object]] = {
        "full_train": make_context(
            all_ids, set(), matrices, receptor_ids, dict(stage10_config["model"])
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
            dict(stage10_config["model"]),
        )

    target_size = int(config["diagnostic"]["target_size"])
    subsets = sorted(
        tuple(sorted(subset))
        for subset in itertools.combinations(receptor_ids, target_size)
    )
    if len(subsets) != int(expected["triple_subset_count"]):
        raise ValueError("Stage 12A subset count differs")
    subset_index = {subset: index for index, subset in enumerate(subsets)}
    additive_features, additive_names = design_matrix(subsets, receptor_ids, False)
    quadratic_features, quadratic_names = design_matrix(subsets, receptor_ids, True)
    features_by_family = {
        "additive": (additive_features, additive_names),
        "quadratic": (quadratic_features, quadratic_names),
    }

    qubo_trials = read_csv(qubo_trials_path)
    direct_trials = read_csv(direct_trials_path)
    v1_rows = {
        row["context_id"]: row
        for row in qubo_trials
        if row["objective_family"] == "pair_synergy_qubo"
        and row["coefficient_source"] == "primary"
        and int(row["target_size"]) == target_size
        and row["aggregation"] == "min_score"
    }
    direct_rows = {
        row["context_id"]: row
        for row in direct_trials
        if int(row["target_size"]) == target_size
        and row["aggregation"] == "min_score"
    }
    if set(v1_rows) != set(contexts) or set(direct_rows) != set(contexts):
        raise ValueError("Stage 12A baseline context grid differs")

    frozen = dict(
        dict(stage10_config["objective_specs"])["pair_synergy_qubo"]
    )["frozen_candidate"]
    weights = {key: float(value) for key, value in dict(frozen["weights"]).items()}
    model_config = dict(stage10_config["model"])
    alpha_grid = [float(value) for value in config["diagnostic"]["ridge_alphas"]]
    subset_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    surrogate_rows: list[dict[str, object]] = []
    scored: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for context_id, context in contexts.items():
        print(f"scoring_context={context_id}", flush=True)
        train_values = score_subsets(context, subsets, "train")
        split_values = {"train": train_values}
        if context_id != "full_train":
            split_values["holdout"] = score_subsets(context, subsets, "validation")
        scored[context_id] = split_values

        terms = terms_for_family(
            context, "pair_synergy_qubo", "primary", "min_score"
        )
        coefficients = build_coefficients(
            terms,
            receptor_ids,
            target_size,
            weights,
            float(model_config["size_penalty"]),
        )
        energies = np.array(
            [coefficient_energy(subset, coefficients) for subset in subsets],
            dtype=float,
        )
        v1_index = int(np.argmin(energies))
        expected_v1 = parse_subset(v1_rows[context_id]["exact_subset"])
        if subsets[v1_index] != expected_v1:
            raise ValueError(f"Stage 12A v1 subset reproduction differs: {context_id}")
        direct_greedy_index = subset_index[
            parse_subset(direct_rows[context_id]["greedy_subset"])
        ]
        direct_exact_index = subset_index[
            parse_subset(direct_rows[context_id]["exact_subset"])
        ]
        for index, subset in enumerate(subsets):
            row: dict[str, object] = {
                "context_id": context_id,
                "subset": "+".join(subset),
                "v1_energy": float(energies[index]),
                **with_prefix("train", subset_metrics(train_values, index)),
                "train_robust_rank": descending_rank(
                    train_values["robust_composite"], index
                ),
            }
            if context_id != "full_train":
                holdout_values = split_values["holdout"]
                row.update(
                    {
                        **with_prefix(
                            "holdout", subset_metrics(holdout_values, index)
                        ),
                        "holdout_robust_rank": descending_rank(
                            holdout_values["robust_composite"], index
                        ),
                    }
                )
            subset_rows.append(row)

        baseline: dict[str, object] = {
            "context_id": context_id,
            "v1_subset": "+".join(subsets[v1_index]),
            "direct_greedy_subset": "+".join(subsets[direct_greedy_index]),
            "direct_exact_subset": "+".join(subsets[direct_exact_index]),
            "v1_train_rank_spearman": safe_spearman(
                -energies, train_values["robust_composite"]
            ),
            "v1_train_rank": descending_rank(
                train_values["robust_composite"], v1_index
            ),
            **with_prefix("v1_train", subset_metrics(train_values, v1_index)),
            **with_prefix(
                "direct_greedy_train",
                subset_metrics(train_values, direct_greedy_index),
            ),
            **with_prefix(
                "direct_exact_train",
                subset_metrics(train_values, direct_exact_index),
            ),
        }
        if context_id != "full_train":
            holdout_values = split_values["holdout"]
            oracle_index = choose_highest(
                holdout_values["robust_composite"], subsets
            )
            baseline.update(
                {
                    "v1_holdout_rank_spearman": safe_spearman(
                        -energies, holdout_values["robust_composite"]
                    ),
                    "train_holdout_rank_spearman": safe_spearman(
                        train_values["robust_composite"],
                        holdout_values["robust_composite"],
                    ),
                    "v1_holdout_rank": descending_rank(
                        holdout_values["robust_composite"], v1_index
                    ),
                    "oracle_holdout_subset": "+".join(subsets[oracle_index]),
                    **with_prefix(
                        "v1_holdout", subset_metrics(holdout_values, v1_index)
                    ),
                    **with_prefix(
                        "direct_greedy_holdout",
                        subset_metrics(holdout_values, direct_greedy_index),
                    ),
                    **with_prefix(
                        "direct_exact_holdout",
                        subset_metrics(holdout_values, direct_exact_index),
                    ),
                    **with_prefix(
                        "oracle_holdout",
                        subset_metrics(holdout_values, oracle_index),
                    ),
                    "v1_holdout_delta_vs_direct_greedy": float(
                        holdout_values["robust_composite"][v1_index]
                        - holdout_values["robust_composite"][direct_greedy_index]
                    ),
                }
            )
        baseline_rows.append(baseline)

        if context_id == "full_train":
            continue
        holdout_values = split_values["holdout"]
        for family, (features, _) in features_by_family.items():
            for alpha in alpha_grid:
                model, predictions = fit_surrogate(
                    features, train_values["robust_composite"], alpha
                )
                selected_index = choose_highest(predictions, subsets)
                selected_holdout = subset_metrics(holdout_values, selected_index)
                surrogate_rows.append(
                    {
                        "context_id": context_id,
                        "model_family": family,
                        "alpha": alpha,
                        "feature_count": features.shape[1],
                        "selected_subset": "+".join(subsets[selected_index]),
                        "train_r2": float(
                            model.score(features, train_values["robust_composite"])
                        ),
                        "train_rank_spearman": safe_spearman(
                            predictions, train_values["robust_composite"]
                        ),
                        "holdout_rank_spearman": safe_spearman(
                            predictions, holdout_values["robust_composite"]
                        ),
                        **with_prefix(
                            "train", subset_metrics(train_values, selected_index)
                        ),
                        **with_prefix("holdout", selected_holdout),
                        "holdout_delta_vs_v1": selected_holdout[
                            "robust_composite"
                        ]
                        - float(holdout_values["robust_composite"][v1_index]),
                        "holdout_delta_vs_direct_greedy": selected_holdout[
                            "robust_composite"
                        ]
                        - float(
                            holdout_values["robust_composite"][
                                direct_greedy_index
                            ]
                        ),
                    }
                )

    aggregate = aggregate_surrogate_trials(surrogate_rows)

    def aggregate_key(value: dict[str, object]) -> tuple[object, ...]:
        return (
            -float(value["mean_holdout_robust_composite"]),
            -float(value["worst_holdout_robust_composite"]),
            -float(value["mean_holdout_rank_spearman"]),
            float(value["alpha"]),
        )

    best_additive = min(
        (row for row in aggregate if row["model_family"] == "additive"),
        key=aggregate_key,
    )
    best_quadratic = min(
        (row for row in aggregate if row["model_family"] == "quadratic"),
        key=aggregate_key,
    )
    selected_alpha = float(best_quadratic["alpha"])
    full_values = scored["full_train"]["train"]
    full_model, full_predictions = fit_surrogate(
        quadratic_features, full_values["robust_composite"], selected_alpha
    )
    full_selected_index = choose_highest(full_predictions, subsets)
    full_selected = subsets[full_selected_index]
    qubo = build_explicit_qubo(
        full_model,
        quadratic_names,
        receptor_ids,
        target_size,
        float(config["diagnostic"]["cardinality_penalty"]),
    )
    exact_subset, exact_energy = exact_all_cardinalities(receptor_ids, qubo)
    if exact_subset != full_selected:
        raise ValueError("penalized provisional v2 QUBO changed the selected subset")

    fold_baselines = [row for row in baseline_rows if row["context_id"] != "full_train"]
    v1_diagnosis = {
        "fold_count": len(fold_baselines),
        "mean_train_rank_spearman": statistics.fmean(
            float(row["v1_train_rank_spearman"]) for row in fold_baselines
        ),
        "mean_holdout_rank_spearman": statistics.fmean(
            float(row["v1_holdout_rank_spearman"]) for row in fold_baselines
        ),
        "mean_train_holdout_rank_spearman": statistics.fmean(
            float(row["train_holdout_rank_spearman"]) for row in fold_baselines
        ),
        "mean_v1_delta_vs_direct_greedy": statistics.fmean(
            float(row["v1_holdout_delta_vs_direct_greedy"])
            for row in fold_baselines
        ),
        "positive_fold_count_vs_direct_greedy": sum(
            float(row["v1_holdout_delta_vs_direct_greedy"]) > 0.0
            for row in fold_baselines
        ),
        "mean_v1_holdout_rank": statistics.fmean(
            int(row["v1_holdout_rank"]) for row in fold_baselines
        ),
    }
    quadratic_pair_value = float(
        best_quadratic["mean_holdout_robust_composite"]
    ) - float(best_additive["mean_holdout_robust_composite"])
    provisional_supported = (
        float(best_quadratic["mean_delta_vs_v1"]) > 0.0
        and float(best_quadratic["mean_delta_vs_direct_greedy"]) > 0.0
        and int(best_quadratic["positive_fold_count_vs_v1"]) >= 3
        and int(best_quadratic["positive_fold_count_vs_direct_greedy"]) >= 3
        and quadratic_pair_value > 0.0
    )
    status = (
        "stage12a_provisional_quadratic_v2_supported_for_external_testing"
        if provisional_supported
        else "stage12a_no_quadratic_v2_gate_retain_v1_for_external_testing"
    )

    outputs = dict(config["outputs"])
    subset_path = rooted(root, str(outputs["subset_diagnostics_csv"]))
    surrogate_path = rooted(root, str(outputs["surrogate_trials_csv"]))
    qubo_path = rooted(root, str(outputs["provisional_v2_qubo_json"]))
    result_path = rooted(root, str(outputs["result_json"]))
    report_path = rooted(root, str(outputs["report_md"]))
    if not overwrite and any(
        path.exists()
        for path in (subset_path, surrogate_path, qubo_path, result_path, report_path)
    ):
        raise FileExistsError("Stage 12A outputs exist; pass --overwrite")
    write_csv(subset_path, subset_rows)
    write_csv(surrogate_path, surrogate_rows)
    provisional_qubo = {
        "schema_version": "1.0",
        "candidate_id": "stage12a-mk14-regularized-quadratic-surrogate-v2-provisional",
        "status": "development_only_not_independently_validated",
        "alpha": selected_alpha,
        "response": config["diagnostic"]["surrogate_response"],
        "selected_subset": list(full_selected),
        "full_train_metrics": subset_metrics(full_values, full_selected_index),
        "full_train_fit": {
            "r2": float(
                full_model.score(quadratic_features, full_values["robust_composite"])
            ),
            "rank_spearman": safe_spearman(
                full_predictions, full_values["robust_composite"]
            ),
        },
        "exact_penalized_qubo_subset": list(exact_subset),
        "exact_penalized_qubo_energy": exact_energy,
        "qubo": qubo,
        "data_boundary": {
            "stage09_train_rows_read": len(ligand_rows),
            "stage11_validation_rows_read": 0,
            "test_rows_read": 0,
        },
    }
    write_json(qubo_path, provisional_qubo)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": status,
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "input_dimensions": {
            "receptor_count": len(receptor_ids),
            "ligand_count": len(ligand_rows),
            "triple_subset_count": len(subsets),
            "outer_fold_count": len(fold_baselines),
        },
        "response_definition": config["diagnostic"]["surrogate_response"],
        "v1_diagnosis": v1_diagnosis,
        "baseline_folds": baseline_rows,
        "surrogate_aggregate": aggregate,
        "surrogate_selection": {
            "selection_rule": config["diagnostic"]["surrogate_selection_rule"],
            "best_additive": best_additive,
            "best_quadratic": best_quadratic,
            "quadratic_mean_holdout_gain_over_additive": quadratic_pair_value,
        },
        "provisional_v2_candidate": {
            "alpha": selected_alpha,
            "selected_subset": "+".join(full_selected),
            "exact_qubo_subset": "+".join(exact_subset),
            "development_support_rule_passed": provisional_supported,
            "qubo_path": qubo_path.relative_to(root).as_posix(),
            "qubo_sha256": file_sha256(qubo_path),
        },
        "outputs": {
            "subset_diagnostics_csv": {
                "path": subset_path.relative_to(root).as_posix(),
                "sha256": file_sha256(subset_path),
            },
            "surrogate_trials_csv": {
                "path": surrogate_path.relative_to(root).as_posix(),
                "sha256": file_sha256(surrogate_path),
            },
        },
        "data_boundary": {
            "stage09_train_rows_read": len(ligand_rows),
            "stage11_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    write_json(result_path, result)
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
