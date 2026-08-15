"""Independently audit the Stage 19h auxiliary-coverage QUBO diagnostic."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_stage19e_cross_target_qubo_v2 import (
    load_target,
    safe_spearman,
    score_subsets,
)
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage05_mk14_uncertainty_qubo_gate import make_context


DEFAULT_CONFIG = Path("configs/stage19h_auxiliary_coverage_qubo.json")
DEFAULT_OUTPUT = Path("data/stage19h_auxiliary_coverage_qubo_audit.json")
CANDIDATE_METHOD = "auxiliary_coverage_nested"
BASELINE_METHODS = (
    "direct_greedy",
    "additive_nested",
    "v1_qubo_exact",
    "stable_singleton_linear",
)
MATRIX_IDS = ("primary", "sensitivity", "seed0", "seed1", "seed2")
METRIC_IDS = (
    "primary",
    "sensitivity",
    "mean_seed",
    "worst_seed",
    "robust_composite",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def rooted(root: Path, value: str) -> Path:
    path = (root / value.replace("\\", "/")).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path leaves repository root: {value}") from error
    return path


def verify_descriptor(root: Path, descriptor: dict[str, Any]) -> Path:
    path = rooted(root, str(descriptor["path"]))
    if not path.is_file() or file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"descriptor differs: {path}")
    if "size_bytes" in descriptor and path.stat().st_size != int(
        descriptor["size_bytes"]
    ):
        raise ValueError(f"descriptor size differs: {path}")
    return path


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def compare_nested(actual: Any, expected: Any, label: str) -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"{label} keys differ")
        for key in actual:
            compare_nested(actual[key], expected[key], f"{label}/{key}")
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{label} lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_nested(left, right, f"{label}/{index}")
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        assert_close(float(actual), float(expected), label)
    elif actual != expected:
        raise ValueError(f"{label} differs: {actual!r} != {expected!r}")


def minmax(values: dict[str, float]) -> dict[str, float]:
    low = min(values.values())
    high = max(values.values())
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-15):
        return {key: 0.0 for key in values}
    return {key: (float(value) - low) / (high - low) for key, value in values.items()}


def pair_key(first: str, second: str) -> str:
    return "__".join(sorted((first, second)))


def rank_fractions(
    context: dict[str, Any], receptor_ids: list[str]
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    rows_by_matrix = {
        matrix_id: {
            str(row["ligand_id"]): row
            for row in context["matrices"][matrix_id]["train"]
        }
        for matrix_id in MATRIX_IDS
    }
    ligand_ids = sorted(rows_by_matrix["primary"])
    ranks = {
        ligand_id: {receptor_id: [] for receptor_id in receptor_ids}
        for ligand_id in ligand_ids
    }
    labels = {
        ligand_id: str(rows_by_matrix["primary"][ligand_id]["label"])
        for ligand_id in ligand_ids
    }
    for matrix_id in MATRIX_IDS:
        for receptor_id in receptor_ids:
            ordered = sorted(
                ligand_ids,
                key=lambda ligand_id: (
                    float(rows_by_matrix[matrix_id][ligand_id][receptor_id]),
                    ligand_id,
                ),
            )
            denominator = max(1, len(ordered) - 1)
            for rank, ligand_id in enumerate(ordered):
                ranks[ligand_id][receptor_id].append(rank / denominator)
    return (
        {
            ligand_id: {
                receptor_id: float(np.median(values))
                for receptor_id, values in receptor_values.items()
            }
            for ligand_id, receptor_values in ranks.items()
        },
        labels,
    )


def independent_terms(
    context: dict[str, Any],
    receptor_ids: list[str],
    coverage_fraction: float,
    bedroc_alpha: float,
    singleton_values: dict[str, float],
) -> dict[str, Any]:
    ranks, labels = rank_fractions(context, receptor_ids)
    active_ids = sorted(key for key, value in labels.items() if value == "active")
    decoy_ids = sorted(key for key, value in labels.items() if value == "decoy")
    active_raw = {
        ligand_id: math.exp(-bedroc_alpha * min(ranks[ligand_id].values()))
        for ligand_id in active_ids
    }
    decoy_raw = {
        ligand_id: math.exp(-bedroc_alpha * min(ranks[ligand_id].values()))
        for ligand_id in decoy_ids
    }
    active_total = sum(active_raw.values())
    decoy_total = sum(decoy_raw.values())
    columns = {
        receptor_id: np.asarray(
            [
                float(row[receptor_id])
                for row in context["matrices"]["primary"]["train"]
            ]
        )
        for receptor_id in receptor_ids
    }
    correlations: dict[str, float] = {}
    for first, second in itertools.combinations(receptor_ids, 2):
        value = float(spearmanr(columns[first], columns[second]).statistic)
        correlations[pair_key(first, second)] = (
            max(0.0, value) if math.isfinite(value) else 0.0
        )
    return {
        "coverage_fraction": coverage_fraction,
        "bedroc_alpha": bedroc_alpha,
        "active_ids": active_ids,
        "decoy_ids": decoy_ids,
        "active_incidence": {
            ligand_id: [
                receptor_id
                for receptor_id in receptor_ids
                if ranks[ligand_id][receptor_id] <= coverage_fraction
            ]
            for ligand_id in active_ids
        },
        "decoy_incidence": {
            ligand_id: [
                receptor_id
                for receptor_id in receptor_ids
                if ranks[ligand_id][receptor_id] <= coverage_fraction
            ]
            for ligand_id in decoy_ids
        },
        "active_weights": {
            ligand_id: value / active_total for ligand_id, value in active_raw.items()
        },
        "decoy_weights": {
            ligand_id: value / decoy_total for ligand_id, value in decoy_raw.items()
        },
        "singleton_utility": minmax(singleton_values),
        "correlations": correlations,
    }


def slack_weights(maximum: int) -> list[int]:
    weights: list[int] = []
    total = 0
    next_weight = 1
    while total < maximum:
        value = min(next_weight, maximum - total)
        weights.append(value)
        total += value
        next_weight *= 2
    return weights


def binary_slack(weights: list[int], value: int) -> list[int]:
    for bits in itertools.product((0, 1), repeat=len(weights)):
        if sum(weight * bit for weight, bit in zip(weights, bits)) == value:
            return list(bits)
    raise ValueError("required slack value is not representable")


def add_linear(coefficients: dict[str, Any], variable: str, value: float) -> None:
    coefficients["linear"][variable] = coefficients["linear"].get(variable, 0.0) + value


def add_quadratic(
    coefficients: dict[str, Any], first: str, second: str, value: float
) -> None:
    if first == second:
        add_linear(coefficients, first, value)
        return
    key = "::".join(sorted((first, second)))
    coefficients["quadratic"][key] = coefficients["quadratic"].get(key, 0.0) + value


def add_square(
    coefficients: dict[str, Any], constant: float, terms: dict[str, float], weight: float
) -> None:
    coefficients["constant"] += weight * constant * constant
    for variable, value in terms.items():
        add_linear(
            coefficients,
            variable,
            weight * (2.0 * constant * value + value * value),
        )
    variables = list(terms)
    for first_index, first in enumerate(variables):
        for second in variables[first_index + 1 :]:
            add_quadratic(
                coefficients,
                first,
                second,
                2.0 * weight * terms[first] * terms[second],
            )


def independent_qubo(
    terms: dict[str, Any],
    receptor_ids: list[str],
    target_size: int,
    params: dict[str, float],
    cardinality_penalty: float,
    constraint_penalty: float,
) -> dict[str, Any]:
    coefficients: dict[str, Any] = {"constant": 0.0, "linear": {}, "quadratic": {}}
    x_names = {receptor_id: f"x__{receptor_id}" for receptor_id in receptor_ids}
    active_y = {
        ligand_id: f"y__{ligand_id}"
        for ligand_id, incidence in terms["active_incidence"].items()
        if incidence
    }
    decoy_z = {
        ligand_id: f"z__{ligand_id}"
        for ligand_id, incidence in terms["decoy_incidence"].items()
        if incidence
    }
    slack_names: list[str] = []
    add_square(
        coefficients,
        -target_size,
        {name: 1.0 for name in x_names.values()},
        cardinality_penalty,
    )
    for ligand_id, incidence in terms["active_incidence"].items():
        if not incidence:
            continue
        expression = {active_y[ligand_id]: 1.0}
        for index, weight in enumerate(slack_weights(len(incidence))):
            name = f"s__{ligand_id}__{index}"
            slack_names.append(name)
            expression[name] = float(weight)
        for receptor_id in incidence:
            expression[x_names[receptor_id]] = -1.0
        add_square(coefficients, 0.0, expression, constraint_penalty)
        add_linear(
            coefficients,
            active_y[ligand_id],
            -float(terms["active_weights"][ligand_id]),
        )
    for ligand_id, incidence in terms["decoy_incidence"].items():
        if not incidence:
            continue
        z_name = decoy_z[ligand_id]
        add_linear(
            coefficients,
            z_name,
            float(params["decoy_weight"]) * float(terms["decoy_weights"][ligand_id]),
        )
        for receptor_id in incidence:
            add_linear(coefficients, x_names[receptor_id], constraint_penalty)
            add_quadratic(
                coefficients, x_names[receptor_id], z_name, -constraint_penalty
            )
    singleton_scale = float(params["singleton_weight"]) / target_size
    for receptor_id in receptor_ids:
        add_linear(
            coefficients,
            x_names[receptor_id],
            -singleton_scale * float(terms["singleton_utility"][receptor_id]),
        )
    pair_denominator = max(1, target_size * (target_size - 1) // 2)
    for first, second in itertools.combinations(receptor_ids, 2):
        add_quadratic(
            coefficients,
            x_names[first],
            x_names[second],
            float(params["redundancy_weight"])
            * float(terms["correlations"].get(pair_key(first, second), 0.0))
            / pair_denominator,
        )
    groups = {
        "x": [x_names[receptor_id] for receptor_id in receptor_ids],
        "active_y": list(active_y.values()),
        "decoy_z": list(decoy_z.values()),
        "active_slack": slack_names,
    }
    return {
        "constant": float(coefficients["constant"]),
        "linear": {key: float(value) for key, value in coefficients["linear"].items()},
        "quadratic": {
            key: float(value) for key, value in coefficients["quadratic"].items()
        },
        "variables": sorted(set().union(*[set(values) for values in groups.values()])),
        "variable_groups": groups,
        "target_size": target_size,
        "decoy_weight": float(params["decoy_weight"]),
        "singleton_weight": float(params["singleton_weight"]),
        "redundancy_weight": float(params["redundancy_weight"]),
        "cardinality_penalty": cardinality_penalty,
        "constraint_penalty": constraint_penalty,
        "convention": (
            "Q(b)=constant+sum_v linear[v]*b_v+"
            "sum_u<v quadratic[u::v]*b_u*b_v; minimize Q"
        ),
    }


def reduced_objective(
    terms: dict[str, Any], subset: tuple[str, ...], params: dict[str, float]
) -> float:
    selected = set(subset)
    active = sum(
        float(terms["active_weights"][ligand_id])
        for ligand_id, incidence in terms["active_incidence"].items()
        if selected.intersection(incidence)
    )
    decoy = sum(
        float(terms["decoy_weights"][ligand_id])
        for ligand_id, incidence in terms["decoy_incidence"].items()
        if selected.intersection(incidence)
    )
    singleton = float(params["singleton_weight"]) * statistics.fmean(
        float(terms["singleton_utility"][receptor_id]) for receptor_id in subset
    )
    redundancy = float(params["redundancy_weight"]) * statistics.fmean(
        float(terms["correlations"].get(pair_key(first, second), 0.0))
        for first, second in itertools.combinations(sorted(subset), 2)
    )
    return float(active - float(params["decoy_weight"]) * decoy + singleton - redundancy)


def assignment(
    terms: dict[str, Any], qubo: dict[str, Any], subset: tuple[str, ...]
) -> dict[str, int]:
    selected = set(subset)
    values = {variable: 0 for variable in qubo["variables"]}
    for receptor_id in selected:
        values[f"x__{receptor_id}"] = 1
    for ligand_id, incidence in terms["active_incidence"].items():
        if not incidence:
            continue
        count = sum(receptor_id in selected for receptor_id in incidence)
        y_value = int(count > 0)
        values[f"y__{ligand_id}"] = y_value
        weights = slack_weights(len(incidence))
        for index, bit in enumerate(binary_slack(weights, count - y_value)):
            values[f"s__{ligand_id}__{index}"] = bit
    for ligand_id, incidence in terms["decoy_incidence"].items():
        if incidence:
            values[f"z__{ligand_id}"] = int(
                any(receptor_id in selected for receptor_id in incidence)
            )
    return values


def validate_constraints(
    terms: dict[str, Any], values: dict[str, int], target_size: int
) -> None:
    if any(value not in (0, 1) for value in values.values()):
        raise ValueError("QUBO assignment is not binary")
    if sum(value for key, value in values.items() if key.startswith("x__")) != target_size:
        raise ValueError("QUBO cardinality constraint differs")
    for ligand_id, incidence in terms["active_incidence"].items():
        if not incidence:
            continue
        count = sum(values[f"x__{receptor_id}"] for receptor_id in incidence)
        y_value = values[f"y__{ligand_id}"]
        slack = sum(
            weight * values[f"s__{ligand_id}__{index}"]
            for index, weight in enumerate(slack_weights(len(incidence)))
        )
        if y_value + slack != count or y_value != int(count > 0):
            raise ValueError(f"active coverage constraint differs: {ligand_id}")
    for ligand_id, incidence in terms["decoy_incidence"].items():
        if not incidence:
            continue
        exposed = int(any(values[f"x__{receptor_id}"] for receptor_id in incidence))
        if values[f"z__{ligand_id}"] != exposed:
            raise ValueError(f"decoy exposure constraint differs: {ligand_id}")


def qubo_energy(qubo: dict[str, Any], values: dict[str, int]) -> float:
    energy = float(qubo["constant"])
    energy += sum(
        float(coefficient) * values.get(variable, 0)
        for variable, coefficient in qubo["linear"].items()
    )
    energy += sum(
        float(coefficient) * values.get(first, 0) * values.get(second, 0)
        for key, coefficient in qubo["quadratic"].items()
        for first, second in [key.split("::", 1)]
    )
    return float(energy)


def parameters(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["coverage_fraction"]),
        float(row["decoy_weight"]),
        float(row["singleton_weight"]),
        float(row["redundancy_weight"]),
    )


def parameter_dict(values: tuple[float, float, float, float]) -> dict[str, float]:
    return dict(
        zip(
            (
                "coverage_fraction",
                "decoy_weight",
                "singleton_weight",
                "redundancy_weight",
            ),
            values,
        )
    )


def objective_key(rows: list[dict[str, Any]]) -> tuple[Any, ...]:
    return (
        -statistics.fmean(float(row["validation_robust_composite"]) for row in rows),
        -min(float(row["validation_robust_composite"]) for row in rows),
        -statistics.fmean(float(row["validation_rank_spearman"]) for row in rows),
        *parameters(rows[0]),
    )


def selected_parameters(rows: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    grouped: dict[tuple[float, float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[parameters(row)].append(row)
    return min(grouped, key=lambda key: objective_key(grouped[key]))


def compare_metrics(
    row: dict[str, Any], values: dict[str, np.ndarray], index: int, prefix: str, label: str
) -> None:
    for metric_id in METRIC_IDS:
        assert_close(
            float(row[f"{prefix}_{metric_id}"]),
            float(values[metric_id][index]),
            f"{label}/{prefix}/{metric_id}",
        )


def paired_comparison(
    rows: list[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    indexed = {
        (row["target_id"], int(row["outer_fold"]), row["method"]): row
        for row in rows
    }
    all_deltas: list[float] = []
    per_target: dict[str, Any] = {}
    for target_id in sorted({str(row["target_id"]) for row in rows}):
        folds = sorted(
            int(row["outer_fold"])
            for row in rows
            if row["target_id"] == target_id and row["method"] == left
        )
        deltas = [
            float(indexed[(target_id, fold, left)]["holdout_robust_composite"])
            - float(indexed[(target_id, fold, right)]["holdout_robust_composite"])
            for fold in folds
        ]
        all_deltas.extend(deltas)
        per_target[target_id] = {
            "fold_count": len(deltas),
            "mean_delta": statistics.fmean(deltas),
            "positive_fold_count": sum(value > 0.0 for value in deltas),
            "fold_deltas": deltas,
        }
    return {
        "direction": f"{left} minus {right}",
        "fold_count": len(all_deltas),
        "mean_delta": statistics.fmean(all_deltas),
        "positive_fold_count": sum(value > 0.0 for value in all_deltas),
        "per_target": per_target,
    }


def certify_all_triples(
    terms: dict[str, Any],
    qubo: dict[str, Any],
    receptor_ids: list[str],
    target_size: int,
    params: dict[str, float],
) -> dict[str, Any]:
    states: list[tuple[tuple[str, ...], float, float]] = []
    baseline: float | None = None
    maximum_residual = 0.0
    for subset in itertools.combinations(receptor_ids, target_size):
        subset = tuple(sorted(subset))
        values = assignment(terms, qubo, subset)
        validate_constraints(terms, values, target_size)
        objective = reduced_objective(terms, subset, params)
        energy = qubo_energy(qubo, values)
        current = energy + objective
        if baseline is None:
            baseline = current
        maximum_residual = max(maximum_residual, abs(current - baseline))
        states.append((subset, objective, energy))
    if maximum_residual > 1e-7:
        raise ValueError("QUBO energy is not equivalent to the reduced objective")
    objective_best = min(states, key=lambda item: (-item[1], item[0]))
    energy_best = min(states, key=lambda item: (item[2], item[0]))
    if objective_best[0] != energy_best[0]:
        raise ValueError("QUBO and reduced objective select different receptor triples")
    return {
        "state_count": len(states),
        "maximum_equivalence_residual": maximum_residual,
        "selected_subset": list(objective_best[0]),
        "selected_objective": objective_best[1],
        "selected_energy": energy_best[2],
        "selected_assignment": assignment(terms, qubo, objective_best[0]),
    }


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = rooted(root, config_path.as_posix())
    config = read_json(config_path)
    verify_descriptor(root, config["implementation"])
    input_paths = {
        key: verify_descriptor(root, descriptor)
        for key, descriptor in config["inputs"].items()
    }
    source_config = read_json(input_paths["stage19e_config"])
    result_path = rooted(root, str(config["outputs"]["result_json"]))
    result = read_json(result_path)
    expected_status = "stage19h_auxiliary_coverage_not_supported_do_not_amend_bace1"
    if result["status"] != expected_status:
        raise ValueError("unexpected Stage 19h result status")
    if result["config"]["sha256"] != file_sha256(config_path):
        raise ValueError("Stage 19h result identifies another config")
    output_paths = {
        key: verify_descriptor(root, descriptor)
        for key, descriptor in result["outputs"].items()
    }
    inner_rows = read_csv(output_paths["inner_trials_csv"])
    outer_rows = read_csv(output_paths["outer_candidate_trials_csv"])
    method_rows = read_csv(output_paths["comparison_methods_csv"])
    model = read_json(output_paths["model_record_json"])

    diagnostic = config["diagnostic"]
    outer_count = int(diagnostic["outer_fold_count"])
    inner_count = int(diagnostic["inner_fold_count"])
    target_size = int(diagnostic["target_size"])
    alpha = float(diagnostic["bedroc_alpha"])
    cardinality_penalty = float(diagnostic["cardinality_penalty"])
    constraint_penalty = float(diagnostic["constraint_penalty"])
    candidates = list(
        itertools.product(
            [float(value) for value in diagnostic["coverage_fractions"]],
            [float(value) for value in diagnostic["decoy_weights"]],
            [float(value) for value in diagnostic["singleton_weights"]],
            [float(value) for value in diagnostic["redundancy_weights"]],
        )
    )
    target_count = len(source_config["targets"])
    expected_counts = (
        target_count * outer_count * inner_count * len(candidates),
        target_count * outer_count * len(candidates),
        target_count * outer_count * (1 + len(BASELINE_METHODS)),
    )
    if (len(inner_rows), len(outer_rows), len(method_rows)) != expected_counts:
        raise ValueError("Stage 19h output row count differs")
    if len({tuple(row.values()) for row in inner_rows}) != len(inner_rows):
        raise ValueError("duplicate Stage 19h inner row")
    if len({tuple(row.values()) for row in outer_rows}) != len(outer_rows):
        raise ValueError("duplicate Stage 19h outer row")

    expected_boundary = {
        "train_rows_read_by_target": {
            target_id: int(spec["expected"]["ligand_count"])
            for target_id, spec in source_config["targets"].items()
        },
        "new_docking_jobs": 0,
        "fresh_validation_rows_read": 0,
        "test_rows_read": 0,
        "bace1_docking_rows_read": 0,
    }
    if result["data_boundary"] != expected_boundary or model["data_boundary"] != expected_boundary:
        raise ValueError("Stage 19h data boundary differs")
    if model["status"] != "development_gate_failed_not_authorized_for_bace1":
        raise ValueError("Stage 19h model authorization differs")

    baseline_source = [
        row
        for row in read_csv(input_paths["stage19f_comparison_methods"])
        if row["method"] in BASELINE_METHODS
    ]
    baseline_output = [row for row in method_rows if row["method"] in BASELINE_METHODS]
    if len(baseline_source) != len(baseline_output):
        raise ValueError("Stage 19h baseline row count differs")
    baseline_index = {
        (row["target_id"], int(row["outer_fold"]), row["method"]): row
        for row in baseline_output
    }
    for source in baseline_source:
        key = (source["target_id"], int(source["outer_fold"]), source["method"])
        recorded = baseline_index[key]
        for field, value in source.items():
            if value == "" or field not in recorded:
                continue
            if field in {"target_id", "method", "selected_subset"}:
                if recorded[field] != value:
                    raise ValueError(f"baseline field differs: {key}/{field}")
                continue
            try:
                assert_close(float(recorded[field]), float(value), f"baseline/{key}/{field}")
            except ValueError as error:
                if recorded[field] != value:
                    raise error

    nested_choices = 0
    selected_outer_metric_sets = 0
    full_certificates: dict[str, Any] = {}
    group_outer_folds: dict[tuple[str, str], set[int]] = defaultdict(set)
    scaffold_outer_folds: dict[tuple[str, str], set[int]] = defaultdict(set)
    group_inner_folds: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    scaffold_inner_folds: dict[tuple[str, int, str], set[int]] = defaultdict(set)

    for target_id, spec in source_config["targets"].items():
        ligands, receptor_ids, matrices = load_target(root, target_id, spec)
        assignments = make_frozen_group_folds(
            ligands, outer_count, int(diagnostic["fold_seed"])
        )
        for row in ligands:
            group_outer_folds[(target_id, row["split_group_id"])].add(
                assignments[row["ligand_id"]]
            )
            scaffold_outer_folds[(target_id, row["scaffold_smiles"])].add(
                assignments[row["ligand_id"]]
            )
        all_ids = {str(row["ligand_id"]) for row in ligands}
        triples = [tuple(sorted(value)) for value in itertools.combinations(receptor_ids, target_size)]
        triple_index = {subset: index for index, subset in enumerate(triples)}
        model_spec = {
            "coverage_fraction": float(spec["v1_qubo"]["coverage_fraction"]),
            "utility_metric": "bedroc",
        }

        for outer_fold in range(outer_count):
            holdout = {
                ligand_id
                for ligand_id, fold in assignments.items()
                if fold == outer_fold
            }
            train = all_ids - holdout
            inner_assignments = make_frozen_group_folds(
                [row for row in ligands if row["ligand_id"] in train],
                inner_count,
                int(diagnostic["inner_fold_seed"]) + outer_fold,
            )
            for row in ligands:
                if row["ligand_id"] not in train:
                    continue
                group_inner_folds[(target_id, outer_fold, row["split_group_id"])].add(
                    inner_assignments[row["ligand_id"]]
                )
                scaffold_inner_folds[(target_id, outer_fold, row["scaffold_smiles"])].add(
                    inner_assignments[row["ligand_id"]]
                )
            fold_inner = [
                row
                for row in inner_rows
                if row["target_id"] == target_id
                and int(row["outer_fold"]) == outer_fold
            ]
            choice = selected_parameters(fold_inner)
            if len(fold_inner) != inner_count * len(candidates):
                raise ValueError(f"{target_id}/{outer_fold} inner row count differs")
            if any(
                len([row for row in fold_inner if parameters(row) == candidate]) != inner_count
                for candidate in candidates
            ):
                raise ValueError(f"{target_id}/{outer_fold} candidate fold coverage differs")
            fold_outer = [
                row
                for row in outer_rows
                if row["target_id"] == target_id
                and int(row["outer_fold"]) == outer_fold
            ]
            selected_outer = [row for row in fold_outer if row["selected_by_inner_cv"] == "True"]
            if len(selected_outer) != 1 or parameters(selected_outer[0]) != choice:
                raise ValueError(f"{target_id}/{outer_fold} nested choice differs")
            nested_choices += 1

            context = make_context(train, holdout, matrices, receptor_ids, model_spec)
            singleton_values = {
                receptor_id: float(
                    score_subsets(
                        context, [(receptor_id,)], receptor_ids, "train", alpha
                    )["robust_composite"][0]
                )
                for receptor_id in receptor_ids
            }
            params = parameter_dict(choice)
            terms = independent_terms(
                context, receptor_ids, params["coverage_fraction"], alpha, singleton_values
            )
            objectives = np.asarray(
                [reduced_objective(terms, subset, params) for subset in triples], dtype=float
            )
            selected_index = min(
                range(len(triples)), key=lambda index: (-float(objectives[index]), triples[index])
            )
            subset = triples[selected_index]
            outer_row = selected_outer[0]
            if "+".join(subset) != outer_row["selected_subset"]:
                raise ValueError(f"{target_id}/{outer_fold} outer subset differs")
            train_values = score_subsets(context, triples, receptor_ids, "train", alpha)
            holdout_values = score_subsets(context, triples, receptor_ids, "validation", alpha)
            compare_metrics(
                outer_row,
                holdout_values,
                selected_index,
                "validation",
                f"{target_id}/{outer_fold}/outer",
            )
            assert_close(
                float(outer_row["validation_rank_spearman"]),
                safe_spearman(objectives, holdout_values["robust_composite"]),
                f"{target_id}/{outer_fold}/outer/rank",
            )
            method_match = [
                row
                for row in method_rows
                if row["target_id"] == target_id
                and int(row["outer_fold"]) == outer_fold
                and row["method"] == CANDIDATE_METHOD
            ]
            if len(method_match) != 1 or method_match[0]["selected_subset"] != "+".join(subset):
                raise ValueError(f"{target_id}/{outer_fold} method row differs")
            compare_metrics(
                method_match[0], train_values, selected_index, "train", f"{target_id}/{outer_fold}"
            )
            compare_metrics(
                method_match[0],
                holdout_values,
                selected_index,
                "holdout",
                f"{target_id}/{outer_fold}",
            )
            rank = 1 + sum(
                float(value) > float(holdout_values["robust_composite"][selected_index])
                for value in holdout_values["robust_composite"]
            )
            if int(method_match[0]["holdout_rank"]) != rank:
                raise ValueError(f"{target_id}/{outer_fold} holdout rank differs")
            selected_outer_metric_sets += 2

        target_outer = [row for row in outer_rows if row["target_id"] == target_id]
        full_choice = selected_parameters(target_outer)
        full_params = parameter_dict(full_choice)
        full_context = make_context(all_ids, set(), matrices, receptor_ids, model_spec)
        full_singletons = {
            receptor_id: float(
                score_subsets(
                    full_context, [(receptor_id,)], receptor_ids, "train", alpha
                )["robust_composite"][0]
            )
            for receptor_id in receptor_ids
        }
        full_terms = independent_terms(
            full_context,
            receptor_ids,
            full_params["coverage_fraction"],
            alpha,
            full_singletons,
        )
        full_qubo = independent_qubo(
            full_terms,
            receptor_ids,
            target_size,
            full_params,
            cardinality_penalty,
            constraint_penalty,
        )
        recorded = model["target_development_models"][target_id]
        compare_nested(full_params, recorded["selected_parameters"], f"{target_id}/parameters")
        compare_nested(full_terms, recorded["terms"], f"{target_id}/terms")
        compare_nested(full_qubo, recorded["qubo"], f"{target_id}/qubo")
        certificate = certify_all_triples(
            full_terms, full_qubo, receptor_ids, target_size, full_params
        )
        if certificate["selected_subset"] != recorded["selected_subset"]:
            raise ValueError(f"{target_id} full-train subset differs")
        compare_nested(
            certificate["selected_assignment"],
            recorded["selected_assignment"],
            f"{target_id}/assignment",
        )
        assert_close(
            certificate["selected_energy"],
            float(recorded["selected_energy"]),
            f"{target_id}/energy",
        )
        assert_close(
            certificate["selected_objective"],
            float(recorded["selected_objective"]),
            f"{target_id}/objective",
        )
        if int(recorded["equivalence_states_evaluated"]) != len(triples):
            raise ValueError(f"{target_id} equivalence state count differs")
        if certificate["maximum_equivalence_residual"] > 1e-7:
            raise ValueError(f"{target_id} QUBO certificate residual differs")
        full_values = score_subsets(full_context, triples, receptor_ids, "train", alpha)
        full_index = triple_index[tuple(recorded["selected_subset"])]
        metrics = {
            metric_id: float(full_values[metric_id][full_index])
            for metric_id in METRIC_IDS
        }
        compare_nested(metrics, recorded["full_train_metrics"], f"{target_id}/metrics")
        rank = 1 + sum(
            float(value) > float(full_values["robust_composite"][full_index])
            for value in full_values["robust_composite"]
        )
        if rank != int(recorded["full_train_rank"]):
            raise ValueError(f"{target_id} full-train rank differs")
        summary = result["full_train_models"][target_id]
        expected_summary = {
            key: value
            for key, value in recorded.items()
            if key not in {"terms", "qubo", "selected_assignment"}
        }
        compare_nested(expected_summary, summary, f"{target_id}/result summary")
        full_certificates[target_id] = {
            "states_certified": certificate["state_count"],
            "maximum_equivalence_residual": certificate[
                "maximum_equivalence_residual"
            ],
            "selected_subset": certificate["selected_subset"],
            "variable_count": len(full_qubo["variables"]),
        }

    if any(len(folds) != 1 for folds in group_outer_folds.values()):
        raise ValueError("an outer split group crosses folds")
    if any(len(folds) != 1 for folds in scaffold_outer_folds.values()):
        raise ValueError("an outer scaffold crosses folds")
    if any(len(folds) != 1 for folds in group_inner_folds.values()):
        raise ValueError("an inner split group crosses folds")
    if any(len(folds) != 1 for folds in scaffold_inner_folds.values()):
        raise ValueError("an inner scaffold crosses folds")

    comparisons = {
        f"auxiliary_coverage_vs_{method}": paired_comparison(
            method_rows, CANDIDATE_METHOD, method
        )
        for method in BASELINE_METHODS
    }
    compare_nested(comparisons, result["paired_comparisons"], "paired comparisons")
    gate_spec = config["development_support_gate"]
    checks = {
        key: (
            all(
                float(value["mean_delta"])
                > float(gate_spec["minimum_target_mean_delta"])
                for value in comparison["per_target"].values()
            )
            and int(comparison["positive_fold_count"])
            >= int(gate_spec["minimum_positive_folds_of_eight"])
        )
        for key, comparison in comparisons.items()
    }
    if checks != result["development_gate"]["comparison_checks"]:
        raise ValueError("Stage 19h comparison checks differ")
    if all(checks.values()) or result["development_gate"]["passed"]:
        raise ValueError("failed Stage 19h gate was not preserved")
    if result["development_gate"]["bace1_method_amendment_authorized"] is not False:
        raise ValueError("Stage 19h authorized a protected-panel amendment")

    protected_paths = [
        str(descriptor["path"]).lower()
        for descriptor in config["inputs"].values()
    ]
    for spec in source_config["targets"].values():
        protected_paths.extend(
            str(descriptor["path"]).lower() for descriptor in spec["inputs"].values()
        )
    forbidden = ("fresh_validation", "locked_test", "bace1_docking")
    if any(marker in path for marker in forbidden for path in protected_paths):
        raise ValueError("Stage 19h input path crosses its data boundary")

    return {
        "schema_version": "1.0",
        "status": "stage19h_auxiliary_coverage_qubo_audit_ok",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "result": {
            "path": result_path.relative_to(root).as_posix(),
            "sha256": file_sha256(result_path),
        },
        "auditor": {
            "path": Path(__file__).resolve().relative_to(root).as_posix(),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "coverage": {
            "target_count": target_count,
            "inner_candidate_rows_reselected": len(inner_rows),
            "outer_candidate_rows_checked": len(outer_rows),
            "nested_outer_choices_reselected": nested_choices,
            "selected_outer_train_or_holdout_metric_sets_recomputed": (
                selected_outer_metric_sets
            ),
            "comparison_rows_checked": len(method_rows),
            "full_train_qubo_states_certified": sum(
                value["states_certified"] for value in full_certificates.values()
            ),
        },
        "full_train_qubo_certificates": full_certificates,
        "checks": {
            "all_input_and_output_hashes_verified": True,
            "all_nested_parameters_reselected": True,
            "selected_outer_subsets_and_metrics_recomputed": True,
            "baseline_rows_match_frozen_stage19f_source": True,
            "outer_scaffolds_fold_disjoint": True,
            "inner_scaffolds_fold_disjoint": True,
            "full_train_terms_and_coefficients_recomputed": True,
            "cardinality_active_and_decoy_constraints_verified": True,
            "all_full_train_triple_energies_certified": True,
            "all_paired_deltas_recomputed": True,
            "failed_gate_reproduced": True,
            "bace1_method_amendment_authorized": False,
            "new_docking_jobs": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "bace1_docking_rows_read": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.config, args.root)
    output = rooted(args.root.resolve(), args.output.as_posix())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
