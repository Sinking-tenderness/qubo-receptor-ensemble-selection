"""Independently audit the Stage 19e nested cross-target diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_stage12a_mk14_qubo_objective_adequacy import (
    exact_all_cardinalities,
    subset_bedroc,
)
from scripts.diagnose_stage19e_cross_target_qubo_v2 import load_target
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage05_mk14_uncertainty_qubo_gate import make_context


DEFAULT_CONFIG = Path(
    "configs/stage19e_cross_target_qubo_v2_nested_diagnostic.json"
)
DEFAULT_OUTPUT = Path(
    "data/stage19e_cross_target_qubo_v2_nested_diagnostic_audit.json"
)
MATRIX_IDS = ("primary", "sensitivity", "seed0", "seed1", "seed2")
SEED_IDS = ("seed0", "seed1", "seed2")
METHODS = (
    "additive_nested",
    "quadratic_nested",
    "v1_qubo_exact",
    "direct_greedy",
    "direct_exact",
    "composite_exact",
    "holdout_oracle",
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
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def scalar_subset_metrics(
    context: dict[str, Any], subset: tuple[str, ...], split: str
) -> dict[str, float]:
    values = {
        matrix_id: subset_bedroc(
            list(context["matrices"][matrix_id][split]), subset
        )
        for matrix_id in MATRIX_IDS
    }
    seeds = [values[seed_id] for seed_id in SEED_IDS]
    values["mean_seed"] = statistics.fmean(seeds)
    values["worst_seed"] = min(seeds)
    values["robust_composite"] = statistics.fmean(
        (values["primary"], values["mean_seed"], values["worst_seed"])
    )
    return values


def alpha_key(rows: list[dict[str, str]], alpha: float) -> tuple[float, ...]:
    selected = [row for row in rows if float(row["alpha"]) == alpha]
    return (
        -statistics.fmean(
            float(row["validation_robust_composite"]) for row in selected
        ),
        -min(float(row["validation_robust_composite"]) for row in selected),
        -statistics.fmean(
            float(row["validation_rank_spearman"]) for row in selected
        ),
        alpha,
    )


def selected_alpha(rows: list[dict[str, str]]) -> float:
    alphas = sorted({float(row["alpha"]) for row in rows})
    return min(alphas, key=lambda alpha: alpha_key(rows, alpha))


def paired_deltas(
    rows: list[dict[str, str]], left: str, right: str
) -> dict[str, Any]:
    indexed = {
        (row["target_id"], int(row["outer_fold"]), row["method"]): row
        for row in rows
    }
    per_target: dict[str, dict[str, Any]] = {}
    all_values: list[float] = []
    for target_id in sorted({row["target_id"] for row in rows}):
        folds = sorted(
            int(row["outer_fold"])
            for row in rows
            if row["target_id"] == target_id and row["method"] == left
        )
        values = [
            float(indexed[(target_id, fold, left)]["holdout_robust_composite"])
            - float(indexed[(target_id, fold, right)]["holdout_robust_composite"])
            for fold in folds
        ]
        all_values.extend(values)
        per_target[target_id] = {
            "mean_delta": statistics.fmean(values),
            "positive_fold_count": sum(value > 0.0 for value in values),
        }
    return {
        "mean_delta": statistics.fmean(all_values),
        "positive_fold_count": sum(value > 0.0 for value in all_values),
        "fold_count": len(all_values),
        "per_target": per_target,
    }


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = rooted(root, config_path.as_posix())
    config = read_json(config_path)
    verify_descriptor(root, config["implementation"])
    for descriptor in config["prior_records"].values():
        verify_descriptor(root, descriptor)

    result_path = rooted(root, config["outputs"]["result_json"])
    result = read_json(result_path)
    if result["status"] != "stage19e_quadratic_v2_not_supported_do_not_amend_bace1":
        raise ValueError("unexpected Stage 19e result status")
    if result["config"]["sha256"] != file_sha256(config_path):
        raise ValueError("result identifies another config")
    if result["data_boundary"] != {
        "train_rows_read_by_target": {"MK14": 696, "PPARG": 668},
        "fresh_validation_rows_read": 0,
        "test_rows_read": 0,
        "bace1_docking_rows_read": 0,
    }:
        raise ValueError("Stage 19e data boundary differs")

    output_paths = {
        key: verify_descriptor(root, descriptor)
        for key, descriptor in result["outputs"].items()
    }
    outer_assignments = read_csv(output_paths["fold_assignments_csv"])
    inner_assignments = read_csv(output_paths["inner_fold_assignments_csv"])
    inner_trials = read_csv(output_paths["inner_trials_csv"])
    outer_alpha_trials = read_csv(output_paths["outer_alpha_trials_csv"])
    method_rows = read_csv(output_paths["outer_method_results_csv"])
    algorithm = read_json(output_paths["algorithm_record_json"])

    diagnostic = config["diagnostic"]
    outer_count = int(diagnostic["outer_fold_count"])
    inner_count = int(diagnostic["inner_fold_count"])
    alphas = [float(value) for value in diagnostic["ridge_alphas"]]
    expected_ligands = sum(
        int(spec["expected"]["ligand_count"])
        for spec in config["targets"].values()
    )
    if len(outer_assignments) != expected_ligands:
        raise ValueError("outer assignment count differs")
    if len(inner_assignments) != expected_ligands * (outer_count - 1):
        raise ValueError("inner assignment count differs")
    if len(inner_trials) != (
        len(config["targets"])
        * outer_count
        * inner_count
        * 2
        * len(alphas)
    ):
        raise ValueError("inner trial count differs")
    if len(outer_alpha_trials) != (
        len(config["targets"]) * outer_count * 2 * len(alphas)
    ):
        raise ValueError("outer alpha trial count differs")
    if len(method_rows) != len(config["targets"]) * outer_count * len(METHODS):
        raise ValueError("outer method count differs")
    if len(
        {
            (row["target_id"], row["outer_fold"], row["method"])
            for row in method_rows
        }
    ) != len(method_rows):
        raise ValueError("outer methods contain duplicates")

    recorded_outer = {
        (row["target_id"], row["ligand_id"]): row
        for row in outer_assignments
    }
    recorded_inner = {
        (row["target_id"], int(row["outer_fold"]), row["ligand_id"]): row
        for row in inner_assignments
    }
    group_outer_folds: dict[tuple[str, str], set[int]] = defaultdict(set)
    scaffold_outer_folds: dict[tuple[str, str], set[int]] = defaultdict(set)
    group_inner_folds: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    scaffold_inner_folds: dict[tuple[str, int, str], set[int]] = defaultdict(set)

    scalar_metric_count = 0
    for target_id, spec in config["targets"].items():
        ligands, receptor_ids, matrices = load_target(root, target_id, spec)
        assignments = make_frozen_group_folds(
            ligands, outer_count, int(diagnostic["fold_seed"])
        )
        for row in ligands:
            recorded = recorded_outer[(target_id, row["ligand_id"])]
            if int(recorded["outer_fold"]) != assignments[row["ligand_id"]]:
                raise ValueError("outer assignment reproduction differs")
            group_outer_folds[(target_id, row["split_group_id"])].add(
                int(recorded["outer_fold"])
            )
            scaffold_outer_folds[(target_id, row["scaffold_smiles"])].add(
                int(recorded["outer_fold"])
            )

        all_ids = {row["ligand_id"] for row in ligands}
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
            train_rows = [row for row in ligands if row["ligand_id"] in train]
            inner = make_frozen_group_folds(
                train_rows,
                inner_count,
                int(diagnostic["inner_fold_seed"]) + outer_fold,
            )
            for row in train_rows:
                recorded = recorded_inner[(target_id, outer_fold, row["ligand_id"])]
                if int(recorded["inner_fold"]) != inner[row["ligand_id"]]:
                    raise ValueError("inner assignment reproduction differs")
                group_inner_folds[(target_id, outer_fold, row["split_group_id"])].add(
                    int(recorded["inner_fold"])
                )
                scaffold_inner_folds[
                    (target_id, outer_fold, row["scaffold_smiles"])
                ].add(int(recorded["inner_fold"]))

            context = make_context(train, holdout, matrices, receptor_ids, model_spec)
            selected_rows = [
                row
                for row in method_rows
                if row["target_id"] == target_id
                and int(row["outer_fold"]) == outer_fold
            ]
            for row in selected_rows:
                subset = tuple(sorted(row["selected_subset"].split("+")))
                for split, prefix in (("train", "train"), ("validation", "holdout")):
                    metrics = scalar_subset_metrics(context, subset, split)
                    for field in (
                        "primary",
                        "sensitivity",
                        "mean_seed",
                        "worst_seed",
                        "robust_composite",
                    ):
                        assert_close(
                            float(row[f"{prefix}_{field}"]),
                            float(metrics[field]),
                            f"{target_id}/{outer_fold}/{row['method']}/{prefix}/{field}",
                        )
                    scalar_metric_count += 1

            for family in ("additive", "quadratic"):
                trials = [
                    row
                    for row in inner_trials
                    if row["target_id"] == target_id
                    and int(row["outer_fold"]) == outer_fold
                    and row["model_family"] == family
                ]
                chosen = selected_alpha(trials)
                method = next(
                    row
                    for row in selected_rows
                    if row["method"] == f"{family}_nested"
                )
                assert_close(
                    chosen,
                    float(method["selected_alpha"]),
                    f"{target_id}/{outer_fold}/{family}/selected alpha",
                )

    if any(len(folds) != 1 for folds in group_outer_folds.values()):
        raise ValueError("an outer split group crosses folds")
    if any(len(folds) != 1 for folds in scaffold_outer_folds.values()):
        raise ValueError("an outer scaffold crosses folds")
    if any(len(folds) != 1 for folds in group_inner_folds.values()):
        raise ValueError("an inner split group crosses folds")
    if any(len(folds) != 1 for folds in scaffold_inner_folds.values()):
        raise ValueError("an inner scaffold crosses folds")

    comparisons = {
        "quadratic_vs_direct_greedy": paired_deltas(
            method_rows, "quadratic_nested", "direct_greedy"
        ),
        "quadratic_vs_additive": paired_deltas(
            method_rows, "quadratic_nested", "additive_nested"
        ),
        "quadratic_vs_v1": paired_deltas(
            method_rows, "quadratic_nested", "v1_qubo_exact"
        ),
    }
    for key, recomputed in comparisons.items():
        recorded = result["paired_comparisons"][key]
        assert_close(
            recomputed["mean_delta"], recorded["mean_delta"], f"{key} mean"
        )
        if recomputed["positive_fold_count"] != recorded["positive_fold_count"]:
            raise ValueError(f"{key} win count differs")
        for target_id, target_values in recomputed["per_target"].items():
            assert_close(
                target_values["mean_delta"],
                recorded["per_target"][target_id]["mean_delta"],
                f"{key}/{target_id} mean",
            )

    gate_spec = config["development_support_gate"]
    gate_passed = all(
        all(
            float(value["mean_delta"]) > float(gate_spec["minimum_target_mean_delta"])
            for value in comparison["per_target"].values()
        )
        and int(comparison["positive_fold_count"])
        >= int(gate_spec["minimum_positive_folds_of_eight"])
        for comparison in comparisons.values()
    )
    if gate_passed or result["gate"]["passed"] or result["gate"][
        "bace1_v2_amendment_authorized"
    ]:
        raise ValueError("failed Stage 19e gate was not preserved")

    if algorithm["status"] != "development_gate_failed_not_authorized_for_bace1":
        raise ValueError("algorithm authorization differs")
    if algorithm["data_boundary"]["bace1_docking_rows_read"] != 0:
        raise ValueError("algorithm record read BACE1 docking rows")
    exact_qubo_count = 0
    for target_id, evidence in algorithm["target_development_fits"].items():
        quadratic = evidence["models"]["quadratic"]
        qubo = quadratic["explicit_qubo"]
        subset, energy = exact_all_cardinalities(sorted(qubo["linear"]), qubo)
        if list(subset) != quadratic["exact_all_cardinalities_subset"]:
            raise ValueError(f"{target_id} explicit QUBO optimum differs")
        if len(subset) != int(config["diagnostic"]["target_size"]):
            raise ValueError(f"{target_id} explicit QUBO cardinality differs")
        assert_close(
            energy,
            float(quadratic["exact_all_cardinalities_energy"]),
            f"{target_id} explicit QUBO energy",
        )
        exact_qubo_count += 1

    return {
        "schema_version": "1.0",
        "status": "stage19e_cross_target_qubo_v2_nested_diagnostic_audit_ok",
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
            "target_count": len(config["targets"]),
            "outer_assignment_count": len(outer_assignments),
            "inner_assignment_count": len(inner_assignments),
            "inner_trial_count": len(inner_trials),
            "outer_alpha_trial_count": len(outer_alpha_trials),
            "outer_method_count": len(method_rows),
            "scalar_train_or_holdout_metric_sets_recomputed": scalar_metric_count,
            "explicit_qubo_optima_reenumerated": exact_qubo_count,
        },
        "checks": {
            "all_input_and_output_hashes_verified": True,
            "outer_scaffolds_fold_disjoint": True,
            "inner_scaffolds_fold_disjoint": True,
            "all_outer_method_metrics_scalar_recomputed": True,
            "all_nested_alphas_reselected": True,
            "all_paired_deltas_recomputed": True,
            "failed_gate_reproduced": True,
            "bace1_amendment_authorized": False,
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
