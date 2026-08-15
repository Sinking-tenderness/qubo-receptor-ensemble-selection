"""Independently audit the frozen Stage 19d PPARG Train-668 outputs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fit_enopt_xgboost_baseline import metrics_for_probabilities
from scripts.normalized_receptor_qubo import coefficient_energy
from scripts.prepare_receptor import file_sha256
from scripts.screen_stage10_mk14_expanded16_qubo_greedy import (
    fixed_cardinality_exact,
    fixed_cardinality_greedy,
)


DEFAULT_CONFIG = Path(
    "configs/stage19d_pparg_train668_frozen_comparator_analysis.json"
)
DEFAULT_OUTPUT = Path(
    "data/stage19d_pparg_train668_frozen_comparator_audit.json"
)
MATRIX_IDS = ("primary", "sensitivity", "seed0", "seed1", "seed2")
METRIC_FIELDS = (
    "roc_auc",
    "pr_auc_average_precision",
    "bedroc_alpha_20",
    "EF1%",
    "EF5%",
    "EF10%",
    "top10_active_count",
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
    if not path.is_file():
        raise ValueError(f"described file is missing: {path}")
    actual_sha = file_sha256(path)
    if actual_sha != str(descriptor["sha256"]).upper():
        raise ValueError(f"described file hash differs: {path}")
    if "size_bytes" in descriptor and path.stat().st_size != int(
        descriptor["size_bytes"]
    ):
        raise ValueError(f"described file size differs: {path}")
    return path


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-11):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = rooted(root, config_path.as_posix())
    config = read_json(config_path)

    if config["target_id"] != "PPARG":
        raise ValueError("Stage 19d target must be PPARG")
    if bool(config["evidence_timing"]["weight_retuning_permitted"]):
        raise ValueError("Stage 19d must preserve the frozen QUBO weights")
    if float(config["qubo_model"]["bedroc_alpha"]) != 20.0:
        raise ValueError("Stage 19d must use BEDROC alpha=20")

    verify_descriptor(root, config["implementation"])
    for descriptor in config["inputs"].values():
        verify_descriptor(root, descriptor)

    outputs = config["outputs"]
    result_path = rooted(root, outputs["result_json"])
    result = read_json(result_path)
    if result["status"] != "stage19d_pparg_train_only_comparison_complete":
        raise ValueError("Stage 19d result is not complete")
    if result["config"]["sha256"] != file_sha256(config_path):
        raise ValueError("Stage 19d result does not identify the current config")
    if result["data_boundary"] != {
        "validation_rows_read": 0,
        "test_rows_read": 0,
    }:
        raise ValueError("Stage 19d crossed the train-only data boundary")

    verified_outputs = {
        key: verify_descriptor(root, descriptor)
        for key, descriptor in result["outputs"].items()
    }
    frozen = read_json(verified_outputs["frozen_methods_json"])
    if frozen["data_boundary"] != {
        "validation_rows_read": 0,
        "test_rows_read": 0,
    }:
        raise ValueError("frozen method artifacts crossed the data boundary")
    for evidence in frozen["models"].values():
        verify_descriptor(root, evidence["model"])

    fold_rows = read_csv(verified_outputs["fold_assignments_csv"])
    expected = config["expected"]
    ligand_count = int(expected["ligand_count"])
    fold_count = int(config["cross_validation"]["fold_count"])
    if len(fold_rows) != ligand_count:
        raise ValueError("fold assignment row count differs")
    if len({row["ligand_id"] for row in fold_rows}) != ligand_count:
        raise ValueError("fold assignments contain duplicate ligands")
    if Counter(row["label"] for row in fold_rows) != Counter(
        expected["label_counts"]
    ):
        raise ValueError("fold assignment labels differ")
    if {int(row["outer_fold"]) for row in fold_rows} != set(range(fold_count)):
        raise ValueError("outer fold IDs differ")

    group_folds: dict[str, set[int]] = defaultdict(set)
    scaffold_folds: dict[str, set[int]] = defaultdict(set)
    ligand_evidence: dict[str, tuple[str, int]] = {}
    for row in fold_rows:
        fold = int(row["outer_fold"])
        group_folds[row["split_group_id"]].add(fold)
        scaffold_folds[row["scaffold_smiles"]].add(fold)
        ligand_evidence[row["ligand_id"]] = (row["label"], fold)
    if any(len(folds) != 1 for folds in group_folds.values()):
        raise ValueError("a split group crosses outer folds")
    if any(len(folds) != 1 for folds in scaffold_folds.values()):
        raise ValueError("a scaffold crosses outer folds")

    robust_methods = list(result["method_ranking_by_oof_robust_key"])
    primary_only_methods = sorted(result["primary_only_geometric_metrics"])
    if len(robust_methods) != int(result["method_count"]):
        raise ValueError("robust method count differs")
    if len(primary_only_methods) != int(
        result["primary_only_geometric_method_count"]
    ):
        raise ValueError("primary-only method count differs")

    expected_prediction_keys = {
        (method, matrix_id, ligand_id)
        for method, matrix_id, ligand_id in itertools.product(
            robust_methods, MATRIX_IDS, ligand_evidence
        )
    }
    expected_prediction_keys.update(
        (method, "primary", ligand_id)
        for method, ligand_id in itertools.product(
            primary_only_methods, ligand_evidence
        )
    )
    prediction_rows = read_csv(verified_outputs["oof_predictions_csv"])
    prediction_keys: set[tuple[str, str, str]] = set()
    predictions: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in prediction_rows:
        key = (row["method"], row["matrix"], row["ligand_id"])
        if key in prediction_keys:
            raise ValueError(f"duplicate OOF prediction: {key}")
        prediction_keys.add(key)
        label, fold = ligand_evidence[row["ligand_id"]]
        if row["label"] != label or int(row["outer_fold"]) != fold:
            raise ValueError(f"OOF prediction metadata differs: {key}")
        score = float(row["ranking_score"])
        if not math.isfinite(score):
            raise ValueError(f"non-finite OOF score: {key}")
        predictions[(row["method"], row["matrix"])][row["ligand_id"]] = score
    if prediction_keys != expected_prediction_keys:
        raise ValueError("OOF prediction coverage differs")

    metric_rows = read_csv(verified_outputs["oof_metrics_csv"])
    metric_by_key = {(row["method"], row["matrix"]): row for row in metric_rows}
    if len(metric_by_key) != len(metric_rows):
        raise ValueError("OOF metrics contain duplicate method/matrix rows")
    if set(metric_by_key) != set(predictions):
        raise ValueError("OOF metric coverage differs from predictions")
    label_matrix = {
        ligand_id: {"label": label}
        for ligand_id, (label, _) in ligand_evidence.items()
    }
    for key, values in predictions.items():
        recomputed = metrics_for_probabilities(label_matrix, values)
        recorded = metric_by_key[key]
        if int(recorded["ligand_count"]) != ligand_count:
            raise ValueError(f"OOF metric ligand count differs: {key}")
        for field in METRIC_FIELDS:
            assert_close(
                float(recorded[field]),
                float(recomputed[field]),
                f"{key} {field}",
            )

    selection_rows = read_csv(verified_outputs["fold_method_selections_csv"])
    expected_selection_keys = {
        (fold, method)
        for fold, method in itertools.product(
            range(fold_count), robust_methods + primary_only_methods
        )
    }
    selection_keys = {
        (int(row["outer_fold"]), row["method"]) for row in selection_rows
    }
    if len(selection_rows) != len(selection_keys):
        raise ValueError("fold selections contain duplicate rows")
    if selection_keys != expected_selection_keys:
        raise ValueError("fold selection coverage differs")

    full_selection_rows = read_csv(verified_outputs["full_train_selections_csv"])
    if len(full_selection_rows) != len(robust_methods) + len(primary_only_methods):
        raise ValueError("full-train selection row count differs")
    if len({row["method"] for row in full_selection_rows}) != len(
        full_selection_rows
    ):
        raise ValueError("full-train selections contain duplicate methods")

    full_qubo = read_json(verified_outputs["full_train_qubo_json"])
    coefficients = full_qubo["coefficients"]
    receptor_ids = sorted(coefficients["linear"])
    target_size = int(coefficients["target_size"])
    exact_subset, exact_energy = fixed_cardinality_exact(
        coefficients, receptor_ids, target_size
    )
    greedy_subset, greedy_energy, _ = fixed_cardinality_greedy(
        coefficients, receptor_ids, target_size
    )
    if exact_subset != tuple(sorted(full_qubo["subsets"]["qubo_exact_top3"])):
        raise ValueError("stored exact QUBO subset is not the global optimum")
    if greedy_subset != tuple(sorted(full_qubo["subsets"]["qubo_greedy_top3"])):
        raise ValueError("stored greedy QUBO subset differs")
    assert_close(exact_energy, float(full_qubo["exact_energy"]), "exact energy")
    assert_close(greedy_energy, float(full_qubo["qubo_greedy_energy"]), "greedy energy")
    assert_close(
        greedy_energy - exact_energy,
        float(full_qubo["qubo_greedy_regret"]),
        "greedy regret",
    )
    assert_close(
        coefficient_energy(exact_subset, coefficients),
        exact_energy,
        "exact subset energy",
    )

    fold_qubo_differences = sum(
        set(row["subset"].split("+"))
        != set(
            next(
                candidate["subset"].split("+")
                for candidate in selection_rows
                if candidate["outer_fold"] == row["outer_fold"]
                and candidate["method"] == "qubo_greedy_top3"
            )
        )
        for row in selection_rows
        if row["method"] == "qubo_exact_top3"
    )
    direct_full = set(result["full_train_subsets"]["direct_greedy_top3"])
    qubo_full = set(result["full_train_subsets"]["qubo_exact_top3"])

    return {
        "schema_version": "1.0",
        "status": "stage19d_pparg_train668_frozen_comparator_audit_ok",
        "target_id": "PPARG",
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
            "ligand_count": ligand_count,
            "split_group_count": len(group_folds),
            "scaffold_count": len(scaffold_folds),
            "outer_fold_count": fold_count,
            "robust_method_count": len(robust_methods),
            "primary_only_method_count": len(primary_only_methods),
            "oof_prediction_count": len(prediction_rows),
            "oof_metric_count": len(metric_rows),
            "fold_selection_count": len(selection_rows),
            "full_selection_count": len(full_selection_rows),
            "model_artifact_count": len(frozen["models"]),
        },
        "checks": {
            "all_input_hashes_verified": True,
            "all_output_hashes_verified": True,
            "all_model_hashes_verified": True,
            "scaffold_groups_are_fold_disjoint": True,
            "oof_coverage_is_exact": True,
            "oof_metrics_recomputed_exactly": True,
            "full_qubo_exact_solution_reenumerated": True,
            "validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "qubo_diagnostics": {
            "full_train_exact_equals_qubo_greedy": exact_subset == greedy_subset,
            "full_train_exact_differs_from_direct_greedy": qubo_full != direct_full,
            "folds_where_exact_differs_from_qubo_greedy": fold_qubo_differences,
            "fold_count": fold_count,
            "noncardinality_quadratic_term_count": int(
                result["full_train_qubo_diagnostics"][
                    "noncardinality_quadratic"
                ]["term_count"]
            ),
        },
        "interpretation": (
            "The artifacts are internally consistent and train-only. The frozen "
            "QUBO defines a genuine quadratic optimization problem, but its OOF "
            "ranking performance does not outperform direct greedy selection or "
            "the strongest transferred literature-family baseline."
        ),
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
