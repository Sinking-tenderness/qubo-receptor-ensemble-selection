"""Independently audit Stage 53 PPARA QUBO-transfer output ledgers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def checked(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage 53 output identity differs: {path}")
    return path


def run(result_path: Path, root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    result_path = result_path.resolve()
    result = read_json(result_path)
    if result["status"] != "stage53_ppara_large_pool_qubo_transfer_complete":
        raise ValueError("Stage 53 source result did not complete")
    config_path = checked(root, result["config"])
    implementation_path = checked(root, result["implementation"])
    config = read_json(config_path)
    if implementation_path.resolve() != (
        root / config["implementation"]["path"]
    ).resolve():
        raise ValueError("Stage 53 implementation path differs")
    paths = {
        key: checked(root, value) for key, value in result["outputs"].items()
    }
    folds = read_csv(paths["fold_assignments_csv"])
    selections = read_csv(paths["selection_metrics_csv"])
    landscape = read_csv(paths["fixed_k_landscape_csv"])
    if len(folds) != 374 or Counter(row["label"] for row in folds) != Counter(
        {"active": 187, "decoy": 187}
    ):
        raise ValueError("Stage 53 fold ledger differs")
    if len(selections) != 60 or Counter(row["scope"] for row in selections) != Counter(
        {"outer_holdout": 48, "full_data": 12}
    ):
        raise ValueError("Stage 53 selection ledger differs")
    if len(landscape) != 192:
        raise ValueError("Stage 53 fixed-k ledger differs")
    if {int(row["subset_size"]) for row in landscape} != set(range(1, 7)):
        raise ValueError("Stage 53 fixed-k range differs")
    expected_fixed_methods = {
        "rank_pair_qubo_exact",
        "rank_pair_strong_classical",
        "rank_pair_direct_greedy",
        "coverage_qubo_exact",
        "coverage_direct_greedy",
        "bedroc_linear_topk",
        "bedroc_nested_greedy",
        "bedroc_random_search",
    }
    if {row["method"] for row in landscape} != expected_fixed_methods:
        raise ValueError("Stage 53 fixed-k methods differ")
    full = {
        row["method"]: row for row in selections if row["scope"] == "full_data"
    }
    if set(full) != set(result["full_data_methods"]):
        raise ValueError("Stage 53 full-data methods differ")
    for method, row in full.items():
        source = result["full_data_methods"][method]
        if row["selected_subset"] != source["selected_subset"] or int(
            row["subset_size"]
        ) != int(source["subset_size"]):
            raise ValueError(f"Stage 53 full-data selection differs: {method}")
        if abs(float(row["evaluation_robust_bedroc"]) - float(source["robust_bedroc"])) > 1e-12:
            raise ValueError(f"Stage 53 full-data metric differs: {method}")

    def fold_values(method: str) -> list[float]:
        return [
            float(row["evaluation_robust_bedroc"])
            for row in selections
            if row["scope"] == "outer_holdout" and row["method"] == method
        ]

    rank = fold_values("rank_pair_qubo_exact")
    single = fold_values("best_single_receptor")
    linear = fold_values("bedroc_linear_topk")
    nested = fold_values("bedroc_nested_greedy")
    observed = result["decision"]
    recalculated = {
        "mean_holdout_rank_pair_qubo": statistics.fmean(rank),
        "mean_holdout_rank_pair_minus_single": statistics.fmean(
            left - right for left, right in zip(rank, single, strict=True)
        ),
        "mean_holdout_rank_pair_minus_linear": statistics.fmean(
            left - right for left, right in zip(rank, linear, strict=True)
        ),
        "mean_holdout_rank_pair_minus_nested_greedy": statistics.fmean(
            left - right for left, right in zip(rank, nested, strict=True)
        ),
    }
    for key, value in recalculated.items():
        if abs(float(observed[key]) - value) > 1e-12:
            raise ValueError(f"Stage 53 decision metric differs: {key}")
    exact_rows = {
        (row["outer_fold"], row["subset_size"]): row
        for row in landscape
        if row["method"] == "rank_pair_qubo_exact"
    }
    strong_rows = {
        (row["outer_fold"], row["subset_size"]): row
        for row in landscape
        if row["method"] == "rank_pair_strong_classical"
    }
    exact_strong_gap_count = sum(
        float(row["train_objective"])
        - float(strong_rows[key]["train_objective"])
        > 1e-12
        for key, row in exact_rows.items()
    )
    if exact_strong_gap_count != int(
        observed["positive_rank_pair_objective_gap_cells_over_strong_classical"]
    ):
        raise ValueError("Stage 53 exact-versus-strong gap count differs")

    fixed_k_rank_means = {
        str(size): statistics.fmean(
            float(row["evaluation_robust_bedroc"])
            for row in landscape
            if row["method"] == "rank_pair_qubo_exact"
            and int(row["subset_size"]) == size
        )
        for size in range(1, 7)
    }
    audit = {
        "schema_version": "1.0",
        "status": "stage53_ppara_large_pool_qubo_transfer_independent_audit_ok",
        "source_result": descriptor(root, result_path),
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, implementation_path),
        "ledger_counts": {
            "fold_assignment_rows": len(folds),
            "selection_rows": len(selections),
            "fixed_k_rows": len(landscape),
        },
        "fixed_k_rank_pair_mean_holdout_bedroc": fixed_k_rank_means,
        "decision_metrics_exact": True,
        "solver_gap_count_exact": True,
        "decision": {
            "frozen_qubo_application_transfer_supported": observed[
                "frozen_qubo_application_transfer_supported"
            ],
            "solver_novelty_detected": observed["solver_novelty_detected"],
            "fresh_validation_authorized": False,
            "same_data_weight_retuning_authorized": False,
            "quantum_hardware_authorized": False,
        },
        "data_boundary": result["data_boundary"],
        "interpretation_boundary": result["interpretation_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.result, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
