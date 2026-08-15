"""Independently audit Stage 54 PPARA failure-diagnosis ledgers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
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
        raise ValueError(f"Stage 54 output identity differs: {path}")
    return path


def as_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"unexpected CSV boolean: {value}")


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("invalid vectors for Spearman correlation")
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right_ranks))
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    return numerator / (left_scale * right_scale)


def assert_close(observed: Any, expected: float, name: str) -> None:
    if abs(float(observed) - expected) > 1e-12:
        raise ValueError(f"Stage 54 metric differs: {name}")


def run(result_path: Path, root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    result_path = result_path.resolve()
    result = read_json(result_path)
    if result["status"] != "stage54_ppara_functional_complementarity_diagnosis_complete":
        raise ValueError("Stage 54 source result did not complete")
    config_path = checked(root, result["config"])
    implementation_path = checked(root, result["implementation"])
    config = read_json(config_path)
    if implementation_path.resolve() != (root / config["implementation"]["path"]).resolve():
        raise ValueError("Stage 54 implementation path differs")
    paths = {key: checked(root, value) for key, value in result["outputs"].items()}
    receptors = read_csv(paths["receptor_diagnostics_csv"])
    pairs = read_csv(paths["pair_diagnostics_csv"])
    folds = read_csv(paths["fold_oracle_diagnostics_csv"])
    future = read_json(paths["future_intake_criteria_json"])

    receptor_ids = [row["receptor_id"] for row in receptors]
    if len(receptors) != 20 or len(set(receptor_ids)) != 20:
        raise ValueError("Stage 54 receptor ledger differs")
    dominant_rows = [row for row in receptors if as_bool(row["is_dominant_single"])]
    if len(dominant_rows) != 1:
        raise ValueError("Stage 54 dominant receptor count differs")
    expected_pairs = {
        frozenset((receptor_ids[left], receptor_ids[right]))
        for left in range(len(receptor_ids))
        for right in range(left + 1, len(receptor_ids))
    }
    observed_pairs = {
        frozenset((row["left_receptor"], row["right_receptor"])) for row in pairs
    }
    if len(pairs) != 190 or observed_pairs != expected_pairs:
        raise ValueError("Stage 54 pair ledger differs")
    if len(folds) != 4 or {int(row["outer_fold"]) for row in folds} != set(range(4)):
        raise ValueError("Stage 54 fold ledger differs")
    if sum(int(row["train_ligand_count"]) for row in folds) != 1122:
        raise ValueError("Stage 54 train fold sizes differ")
    if sum(int(row["holdout_ligand_count"]) for row in folds) != 374:
        raise ValueError("Stage 54 holdout fold sizes differ")

    diagnosis = result["diagnosis"]
    dominant = dominant_rows[0]
    best_pair = max(pairs, key=lambda row: float(row["pair_robust_bedroc"]))
    additions = [row for row in receptors if not as_bool(row["is_dominant_single"])]
    pair_gains = [float(row["pair_bedroc_gain_over_best_member"]) for row in pairs]
    pair_coefficients = [
        float(row["rank_pair_qubo_complement_coefficient"]) for row in pairs
    ]
    rescue_balances = [
        float(row["normalized_rescue_minus_promotion"]) for row in additions
    ]
    holdout_gains = [float(row["holdout_pair_gain"]) for row in folds]
    recalculated = {
        "dominant_single_robust_bedroc": float(dominant["single_robust_bedroc"]),
        "best_pair_robust_bedroc": float(best_pair["pair_robust_bedroc"]),
        "best_pair_gain_over_best_member": float(
            best_pair["pair_bedroc_gain_over_best_member"]
        ),
        "positive_pair_bedroc_gain_count": sum(value > 0 for value in pair_gains),
        "positive_dominant_addition_bedroc_count": sum(
            float(row["bedroc_delta_over_dominant"]) > 0 for row in additions
        ),
        "positive_rescue_balance_dominant_addition_count": sum(
            value > 0 for value in rescue_balances
        ),
        "median_dominant_addition_rescue_balance": statistics.median(rescue_balances),
        "mean_holdout_oracle_pair_gain": statistics.fmean(holdout_gains),
        "positive_holdout_oracle_pair_gain_fold_count": sum(
            value > 0 for value in holdout_gains
        ),
        "median_pair_rank_spearman": statistics.median(
            float(row["mean_seed_rank_spearman"]) for row in pairs
        ),
        "rank_pair_coefficient_vs_bedroc_gain_spearman": spearman(
            pair_coefficients, pair_gains
        ),
    }
    if diagnosis["dominant_single_receptor"] != dominant["receptor_id"]:
        raise ValueError("Stage 54 dominant receptor identity differs")
    if diagnosis["best_pair"] != best_pair["pair"]:
        raise ValueError("Stage 54 best pair identity differs")
    if int(diagnosis["pair_count"]) != len(pairs):
        raise ValueError("Stage 54 pair count differs")
    for key, value in recalculated.items():
        assert_close(diagnosis[key], value, key)

    criteria = config["future_target_intake_criteria"]
    checks = {
        "minimum_full_best_pair_gain": recalculated["best_pair_gain_over_best_member"]
        >= float(criteria["minimum_full_best_pair_gain"]),
        "minimum_positive_holdout_pair_gain_folds": recalculated[
            "positive_holdout_oracle_pair_gain_fold_count"
        ]
        >= int(criteria["minimum_positive_holdout_pair_gain_folds"]),
        "minimum_mean_holdout_pair_gain": recalculated["mean_holdout_oracle_pair_gain"]
        >= float(criteria["minimum_mean_holdout_pair_gain"]),
        "minimum_positive_rescue_balance_additions": recalculated[
            "positive_rescue_balance_dominant_addition_count"
        ]
        >= int(criteria["minimum_positive_rescue_balance_additions"]),
        "maximum_median_pair_rank_redundancy": recalculated[
            "median_pair_rank_spearman"
        ]
        <= float(criteria["maximum_median_pair_rank_redundancy"]),
    }
    if checks != result["decision"]["future_intake_checks_on_ppara"]:
        raise ValueError("Stage 54 prospective intake checks differ")
    if future["criteria"] != criteria or future["ppara_retrospective_check"] != checks:
        raise ValueError("Stage 54 frozen future criteria record differs")
    if bool(future["ppara_pass"]) != all(checks.values()):
        raise ValueError("Stage 54 prospective pass decision differs")
    if any(
        (
            result["decision"]["ppara_same_data_retuning_authorized"],
            result["decision"]["ppara_fresh_validation_authorized"],
            result["decision"]["quantum_hardware_authorized"],
            result["decision"]["ppara_would_pass_future_intake_criteria"],
        )
    ):
        raise ValueError("Stage 54 authorization boundary differs")
    if result["data_boundary"] != {
        "train_rows_read": 374,
        "fresh_validation_rows_read": 0,
        "locked_test_rows_read": 0,
        "new_docking_jobs": 0,
        "quantum_hardware_jobs": 0,
    }:
        raise ValueError("Stage 54 data boundary differs")

    audit = {
        "schema_version": "1.0",
        "status": "stage54_ppara_functional_complementarity_independent_audit_ok",
        "source_result": descriptor(root, result_path),
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, implementation_path),
        "ledger_counts": {
            "receptor_rows": len(receptors),
            "pair_rows": len(pairs),
            "fold_rows": len(folds),
        },
        "diagnosis_metrics_exact": True,
        "prospective_intake_checks_exact": True,
        "failure_mechanism": {
            "single_receptor_dominance_confirmed": diagnosis[
                "single_receptor_dominance_confirmed"
            ],
            "decoy_promotion_failure_confirmed": diagnosis[
                "decoy_promotion_failure_confirmed"
            ],
            "rank_pair_objective_alignment_weak": diagnosis[
                "rank_pair_objective_alignment_weak"
            ],
        },
        "decision": {
            "ppara_would_pass_future_intake_criteria": False,
            "ppara_same_data_retuning_authorized": False,
            "ppara_fresh_validation_authorized": False,
            "quantum_hardware_authorized": False,
            "small_pilot_required_before_full_matrix": True,
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
