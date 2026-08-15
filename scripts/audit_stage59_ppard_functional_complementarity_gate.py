"""Independently audit the Stage59 PPARD functional-complementarity gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
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
        raise ValueError(f"Stage59 output identity differs: {path}")
    return path


def as_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"unexpected Stage59 CSV boolean: {value}")


def assert_close(observed: Any, expected: float, name: str) -> None:
    if abs(float(observed) - expected) > 1e-12:
        raise ValueError(f"Stage59 metric differs: {name}")


def run(result_path: Path, root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    result_path = result_path.resolve()
    result = read_json(result_path)
    if result.get("status") != "stage59_ppard_functional_complementarity_gate_complete":
        raise ValueError("Stage59 source result did not complete")
    config_path = checked(root, result["config"])
    runner_path = checked(root, result["implementation"])
    config = read_json(config_path)
    auditor = checked(root, dict(config["implementation"])["independent_auditor"])
    if auditor.resolve() != Path(__file__).resolve():
        raise ValueError("Stage59 auditor identity differs")
    if runner_path.resolve() != (root / config["implementation"]["runner"]["path"]).resolve():
        raise ValueError("Stage59 runner identity differs")
    matrix_audit = checked(root, result["matrix_audit"])
    if read_json(matrix_audit).get("status") != (
        "independent_stage58b_ppard_pilot96_unidock_matrix_audit_ok"
    ):
        raise ValueError("Stage59 source matrix audit differs")
    paths = {key: checked(root, value) for key, value in result["outputs"].items()}
    receptors = read_csv(paths["receptor_metrics_csv"])
    pairs = read_csv(paths["pair_metrics_csv"])
    folds = read_csv(paths["fold_gate_csv"])
    receptor_ids = [row["receptor_id"] for row in receptors]
    if len(receptors) != 29 or len(set(receptor_ids)) != 29:
        raise ValueError("Stage59 receptor ledger differs")
    dominant_rows = [row for row in receptors if as_bool(row["is_dominant_single"])]
    if len(dominant_rows) != 1:
        raise ValueError("Stage59 dominant receptor count differs")
    expected_pairs = {
        frozenset((left, right)) for left, right in itertools.combinations(receptor_ids, 2)
    }
    observed_pairs = {
        frozenset((row["left_receptor"], row["right_receptor"])) for row in pairs
    }
    if len(pairs) != 406 or observed_pairs != expected_pairs:
        raise ValueError("Stage59 pair ledger differs")
    if len(folds) != 4 or {int(row["outer_fold"]) for row in folds} != set(range(4)):
        raise ValueError("Stage59 fold ledger differs")
    if sum(int(row["train_ligand_count"]) for row in folds) != 288 or sum(
        int(row["holdout_ligand_count"]) for row in folds
    ) != 96:
        raise ValueError("Stage59 fold dimensions differ")
    for row in pairs:
        expected_gain = float(row["pair_robust_bedroc_composite"]) - float(
            row["best_member_robust_bedroc"]
        )
        assert_close(row["pair_bedroc_gain_over_best_member"], expected_gain, row["pair"])
        expected_balance = float(row["normalized_active_rescue"]) - float(
            row["normalized_decoy_promotion"]
        )
        assert_close(row["normalized_rescue_minus_promotion"], expected_balance, row["pair"])

    dominant = dominant_rows[0]
    best_pair = max(pairs, key=lambda row: float(row["pair_robust_bedroc_composite"]))
    additions = [row for row in receptors if not as_bool(row["is_dominant_single"])]
    holdout_gains = [float(row["holdout_pair_gain"]) for row in folds]
    recalculated = {
        "dominant_single_robust_bedroc": float(dominant["single_robust_bedroc_composite"]),
        "best_pair_robust_bedroc": float(best_pair["pair_robust_bedroc_composite"]),
        "best_pair_gain_over_best_member": float(best_pair["pair_bedroc_gain_over_best_member"]),
        "positive_pair_bedroc_gain_count": sum(
            float(row["pair_bedroc_gain_over_best_member"]) > 0 for row in pairs
        ),
        "positive_dominant_addition_bedroc_count": sum(
            float(row["bedroc_delta_over_dominant"]) > 0 for row in additions
        ),
        "positive_rescue_balance_dominant_addition_count": sum(
            float(row["normalized_rescue_minus_promotion"]) > 0 for row in additions
        ),
        "median_dominant_addition_rescue_balance": statistics.median(
            float(row["normalized_rescue_minus_promotion"]) for row in additions
        ),
        "mean_holdout_pair_gain": statistics.fmean(holdout_gains),
        "positive_holdout_pair_gain_fold_count": sum(value > 0 for value in holdout_gains),
        "median_pair_rank_spearman": statistics.median(
            float(row["mean_seed_rank_spearman"]) for row in pairs
        ),
    }
    diagnosis = dict(result["diagnosis"])
    if diagnosis["dominant_single_receptor"] != dominant["receptor_id"]:
        raise ValueError("Stage59 dominant receptor identity differs")
    if diagnosis["best_pair"] != best_pair["pair"] or int(diagnosis["pair_count"]) != 406:
        raise ValueError("Stage59 best-pair identity or count differs")
    for key, value in recalculated.items():
        assert_close(diagnosis[key], value, key)

    criteria = dict(config["gate"])
    checks = {
        "minimum_full_best_pair_gain": recalculated["best_pair_gain_over_best_member"]
        >= float(criteria["minimum_full_best_pair_gain"]),
        "minimum_positive_holdout_pair_gain_folds": recalculated[
            "positive_holdout_pair_gain_fold_count"
        ] >= int(criteria["minimum_positive_holdout_pair_gain_folds"]),
        "minimum_mean_holdout_pair_gain": recalculated["mean_holdout_pair_gain"]
        >= float(criteria["minimum_mean_holdout_pair_gain"]),
        "minimum_positive_rescue_balance_additions": recalculated[
            "positive_rescue_balance_dominant_addition_count"
        ] >= int(criteria["minimum_positive_rescue_balance_additions"]),
        "maximum_median_pair_rank_redundancy": recalculated["median_pair_rank_spearman"]
        <= float(criteria["maximum_median_pair_rank_redundancy"]),
    }
    if checks != result["gate"]["observed_checks"] or checks != result["decision"]["gate_checks"]:
        raise ValueError("Stage59 gate checks differ")
    gate_pass = all(checks.values())
    decision = dict(result["decision"])
    if bool(decision["functional_complementarity_gate_passed"]) != gate_pass:
        raise ValueError("Stage59 overall gate differs")
    if bool(decision["transferred_qubo_objective_freeze_authorized"]) != gate_pass or bool(
        decision["remaining_development_train_docking_authorized"]
    ) != gate_pass:
        raise ValueError("Stage59 downstream authorization differs")
    if any(
        decision[key]
        for key in (
            "same_pilot_objective_or_threshold_retuning_authorized",
            "fresh_validation_authorized",
            "locked_test_authorized",
            "quantum_hardware_authorized",
        )
    ):
        raise ValueError("Stage59 protected authorization boundary differs")
    expected_boundary = {
        "pilot_train_rows_read": 96,
        "remaining_development_train_rows_read": 0,
        "fresh_validation_rows_read": 0,
        "locked_test_rows_read": 0,
        "new_docking_jobs": 0,
        "quantum_hardware_jobs": 0,
    }
    if result["data_boundary"] != expected_boundary:
        raise ValueError("Stage59 data boundary differs")

    audit = {
        "schema_version": "1.0",
        "status": "stage59_ppard_functional_complementarity_independent_audit_ok",
        "source_result": descriptor(root, result_path),
        "config": descriptor(root, config_path),
        "runner": descriptor(root, runner_path),
        "ledger_counts": {"receptor_rows": 29, "pair_rows": 406, "fold_rows": 4},
        "diagnosis_metrics_exact": True,
        "preregistered_gate_checks_exact": True,
        "functional_complementarity_gate_passed": gate_pass,
        "gate_checks": checks,
        "decision": {
            "transferred_qubo_objective_freeze_authorized": gate_pass,
            "remaining_development_train_docking_authorized": gate_pass,
            "fresh_validation_authorized": False,
            "quantum_hardware_authorized": False,
        },
        "data_boundary": expected_boundary,
        "interpretation_boundary": result["interpretation_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="ascii")
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
