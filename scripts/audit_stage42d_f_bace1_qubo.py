"""Independently audit the Stage 42d-f BACE1 QUBO analyses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_result(root: Path, result: dict[str, object], expected_status: str) -> None:
    if result.get("status") != expected_status:
        raise ValueError(f"analysis status differs: {expected_status}")
    for key in ("config", "implementation"):
        descriptor = dict(result[key])
        path = root / str(descriptor["path"])
        if not path.is_file() or sha256(path) != str(descriptor["sha256"]).upper():
            raise ValueError(f"analysis identity differs: {path}")
    boundary = dict(result["data_boundary"])
    if int(boundary["fresh_validation_rows_read"]) != 0 or int(boundary["locked_test_rows_read"]) != 0:
        raise ValueError("Stage 42 analysis crossed a protected boundary")
    if int(boundary["new_docking_jobs"]) != 0 or int(boundary["quantum_hardware_jobs"]) != 0:
        raise ValueError("Stage 42 analysis started an unauthorized job")


def run(root: Path, output: Path) -> dict[str, object]:
    root = root.resolve()
    stage42c = read_json(root / "data/stage42c_bace1_train266_unidock113_production_audit.json")
    if stage42c.get("status") != "independent_stage42c_bace1_train266_unidock_matrix_audit_ok":
        raise ValueError("Stage 42c source matrix audit differs")

    stage42d = read_json(root / "data/stage42d_bace1_large_pool_qubo_screen_result.json")
    verify_result(root, stage42d, "stage42d_bace1_large_pool_qubo_screen_complete")
    fold42d = read_csv(root / "results/runs/stage42d_bace1_large_pool_qubo_screen/fold_metrics.csv")
    selected42d = read_csv(root / "results/runs/stage42d_bace1_large_pool_qubo_screen/selection_metrics.csv")
    if len(fold42d) != 4 or any(int(row["state_count"]) != 1676115 for row in fold42d):
        raise ValueError("Stage 42d exact fold ledger differs")
    if any(abs(float(row["train_exact_minus_classical_gap"])) > 1e-12 for row in fold42d):
        raise ValueError("Stage 42d source gap differs")
    if stage42d["decision"]["frozen_objective_supported_on_bace1"]:
        raise ValueError("Stage 42d failed objective was promoted")
    exact42d = next(row for row in selected42d if row["method"] == "exact_qubo_objective_optimum")
    single42d = next(row for row in selected42d if row["method"] == "best_single_receptor")
    old_bedroc_delta = float(exact42d["robust_bedroc_composite"]) - float(single42d["robust_bedroc_composite"])
    if old_bedroc_delta >= 0.0:
        raise ValueError("Stage 42d old-objective BEDROC failure differs")

    stage42e = read_json(root / "data/stage42e_bace1_qubo_rank_alignment_diagnosis_result.json")
    verify_result(root, stage42e, "stage42e_bace1_qubo_rank_alignment_diagnosis_complete")
    diagnosis = dict(stage42e["diagnosis"])
    if not diagnosis["cardinality_pressure_detected"] or int(diagnosis["objective_optimal_k"]) != 6:
        raise ValueError("Stage 42e cardinality diagnosis differs")
    if int(diagnosis["bedroc_optimal_k"]) != 2 or diagnosis["old_objective_retuning_authorized"]:
        raise ValueError("Stage 42e decision boundary differs")

    stage42f = read_json(root / "data/stage42f_bace1_rank_sensitive_pair_qubo_result.json")
    verify_result(root, stage42f, "stage42f_bace1_rank_sensitive_pair_qubo_complete")
    folds42f = read_csv(root / "results/runs/stage42f_bace1_rank_sensitive_pair_qubo/fold_metrics.csv")
    full42f = read_csv(root / "results/runs/stage42f_bace1_rank_sensitive_pair_qubo/full_metrics.csv")
    if len(folds42f) != 24 or len(full42f) != 6:
        raise ValueError("Stage 42f fixed-k dimensions differ")
    for row in [*folds42f, *full42f]:
        gap_key = "train_exact_minus_classical_gap" if "train_exact_minus_classical_gap" in row else "exact_minus_classical_gap"
        if abs(float(row[gap_key])) > 1e-12 or row["exact_subset"] != row["classical_subset"]:
            raise ValueError("Stage 42f exact/classical equality differs")
    if any(not math.isfinite(float(row["exact_robust_bedroc"])) for row in full42f):
        raise ValueError("Stage 42f BEDROC value is invalid")
    best_combination = max(float(row["exact_robust_bedroc"]) for row in full42f)
    best_single = float(next(row for row in full42f if int(row["subset_size"]) == 1)["exact_robust_bedroc"])
    if best_combination - best_single < 0.019 - 1e-12:
        raise ValueError("Stage 42f combination gain differs")
    decision42f = dict(stage42f["decision"])
    if decision42f["rank_sensitive_pair_qubo_supported"] or decision42f["fresh_validation_authorized"]:
        raise ValueError("Stage 42f no-go decision differs")

    result = {
        "schema_version": "1.0",
        "audit_id": "stage42d-f-bace1-qubo-independent-audit-v1",
        "status": "stage42d_f_bace1_qubo_independent_audit_ok",
        "stage42c_pair_count": int(stage42c["pair_count"]),
        "stage42d_exact_fold_count": len(fold42d),
        "stage42d_old_objective_over_single_bedroc_delta": old_bedroc_delta,
        "stage42e_objective_optimal_k": int(diagnosis["objective_optimal_k"]),
        "stage42e_bedroc_optimal_k": int(diagnosis["bedroc_optimal_k"]),
        "stage42f_fold_k_cell_count": len(folds42f),
        "stage42f_full_k_count": len(full42f),
        "stage42f_exact_classical_positive_gap_cell_count": 0,
        "stage42f_best_combination_over_single_bedroc_gain": best_combination - best_single,
        "stage42f_rank_sensitive_pair_qubo_supported": False,
        "data_boundary": {
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "decision": "Retain the rank-sensitive pair QUBO as a useful receptor-combination model, but do not claim or test solver advantage on protected data because strong classical search found every exact optimum.",
    }
    output = output if output.is_absolute() else root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/stage42f_bace1_rank_sensitive_pair_qubo_audit.json"))
    args = parser.parse_args()
    run(args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
