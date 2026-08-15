"""Independently audit the Stage102B adaptive-cardinality development run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage102b_marginal_model_execution as stage102b


POLICIES = ("mechanistic_bootstrap_lcb", "target_held_out_l2_ridge")
TARGETS = ("BACE1", "EGFR", "FA10", "MK14", "PPARA", "PPARD", "PPARG")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_close(actual: float, expected: float, message: str) -> None:
    require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), message)


def as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    require(value in {"True", "False"}, f"expected Boolean CSV value, received {value!r}")
    return value == "True"


def audit(root: Path, config_path: Path) -> dict[str, Any]:
    config = read_json(root / config_path)
    outputs = config["outputs"]
    stage102b.validate_parent_hashes(root, config)
    stage102b.validate_stage102a_inputs(root, config)

    result = read_json(root / outputs["result_json"])
    edges = read_csv(root / outputs["edge_csv"])
    folds = read_csv(root / outputs["fold_csv"])
    summaries = read_csv(root / outputs["target_csv"])
    models = read_json(root / outputs["model_json"])

    require(result["status"] == "stage102b_marginal_model_execution_complete", "unexpected result status")
    require(tuple(result["target_ids"]) == TARGETS, "unexpected target list")
    require(len(edges) == 70, f"expected 70 marginal edges, received {len(edges)}")
    require(len(folds) == 700, f"expected 700 fold-decision rows, received {len(folds)}")
    require(len(summaries) == 140, f"expected 140 target-summary rows, received {len(summaries)}")
    require(set(models) == set(TARGETS), "held-target Ridge model coverage is incomplete")

    expected_edge_keys = {(target, fold, current) for target in TARGETS for fold in range(1, 6) for current in (2, 3)}
    edge_keys = {(row["target_id"], int(row["outer_fold"]), int(row["to_k"])) for row in edges}
    require(edge_keys == expected_edge_keys, "marginal-edge target/fold/cardinality coverage is incomplete")
    require(len(edge_keys) == len(edges), "marginal-edge rows are duplicated")

    numeric_fields = (
        "inner_mean_gain",
        "inner_gain_se",
        "bootstrap_mean_gain",
        "bootstrap_ci95_lower",
        "bootstrap_positive_probability",
        "active_rescue_contrast_top1",
        "active_rescue_contrast_top5",
        "aggregate_rank_spearman",
        "qubo_optimum_gap",
        "outer_gain",
        "ridge_predicted_outer_gain",
    )
    for row in edges:
        target = row["target_id"]
        require(target in TARGETS, f"unknown edge target: {target}")
        require(int(row["from_k"]) + 1 == int(row["to_k"]), "edge is not an adjacent cardinality transition")
        require(int(row["to_k"]) in {2, 3}, "unexpected candidate cardinality")
        require(all(math.isfinite(float(row[field])) for field in numeric_fields), "non-finite edge feature")
        training_targets = tuple(row["ridge_training_targets"].split("|"))
        require(target not in training_targets, "held target leaked into its Ridge training set")
        require(set(training_targets) == set(TARGETS) - {target}, "incorrect Ridge training-target set")
        expected_mechanistic = (
            float(row["bootstrap_ci95_lower"]) > 0.0
            and (float(row["active_rescue_contrast_top1"]) + float(row["active_rescue_contrast_top5"])) / 2.0 > 0.0
        )
        require(
            as_bool(row["mechanistic_bootstrap_lcb_continue"]) == expected_mechanistic,
            "mechanistic continuation decision does not match its frozen rule",
        )
        require(
            as_bool(row["target_held_out_l2_ridge_continue"]) == (float(row["ridge_predicted_outer_gain"]) > 0.0),
            "Ridge continuation decision does not match its frozen rule",
        )

    expected_fold_coverage = {
        (target, fold, policy, method)
        for target in TARGETS
        for fold in range(1, 6)
        for policy, methods in {
            "single": ("exact_qubo",),
            "fixed_k2": ("exact_qubo", "direct_bedroc_forward_greedy", "mean_singleton_topk"),
            "fixed_k3": ("exact_qubo", "direct_bedroc_forward_greedy", "mean_singleton_topk"),
            "one_standard_error_smallest_k": ("exact_qubo", "direct_bedroc_forward_greedy", "mean_singleton_topk"),
            "stage100_sequential_lcb": ("exact_qubo", "direct_bedroc_forward_greedy", "mean_singleton_topk"),
            "mechanistic_bootstrap_lcb": ("exact_qubo", "direct_bedroc_forward_greedy", "mean_singleton_topk"),
            "target_held_out_l2_ridge": ("exact_qubo", "direct_bedroc_forward_greedy", "mean_singleton_topk"),
            "outer_oracle_k": ("exact_qubo",),
        }.items()
        for method in methods
    }
    observed_fold_coverage = {
        (row["target_id"], int(row["outer_fold"]), row["policy"], row["selection_method"])
        for row in folds
    }
    require(observed_fold_coverage == expected_fold_coverage, "fold-decision policy coverage is incomplete")
    require(len(observed_fold_coverage) == len(folds), "fold-decision rows are duplicated")
    for row in folds:
        is_oracle = row["policy"] == "outer_oracle_k"
        require(as_bool(row["uses_outer_labels_for_selection"]) == is_oracle, "outer-label flag is inconsistent")
        if row["policy"] in POLICIES:
            require(not as_bool(row["uses_outer_labels_for_selection"]), "candidate selector used outer labels")

    summary_rows = [
        {
            **row,
            "mean_gain_over_train_selected_single": float(row["mean_gain_over_train_selected_single"]),
            "selected_k_values": row["selected_k_values"],
        }
        for row in summaries
    ]
    edge_rows_for_gate = [
        {
            **row,
            "mechanistic_bootstrap_lcb_continue": as_bool(row["mechanistic_bootstrap_lcb_continue"]),
            "target_held_out_l2_ridge_continue": as_bool(row["target_held_out_l2_ridge_continue"]),
        }
        for row in edges
    ]
    recomputed = {
        policy: stage102b.policy_gate(policy, summary_rows, edge_rows_for_gate, config)
        for policy in POLICIES
    }
    for policy in POLICIES:
        stored = result["candidate_decisions"][policy]
        require(stored["checks"] == recomputed[policy]["checks"], f"{policy}: gate checks do not reproduce")
        require(stored["passes"] == recomputed[policy]["passes"], f"{policy}: gate decision does not reproduce")
        for key, expected in recomputed[policy]["metrics"].items():
            actual = stored["metrics"][key]
            if isinstance(expected, float):
                require_close(float(actual), expected, f"{policy}: mismatched metric {key}")
            else:
                require(actual == expected, f"{policy}: mismatched metric {key}")

    require(result["decision"]["phase_a_gate_passes"] is False, "Stage102B incorrectly passed the Phase-A gate")
    require(result["selected_candidate"] is None, "a failed Stage102B run selected a candidate")
    require(result["decision"]["parp1_released"] is False, "PARP1 was incorrectly released")
    require(result["decision"]["quantum_hardware_authorized"] is False, "quantum hardware was incorrectly authorized")
    for key, value in result["data_boundary"].items():
        require(value == 0, f"data boundary was breached: {key}={value}")

    return {
        "schema_version": "1.0",
        "status": "stage102b_independent_audit_ok",
        "target_count": len(TARGETS),
        "marginal_edge_count": len(edges),
        "fold_decision_count": len(folds),
        "target_summary_count": len(summaries),
        "candidate_policies": list(POLICIES),
        "candidate_passes": {policy: result["candidate_decisions"][policy]["passes"] for policy in POLICIES},
        "outer_labels_used_by_candidate_selectors": False,
        "parp1_released": False,
        "quantum_hardware_authorized": False,
        "data_boundary": result["data_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/stage102b_marginal_model_execution_amendment01.json")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    record = audit(root, args.config)
    config = read_json(root / args.config)
    output = root / config["outputs"]["audit_json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
