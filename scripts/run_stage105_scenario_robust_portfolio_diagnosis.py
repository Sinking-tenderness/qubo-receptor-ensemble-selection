"""Diagnose a scenario-robust constrained portfolio rule on existing score cubes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.sparse import coo_matrix, vstack


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_stage102a_phase_a_results as stage102a
from scripts import run_stage05_mk14_method_gate as stage05
from scripts import run_stage42d_bace1_large_pool_qubo_screen as stage42d
from scripts import run_stage42f_bace1_rank_sensitive_pair_qubo as stage42f
from scripts import run_stage64_cross_target_uncertainty_shrunk_qubo as stage64
from scripts import run_stage68_quality_plateau_portfolio_qubo as stage68


@dataclass(frozen=True)
class TargetData:
    target_id: str
    receptor_ids: list[str]
    ligand_rows: list[dict[str, str]]
    scores: np.ndarray
    labels: np.ndarray
    outer_folds: np.ndarray


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_hashes(root: Path, config: dict[str, Any]) -> None:
    for key, expected in config["parent"].items():
        if not key.endswith("_sha256"):
            continue
        path_key = key.removesuffix("_sha256")
        if sha256(root / config["parent"][path_key]) != expected:
            raise ValueError(f"parent hash mismatch: {path_key}")
    for target_id, spec in config["inputs"].items():
        for name in ("scores", "manifest"):
            if sha256(root / spec[name]) != spec[f"{name}_sha256"]:
                raise ValueError(f"{target_id} {name} hash mismatch")


def historical_targets(root: Path, config: dict[str, Any]) -> dict[str, TargetData]:
    stage64_config = read_json(root / config["parent"]["stage64_config"])
    loaded: dict[str, TargetData] = {}
    for target_id in config["cohort"]["historical_targets"]:
        raw = stage64.load_target(root, target_id, stage64_config["targets"][target_id])
        loaded[target_id] = TargetData(
            target_id=target_id,
            receptor_ids=raw["receptor_ids"],
            ligand_rows=raw["ligands"],
            scores=raw["scores"],
            labels=raw["labels"],
            outer_folds=np.asarray([raw["outer"][ligand_id] for ligand_id in raw["ligand_ids"]], dtype=int),
        )
    return loaded


def phase_a_targets(root: Path, config: dict[str, Any]) -> dict[str, TargetData]:
    received_root = root / "analysis/stage102a_received_20260813/core"
    loaded: dict[str, TargetData] = {}
    for target_id, spec in config["inputs"].items():
        receptors, rows, _ = stage102a.load_target(received_root, target_id)
        seeds = stage102a.load_seed_values(received_root, target_id, receptors, rows)
        if len(receptors) != int(spec["expected_receptor_count"]) or len(rows) != int(spec["expected_ligand_count"]):
            raise ValueError(f"{target_id}: unexpected input dimensions")
        loaded[target_id] = TargetData(
            target_id=target_id,
            receptor_ids=receptors,
            ligand_rows=rows,
            scores=np.stack([seeds[seed] for seed in ("seed0", "seed1", "seed2")]),
            labels=np.asarray([int(row["label"] == "active") for row in rows], dtype=int),
            outer_folds=np.asarray([int(row["outer_fold"]) for row in rows], dtype=int),
        )
    return loaded


def scenario_utilities(
    train_scores: np.ndarray,
    train_labels: np.ndarray,
    train_rows: list[dict[str, str]],
    alpha: float,
    block_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full-train and frozen jackknife singleton utilities."""
    full_ranks = stage42d.rank_cube(train_scores, np.ones(train_scores.shape[1], dtype=bool))
    full_utility, _ = stage42f.pair_coefficients(full_ranks, train_labels, alpha)
    assignments = stage05.make_frozen_group_folds(train_rows, block_count, seed)
    fold_ids = np.asarray([assignments[row["ligand_id"]] for row in train_rows], dtype=int)
    utilities: list[np.ndarray] = []
    for block in range(block_count):
        keep = fold_ids != block
        if not np.any(train_labels[keep] == 1) or not np.any(train_labels[keep] == 0):
            raise ValueError("jackknife scenario removed a complete class")
        ranks = stage42d.rank_cube(train_scores, keep)
        utility, _ = stage42f.pair_coefficients(ranks[:, keep, :], train_labels[keep], alpha)
        utilities.append(utility)
    return full_utility, np.stack(utilities)


def solve_scenario_robust(
    workspace: stage68.PortfolioMilp,
    utility: np.ndarray,
    scenario_utility: np.ndarray,
    baseline: tuple[int, ...],
    subset_size: int,
    time_limit_seconds: float,
) -> tuple[tuple[int, ...], dict[str, Any], np.ndarray]:
    """Minimize redundancy while matching the pair-off baseline in every scenario."""
    scenario_count = scenario_utility.shape[0]
    quality = np.zeros((scenario_count, workspace.variable_count), dtype=float)
    quality[:, : workspace.receptor_count] = scenario_utility
    floors = np.mean(scenario_utility[:, list(baseline)], axis=1)
    matrix = vstack(
        [workspace.edge_constraints, coo_matrix(workspace.cardinality), coo_matrix(quality)]
    ).tocsr()
    lower = np.concatenate(
        [np.full(workspace.edge_count, -1.0), np.asarray([subset_size]), floors * subset_size]
    )
    upper = np.concatenate(
        [np.full(workspace.edge_count, np.inf), np.asarray([subset_size]), np.full(scenario_count, np.inf)]
    )
    subset, record = workspace._solve(
        matrix,
        lower,
        upper,
        utility,
        time_limit_seconds,
    )
    margins = np.mean(scenario_utility[:, list(subset)], axis=1) - floors
    if np.min(margins) < -1e-10:
        raise ValueError("scenario-robust MILP returned an infeasible subset")
    return subset, record, margins


def metric_row(
    data: TargetData,
    fold: int,
    subset_size: int,
    solver: str,
    subset: tuple[int, ...],
    utility: np.ndarray,
    scenario_margins: np.ndarray,
    redundancy: np.ndarray,
    ranks: np.ndarray,
    test_mask: np.ndarray,
    alpha: float,
    record: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = stage42d.bedroc_metrics(ranks[:, test_mask, :], data.labels[test_mask], subset, alpha)
    row = {
        "target_id": data.target_id,
        "outer_fold": fold,
        "subset_size": subset_size,
        "solver_id": solver,
        "selected_subset": stage68.subset_name(subset, data.receptor_ids),
        "train_mean_singleton_utility": float(np.mean(utility[list(subset)])),
        "minimum_jackknife_quality_margin": float(np.min(scenario_margins)),
        "mean_jackknife_quality_margin": float(np.mean(scenario_margins)),
        "stable_redundancy_sum": stage68.redundancy_sum(subset, redundancy),
        "stable_redundancy_mean": stage68.redundancy_mean(subset, redundancy),
        "holdout_primary_bedroc": metrics["primary_bedroc"],
        "holdout_mean_seed_bedroc": metrics["mean_seed_bedroc"],
        "holdout_worst_seed_bedroc": metrics["worst_seed_bedroc"],
        "holdout_robust_bedroc": metrics["robust_bedroc_composite"],
        "uses_outer_labels_for_selection": False,
    }
    if record is not None:
        row.update({"milp_status": record["status"], "milp_gap": record["mip_gap"], "milp_node_count": record["mip_node_count"]})
    return row


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): row
        for row in rows
        if row["solver_id"] == "pair_off_baseline"
    }
    result: list[dict[str, Any]] = []
    for target_id in sorted({str(row["target_id"]) for row in rows}):
        for subset_size in sorted({int(row["subset_size"]) for row in rows}):
            selected = [row for row in rows if row["target_id"] == target_id and int(row["subset_size"]) == subset_size and row["solver_id"] == "scenario_robust_milp_certificate"]
            gains = [float(row["holdout_robust_bedroc"]) - float(baseline[(target_id, int(row["outer_fold"]), subset_size)]["holdout_robust_bedroc"]) for row in selected]
            reductions = [float(baseline[(target_id, int(row["outer_fold"]), subset_size)]["stable_redundancy_mean"]) - float(row["stable_redundancy_mean"]) for row in selected]
            changes = [row["selected_subset"] != baseline[(target_id, int(row["outer_fold"]), subset_size)]["selected_subset"] for row in selected]
            result.append(
                {
                    "target_id": target_id,
                    "subset_size": subset_size,
                    "fold_count": len(selected),
                    "changed_subset_fold_count": int(sum(changes)),
                    "mean_holdout_robust_bedroc": float(np.mean([row["holdout_robust_bedroc"] for row in selected])),
                    "mean_gain_over_pair_off": float(np.mean(gains)),
                    "worst_gain_over_pair_off": float(np.min(gains)),
                    "mean_stable_redundancy_reduction": float(np.mean(reductions)),
                    "minimum_stable_redundancy_reduction": float(np.min(reductions)),
                    "minimum_jackknife_quality_margin": float(min(float(row["minimum_jackknife_quality_margin"]) for row in selected)),
                    "noninferior_fold_count_at_0p01": int(sum(gain >= -0.01 for gain in gains)),
                }
            )
    return result


def render_report(result: dict[str, Any], summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage105 scenario-robust portfolio diagnosis",
        "",
        "For every outer training split, Stage105 requires the selected subset to match or exceed the full-train pair-off baseline's singleton utility in each of four frozen scaffold-jackknife scenarios. It then minimizes Stage68 three-seed stable redundancy. This is posthoc mechanism diagnosis only.",
        "",
        "| Target | k | Changed folds | Robust BEDROC gain | Redundancy reduction | Worst gain |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['target_id']} | {row['subset_size']} | {row['changed_subset_fold_count']}/{row['fold_count']} | "
            f"{row['mean_gain_over_pair_off']:+.6f} | {row['mean_stable_redundancy_reduction']:+.6f} | "
            f"{row['worst_gain_over_pair_off']:+.6f} |"
        )
    lines.extend(["", "## Decision", "", result["decision"]["next_action"], ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage105_scenario_robust_portfolio_diagnosis.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = read_json(root / args.config)
    validate_hashes(root, config)
    rule = config["frozen_rule"]
    alpha = float(rule["bedroc_alpha"])
    targets = {**historical_targets(root, config), **phase_a_targets(root, config)}
    rows: list[dict[str, Any]] = []
    certificate_count = 0
    for data in targets.values():
        for fold in sorted(set(data.outer_folds.tolist())):
            train_mask = data.outer_folds != fold
            test_mask = ~train_mask
            train_rows = [row for row, keep in zip(data.ligand_rows, train_mask) if keep]
            utility, scenario_utility = scenario_utilities(
                data.scores[:, train_mask, :],
                data.labels[train_mask],
                train_rows,
                alpha,
                int(rule["jackknife_block_count"]),
                int(rule["jackknife_seed_base"]) + fold,
            )
            ranks = stage42d.rank_cube(data.scores, train_mask)
            redundancy = stage68.stable_redundancy(ranks, train_mask)
            workspace = stage68.PortfolioMilp(redundancy)
            for subset_size in (int(value) for value in rule["subset_sizes"]):
                baseline = stage68.pair_off_subset(utility, subset_size)
                baseline_margins = np.zeros(scenario_utility.shape[0], dtype=float)
                rows.append(metric_row(data, fold, subset_size, "pair_off_baseline", baseline, utility, baseline_margins, redundancy, ranks, test_mask, alpha, None))
                selected, record, margins = solve_scenario_robust(
                    workspace, utility, scenario_utility, baseline, subset_size, 60.0
                )
                certificate_count += 1
                rows.append(metric_row(data, fold, subset_size, "scenario_robust_milp_certificate", selected, utility, margins, redundancy, ranks, test_mask, alpha, record))
            print(json.dumps({"target_id": data.target_id, "outer_fold": fold, "status": "complete"}), flush=True)
    summaries = summarize(rows)
    gate = config["evaluation"]["diagnostic_gate"]
    target_means = {}
    for target_id in sorted(targets):
        target_rows = [row for row in summaries if row["target_id"] == target_id]
        target_means[target_id] = {
            "gain": float(np.mean([row["mean_gain_over_pair_off"] for row in target_rows])),
            "redundancy": float(np.mean([row["mean_stable_redundancy_reduction"] for row in target_rows])),
        }
    gains = [value["gain"] for value in target_means.values()]
    reductions = [value["redundancy"] for value in target_means.values()]
    checks = {
        "mean_target_gain": float(np.mean(gains)) >= float(gate["minimum_mean_target_gain_over_pair_off"]),
        "worst_target_gain": float(np.min(gains)) >= float(gate["minimum_worst_target_gain_over_pair_off"]),
        "target_count_within_0p01": int(sum(value >= -0.01 for value in gains)) >= int(gate["minimum_target_count_within_0p01"]),
        "mean_redundancy_reduction": float(np.mean(reductions)) >= float(gate["minimum_mean_stable_redundancy_reduction"]),
        "target_count_nonnegative_redundancy": int(sum(value >= 0.0 for value in reductions)) >= int(gate["minimum_target_count_with_nonnegative_redundancy_reduction"]),
    }
    result = {
        "schema_version": "1.0",
        "status": "stage105_scenario_robust_portfolio_diagnosis_complete",
        "evidence_status": "posthoc six-target mechanism diagnosis using only already consumed development and Phase-A matrices; no independent confirmation is created.",
        "target_ids": sorted(targets),
        "certificate_count": certificate_count,
        "target_mean_gain_and_redundancy": target_means,
        "diagnostic_gate": {"checks": checks, "passes": bool(all(checks.values()))},
        "decision": {
            "replacement_objective_authorized": False,
            "new_target_protocol_authorized": False,
            "parp1_released": False,
            "quantum_hardware_authorized": False,
            "next_action": "Do not retune this scenario constraint on the same six targets. Use this result only to decide whether a separately reviewed untouched-target protocol is scientifically justified; no current protected dataset or hardware task is released.",
        },
        "data_boundary": config["data_boundary"],
        "interpretation": "The all-scenario constraint protects jackknife singleton utility, not outer BEDROC by assumption. Outer metrics remain diagnostic only.",
    }
    outputs = config["outputs"]
    write_csv(root / outputs["fold_csv"], rows)
    write_csv(root / outputs["target_csv"], summaries)
    result_path = root / outputs["result_json"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    report_path = root / outputs["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result, summaries), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
