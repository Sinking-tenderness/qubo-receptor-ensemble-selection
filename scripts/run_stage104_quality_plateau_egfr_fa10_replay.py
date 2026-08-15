"""Replay the frozen Stage68 portfolio rule on existing EGFR and FA10 matrices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_stage102a_phase_a_results as stage102a
from scripts import run_stage42d_bace1_large_pool_qubo_screen as stage42d
from scripts import run_stage64_cross_target_uncertainty_shrunk_qubo as stage64
from scripts import run_stage68_quality_plateau_portfolio_qubo as stage68


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_hashes(root: Path, config: dict[str, Any]) -> None:
    for key, expected in config["parent"].items():
        if key.endswith("_sha256"):
            path_key = key.removesuffix("_sha256")
            actual = sha256(root / config["parent"][path_key])
            if actual != expected:
                raise ValueError(f"parent hash mismatch: {path_key}")
    for target, spec in config["inputs"].items():
        for name in ("scores", "manifest"):
            actual = sha256(root / spec[name])
            if actual != spec[f"{name}_sha256"]:
                raise ValueError(f"{target} {name} hash mismatch")


def build_cube(
    root: Path, target: str, spec: dict[str, Any]
) -> tuple[list[str], list[dict[str, str]], np.ndarray]:
    received_root = root / "analysis/stage102a_received_20260813/core"
    receptors, rows, _ = stage102a.load_target(received_root, target)
    seeds = stage102a.load_seed_values(received_root, target, receptors, rows)
    if len(rows) != int(spec["expected_ligand_count"]) or len(receptors) != int(spec["expected_receptor_count"]):
        raise ValueError(f"{target}: unexpected input dimensions")
    if sorted(seeds) != ["seed0", "seed1", "seed2"]:
        raise ValueError(f"{target}: unexpected seed IDs")
    scores = np.stack([seeds[seed] for seed in ("seed0", "seed1", "seed2")])
    if not np.isfinite(scores).all():
        raise ValueError(f"{target}: non-finite seed matrix")
    return receptors, rows, scores


def metric_row(
    target: str,
    fold: int,
    k: int,
    solver: str,
    subset: tuple[int, ...],
    receptors: list[str],
    utility: np.ndarray,
    quality_floor: float,
    redundancy: np.ndarray,
    ranks: np.ndarray,
    labels: np.ndarray,
    test_mask: np.ndarray,
    alpha: float,
    record: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = stage42d.bedroc_metrics(ranks[:, test_mask, :], labels[test_mask], subset, alpha)
    result = {
        "target_id": target,
        "outer_fold": fold,
        "subset_size": k,
        "solver_id": solver,
        "selected_subset": stage68.subset_name(subset, receptors),
        "train_mean_singleton_utility": float(np.mean(utility[list(subset)])),
        "train_quality_floor": quality_floor,
        "train_quality_margin": float(np.mean(utility[list(subset)])) - quality_floor,
        "stable_redundancy_sum": stage68.redundancy_sum(subset, redundancy),
        "stable_redundancy_mean": stage68.redundancy_mean(subset, redundancy),
        "holdout_primary_bedroc": metrics["primary_bedroc"],
        "holdout_mean_seed_bedroc": metrics["mean_seed_bedroc"],
        "holdout_worst_seed_bedroc": metrics["worst_seed_bedroc"],
        "holdout_robust_bedroc": metrics["robust_bedroc_composite"],
        "uses_outer_labels_for_selection": False,
    }
    if record is not None:
        result.update(
            {
                "milp_status": record["status"],
                "milp_gap": record["mip_gap"],
                "milp_node_count": record["mip_node_count"],
            }
        )
    return result


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    baseline = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): row
        for row in rows
        if row["solver_id"] == "pair_off_baseline"
    }
    for target in sorted({str(row["target_id"]) for row in rows}):
        for k in sorted({int(row["subset_size"]) for row in rows}):
            selected = [
                row
                for row in rows
                if row["target_id"] == target and int(row["subset_size"]) == k and row["solver_id"] == "continuous_milp_certificate"
            ]
            gains = [
                float(row["holdout_robust_bedroc"])
                - float(baseline[(target, int(row["outer_fold"]), k)]["holdout_robust_bedroc"])
                for row in selected
            ]
            reductions = [
                float(baseline[(target, int(row["outer_fold"]), k)]["stable_redundancy_mean"])
                - float(row["stable_redundancy_mean"])
                for row in selected
            ]
            result.append(
                {
                    "target_id": target,
                    "subset_size": k,
                    "fold_count": len(selected),
                    "mean_holdout_robust_bedroc": float(np.mean([row["holdout_robust_bedroc"] for row in selected])),
                    "mean_gain_over_pair_off": float(np.mean(gains)),
                    "worst_gain_over_pair_off": float(np.min(gains)),
                    "mean_stable_redundancy_reduction": float(np.mean(reductions)),
                    "minimum_stable_redundancy_reduction": float(np.min(reductions)),
                    "noninferior_fold_count_at_0p01": int(sum(value >= -0.01 for value in gains)),
                }
            )
    return result


def render_report(result: dict[str, Any], summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage104 frozen quality-plateau replay",
        "",
        "Stage104 replays the Stage68 frozen `uncertainty_0p5x` quality-floor stable-redundancy portfolio rule on already collected EGFR and FA10 Phase-A matrices. It is posthoc transfer evidence, not independent confirmation.",
        "",
        "| Target | k | Robust BEDROC | Gain over pair-off | Redundancy reduction |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['target_id']} | {row['subset_size']} | {row['mean_holdout_robust_bedroc']:.6f} | "
            f"{row['mean_gain_over_pair_off']:+.6f} | {row['mean_stable_redundancy_reduction']:+.6f} |"
        )
    lines.extend(["", "## Decision", "", result["decision"]["next_action"], ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage104_quality_plateau_egfr_fa10_replay.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = read_json(root / args.config)
    validate_hashes(root, config)
    rule = config["frozen_stage68_rule"]
    alpha = float(rule["bedroc_alpha"])
    rows: list[dict[str, Any]] = []
    for target, spec in config["inputs"].items():
        receptors, ligands, scores = build_cube(root, target, spec)
        labels = np.asarray([int(row["label"] == "active") for row in ligands], dtype=int)
        folds = np.asarray([int(row["outer_fold"]) for row in ligands], dtype=int)
        for fold in sorted(set(folds.tolist())):
            train_mask = folds != fold
            test_mask = ~train_mask
            ranks = stage42d.rank_cube(scores, train_mask)
            train_rows = [row for row, keep in zip(ligands, train_mask) if keep]
            statistics_ = stage64.jackknife_pair_statistics(
                scores[:, train_mask, :],
                labels[train_mask],
                train_rows,
                alpha,
                int(rule["jackknife_block_count"]),
                int(rule["jackknife_seed_base"]) + fold,
            )
            utility = statistics_["full_singleton"]
            spread = statistics_["singleton_spread"]
            redundancy = stage68.stable_redundancy(ranks, train_mask)
            workspace = stage68.PortfolioMilp(redundancy)
            for k in (int(value) for value in rule["subset_sizes"]):
                baseline_plateau = stage68.quality_plateau(utility, spread, k, 0.0)
                baseline = baseline_plateau["baseline_subset"]
                rows.append(metric_row(target, fold, k, "pair_off_baseline", baseline, receptors, utility, baseline_plateau["quality_floor"], redundancy, ranks, labels, test_mask, alpha, None))
                plateau = stage68.quality_plateau(utility, spread, k, float(rule["uncertainty_multiplier"]))
                selected, record = workspace.solve_lower_quality(
                    utility, k, k * plateau["quality_floor"], 60.0
                )
                rows.append(metric_row(target, fold, k, "continuous_milp_certificate", selected, receptors, utility, plateau["quality_floor"], redundancy, ranks, labels, test_mask, alpha, record))
            print(json.dumps({"target_id": target, "outer_fold": fold, "status": "complete"}), flush=True)
    summaries = summarize(rows)
    transfer_checks = [
        row["mean_gain_over_pair_off"] >= -0.01 and row["mean_stable_redundancy_reduction"] >= 0.0
        for row in summaries
    ]
    result = {
        "schema_version": "1.0",
        "status": "stage104_quality_plateau_egfr_fa10_replay_complete",
        "evidence_status": "posthoc frozen-rule transfer replay on already used Phase-A targets; it cannot serve as independent confirmation or change the Stage102B NO-GO.",
        "target_ids": sorted(config["inputs"]),
        "frozen_rule": rule,
        "transfer_checks": {
            "all_target_k_cells_quality_noninferior_and_redundancy_nonnegative": bool(all(transfer_checks)),
            "passing_cell_count": int(sum(transfer_checks)),
            "cell_count": len(transfer_checks),
        },
        "decision": {
            "new_target_protocol_authorized": False,
            "parp1_released": False,
            "quantum_hardware_authorized": False,
            "next_action": "Use the replay only to judge whether the Stage68 constrained-selection application transfers to these contrasting targets. Do not alter the frozen rule or claim independent efficacy from this posthoc replay.",
        },
        "data_boundary": config["data_boundary"],
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
