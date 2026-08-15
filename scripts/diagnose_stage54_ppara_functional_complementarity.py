"""Diagnose PPARA single-receptor dominance and decoy-promotion failure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage42d_bace1_large_pool_qubo_screen import bedroc_metrics, rank_cube
from scripts.run_stage42f_bace1_rank_sensitive_pair_qubo import pair_coefficients
from scripts.run_stage53_ppara_large_pool_qubo_transfer import build_score_cube


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def verified(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage 54 input identity differs: {path}")
    return path


def finite_spearman(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return 0.0
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else 0.0


def robust_bedroc(
    ranks: np.ndarray, labels: np.ndarray, subset: tuple[int, ...], alpha: float
) -> float:
    return float(
        bedroc_metrics(ranks, labels, subset, alpha)["robust_bedroc_composite"]
    )


def majority_hit(ranks: np.ndarray, subset: tuple[int, ...], threshold: float) -> np.ndarray:
    per_seed = np.any(ranks[:, :, subset] <= threshold, axis=2)
    return np.sum(per_seed, axis=0) >= 2


def best_subset(
    ranks: np.ndarray,
    labels: np.ndarray,
    subsets: list[tuple[int, ...]],
    alpha: float,
) -> tuple[int, ...]:
    return min(
        subsets,
        key=lambda subset: (-robust_bedroc(ranks, labels, subset, alpha), subset),
    )


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 54 implementation identity differs")
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    stage53 = read_json(inputs["stage53_result"])
    stage53_audit = read_json(inputs["stage53_audit"])
    if stage53["status"] != "stage53_ppara_large_pool_qubo_transfer_complete":
        raise ValueError("Stage 53 source result did not complete")
    if stage53["decision"]["frozen_qubo_application_transfer_supported"]:
        raise ValueError("Stage 54 is only authorized after Stage 53 no-go")
    if stage53_audit["status"] != (
        "stage53_ppara_large_pool_qubo_transfer_independent_audit_ok"
    ):
        raise ValueError("Stage 53 independent audit did not pass")
    outputs = {key: root / value for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage 54 outputs exist; pass --overwrite")

    ligands = read_csv(inputs["ligand_manifest"])
    receptors = read_csv(inputs["receptor_manifest"])
    scores = read_csv(inputs["corrected_scores"])
    assignments = read_csv(inputs["fold_assignments"])
    ligand_ids = [row["ligand_id"] for row in ligands]
    receptor_ids = [row["conformer_id"] for row in receptors]
    labels = np.asarray([int(row["label"] == "active") for row in ligands], dtype=int)
    cube = build_score_cube(scores, ligand_ids, receptor_ids)
    fold_by_ligand = {row["ligand_id"]: int(row["outer_fold"]) for row in assignments}
    alpha = float(config["diagnostic"]["bedroc_alpha"])
    threshold = float(config["diagnostic"]["favorable_rank_fraction"])
    full_mask = np.ones(len(ligands), dtype=bool)
    ranks = rank_cube(cube, full_mask)
    singles = [(index,) for index in range(len(receptors))]
    pairs = list(itertools.combinations(range(len(receptors)), 2))
    single_values = {
        subset[0]: robust_bedroc(ranks, labels, subset, alpha) for subset in singles
    }
    dominant_index = min(single_values, key=lambda index: (-single_values[index], index))
    dominant = (dominant_index,)
    dominant_hits = majority_hit(ranks, dominant, threshold)
    active_mask = labels == 1
    decoy_mask = labels == 0
    active_count = int(active_mask.sum())
    decoy_count = int(decoy_mask.sum())

    receptor_rows: list[dict[str, Any]] = []
    for index, receptor_id in enumerate(receptor_ids):
        pair = dominant if index == dominant_index else tuple(sorted((dominant_index, index)))
        pair_hits = majority_hit(ranks, pair, threshold)
        rescued = int(np.sum(~dominant_hits & pair_hits & active_mask))
        promoted = int(np.sum(~dominant_hits & pair_hits & decoy_mask))
        pair_bedroc = robust_bedroc(ranks, labels, pair, alpha)
        receptor_rows.append(
            {
                "receptor_id": receptor_id,
                "single_robust_bedroc": single_values[index],
                "single_rank": 1
                + sorted(single_values, key=lambda item: (-single_values[item], item)).index(index),
                "is_dominant_single": index == dominant_index,
                "dominant_plus_receptor_subset": "+".join(
                    receptor_ids[item] for item in pair
                ),
                "dominant_plus_receptor_robust_bedroc": pair_bedroc,
                "bedroc_delta_over_dominant": pair_bedroc
                - single_values[dominant_index],
                "active_majority_hits_rescued": rescued,
                "decoy_majority_hits_promoted": promoted,
                "normalized_active_rescue": rescued / active_count,
                "normalized_decoy_promotion": promoted / decoy_count,
                "normalized_rescue_minus_promotion": rescued / active_count
                - promoted / decoy_count,
            }
        )

    singleton_coeff, complement = pair_coefficients(ranks, labels, alpha)
    pair_rows: list[dict[str, Any]] = []
    for left, right in pairs:
        subset = (left, right)
        value = robust_bedroc(ranks, labels, subset, alpha)
        better_single = max(single_values[left], single_values[right])
        baseline_index = left if single_values[left] >= single_values[right] else right
        baseline_hits = majority_hit(ranks, (baseline_index,), threshold)
        pair_hits = majority_hit(ranks, subset, threshold)
        rescued = int(np.sum(~baseline_hits & pair_hits & active_mask))
        promoted = int(np.sum(~baseline_hits & pair_hits & decoy_mask))
        seed_correlations = [
            finite_spearman(ranks[seed, :, left], ranks[seed, :, right])
            for seed in range(3)
        ]
        pair_rows.append(
            {
                "left_receptor": receptor_ids[left],
                "right_receptor": receptor_ids[right],
                "pair": receptor_ids[left] + "+" + receptor_ids[right],
                "pair_robust_bedroc": value,
                "best_member_robust_bedroc": better_single,
                "pair_bedroc_gain_over_best_member": value - better_single,
                "rank_pair_qubo_complement_coefficient": float(complement[left, right]),
                "active_majority_hits_rescued": rescued,
                "decoy_majority_hits_promoted": promoted,
                "normalized_active_rescue": rescued / active_count,
                "normalized_decoy_promotion": promoted / decoy_count,
                "normalized_rescue_minus_promotion": rescued / active_count
                - promoted / decoy_count,
                "mean_seed_rank_spearman": statistics.fmean(seed_correlations),
                "minimum_seed_rank_spearman": min(seed_correlations),
            }
        )

    fold_rows: list[dict[str, Any]] = []
    for fold in range(int(config["diagnostic"]["outer_fold_count"])):
        train_mask = np.asarray([fold_by_ligand[value] != fold for value in ligand_ids])
        holdout_mask = ~train_mask
        fold_ranks = rank_cube(cube, train_mask)
        best_single = best_subset(
            fold_ranks[:, train_mask, :], labels[train_mask], singles, alpha
        )
        best_pair = best_subset(
            fold_ranks[:, train_mask, :], labels[train_mask], pairs, alpha
        )
        train_single = robust_bedroc(
            fold_ranks[:, train_mask, :], labels[train_mask], best_single, alpha
        )
        train_pair = robust_bedroc(
            fold_ranks[:, train_mask, :], labels[train_mask], best_pair, alpha
        )
        holdout_single = robust_bedroc(
            fold_ranks[:, holdout_mask, :], labels[holdout_mask], best_single, alpha
        )
        holdout_pair = robust_bedroc(
            fold_ranks[:, holdout_mask, :], labels[holdout_mask], best_pair, alpha
        )
        fold_rows.append(
            {
                "outer_fold": fold,
                "train_ligand_count": int(train_mask.sum()),
                "holdout_ligand_count": int(holdout_mask.sum()),
                "train_best_single": receptor_ids[best_single[0]],
                "train_best_pair": "+".join(receptor_ids[item] for item in best_pair),
                "train_best_single_bedroc": train_single,
                "train_best_pair_bedroc": train_pair,
                "train_pair_gain": train_pair - train_single,
                "holdout_best_single_bedroc": holdout_single,
                "holdout_best_pair_bedroc": holdout_pair,
                "holdout_pair_gain": holdout_pair - holdout_single,
            }
        )

    best_pair_row = max(pair_rows, key=lambda row: float(row["pair_robust_bedroc"]))
    dominant_additions = [row for row in receptor_rows if not row["is_dominant_single"]]
    pair_bedroc_gains = [float(row["pair_bedroc_gain_over_best_member"]) for row in pair_rows]
    pair_coefficients_values = [
        float(row["rank_pair_qubo_complement_coefficient"]) for row in pair_rows
    ]
    alignment = finite_spearman(pair_coefficients_values, pair_bedroc_gains)
    holdout_gains = [float(row["holdout_pair_gain"]) for row in fold_rows]
    criteria = config["future_target_intake_criteria"]
    observed_checks = {
        "minimum_full_best_pair_gain": float(
            best_pair_row["pair_bedroc_gain_over_best_member"]
        )
        >= float(criteria["minimum_full_best_pair_gain"]),
        "minimum_positive_holdout_pair_gain_folds": sum(value > 0 for value in holdout_gains)
        >= int(criteria["minimum_positive_holdout_pair_gain_folds"]),
        "minimum_mean_holdout_pair_gain": statistics.fmean(holdout_gains)
        >= float(criteria["minimum_mean_holdout_pair_gain"]),
        "minimum_positive_rescue_balance_additions": sum(
            float(row["normalized_rescue_minus_promotion"]) > 0
            for row in dominant_additions
        )
        >= int(criteria["minimum_positive_rescue_balance_additions"]),
        "maximum_median_pair_rank_redundancy": statistics.median(
            float(row["mean_seed_rank_spearman"]) for row in pair_rows
        )
        <= float(criteria["maximum_median_pair_rank_redundancy"]),
    }
    future_candidate_pass = all(observed_checks.values())
    diagnosis = {
        "dominant_single_receptor": receptor_ids[dominant_index],
        "dominant_single_robust_bedroc": single_values[dominant_index],
        "best_pair": best_pair_row["pair"],
        "best_pair_robust_bedroc": best_pair_row["pair_robust_bedroc"],
        "best_pair_gain_over_best_member": best_pair_row[
            "pair_bedroc_gain_over_best_member"
        ],
        "positive_pair_bedroc_gain_count": sum(value > 0 for value in pair_bedroc_gains),
        "pair_count": len(pair_rows),
        "positive_dominant_addition_bedroc_count": sum(
            float(row["bedroc_delta_over_dominant"]) > 0 for row in dominant_additions
        ),
        "positive_rescue_balance_dominant_addition_count": sum(
            float(row["normalized_rescue_minus_promotion"]) > 0
            for row in dominant_additions
        ),
        "median_dominant_addition_rescue_balance": statistics.median(
            float(row["normalized_rescue_minus_promotion"])
            for row in dominant_additions
        ),
        "mean_holdout_oracle_pair_gain": statistics.fmean(holdout_gains),
        "positive_holdout_oracle_pair_gain_fold_count": sum(
            value > 0 for value in holdout_gains
        ),
        "median_pair_rank_spearman": statistics.median(
            float(row["mean_seed_rank_spearman"]) for row in pair_rows
        ),
        "rank_pair_coefficient_vs_bedroc_gain_spearman": alignment,
        "single_receptor_dominance_confirmed": not (
            observed_checks["minimum_full_best_pair_gain"]
            and observed_checks["minimum_positive_holdout_pair_gain_folds"]
            and observed_checks["minimum_mean_holdout_pair_gain"]
        ),
        "decoy_promotion_failure_confirmed": statistics.median(
            float(row["normalized_rescue_minus_promotion"])
            for row in dominant_additions
        )
        <= 0,
        "rank_pair_objective_alignment_weak": alignment
        < float(config["diagnostic"]["minimum_supported_pair_alignment_spearman"]),
    }
    decision = {
        "failure_mechanism_resolved": bool(
            diagnosis["single_receptor_dominance_confirmed"]
            or diagnosis["decoy_promotion_failure_confirmed"]
            or diagnosis["rank_pair_objective_alignment_weak"]
        ),
        "ppara_same_data_retuning_authorized": False,
        "ppara_fresh_validation_authorized": False,
        "future_target_intake_criteria_frozen": True,
        "ppara_would_pass_future_intake_criteria": future_candidate_pass,
        "future_intake_checks_on_ppara": observed_checks,
        "small_pilot_required_before_full_matrix": True,
        "quantum_hardware_authorized": False,
    }

    write_csv(outputs["receptor_diagnostics_csv"], receptor_rows)
    write_csv(outputs["pair_diagnostics_csv"], pair_rows)
    write_csv(outputs["fold_oracle_diagnostics_csv"], fold_rows)
    future_record = {
        "schema_version": "1.0",
        "status": "stage54_future_target_intake_criteria_frozen",
        "criteria": criteria,
        "workflow": [
            "retain only structures passing hard preparation and pocket-validity gates",
            "do not compress the valid pool by outcome-informed max-min selection",
            "dock a small preregistered active/decoy pilot panel across the candidate pool",
            "apply the frozen functional-complementarity intake criteria",
            "build the full train matrix only when every intake check passes",
            "freeze QUBO objective and stopping rule before full-matrix outcomes",
        ],
        "ppara_retrospective_check": observed_checks,
        "ppara_pass": future_candidate_pass,
        "interpretation_boundary": (
            "These criteria are prospective design requirements for a new target. "
            "They cannot rescue or retune PPARA."
        ),
    }
    write_json(outputs["future_intake_criteria_json"], future_record)
    report = [
        "# Stage54 PPARA functional-complementarity diagnosis",
        "",
        f"Dominant single receptor: {diagnosis['dominant_single_receptor']} "
        f"({diagnosis['dominant_single_robust_bedroc']:.6f}).",
        f"Best full-data pair gain: {diagnosis['best_pair_gain_over_best_member']:+.6f}.",
        f"Mean scaffold-holdout oracle-pair gain: {diagnosis['mean_holdout_oracle_pair_gain']:+.6f}.",
        f"Positive oracle-pair holdout folds: {diagnosis['positive_holdout_oracle_pair_gain_fold_count']}/4.",
        f"Median dominant-addition rescue balance: {diagnosis['median_dominant_addition_rescue_balance']:+.6f}.",
        f"Pair coefficient/BEDROC-gain Spearman: {alignment:+.6f}.",
        "",
        f"Failure mechanism resolved: **{'YES' if decision['failure_mechanism_resolved'] else 'NO'}**.",
        f"PPARA passes future intake criteria: **{'YES' if future_candidate_pass else 'NO'}**.",
        "",
        config["interpretation_boundary"],
        "",
    ]
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report), encoding="ascii")
    result = {
        "schema_version": "1.0",
        "status": "stage54_ppara_functional_complementarity_diagnosis_complete",
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, implementation),
        "diagnosis": diagnosis,
        "decision": decision,
        "data_boundary": {
            "train_rows_read": len(ligand_ids),
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "outputs": {
            key: descriptor(root, path)
            for key, path in outputs.items()
            if key != "result_json"
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    print(json.dumps({"diagnosis": diagnosis, "decision": decision}, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    run(config_path, root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
