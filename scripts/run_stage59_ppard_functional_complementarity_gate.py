"""Apply the preregistered PPARD Pilot-96 functional-complementarity gate."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import write_json  # noqa: F401 (deduped)
import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage42d_bace1_large_pool_qubo_screen import bedroc_metrics, rank_cube


SEED_IDS = ("seed0", "seed1", "seed2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))




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
        raise ValueError(f"Stage59 input identity differs: {path}")
    return path


def build_score_cube(
    rows: list[dict[str, str]], ligand_ids: list[str], receptor_ids: list[str]
) -> np.ndarray:
    ligand_index = {value: index for index, value in enumerate(ligand_ids)}
    receptor_index = {value: index for index, value in enumerate(receptor_ids)}
    seed_index = {value: index for index, value in enumerate(SEED_IDS)}
    cube = np.full((3, len(ligand_ids), len(receptor_ids)), np.nan, dtype=float)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if row["target_id"] != "PPARD":
            raise ValueError("Stage59 score target differs")
        key = (row["seed_id"], row["ligand_id"], row["receptor_id"])
        if key in seen:
            raise ValueError(f"duplicate Stage59 score key: {key}")
        if key[0] not in seed_index or key[1] not in ligand_index or key[2] not in receptor_index:
            raise ValueError(f"unexpected Stage59 score key: {key}")
        seen.add(key)
        cube[seed_index[key[0]], ligand_index[key[1]], receptor_index[key[2]]] = float(
            row["gpu_score"]
        )
    expected = 3 * len(ligand_ids) * len(receptor_ids)
    if len(seen) != expected or not np.isfinite(cube).all():
        raise ValueError("Stage59 score cube is incomplete")
    return cube


def robust_bedroc(
    ranks: np.ndarray, labels: np.ndarray, subset: tuple[int, ...], alpha: float
) -> float:
    return float(bedroc_metrics(ranks, labels, subset, alpha)["robust_bedroc_composite"])


def majority_hit(ranks: np.ndarray, subset: tuple[int, ...], threshold: float) -> np.ndarray:
    per_seed = np.any(ranks[:, :, subset] <= threshold, axis=2)
    return np.sum(per_seed, axis=0) >= 2


def best_subset(
    ranks: np.ndarray,
    labels: np.ndarray,
    subsets: list[tuple[int, ...]],
    alpha: float,
) -> tuple[int, ...]:
    return min(subsets, key=lambda subset: (-robust_bedroc(ranks, labels, subset, alpha), subset))


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return 0.0
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else 0.0


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, dict(config["implementation"])["runner"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage59 implementation identity differs")
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    matrix_summary = read_json(inputs["stage58b_summary"])
    matrix_audit = read_json(inputs["stage58b_audit"])
    target_amendment = read_json(inputs["stage58c_target_id_amendment"])
    target_amendment_audit = read_json(inputs["stage58c_target_id_audit"])
    preregistration = read_json(inputs["stage55_preregistration"])
    frozen_criteria = read_json(inputs["stage54_frozen_intake_criteria"])
    if matrix_summary.get("status") != "stage58b_ppard_pilot96_unidock_matrix_ok":
        raise ValueError("Stage58b matrix did not complete")
    if matrix_audit.get("status") != (
        "independent_stage58b_ppard_pilot96_unidock_matrix_audit_ok"
    ):
        raise ValueError("Stage58b independent matrix audit did not pass")
    if any(int(value) != 0 for value in dict(matrix_audit["data_boundary"]).values()):
        raise ValueError("Stage58b crossed a protected data boundary")
    if target_amendment.get("status") != "stage58c_ppard_target_id_amendment_ok" or (
        target_amendment_audit.get("status")
        != "stage58c_ppard_target_id_amendment_independent_audit_ok"
    ):
        raise ValueError("Stage58c target-id amendment was not independently audited")
    if (
        target_amendment_audit["docking_scores_changed"] != 0
        or target_amendment_audit["pose_fields_changed"] != 0
        or target_amendment_audit["target_id_after"] != ["PPARD"]
    ):
        raise ValueError("Stage58c changed more than the inherited target ID")
    if preregistration.get("experiment_id") != (
        "stage55-ppard-small-pilot-preregistration-20260805-v1"
    ):
        raise ValueError("Stage55 PPARD pilot preregistration differs")
    frozen_protocol = dict(preregistration["frozen_protocol"])
    if dict(config["analysis"]) != dict(frozen_protocol["functional_diagnosis"]):
        raise ValueError("Stage59 functional analysis differs from Stage55")
    if dict(config["gate"]) != dict(frozen_protocol["functional_gate"]):
        raise ValueError("Stage59 functional gate differs from Stage55")
    if dict(config["gate"]) != dict(frozen_criteria["criteria"]):
        raise ValueError("Stage59 functional gate differs from the Stage54 transfer record")

    outputs = {key: root / value for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage59 outputs exist; pass --overwrite")
    ligands = read_csv(inputs["ligand_manifest"])
    receptors = read_csv(inputs["receptor_manifest"])
    scores = read_csv(inputs["scores_csv"])
    ligand_ids = [row["ligand_id"] for row in ligands]
    receptor_ids = [row["conformer_id"] for row in receptors]
    labels = np.asarray([int(row["label"] == "active") for row in ligands], dtype=int)
    if (
        len(ligands) != 96
        or Counter(labels) != Counter({0: 48, 1: 48})
        or len(receptors) != 29
        or len(set(receptor_ids)) != 29
    ):
        raise ValueError("Stage59 input dimensions differ")
    if {row["split"] for row in ligands} != {"train"} or {
        row["pilot_role"] for row in ligands
    } != {"development_train_pilot"}:
        raise ValueError("Stage59 crossed the Pilot-96 boundary")
    folds = np.asarray([int(row["pilot_outer_fold"]) for row in ligands], dtype=int)
    if set(folds) != set(range(4)):
        raise ValueError("Stage59 outer-fold IDs differ")
    for fold in range(4):
        local = Counter(ligands[index]["label"] for index in np.flatnonzero(folds == fold))
        if local != Counter({"active": 12, "decoy": 12}):
            raise ValueError(f"Stage59 fold {fold} is not balanced")

    cube = build_score_cube(scores, ligand_ids, receptor_ids)
    alpha = float(config["analysis"]["bedroc_alpha"])
    threshold = float(config["analysis"]["favorable_rank_fraction"])
    ranks = rank_cube(cube, np.ones(len(ligands), dtype=bool))
    singles = [(index,) for index in range(len(receptors))]
    pairs = list(itertools.combinations(range(len(receptors)), 2))
    single_metrics = {
        subset[0]: bedroc_metrics(ranks, labels, subset, alpha) for subset in singles
    }
    single_values = {
        index: float(metrics["robust_bedroc_composite"])
        for index, metrics in single_metrics.items()
    }
    dominant_index = min(single_values, key=lambda index: (-single_values[index], index))
    dominant = (dominant_index,)
    dominant_hits = majority_hit(ranks, dominant, threshold)
    active_mask = labels == 1
    decoy_mask = labels == 0

    receptor_rows: list[dict[str, Any]] = []
    for index, receptor_id in enumerate(receptor_ids):
        pair = dominant if index == dominant_index else tuple(sorted((dominant_index, index)))
        pair_hits = majority_hit(ranks, pair, threshold)
        rescued = int(np.sum(~dominant_hits & pair_hits & active_mask))
        promoted = int(np.sum(~dominant_hits & pair_hits & decoy_mask))
        pair_value = robust_bedroc(ranks, labels, pair, alpha)
        receptor_rows.append(
            {
                "receptor_id": receptor_id,
                **{f"single_{key}": value for key, value in single_metrics[index].items()},
                "single_rank": 1
                + sorted(single_values, key=lambda item: (-single_values[item], item)).index(index),
                "is_dominant_single": index == dominant_index,
                "dominant_plus_receptor_subset": "+".join(receptor_ids[item] for item in pair),
                "dominant_plus_receptor_robust_bedroc": pair_value,
                "bedroc_delta_over_dominant": pair_value - single_values[dominant_index],
                "active_majority_hits_rescued": rescued,
                "decoy_majority_hits_promoted": promoted,
                "normalized_active_rescue": rescued / 48,
                "normalized_decoy_promotion": promoted / 48,
                "normalized_rescue_minus_promotion": (rescued - promoted) / 48,
            }
        )

    pair_rows: list[dict[str, Any]] = []
    for left, right in pairs:
        subset = (left, right)
        metrics = bedroc_metrics(ranks, labels, subset, alpha)
        better_single = max(single_values[left], single_values[right])
        baseline_index = left if single_values[left] >= single_values[right] else right
        baseline_hits = majority_hit(ranks, (baseline_index,), threshold)
        pair_hits = majority_hit(ranks, subset, threshold)
        rescued = int(np.sum(~baseline_hits & pair_hits & active_mask))
        promoted = int(np.sum(~baseline_hits & pair_hits & decoy_mask))
        correlations = [
            finite_spearman(ranks[seed, :, left], ranks[seed, :, right])
            for seed in range(3)
        ]
        pair_rows.append(
            {
                "left_receptor": receptor_ids[left],
                "right_receptor": receptor_ids[right],
                "pair": receptor_ids[left] + "+" + receptor_ids[right],
                **{f"pair_{key}": value for key, value in metrics.items()},
                "best_member_robust_bedroc": better_single,
                "pair_bedroc_gain_over_best_member": float(
                    metrics["robust_bedroc_composite"]
                ) - better_single,
                "active_majority_hits_rescued": rescued,
                "decoy_majority_hits_promoted": promoted,
                "normalized_active_rescue": rescued / 48,
                "normalized_decoy_promotion": promoted / 48,
                "normalized_rescue_minus_promotion": (rescued - promoted) / 48,
                "mean_seed_rank_spearman": statistics.fmean(correlations),
                "minimum_seed_rank_spearman": min(correlations),
            }
        )

    fold_rows: list[dict[str, Any]] = []
    for fold in range(4):
        train_mask = folds != fold
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

    best_pair_row = max(
        pair_rows, key=lambda row: float(row["pair_robust_bedroc_composite"])
    )
    additions = [row for row in receptor_rows if not row["is_dominant_single"]]
    holdout_gains = [float(row["holdout_pair_gain"]) for row in fold_rows]
    criteria = dict(config["gate"])
    observed = {
        "minimum_full_best_pair_gain": float(
            best_pair_row["pair_bedroc_gain_over_best_member"]
        ) >= float(criteria["minimum_full_best_pair_gain"]),
        "minimum_positive_holdout_pair_gain_folds": sum(value > 0 for value in holdout_gains)
        >= int(criteria["minimum_positive_holdout_pair_gain_folds"]),
        "minimum_mean_holdout_pair_gain": statistics.fmean(holdout_gains)
        >= float(criteria["minimum_mean_holdout_pair_gain"]),
        "minimum_positive_rescue_balance_additions": sum(
            float(row["normalized_rescue_minus_promotion"]) > 0 for row in additions
        ) >= int(criteria["minimum_positive_rescue_balance_additions"]),
        "maximum_median_pair_rank_redundancy": statistics.median(
            float(row["mean_seed_rank_spearman"]) for row in pair_rows
        ) <= float(criteria["maximum_median_pair_rank_redundancy"]),
    }
    gate_pass = all(observed.values())
    diagnosis = {
        "dominant_single_receptor": receptor_ids[dominant_index],
        "dominant_single_robust_bedroc": single_values[dominant_index],
        "best_pair": best_pair_row["pair"],
        "best_pair_robust_bedroc": best_pair_row["pair_robust_bedroc_composite"],
        "best_pair_gain_over_best_member": best_pair_row[
            "pair_bedroc_gain_over_best_member"
        ],
        "positive_pair_bedroc_gain_count": sum(
            float(row["pair_bedroc_gain_over_best_member"]) > 0 for row in pair_rows
        ),
        "pair_count": len(pair_rows),
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
            float(row["mean_seed_rank_spearman"]) for row in pair_rows
        ),
    }
    decision = {
        "functional_complementarity_gate_passed": gate_pass,
        "gate_checks": observed,
        "transferred_qubo_objective_freeze_authorized": gate_pass,
        "remaining_development_train_docking_authorized": gate_pass,
        "same_pilot_objective_or_threshold_retuning_authorized": False,
        "fresh_validation_authorized": False,
        "locked_test_authorized": False,
        "quantum_hardware_authorized": False,
        "route": (
            "freeze_transferred_qubo_then_dock_remaining_development_train"
            if gate_pass
            else "stop_ppard_full_development_route_and_select_next_unseen_target"
        ),
    }
    write_csv(outputs["receptor_metrics_csv"], receptor_rows)
    write_csv(outputs["pair_metrics_csv"], pair_rows)
    write_csv(outputs["fold_gate_csv"], fold_rows)
    result = {
        "schema_version": "1.0",
        "status": "stage59_ppard_functional_complementarity_gate_complete",
        "experiment_class": "prospective preregistered small-pilot gate",
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, implementation),
        "matrix_audit": descriptor(root, inputs["stage58b_audit"]),
        "coverage": {
            "receptor_count": len(receptors),
            "ligand_count": len(ligands),
            "seed_count": 3,
            "single_receptor_count": len(singles),
            "pair_count": len(pairs),
            "outer_fold_count": 4,
        },
        "diagnosis": diagnosis,
        "gate": {"criteria": criteria, "observed_checks": observed, "all_checks_required": True},
        "decision": decision,
        "data_boundary": {
            "pilot_train_rows_read": 96,
            "remaining_development_train_rows_read": 0,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "outputs": {
            "receptor_metrics_csv": descriptor(root, outputs["receptor_metrics_csv"]),
            "pair_metrics_csv": descriptor(root, outputs["pair_metrics_csv"]),
            "fold_gate_csv": descriptor(root, outputs["fold_gate_csv"]),
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    check_rows = [
        ("Full best-pair gain", diagnosis["best_pair_gain_over_best_member"], criteria["minimum_full_best_pair_gain"], observed["minimum_full_best_pair_gain"]),
        ("Positive holdout folds", diagnosis["positive_holdout_pair_gain_fold_count"], criteria["minimum_positive_holdout_pair_gain_folds"], observed["minimum_positive_holdout_pair_gain_folds"]),
        ("Mean holdout pair gain", diagnosis["mean_holdout_pair_gain"], criteria["minimum_mean_holdout_pair_gain"], observed["minimum_mean_holdout_pair_gain"]),
        ("Positive rescue-balance additions", diagnosis["positive_rescue_balance_dominant_addition_count"], criteria["minimum_positive_rescue_balance_additions"], observed["minimum_positive_rescue_balance_additions"]),
        ("Median pair rank redundancy", diagnosis["median_pair_rank_spearman"], criteria["maximum_median_pair_rank_redundancy"], observed["maximum_median_pair_rank_redundancy"]),
    ]
    report = [
        "# Stage59 PPARD functional-complementarity gate",
        "",
        f"Dominant single: {diagnosis['dominant_single_receptor']} ({diagnosis['dominant_single_robust_bedroc']:.6f}).",
        f"Best pair: {diagnosis['best_pair']} ({float(diagnosis['best_pair_robust_bedroc']):.6f}).",
        "",
        "| Frozen check | Observed | Threshold | Result |",
        "|---|---:|---:|---|",
    ]
    for name, value, threshold_value, passed in check_rows:
        report.append(
            f"| {name} | {float(value):.6f} | {float(threshold_value):.6f} | "
            f"{'PASS' if passed else 'FAIL'} |"
        )
    report.extend(
        [
            "",
            f"Overall preregistered gate: **{'PASS' if gate_pass else 'NO-GO'}**.",
            "",
            config["interpretation_boundary"],
            "",
        ]
    )
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report), encoding="ascii")
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
