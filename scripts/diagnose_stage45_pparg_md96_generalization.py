"""Diagnose why the Stage44 PPARG MD-96 QUBO failed scaffold transfer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage42d_bace1_large_pool_qubo_screen import bedroc_metrics, rank_cube
from scripts.run_stage42f_bace1_rank_sensitive_pair_qubo import pair_coefficients, qubo_value
from scripts.run_stage44_pparg_md96_rank_sensitive_qubo import build_score_cube


SEED_IDS = ("seed0", "seed1", "seed2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def verified(root: Path, value: dict[str, Any]) -> Path:
    path = root / value["path"]
    if not path.is_file() or sha256(path) != value["sha256"].upper():
        raise ValueError(f"Stage45 input identity differs: {path}")
    return path


def safe_spearman(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


def parse_subset(value: str, receptor_index: dict[str, int]) -> tuple[int, ...]:
    return tuple(sorted(receptor_index[item] for item in value.split("+") if item))


def jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    first, second = set(left), set(right)
    return len(first & second) / len(first | second)


def sampled_subsets(receptor_count: int, config: dict[str, Any]) -> dict[int, list[tuple[int, ...]]]:
    output = {
        1: list(itertools.combinations(range(receptor_count), 1)),
        2: list(itertools.combinations(range(receptor_count), 2)),
    }
    target = min(int(config["landscape_sampling"]["k3_sample_count"]), math.comb(receptor_count, 3))
    generator = random.Random(int(config["landscape_sampling"]["seed"]))
    sampled: set[tuple[int, ...]] = set()
    while len(sampled) < target:
        sampled.add(tuple(sorted(generator.sample(range(receptor_count), 3))))
    output[3] = sorted(sampled)
    return output


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage45 implementation identity differs")
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    stage44 = read_json(inputs["stage44_result"])
    audit = read_json(inputs["stage44_audit"])
    if stage44.get("status") != "stage44_pparg_md96_rank_sensitive_qubo_complete" or audit.get("status") != "stage44_pparg_md96_rank_sensitive_qubo_independent_audit_ok":
        raise ValueError("Stage44 result or audit did not pass")
    outputs = {key: root / value for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage45 outputs exist; pass --overwrite")

    ligands = read_csv(inputs["ligand_manifest"])
    receptors = read_csv(inputs["receptor_manifest"])
    scores = read_csv(inputs["scores"])
    assignments = read_csv(inputs["fold_assignments"])
    stage44_metrics = read_csv(inputs["stage44_metrics"])
    ligand_ids = [row["ligand_id"] for row in ligands]
    receptor_ids = [row["conformer_id"] for row in receptors]
    receptor_index = {value: index for index, value in enumerate(receptor_ids)}
    labels = np.asarray([int(row["label"] == "active") for row in ligands], dtype=int)
    cube = build_score_cube(scores, ligand_ids, receptor_ids)
    fold_by_ligand = {row["ligand_id"]: int(row["outer_fold"]) for row in assignments}
    alpha = float(config["objective"]["bedroc_alpha"])
    upper = np.triu_indices(len(receptors), 1)

    coefficient_rows: list[dict[str, Any]] = []
    singleton_by_scope: dict[str, np.ndarray] = {}
    pair_by_scope: dict[str, np.ndarray] = {}
    ranks_by_scope: dict[str, np.ndarray] = {}
    masks_by_scope: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for fold in range(4):
        train_mask = np.asarray([fold_by_ligand[value] != fold for value in ligand_ids], dtype=bool)
        holdout_mask = ~train_mask
        ranks = rank_cube(cube, train_mask)
        singleton, complement = pair_coefficients(ranks[:, train_mask, :], labels[train_mask], alpha)
        key = f"fold{fold}"
        singleton_by_scope[key], pair_by_scope[key], ranks_by_scope[key] = singleton, complement, ranks
        masks_by_scope[key] = (train_mask, holdout_mask)
    full_mask = np.ones(len(ligands), dtype=bool)
    full_ranks = rank_cube(cube, full_mask)
    full_singleton, full_pair = pair_coefficients(full_ranks, labels, alpha)
    singleton_by_scope["full"], pair_by_scope["full"], ranks_by_scope["full"] = full_singleton, full_pair, full_ranks

    fold_keys = [f"fold{fold}" for fold in range(4)]
    coefficient_correlations: list[dict[str, Any]] = []
    for left, right in itertools.combinations(fold_keys, 2):
        coefficient_correlations.append({
            "left": left, "right": right,
            "singleton_spearman": safe_spearman(singleton_by_scope[left], singleton_by_scope[right]),
            "pair_spearman": safe_spearman(pair_by_scope[left][upper], pair_by_scope[right][upper]),
            "singleton_sign_agreement": float(np.mean(np.sign(singleton_by_scope[left]) == np.sign(singleton_by_scope[right]))),
            "pair_sign_agreement": float(np.mean(np.sign(pair_by_scope[left][upper]) == np.sign(pair_by_scope[right][upper]))),
        })

    for index, receptor_id in enumerate(receptor_ids):
        fold_values = [float(singleton_by_scope[key][index]) for key in fold_keys]
        fold_ranks = [int(np.where(np.argsort(-singleton_by_scope[key], kind="stable") == index)[0][0]) + 1 for key in fold_keys]
        coefficient_rows.append({
            "receptor_id": receptor_id, "full_singleton_coefficient": float(full_singleton[index]),
            "mean_fold_singleton_coefficient": statistics.fmean(fold_values),
            "fold_singleton_coefficient_sd": statistics.pstdev(fold_values),
            "mean_fold_rank": statistics.fmean(fold_ranks), "fold_rank_sd": statistics.pstdev(fold_ranks),
            **{f"fold{fold}_coefficient": fold_values[fold] for fold in range(4)},
            **{f"fold{fold}_rank": fold_ranks[fold] for fold in range(4)},
        })

    subsets = sampled_subsets(len(receptors), config)
    landscape_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    for fold in range(4):
        key = f"fold{fold}"
        singleton, complement, ranks = singleton_by_scope[key], pair_by_scope[key], ranks_by_scope[key]
        train_mask, holdout_mask = masks_by_scope[key]
        for size in (1, 2, 3):
            objective_values: list[float] = []
            train_values: list[float] = []
            holdout_values: list[float] = []
            for subset in subsets[size]:
                objective = qubo_value(subset, singleton, complement)
                train_bedroc = bedroc_metrics(ranks[:, train_mask, :], labels[train_mask], subset, alpha)["robust_bedroc_composite"]
                holdout_bedroc = bedroc_metrics(ranks[:, holdout_mask, :], labels[holdout_mask], subset, alpha)["robust_bedroc_composite"]
                objective_values.append(objective)
                train_values.append(train_bedroc)
                holdout_values.append(holdout_bedroc)
            correlation_rows.append({
                "fold": fold, "subset_size": size, "state_count": len(subsets[size]),
                "objective_vs_train_bedroc_spearman": safe_spearman(objective_values, train_values),
                "objective_vs_holdout_bedroc_spearman": safe_spearman(objective_values, holdout_values),
                "train_vs_holdout_bedroc_spearman": safe_spearman(train_values, holdout_values),
                "mean_train_bedroc": statistics.fmean(train_values),
                "mean_holdout_bedroc": statistics.fmean(holdout_values),
            })
            print(f"Stage45 fold {fold + 1}/4 k={size} complete", flush=True)

    selection_rows: list[dict[str, Any]] = []
    subsets_by_k: dict[int, dict[str, tuple[int, ...]]] = {3: {}, 6: {}}
    method_by_k = {3: "exact", 6: "strong_classical"}
    for size in (3, 6):
        method = method_by_k[size]
        full_row = next(row for row in stage44_metrics if row["scope"] == "full_data" and row["method"] == method and int(row["subset_size"]) == size)
        subsets_by_k[size]["full"] = parse_subset(full_row["selected_subset"], receptor_index)
        for fold in range(4):
            row = next(row for row in stage44_metrics if row["scope"] == "outer_holdout" and int(row["fold"]) == fold and row["method"] == method and int(row["subset_size"]) == size)
            subsets_by_k[size][f"fold{fold}"] = parse_subset(row["selected_subset"], receptor_index)
        fold_subsets = [subsets_by_k[size][key] for key in fold_keys]
        pairwise = [jaccard(left, right) for left, right in itertools.combinations(fold_subsets, 2)]
        full_overlap = [jaccard(subsets_by_k[size]["full"], value) for value in fold_subsets]
        selection_rows.append({
            "subset_size": size, "method": method,
            "mean_pairwise_fold_jaccard": statistics.fmean(pairwise),
            "mean_full_to_fold_jaccard": statistics.fmean(full_overlap),
            "minimum_full_to_fold_jaccard": min(full_overlap),
            "unique_fold_subset_count": len(set(fold_subsets)),
            "full_subset": "+".join(receptor_ids[index] for index in subsets_by_k[size]["full"]),
        })

    k6_gains = []
    for fold in range(4):
        single = next(row for row in stage44_metrics if row["scope"] == "outer_holdout" and int(row["fold"]) == fold and row["method"] == "exact" and int(row["subset_size"]) == 1)
        k6 = next(row for row in stage44_metrics if row["scope"] == "outer_holdout" and int(row["fold"]) == fold and row["method"] == "strong_classical" and int(row["subset_size"]) == 6)
        k6_gains.append(float(k6["robust_bedroc_composite"]) - float(single["robust_bedroc_composite"]))

    median_objective_holdout = statistics.median(float(row["objective_vs_holdout_bedroc_spearman"]) for row in correlation_rows)
    median_train_holdout = statistics.median(float(row["train_vs_holdout_bedroc_spearman"]) for row in correlation_rows)
    k3_selection = next(row for row in selection_rows if row["subset_size"] == 3)
    k6_selection = next(row for row in selection_rows if row["subset_size"] == 6)
    thresholds = config["diagnostic_thresholds"]
    diagnosis = {
        "rank_generalization_failure": median_objective_holdout < float(thresholds["minimum_median_objective_holdout_spearman"]),
        "train_holdout_landscape_instability": median_train_holdout < float(thresholds["minimum_median_train_holdout_spearman"]),
        "k3_selection_instability": float(k3_selection["mean_pairwise_fold_jaccard"]) < float(thresholds["minimum_mean_pairwise_fold_jaccard"]),
        "k6_exploratory_signal": {
            "mean_holdout_gain": statistics.fmean(k6_gains), "positive_fold_count": sum(value > 0 for value in k6_gains),
            "mean_pairwise_fold_jaccard": k6_selection["mean_pairwise_fold_jaccard"],
            "eligible_for_new_target_preregistration": (
                statistics.fmean(k6_gains) >= float(thresholds["minimum_k6_mean_holdout_gain"])
                and sum(value > 0 for value in k6_gains) >= int(thresholds["minimum_k6_positive_fold_count"])
                and float(k6_selection["mean_pairwise_fold_jaccard"]) >= float(thresholds["minimum_mean_pairwise_fold_jaccard"])
            ),
        },
    }
    decision = {
        "failure_mechanism_resolved": bool(diagnosis["rank_generalization_failure"] or diagnosis["train_holdout_landscape_instability"] or diagnosis["k3_selection_instability"]),
        "same_pparg_retuning_authorized": False,
        "cross_target_stability_redesign_authorized": True,
        "k6_new_target_preregistration_authorized": diagnosis["k6_exploratory_signal"]["eligible_for_new_target_preregistration"],
        "fresh_validation_authorized": False,
        "quantum_hardware_authorized": False,
    }
    write_csv(outputs["coefficient_stability_csv"], coefficient_rows)
    write_csv(outputs["coefficient_correlations_csv"], coefficient_correlations)
    write_csv(outputs["landscape_correlations_csv"], correlation_rows)
    write_csv(outputs["selection_stability_csv"], selection_rows)
    report = [
        "# Stage45 PPARG MD-96 generalization diagnosis", "",
        f"Median objective-to-holdout Spearman: {median_objective_holdout:.6f}.",
        f"Median train-to-holdout BEDROC Spearman: {median_train_holdout:.6f}.",
        f"k=3 mean pairwise fold Jaccard: {float(k3_selection['mean_pairwise_fold_jaccard']):.6f}.",
        f"k=6 exploratory mean holdout gain: {statistics.fmean(k6_gains):+.6f}; positive folds {sum(value > 0 for value in k6_gains)}/4.", "",
        f"Failure mechanism resolved: **{'YES' if decision['failure_mechanism_resolved'] else 'NO'}**.",
        f"k=6 eligible for new-target preregistration: **{'YES' if decision['k6_new_target_preregistration_authorized'] else 'NO'}**.", "",
        config["interpretation_boundary"], "",
    ]
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report), encoding="ascii")
    result = {
        "schema_version": "1.0", "status": "stage45_pparg_md96_generalization_diagnosis_complete",
        "config": descriptor(root, config_path), "implementation": descriptor(root, Path(__file__).resolve()),
        "summary": {
            "median_objective_holdout_spearman": median_objective_holdout,
            "median_train_holdout_bedroc_spearman": median_train_holdout,
            "median_singleton_fold_spearman": statistics.median(float(row["singleton_spearman"]) for row in coefficient_correlations),
            "median_pair_fold_spearman": statistics.median(float(row["pair_spearman"]) for row in coefficient_correlations),
            "k3_selection_stability": k3_selection, "k6_selection_stability": k6_selection,
        },
        "diagnosis": diagnosis, "decision": decision,
        "data_boundary": {"train_rows_read": 160, "fresh_validation_rows_read": 0, "test_rows_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0},
        "outputs": {key: descriptor(root, path) for key, path in outputs.items() if key != "result_json"},
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    print(json.dumps({"summary": result["summary"], "diagnosis": diagnosis, "decision": decision}, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage45_pparg_md96_generalization_diagnosis.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    run(config_path, root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
