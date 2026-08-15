"""Diagnose the failed Stage32b PPARG MD-pair confirmation without retuning."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.stage32b_common import SCENARIOS, SEEDS, descriptor, load_score_matrices, read_csv, read_json, vectorized_bedroc, write_csv


def transform(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    for column in range(reference.shape[1]):
        frozen = np.sort(reference[:, column], kind="stable")
        left = np.searchsorted(frozen, values[:, column], side="left")
        right = np.searchsorted(frozen, values[:, column], side="right")
        output[:, column] = (left + 0.5 * (right - left) + 0.5) / (len(frozen) + 1.0)
    return output


def stable_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def bedroc(scores: np.ndarray, labels: np.ndarray, alpha: float) -> float:
    return float(vectorized_bedroc(scores[:, None], labels, alpha)[0])


def summarize_split(
    split: str,
    scenario: str,
    normalized: np.ndarray,
    labels: np.ndarray,
    single_column: int,
    alpha: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    single = normalized[:, single_column]
    extra = normalized[:, 1 - single_column]
    pair = np.minimum(single, extra)
    improvement = single - pair
    extra_wins = improvement > 1e-12
    single_rank = stable_ranks(single)
    pair_rank = stable_ranks(pair)
    rank_shift = pair_rank - single_rank
    single_bedroc = bedroc(single, labels, alpha)
    pair_bedroc = bedroc(pair, labels, alpha)
    row: dict[str, Any] = {
        "split": split,
        "scenario": scenario,
        "row_count": len(labels),
        "active_count": int(labels.sum()),
        "decoy_count": int(len(labels) - labels.sum()),
        "single_bedroc20": single_bedroc,
        "pair_bedroc20": pair_bedroc,
        "pair_minus_single_bedroc20": pair_bedroc - single_bedroc,
    }
    for label_name, label_value in (("active", 1), ("decoy", 0)):
        mask = labels == label_value
        row[f"{label_name}_extra_win_count"] = int(extra_wins[mask].sum())
        row[f"{label_name}_extra_win_rate"] = float(extra_wins[mask].mean())
        row[f"{label_name}_mean_fraction_improvement"] = float(improvement[mask].mean())
        row[f"{label_name}_median_fraction_improvement"] = float(np.median(improvement[mask]))
        row[f"{label_name}_mean_rank_shift"] = float(rank_shift[mask].mean())
        row[f"{label_name}_median_rank_shift"] = float(np.median(rank_shift[mask]))
    row["active_minus_decoy_extra_win_rate"] = row["active_extra_win_rate"] - row["decoy_extra_win_rate"]
    row["active_minus_decoy_mean_fraction_improvement"] = row["active_mean_fraction_improvement"] - row["decoy_mean_fraction_improvement"]
    for fraction in (0.01, 0.05):
        cutoff = max(1, int(math.ceil(len(labels) * fraction)))
        key = f"top{int(fraction * 100)}pct"
        single_top = single_rank <= cutoff
        pair_top = pair_rank <= cutoff
        row[f"{key}_cutoff"] = cutoff
        row[f"single_{key}_active_count"] = int(np.sum(single_top & (labels == 1)))
        row[f"pair_{key}_active_count"] = int(np.sum(pair_top & (labels == 1)))
        row[f"pair_{key}_new_decoy_count"] = int(np.sum(pair_top & ~single_top & (labels == 0)))
        row[f"pair_{key}_lost_active_count"] = int(np.sum(single_top & ~pair_top & (labels == 1)))
    vectors = {"single": single, "extra": extra, "pair": pair, "improvement": improvement, "extra_wins": extra_wins, "single_rank": single_rank, "pair_rank": pair_rank, "rank_shift": rank_shift}
    return row, vectors


def diagnose(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    stage32b = read_json(root / config["inputs"]["stage32b_result"])
    audit = read_json(root / config["inputs"]["stage32b_audit"])
    selection = read_json(root / config["inputs"]["train_selection"])
    if stage32b.get("status") != "stage32b_pparg_md_pair_fresh_validation_evaluation_complete" or audit.get("status") != "stage32b_pparg_md_pair_fresh_validation_audit_ok":
        raise ValueError("Stage32b result or audit gate differs")
    if stage32b["confirmation_gate"]["passed"]:
        raise ValueError("Stage32c is only valid after a failed Stage32b gate")
    pair_ids = list(selection["selected_pair"]["receptor_ids"])
    single_id = selection["selected_singleton"]["receptor_ids"][0]
    single_column = pair_ids.index(single_id)
    train_ligands = read_csv(root / config["inputs"]["train_ligand_manifest"])
    validation_ligands = read_csv(root / config["inputs"]["validation_ligand_manifest"])
    all_receptors = [row["conformer_id"] for row in read_csv(root / config["inputs"]["train_receptor_manifest"])]
    train_all = load_score_matrices(
        root / config["inputs"]["train_median_matrix_csv"],
        root / config["inputs"]["train_minimum_matrix_csv"],
        root / config["inputs"]["train_scores_csv"],
        train_ligands,
        all_receptors,
    )
    pair_columns = [all_receptors.index(value) for value in pair_ids]
    train = {scenario: matrix[:, pair_columns] for scenario, matrix in train_all.items()}
    validation = load_score_matrices(
        root / config["inputs"]["validation_median_matrix_csv"],
        root / config["inputs"]["validation_minimum_matrix_csv"],
        root / config["inputs"]["validation_scores_csv"],
        validation_ligands,
        pair_ids,
    )
    train_labels = np.asarray([int(row["label"] == "active") for row in train_ligands], dtype=np.int8)
    validation_labels = np.asarray([int(row["label"] == "active") for row in validation_ligands], dtype=np.int8)
    alpha = float(config["diagnostic"]["bedroc_alpha"])
    summary_rows = []
    validation_primary_vectors = None
    for split, labels, matrices in (("train", train_labels, train), ("validation", validation_labels, validation)):
        for scenario in SCENARIOS:
            normalized = transform(train[scenario], matrices[scenario])
            row, vectors = summarize_split(split, scenario, normalized, labels, single_column, alpha)
            summary_rows.append(row)
            if split == "validation" and scenario == "primary":
                validation_primary_vectors = vectors
    by_key = {(row["split"], row["scenario"]): row for row in summary_rows}
    train_primary = by_key[("train", "primary")]
    validation_primary = by_key[("validation", "primary")]
    all_seed_gains_negative = all(by_key[("validation", seed)]["pair_minus_single_bedroc20"] < 0 for seed in SEEDS)
    win_rate_not_active_selective = validation_primary["active_minus_decoy_extra_win_rate"] <= 0
    mean_improvement_not_active_selective = validation_primary["active_minus_decoy_mean_fraction_improvement"] <= 0
    nonselective_decoy_promotion = validation_primary["pair_minus_single_bedroc20"] < 0 and all_seed_gains_negative and (win_rate_not_active_selective or mean_improvement_not_active_selective)
    stop_route = nonselective_decoy_promotion
    vectors = validation_primary_vectors
    ligand_rows = []
    for index, ligand in enumerate(validation_ligands):
        ligand_rows.append({
            "ligand_id": ligand["ligand_id"],
            "label": ligand["label"],
            "split_group_id": ligand["split_group_id"],
            "single_fraction": float(vectors["single"][index]),
            "extra_fraction": float(vectors["extra"][index]),
            "pair_fraction": float(vectors["pair"][index]),
            "extra_receptor_wins": bool(vectors["extra_wins"][index]),
            "fraction_improvement": float(vectors["improvement"][index]),
            "single_rank": int(vectors["single_rank"][index]),
            "pair_rank": int(vectors["pair_rank"][index]),
            "pair_minus_single_rank": int(vectors["rank_shift"][index]),
            "single_top1pct": bool(vectors["single_rank"][index] <= 16),
            "pair_top1pct": bool(vectors["pair_rank"][index] <= 16),
            "single_top5pct": bool(vectors["single_rank"][index] <= 79),
            "pair_top5pct": bool(vectors["pair_rank"][index] <= 79),
        })
    outputs = {key: root / value for key, value in config["outputs"].items()}
    write_csv(outputs["ligand_diagnostic_csv"], ligand_rows)
    write_csv(outputs["scenario_summary_csv"], summary_rows)
    result = {
        "schema_version": "1.0",
        "status": "stage32c_pparg_md_pair_failure_diagnostic_complete",
        "experiment_id": config["experiment_id"],
        "config": descriptor(root, config_path),
        "frozen_selection": {"single_receptor": single_id, "pair_receptors": pair_ids},
        "primary_train": train_primary,
        "primary_validation": validation_primary,
        "generalization": {
            "train_pair_minus_single_bedroc20": train_primary["pair_minus_single_bedroc20"],
            "validation_pair_minus_single_bedroc20": validation_primary["pair_minus_single_bedroc20"],
            "validation_minus_train_gain": validation_primary["pair_minus_single_bedroc20"] - train_primary["pair_minus_single_bedroc20"],
            "all_three_validation_seed_gains_negative": all_seed_gains_negative,
        },
        "mechanism": {
            "win_rate_not_active_selective": win_rate_not_active_selective,
            "mean_improvement_not_active_selective": mean_improvement_not_active_selective,
            "nonselective_decoy_promotion": nonselective_decoy_promotion,
        },
        "decision": {
            "stop_frozen_min_aggregation_md_pair_efficacy_route": stop_route,
            "replacement_tuning_on_validation_authorized": False,
            "locked_test_authorized": False,
            "new_docking_authorized": False,
            "quantum_hardware_authorized": False,
        },
        "outputs": {key: descriptor(root, path) for key, path in outputs.items() if key in {"ligand_diagnostic_csv", "scenario_summary_csv"}},
        "data_boundary": {"train_rows_read": 160, "fresh_validation_rows_read": 1576, "locked_test_rows_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0},
        "interpretation_boundary": config["interpretation_boundary"],
    }
    outputs["result_json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["result_json"].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    report = [
        "# Stage 32c: PPARG MD-pair failure diagnostic",
        "",
        "| Quantity | Train | Fresh validation |",
        "|---|---:|---:|",
        f"| Pair minus single BEDROC20 | {train_primary['pair_minus_single_bedroc20']:+.6f} | {validation_primary['pair_minus_single_bedroc20']:+.6f} |",
        f"| Active extra-receptor win rate | {train_primary['active_extra_win_rate']:.3f} | {validation_primary['active_extra_win_rate']:.3f} |",
        f"| Decoy extra-receptor win rate | {train_primary['decoy_extra_win_rate']:.3f} | {validation_primary['decoy_extra_win_rate']:.3f} |",
        f"| Active mean fraction improvement | {train_primary['active_mean_fraction_improvement']:.4f} | {validation_primary['active_mean_fraction_improvement']:.4f} |",
        f"| Decoy mean fraction improvement | {train_primary['decoy_mean_fraction_improvement']:.4f} | {validation_primary['decoy_mean_fraction_improvement']:.4f} |",
        "",
        f"Nonselective decoy promotion: **{'YES' if nonselective_decoy_promotion else 'NO'}**.",
        f"Stop frozen min-aggregation MD-pair efficacy route: **{'YES' if stop_route else 'NO'}**.",
        "",
        config["interpretation_boundary"],
    ]
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage32c_pparg_md_pair_failure_diagnostic.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    diagnose(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
