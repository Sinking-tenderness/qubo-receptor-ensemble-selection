"""Freeze the Stage32b best PPARG MD singleton and pair using training only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.stage32b_common import (
    SCENARIOS,
    SEEDS,
    descriptor,
    load_score_matrices,
    normalize_from_train,
    read_csv,
    read_json,
    vectorized_bedroc,
)


def subset_metrics(
    normalized: dict[str, np.ndarray],
    labels: np.ndarray,
    subsets: list[tuple[int, ...]],
    alpha: float,
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for scenario in SCENARIOS:
        scores = np.stack([normalized[scenario][:, subset].min(axis=1) for subset in subsets], axis=1)
        values[scenario] = vectorized_bedroc(scores, labels, alpha)
    seed_values = np.vstack([values[seed] for seed in SEEDS])
    values["mean_seed"] = seed_values.mean(axis=0)
    values["worst_seed"] = seed_values.min(axis=0)
    values["robust"] = (values["primary"] + values["mean_seed"] + values["worst_seed"]) / 3.0
    return values


def best_index(subsets: list[tuple[int, ...]], values: np.ndarray, receptor_ids: list[str], tolerance: float = 1e-12) -> int:
    best = 0
    for index in range(1, len(subsets)):
        current_name = tuple(sorted(receptor_ids[value] for value in subsets[index]))
        best_name = tuple(sorted(receptor_ids[value] for value in subsets[best]))
        if values[index] > values[best] + tolerance or (abs(values[index] - values[best]) <= tolerance and current_name < best_name):
            best = index
    return best


def select(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    audit = read_json(root / config["inputs"]["stage32a_audit"])
    if audit.get("status") != "stage32a_pparg_md_functional_landscape_audit_ok":
        raise ValueError("Stage32a audit gate differs")
    ligands = read_csv(root / config["inputs"]["stage32_train_ligand_manifest"])
    receptors = read_csv(root / config["inputs"]["stage32_prepared_receptor_manifest"])
    receptor_ids = [row["conformer_id"] for row in receptors]
    labels = np.asarray([int(row["label"] == "active") for row in ligands], dtype=np.int8)
    matrices = load_score_matrices(
        root / config["inputs"]["stage32_median_matrix_csv"],
        root / config["inputs"]["stage32_minimum_matrix_csv"],
        root / config["inputs"]["stage32_scores_csv"],
        ligands,
        receptor_ids,
    )
    normalized = {scenario: normalize_from_train(matrix) for scenario, matrix in matrices.items()}
    singles = [(index,) for index in range(len(receptor_ids))]
    pairs = [(first, second) for first in range(len(receptor_ids)) for second in range(first + 1, len(receptor_ids))]
    alpha = float(config["train_selection"]["bedroc_alpha"])
    single_values = subset_metrics(normalized, labels, singles, alpha)
    pair_values = subset_metrics(normalized, labels, pairs, alpha)
    single_index = best_index(singles, single_values["robust"], receptor_ids)
    pair_index = best_index(pairs, pair_values["robust"], receptor_ids)
    selected_single = singles[single_index]
    selected_pair = pairs[pair_index]

    def metric_record(values: dict[str, np.ndarray], index: int) -> dict[str, float]:
        return {key: float(values[key][index]) for key in ("primary", "sensitivity", "seed0", "seed1", "seed2", "mean_seed", "worst_seed", "robust")}

    output = root / config["outputs"]["train_selection_json"]
    result = {
        "schema_version": "1.0",
        "status": "stage32b_pparg_md_pair_train_selection_frozen",
        "experiment_id": config["experiment_id"],
        "config": descriptor(root, config_path),
        "training_boundary": {"train_rows_read": len(ligands), "fresh_validation_score_rows_read": 0, "locked_test_rows_read": 0},
        "coverage": {"receptor_count": len(receptor_ids), "singleton_count": len(singles), "pair_count": len(pairs), "active_count": int(labels.sum()), "decoy_count": int(len(labels) - labels.sum())},
        "selected_singleton": {
            "receptor_ids": [receptor_ids[index] for index in selected_single],
            "metrics": metric_record(single_values, single_index),
        },
        "selected_pair": {
            "receptor_ids": sorted(receptor_ids[index] for index in selected_pair),
            "metrics": metric_record(pair_values, pair_index),
        },
        "train_pair_minus_single_robust": float(pair_values["robust"][pair_index] - single_values["robust"][single_index]),
        "next_gate": "prepare the untouched Stage19a PPARG fresh-validation panel and execute only the frozen selected pair across three seeds",
        "decision_boundary": config["decision_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage32b_pparg_md_pair_fresh_validation.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    select(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
