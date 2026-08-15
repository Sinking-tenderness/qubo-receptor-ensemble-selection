"""Apply the frozen Stage32b PPARG MD-pair confirmation analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.stage32b_common import SCENARIOS, SEEDS, descriptor, load_score_matrices, read_csv, read_json, vectorized_bedroc


def transform_with_reference(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    denominator = reference.shape[0] + 1.0
    for receptor in range(reference.shape[1]):
        frozen = np.sort(reference[:, receptor], kind="stable")
        left = np.searchsorted(frozen, values[:, receptor], side="left")
        right = np.searchsorted(frozen, values[:, receptor], side="right")
        output[:, receptor] = (left + 0.5 * (right - left) + 0.5) / denominator
    return output


def load_validation_matrices(root: Path, config: dict[str, Any], ligands: list[dict[str, str]], receptor_ids: list[str]) -> dict[str, np.ndarray]:
    ligand_index = {row["ligand_id"]: index for index, row in enumerate(ligands)}
    receptor_index = {value: index for index, value in enumerate(receptor_ids)}
    matrices = {scenario: np.full((len(ligands), len(receptor_ids)), np.nan) for scenario in SCENARIOS}
    for scenario, key in (("primary", "median_matrix_csv"), ("sensitivity", "minimum_matrix_csv")):
        for row in read_csv(root / config["outputs"][key]):
            matrices[scenario][ligand_index[row["ligand_id"]]] = [float(row[value]) for value in receptor_ids]
    seen = set()
    for row in read_csv(root / config["outputs"]["scores_csv"]):
        key = (row["seed_id"], row["ligand_id"], row["receptor_id"])
        if key in seen:
            raise ValueError("duplicate Stage32b score key")
        seen.add(key)
        matrices[row["seed_id"]][ligand_index[row["ligand_id"]], receptor_index[row["receptor_id"]]] = float(row["gpu_score"])
    if len(seen) != 9456 or any(not np.all(np.isfinite(matrix)) for matrix in matrices.values()):
        raise ValueError("Stage32b validation matrix coverage differs")
    return matrices


def bedroc(scores: np.ndarray, labels: np.ndarray, alpha: float) -> float:
    return float(vectorized_bedroc(scores[:, None], labels, alpha)[0])


def enrichment_factor(scores: np.ndarray, labels: np.ndarray, fraction: float) -> float:
    cutoff = max(1, int(math.ceil(len(labels) * fraction)))
    order = np.argsort(scores, kind="stable")[:cutoff]
    return float(labels[order].mean() / labels.mean())


def bootstrap_gain(single_scores: np.ndarray, pair_scores: np.ndarray, labels: np.ndarray, alpha: float, count: int, seed_text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("ascii")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    active = np.flatnonzero(labels == 1)
    decoy = np.flatnonzero(labels == 0)
    gains = np.empty(count, dtype=float)
    for replicate in range(count):
        sample = np.concatenate((rng.choice(active, len(active), replace=True), rng.choice(decoy, len(decoy), replace=True)))
        sample = sample[rng.permutation(len(sample))]
        sample_labels = labels[sample]
        gains[replicate] = bedroc(pair_scores[sample], sample_labels, alpha) - bedroc(single_scores[sample], sample_labels, alpha)
    return gains


def evaluate(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    summary = read_json(root / config["outputs"]["summary_json"])
    if summary.get("status") != "stage32b_pparg_md_pair_fresh_validation_matrix_ok":
        raise ValueError("Stage32b matrix is not complete")
    selection = read_json(root / config["outputs"]["train_selection_json"])
    pair_ids = list(selection["selected_pair"]["receptor_ids"])
    single_id = selection["selected_singleton"]["receptor_ids"][0]
    if single_id not in pair_ids:
        raise ValueError("Stage32b frozen singleton is outside selected pair")

    train_ligands = read_csv(root / config["inputs"]["stage32_train_ligand_manifest"])
    all_receptors = read_csv(root / config["inputs"]["stage32_prepared_receptor_manifest"])
    all_receptor_ids = [row["conformer_id"] for row in all_receptors]
    train_all = load_score_matrices(
        root / config["inputs"]["stage32_median_matrix_csv"],
        root / config["inputs"]["stage32_minimum_matrix_csv"],
        root / config["inputs"]["stage32_scores_csv"],
        train_ligands,
        all_receptor_ids,
    )
    columns = [all_receptor_ids.index(value) for value in pair_ids]
    train = {scenario: matrix[:, columns] for scenario, matrix in train_all.items()}
    validation_ligands = read_csv(root / config["outputs"]["prepared_ligand_manifest"])
    labels = np.asarray([int(row["label"] == "active") for row in validation_ligands], dtype=np.int8)
    validation_raw = load_validation_matrices(root, config, validation_ligands, pair_ids)
    validation = {scenario: transform_with_reference(train[scenario], matrix) for scenario, matrix in validation_raw.items()}
    single_column = pair_ids.index(single_id)
    alpha = float(config["train_selection"]["bedroc_alpha"])
    scenario_metrics: dict[str, dict[str, float]] = {}
    score_vectors: dict[str, dict[str, np.ndarray]] = {}
    for scenario in SCENARIOS:
        single_scores = validation[scenario][:, single_column]
        pair_scores = validation[scenario].min(axis=1)
        score_vectors[scenario] = {"single": single_scores, "pair": pair_scores}
        single_value = bedroc(single_scores, labels, alpha)
        pair_value = bedroc(pair_scores, labels, alpha)
        scenario_metrics[scenario] = {"single_bedroc20": single_value, "pair_bedroc20": pair_value, "pair_minus_single_bedroc20": pair_value - single_value}
    primary_single = score_vectors["primary"]["single"]
    primary_pair = score_vectors["primary"]["pair"]
    secondary = {
        "single_ef1pct": enrichment_factor(primary_single, labels, 0.01),
        "pair_ef1pct": enrichment_factor(primary_pair, labels, 0.01),
        "single_ef5pct": enrichment_factor(primary_single, labels, 0.05),
        "pair_ef5pct": enrichment_factor(primary_pair, labels, 0.05),
    }
    bootstrap_count = 2000
    gains = bootstrap_gain(primary_single, primary_pair, labels, alpha, bootstrap_count, "stage32b-pparg-md-pair-bootstrap-v1")
    lower, upper = np.quantile(gains, [0.025, 0.975]).tolist()
    positive_seeds = sum(scenario_metrics[seed]["pair_minus_single_bedroc20"] > 0 for seed in SEEDS)
    gate = config["confirmation_analysis"]["confirmation_gate"]
    conditions = {
        "primary_gain_at_least_0_02": scenario_metrics["primary"]["pair_minus_single_bedroc20"] >= float(gate["minimum_primary_bedroc20_gain"]),
        "at_least_two_positive_individual_seeds": positive_seeds >= int(gate["minimum_positive_individual_seed_count"]),
        "bootstrap_95ci_lower_bound_above_zero": lower > float(gate["minimum_stratified_bootstrap_95ci_lower_bound"]),
    }
    passed = all(conditions.values())
    result = {
        "schema_version": "1.0",
        "status": "stage32b_pparg_md_pair_fresh_validation_evaluation_complete",
        "experiment_id": config["experiment_id"],
        "config": descriptor(root, config_path),
        "matrix_summary": descriptor(root, root / config["outputs"]["summary_json"]),
        "frozen_selection": {"single_receptor": single_id, "pair_receptors": pair_ids},
        "coverage": {"fresh_validation_ligands": len(labels), "active": int(labels.sum()), "decoy": int(len(labels) - labels.sum()), "seed_count": 3, "locked_test_rows": 0},
        "scenario_metrics": scenario_metrics,
        "secondary_primary_metrics": secondary,
        "paired_bootstrap": {"replicate_count": bootstrap_count, "mean_gain": float(gains.mean()), "median_gain": float(np.median(gains)), "ci95_lower": float(lower), "ci95_upper": float(upper)},
        "confirmation_gate": {"conditions": conditions, "positive_individual_seed_count": positive_seeds, "passed": passed},
        "decision": {"pparg_md_pair_complementarity_confirmed": passed, "locked_test_authorized": False, "stage32a_qubo_efficacy_gate_changed": False, "quantum_hardware_authorized": False},
        "data_boundary": {"train_rows_read_for_frozen_cdf": 160, "fresh_validation_rows_read": 1576, "locked_test_rows_read": 0, "quantum_hardware_jobs": 0},
        "decision_boundary": config["decision_boundary"],
    }
    output = root / config["outputs"]["evaluation_json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    report = [
        "# Stage 32b: PPARG MD pair fresh-validation confirmation",
        "",
        f"Frozen single: `{single_id}`",
        f"Frozen pair: `{' + '.join(pair_ids)}`",
        "",
        "| Scenario | Single BEDROC20 | Pair BEDROC20 | Gain |",
        "|---|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        row = scenario_metrics[scenario]
        report.append(f"| {scenario} | {row['single_bedroc20']:.6f} | {row['pair_bedroc20']:.6f} | {row['pair_minus_single_bedroc20']:+.6f} |")
    report += [
        "",
        f"Paired bootstrap 95% CI: **[{lower:+.6f}, {upper:+.6f}]**.",
        f"Confirmation gate: **{'PASS' if passed else 'NO-GO'}**.",
        "",
        config["decision_boundary"],
    ]
    report_path = root / config["outputs"]["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage32b_pparg_md_pair_fresh_validation.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    evaluate(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
