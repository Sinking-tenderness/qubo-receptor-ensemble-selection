"""Independently audit the Stage32b PPARG MD-pair fresh validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.stage32b_common import descriptor, read_csv, read_json, sha256


SCENARIOS = ("primary", "sensitivity", "seed0", "seed1", "seed2")
SEEDS = ("seed0", "seed1", "seed2")


def scalar_bedroc(scores: np.ndarray, labels: np.ndarray, alpha: float) -> float:
    total = len(labels)
    active_total = int(labels.sum())
    order = np.argsort(scores, kind="stable")
    weights = np.exp(-alpha * np.arange(1, total + 1, dtype=float) / total)
    random_expected = active_total * float(weights.mean())
    observed = float(np.sum(labels[order] * weights)) / random_expected
    maximum = float(weights[:active_total].sum()) / random_expected
    minimum = float(weights[-active_total:].sum()) / random_expected
    return (observed - minimum) / (maximum - minimum)


def frozen_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    for column in range(reference.shape[1]):
        ordered = np.sort(reference[:, column], kind="stable")
        left = np.searchsorted(ordered, values[:, column], side="left")
        right = np.searchsorted(ordered, values[:, column], side="right")
        output[:, column] = (left + 0.5 * (right - left) + 0.5) / (len(ordered) + 1.0)
    return output


def matrix_from_rows(rows: list[dict[str, str]], ligand_ids: list[str], receptor_ids: list[str]) -> np.ndarray:
    by_id = {row["ligand_id"]: row for row in rows}
    return np.asarray([[float(by_id[ligand_id][receptor]) for receptor in receptor_ids] for ligand_id in ligand_ids], dtype=float)


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    summary_path = root / config["outputs"]["summary_json"]
    result_path = root / config["outputs"]["evaluation_json"]
    summary = read_json(summary_path)
    result = read_json(result_path)
    if summary.get("status") != "stage32b_pparg_md_pair_fresh_validation_matrix_ok":
        raise ValueError("unexpected Stage32b matrix status")
    if result.get("status") != "stage32b_pparg_md_pair_fresh_validation_evaluation_complete":
        raise ValueError("unexpected Stage32b evaluation status")
    if summary["config"]["sha256"] != sha256(config_path) or result["config"]["sha256"] != sha256(config_path):
        raise ValueError("Stage32b frozen config hash differs")
    for record in summary["outputs"].values():
        path = root / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"] or path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"Stage32b descriptor differs: {record['path']}")

    batch_rows = read_csv(root / config["outputs"]["batch_runs_csv"])
    score_rows = read_csv(root / config["outputs"]["scores_csv"])
    if len(batch_rows) != 6 or Counter(row["status"] for row in batch_rows) != Counter({"ok": 6}):
        raise ValueError("Stage32b batch coverage differs")
    if sum(int(row["pose_integrity_failure_count"]) for row in batch_rows) != 0 or sum(int(row["unresolved_warning_event_count"]) for row in batch_rows) != 0:
        raise ValueError("Stage32b technical integrity gate differs")
    keys = {(row["seed_id"], row["receptor_id"], row["ligand_id"]) for row in score_rows}
    if len(score_rows) != 9456 or len(keys) != 9456:
        raise ValueError("Stage32b score coverage differs")

    validation_ligands = read_csv(root / config["outputs"]["prepared_ligand_manifest"])
    ligand_ids = [row["ligand_id"] for row in validation_ligands]
    labels = np.asarray([int(row["label"] == "active") for row in validation_ligands], dtype=np.int8)
    receptor_ids = list(result["frozen_selection"]["pair_receptors"])
    score_map = {(row["seed_id"], row["ligand_id"], row["receptor_id"]): float(row["gpu_score"]) for row in score_rows}
    validation_seed = {
        seed: np.asarray([[score_map[(seed, ligand, receptor)] for receptor in receptor_ids] for ligand in ligand_ids], dtype=float)
        for seed in SEEDS
    }
    cube = np.stack([validation_seed[seed] for seed in SEEDS])
    validation = {
        "primary": np.median(cube, axis=0),
        "sensitivity": np.min(cube, axis=0),
        **validation_seed,
    }
    reported_primary = matrix_from_rows(read_csv(root / config["outputs"]["median_matrix_csv"]), ligand_ids, receptor_ids)
    reported_minimum = matrix_from_rows(read_csv(root / config["outputs"]["minimum_matrix_csv"]), ligand_ids, receptor_ids)
    maximum_matrix_difference = max(float(np.max(np.abs(validation["primary"] - reported_primary))), float(np.max(np.abs(validation["sensitivity"] - reported_minimum))))

    train_ligands = read_csv(root / config["inputs"]["stage32_train_ligand_manifest"])
    train_ids = [row["ligand_id"] for row in train_ligands]
    train_scores = read_csv(root / config["inputs"]["stage32_scores_csv"])
    train_map = {(row["seed_id"], row["ligand_id"], row["receptor_id"]): float(row["gpu_score"]) for row in train_scores}
    train_seed = {
        seed: np.asarray([[train_map[(seed, ligand, receptor)] for receptor in receptor_ids] for ligand in train_ids], dtype=float)
        for seed in SEEDS
    }
    train = {
        "primary": matrix_from_rows(read_csv(root / config["inputs"]["stage32_median_matrix_csv"]), train_ids, receptor_ids),
        "sensitivity": matrix_from_rows(read_csv(root / config["inputs"]["stage32_minimum_matrix_csv"]), train_ids, receptor_ids),
        **train_seed,
    }
    single_column = receptor_ids.index(result["frozen_selection"]["single_receptor"])
    alpha = float(config["train_selection"]["bedroc_alpha"])
    maximum_metric_difference = 0.0
    primary_vectors = None
    for scenario in SCENARIOS:
        normalized = frozen_cdf(train[scenario], validation[scenario])
        single = normalized[:, single_column]
        pair = normalized.min(axis=1)
        recomputed = {
            "single_bedroc20": scalar_bedroc(single, labels, alpha),
            "pair_bedroc20": scalar_bedroc(pair, labels, alpha),
        }
        recomputed["pair_minus_single_bedroc20"] = recomputed["pair_bedroc20"] - recomputed["single_bedroc20"]
        maximum_metric_difference = max(maximum_metric_difference, *(abs(recomputed[key] - float(result["scenario_metrics"][scenario][key])) for key in recomputed))
        if scenario == "primary":
            primary_vectors = (single, pair)

    single, pair = primary_vectors
    seed_value = int.from_bytes(hashlib.sha256(b"stage32b-pparg-md-pair-bootstrap-v1").digest()[:8], "big")
    rng = np.random.default_rng(seed_value)
    active = np.flatnonzero(labels == 1)
    decoy = np.flatnonzero(labels == 0)
    gains = np.empty(2000)
    for replicate in range(2000):
        sample = np.concatenate((rng.choice(active, len(active), replace=True), rng.choice(decoy, len(decoy), replace=True)))
        sample = sample[rng.permutation(len(sample))]
        gains[replicate] = scalar_bedroc(pair[sample], labels[sample], alpha) - scalar_bedroc(single[sample], labels[sample], alpha)
    lower, upper = np.quantile(gains, [0.025, 0.975])
    maximum_bootstrap_difference = max(abs(float(lower) - float(result["paired_bootstrap"]["ci95_lower"])), abs(float(upper) - float(result["paired_bootstrap"]["ci95_upper"])), abs(float(gains.mean()) - float(result["paired_bootstrap"]["mean_gain"])))
    expected_conditions = {
        "primary_gain_at_least_0_02": result["scenario_metrics"]["primary"]["pair_minus_single_bedroc20"] >= 0.02,
        "at_least_two_positive_individual_seeds": sum(result["scenario_metrics"][seed]["pair_minus_single_bedroc20"] > 0 for seed in SEEDS) >= 2,
        "bootstrap_95ci_lower_bound_above_zero": float(lower) > 0,
    }
    if expected_conditions != result["confirmation_gate"]["conditions"] or bool(all(expected_conditions.values())) != result["confirmation_gate"]["passed"]:
        raise ValueError("Stage32b confirmation decision differs")
    checks = {
        "config_and_output_descriptors_verified": True,
        "all_six_batches_verified": True,
        "all_9456_unique_scores_verified": True,
        "pose_and_warning_integrity_verified": True,
        "median_and_minimum_matrices_recomputed": maximum_matrix_difference == 0.0,
        "all_five_bedroc_scenarios_recomputed": maximum_metric_difference <= 1e-12,
        "all_2000_bootstrap_replicates_recomputed": maximum_bootstrap_difference <= 1e-12,
        "confirmation_gate_recomputed": True,
        "locked_test_and_hardware_boundary_verified": int(result["data_boundary"]["locked_test_rows_read"]) == 0 and int(result["data_boundary"]["quantum_hardware_jobs"]) == 0,
    }
    if not all(checks.values()):
        raise ValueError(f"Stage32b audit failed: {checks}")
    audit_result = {
        "schema_version": "1.0",
        "status": "stage32b_pparg_md_pair_fresh_validation_audit_ok",
        "config": descriptor(root, config_path),
        "matrix_summary": descriptor(root, summary_path),
        "evaluation_result": descriptor(root, result_path),
        "checks": checks,
        "coverage": {"batch_count": 6, "score_row_count": 9456, "fresh_validation_ligand_count": 1576, "bootstrap_replicate_count": 2000, "maximum_matrix_difference": maximum_matrix_difference, "maximum_metric_difference": maximum_metric_difference, "maximum_bootstrap_difference": maximum_bootstrap_difference},
        "decision": result["decision"],
        "data_boundary": result["data_boundary"],
        "decision_boundary": config["decision_boundary"],
    }
    output = root / "data/stage32b_pparg_md_pair_fresh_validation_audit.json"
    output.write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(audit_result, indent=2, sort_keys=True))
    return audit_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage32b_pparg_md_pair_fresh_validation.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    audit(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
