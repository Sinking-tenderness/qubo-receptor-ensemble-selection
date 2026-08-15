"""Independently audit Stage66 auxiliary coverage QUBO outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any


TOLERANCE = 1e-10
ENERGY_TOLERANCE = 1e-7
SOLVER_QUBO = "auxiliary_qubo_beam_swap"
SOLVER_GREEDY = "same_objective_direct_greedy"
SOLVER_PAIR_OFF = "pair_off_baseline"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def checked(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage66 identity differs: {path}")
    if "size_bytes" in value and path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage66 size differs: {path}")
    return path


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value) == "True":
        return True
    if str(value) == "False":
        return False
    raise ValueError(f"invalid Boolean value: {value}")


def subset_set(value: str) -> set[str]:
    return {item for item in value.split("+") if item}


def pairwise_jaccard(values: list[str]) -> float:
    sets = [subset_set(value) for value in values]
    pairs = list(itertools.combinations(sets, 2))
    return (
        statistics.fmean(len(left & right) / len(left | right) for left, right in pairs)
        if pairs
        else 1.0
    )


def compare(observed: Any, expected: Any, field: str) -> None:
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not math.isclose(
            float(observed), float(expected), rel_tol=0.0, abs_tol=TOLERANCE
        ):
            raise ValueError(f"Stage66 numeric value differs: {field}")
    elif str(observed) != str(expected):
        raise ValueError(f"Stage66 value differs: {field}")


def qubo_energy(qubo: dict[str, Any], assignment: dict[str, Any]) -> float:
    value = float(qubo["constant"])
    value += sum(
        float(coefficient) * int(assignment.get(variable, 0))
        for variable, coefficient in qubo["linear"].items()
    )
    value += sum(
        float(coefficient)
        * int(assignment.get(first, 0))
        * int(assignment.get(second, 0))
        for key, coefficient in qubo["quadratic"].items()
        for first, second in [key.split("::", 1)]
    )
    return float(value)


def recompute_target_rows(
    rows: list[dict[str, str]], candidates: list[dict[str, Any]], targets: list[str]
) -> list[dict[str, Any]]:
    pair_off = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): float(
            row["holdout_robust_bedroc"]
        )
        for row in rows
        if row["solver_id"] == SOLVER_PAIR_OFF
    }
    greedy = {
        (
            row["target_id"],
            int(row["outer_fold"]),
            row["candidate_id"],
            int(row["subset_size"]),
        ): row
        for row in rows
        if row["solver_id"] == SOLVER_GREEDY
    }
    output: list[dict[str, Any]] = []
    for target_id in targets:
        for candidate in candidates:
            selected = [
                row
                for row in rows
                if row["target_id"] == target_id
                and row["candidate_id"] == candidate["candidate_id"]
                and row["solver_id"] == SOLVER_QUBO
                and int(row["subset_size"]) >= 2
            ]
            pair_gains = [
                float(row["holdout_robust_bedroc"])
                - pair_off[
                    (target_id, int(row["outer_fold"]), int(row["subset_size"]))
                ]
                for row in selected
            ]
            greedy_rows = [
                greedy[
                    (
                        target_id,
                        int(row["outer_fold"]),
                        candidate["candidate_id"],
                        int(row["subset_size"]),
                    )
                ]
                for row in selected
            ]
            objective_gains = [
                float(row["train_set_objective"])
                - float(greedy_row["train_set_objective"])
                for row, greedy_row in zip(selected, greedy_rows)
            ]
            output.append(
                {
                    "target_id": target_id,
                    "candidate_id": candidate["candidate_id"],
                    "fixed_k_cell_count": len(selected),
                    "mean_fixed_k_holdout_robust_bedroc": statistics.fmean(
                        float(row["holdout_robust_bedroc"]) for row in selected
                    ),
                    "mean_gain_over_pair_off": statistics.fmean(pair_gains),
                    "minimum_fold_k_gain_over_pair_off": min(pair_gains),
                    "nonnegative_fold_k_gain_over_pair_off_count": sum(
                        value >= -TOLERANCE for value in pair_gains
                    ),
                    "mean_gain_over_same_objective_greedy": statistics.fmean(
                        float(row["holdout_robust_bedroc"])
                        - float(greedy_row["holdout_robust_bedroc"])
                        for row, greedy_row in zip(selected, greedy_rows)
                    ),
                    "mean_train_objective_gain_over_greedy": statistics.fmean(
                        objective_gains
                    ),
                    "minimum_train_objective_gain_over_greedy": min(objective_gains),
                    "selection_difference_count_vs_greedy": sum(
                        row["selected_subset"] != greedy_row["selected_subset"]
                        for row, greedy_row in zip(selected, greedy_rows)
                    ),
                    "mean_fixed_k_selection_jaccard": statistics.fmean(
                        pairwise_jaccard(
                            [
                                row["selected_subset"]
                                for row in selected
                                if int(row["subset_size"]) == subset_size
                            ]
                        )
                        for subset_size in range(2, 7)
                    ),
                }
            )
    return output


def recompute_global_rows(
    target_rows: list[dict[str, Any]], candidates: list[dict[str, Any]], targets: list[str]
) -> list[dict[str, Any]]:
    lookup = {
        (row["target_id"], row["candidate_id"]): row for row in target_rows
    }
    output: list[dict[str, Any]] = []
    for order, candidate in enumerate(candidates):
        selected = [lookup[(target, candidate["candidate_id"])] for target in targets]
        gains = [float(row["mean_gain_over_pair_off"]) for row in selected]
        output.append(
            {
                "candidate_order": order,
                "candidate_id": candidate["candidate_id"],
                "mean_target_gain_over_pair_off": statistics.fmean(gains),
                "worst_target_gain_over_pair_off": min(gains),
                "nonnegative_target_count_over_pair_off": sum(
                    value >= -TOLERANCE for value in gains
                ),
                "positive_target_count_over_pair_off": sum(
                    value > TOLERANCE for value in gains
                ),
                "mean_target_gain_over_same_objective_greedy": statistics.fmean(
                    float(row["mean_gain_over_same_objective_greedy"])
                    for row in selected
                ),
                "minimum_train_objective_gain_over_greedy": min(
                    float(row["minimum_train_objective_gain_over_greedy"])
                    for row in selected
                ),
                "selection_difference_count_vs_greedy": sum(
                    int(row["selection_difference_count_vs_greedy"])
                    for row in selected
                ),
                "mean_target_selection_jaccard": statistics.fmean(
                    float(row["mean_fixed_k_selection_jaccard"])
                    for row in selected
                ),
            }
        )
    return output


def run(
    config_path: Path, result_path: Path, root: Path, output_path: Path
) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path.resolve())
    result = read_json(result_path.resolve())
    if result.get("status") != "stage66_cross_target_auxiliary_coverage_qubo_complete":
        raise ValueError("Stage66 source result did not complete")
    if checked(root, result["config"]).resolve() != config_path.resolve():
        raise ValueError("Stage66 result config differs")
    for value in config["implementation"].values():
        checked(root, value)
    for value in config["inputs"].values():
        checked(root, value)
    output_paths = {key: checked(root, value) for key, value in result["outputs"].items()}
    rows = read_csv(output_paths["fixed_k_metrics_csv"])
    observed_target = read_csv(output_paths["target_summary_csv"])
    observed_global = read_csv(output_paths["global_summary_csv"])
    loto = read_csv(output_paths["loto_summary_csv"])
    model = read_json(output_paths["model_record_json"])
    candidates = [dict(value) for value in config["candidate_grid"]]
    targets = [str(value) for value in config["development"]["target_order"]]
    expected_count = len(targets) * 4 * len(candidates) * 6 * 2 + 96
    if len(rows) != expected_count or len(loto) != 4:
        raise ValueError("Stage66 row dimensions differ")
    source_rows = read_csv(checked(root, config["inputs"]["stage65_fixed_k_metrics"]))
    source_pair_off = {
        (row["target_id"], row["outer_fold"], row["subset_size"]): row
        for row in source_rows
        if row["candidate_id"] == "pair_off"
    }
    observed_pair_off = {
        (row["target_id"], row["outer_fold"], row["subset_size"]): row
        for row in rows
        if row["solver_id"] == SOLVER_PAIR_OFF
    }
    if set(source_pair_off) != set(observed_pair_off):
        raise ValueError("Stage66 pair-off key set differs")
    for key, source in source_pair_off.items():
        observed = observed_pair_off[key]
        if observed["selected_subset"] != source["selected_subset"]:
            raise ValueError(f"Stage66 pair-off subset differs: {key}")
        for field in (
            "holdout_primary_bedroc",
            "holdout_mean_seed_bedroc",
            "holdout_worst_seed_bedroc",
            "holdout_robust_bedroc",
        ):
            compare(observed[field], source[field], f"pair_off.{key}.{field}")
    greedy = {
        (
            row["target_id"], row["outer_fold"], row["candidate_id"], row["subset_size"]
        ): row
        for row in rows
        if row["solver_id"] == SOLVER_GREEDY
    }
    qubo_rows = [row for row in rows if row["solver_id"] == SOLVER_QUBO]
    for row in qubo_rows:
        key = (
            row["target_id"], row["outer_fold"], row["candidate_id"], row["subset_size"]
        )
        if float(row["train_set_objective"]) + TOLERANCE < float(
            greedy[key]["train_set_objective"]
        ):
            raise ValueError(f"Stage66 beam search is inferior to direct greedy: {key}")
    expected_target = recompute_target_rows(rows, candidates, targets)
    if len(observed_target) != len(expected_target):
        raise ValueError("Stage66 target summary length differs")
    target_lookup = {
        (row["target_id"], row["candidate_id"]): row for row in observed_target
    }
    for expected in expected_target:
        observed = target_lookup[(expected["target_id"], expected["candidate_id"])]
        for field, value in expected.items():
            compare(observed[field], value, f"target.{expected['target_id']}.{field}")
    expected_global = recompute_global_rows(expected_target, candidates, targets)
    global_lookup = {row["candidate_id"]: row for row in observed_global}
    for expected in expected_global:
        observed = global_lookup[expected["candidate_id"]]
        for field, value in expected.items():
            compare(observed[field], value, f"global.{expected['candidate_id']}.{field}")
    maximum_residual = 0.0
    for target_id, record in model["targets"].items():
        qubo = record["qubo"]
        if canonical_sha256(qubo) != record["qubo_sha256"]:
            raise ValueError(f"Stage66 QUBO hash differs: {target_id}")
        energy = qubo_energy(qubo, record["selected_assignment"])
        if abs(energy - float(record["selected_energy"])) > ENERGY_TOLERANCE:
            raise ValueError(f"Stage66 selected energy differs: {target_id}")
        residual = abs(energy + float(record["selected_objective"]))
        if residual > ENERGY_TOLERANCE:
            raise ValueError(f"Stage66 QUBO certificate differs: {target_id}")
        maximum_residual = max(maximum_residual, residual)
    selected = result["selected_candidate"]
    selected_global = global_lookup[selected["candidate_id"]]
    for field in (
        "mean_target_gain_over_pair_off",
        "worst_target_gain_over_pair_off",
        "nonnegative_target_count_over_pair_off",
        "mean_target_gain_over_same_objective_greedy",
        "selection_difference_count_vs_greedy",
    ):
        compare(selected[field], selected_global[field], f"selected.{field}")
    checks = result["freeze_gate"]["checks"]
    if result["freeze_gate"]["coverage_objective_freeze_authorized"] != all(
        bool(value) for value in checks.values()
    ):
        raise ValueError("Stage66 freeze decision differs")
    audit = {
        "schema_version": "1.0",
        "status": "stage66_cross_target_auxiliary_coverage_qubo_independent_audit_ok",
        "source_result": {
            "path": result_path.resolve().relative_to(root).as_posix(),
            "sha256": sha256(result_path),
            "size_bytes": result_path.stat().st_size,
        },
        "config": result["config"],
        "all_output_hashes_exact": True,
        "row_counts": {
            "fixed_k_metrics": len(rows),
            "target_summary": len(observed_target),
            "global_summary": len(observed_global),
            "loto_summary": len(loto),
        },
        "pair_off_reproduction_cells_independently_verified": len(observed_pair_off),
        "same_objective_search_cells_independently_verified": len(qubo_rows),
        "candidate_summaries_independently_recomputed": True,
        "qubo_models_independently_energy_checked": len(model["targets"]),
        "maximum_selected_state_energy_residual": maximum_residual,
        "decision_gate_independently_recomputed": True,
        "coverage_objective_freeze_authorized": result["freeze_gate"][
            "coverage_objective_freeze_authorized"
        ],
        "data_boundary_exact": result["data_boundary"]
        == {
            "fresh_validation_rows_read": 0,
            "historical_development_targets_read": 4,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "interpretation_boundary": result["interpretation_boundary"],
    }
    if not audit["data_boundary_exact"]:
        raise ValueError("Stage66 data boundary differs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage66_cross_target_auxiliary_coverage_qubo.json"),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("data/stage66_cross_target_auxiliary_coverage_qubo_result.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage66_cross_target_auxiliary_coverage_qubo_audit.json"),
    )
    args = parser.parse_args()
    run(args.config, args.result, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
