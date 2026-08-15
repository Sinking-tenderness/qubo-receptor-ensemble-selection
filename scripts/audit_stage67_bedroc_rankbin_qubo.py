"""Independently audit Stage67 rank-bin QUBO outputs."""

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
SOLVER_BEAM = "objective_beam_swap"
SOLVER_GREEDY = "same_objective_direct_greedy"
SOLVER_PAIR_OFF = "pair_off_baseline"
CONTINUOUS_ID = "continuous_reference"


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
        raise ValueError(f"Stage67 identity differs: {path}")
    if "size_bytes" in value and path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage67 size differs: {path}")
    return path


def pairwise_jaccard(values: list[str]) -> float:
    sets = [{item for item in value.split("+") if item} for value in values]
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
            raise ValueError(f"Stage67 numeric value differs: {field}")
    elif str(observed) != str(expected):
        raise ValueError(f"Stage67 value differs: {field}")


def recompute_target_rows(
    rows: list[dict[str, str]], objective_ids: list[str], targets: list[str]
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
            row["objective_id"],
            int(row["subset_size"]),
        ): row
        for row in rows
        if row["solver_id"] == SOLVER_GREEDY
    }
    continuous = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): row
        for row in rows
        if row["solver_id"] == SOLVER_BEAM
        and row["objective_id"] == CONTINUOUS_ID
    }
    output: list[dict[str, Any]] = []
    for target_id in targets:
        for current_id in objective_ids:
            selected = [
                row
                for row in rows
                if row["target_id"] == target_id
                and row["objective_id"] == current_id
                and row["solver_id"] == SOLVER_BEAM
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
                        current_id,
                        int(row["subset_size"]),
                    )
                ]
                for row in selected
            ]
            references = [
                continuous[
                    (target_id, int(row["outer_fold"]), int(row["subset_size"]))
                ]
                for row in selected
            ]
            output.append(
                {
                    "target_id": target_id,
                    "objective_id": current_id,
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
                    "minimum_train_objective_gain_over_greedy": min(
                        float(row["train_objective"])
                        - float(greedy_row["train_objective"])
                        for row, greedy_row in zip(selected, greedy_rows)
                    ),
                    "selection_difference_count_vs_greedy": sum(
                        row["selected_subset"] != greedy_row["selected_subset"]
                        for row, greedy_row in zip(selected, greedy_rows)
                    ),
                    "mean_absolute_train_quantization_error": statistics.fmean(
                        abs(float(row["train_quantization_error"])) for row in selected
                    ),
                    "mean_subset_jaccard_vs_continuous": statistics.fmean(
                        len(
                            set(row["selected_subset"].split("+"))
                            & set(reference["selected_subset"].split("+"))
                        )
                        / len(
                            set(row["selected_subset"].split("+"))
                            | set(reference["selected_subset"].split("+"))
                        )
                        for row, reference in zip(selected, references)
                    ),
                    "mean_holdout_bedroc_gap_vs_continuous": statistics.fmean(
                        float(row["holdout_robust_bedroc"])
                        - float(reference["holdout_robust_bedroc"])
                        for row, reference in zip(selected, references)
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
    target_rows: list[dict[str, Any]], objective_ids: list[str], targets: list[str]
) -> list[dict[str, Any]]:
    lookup = {
        (row["target_id"], row["objective_id"]): row for row in target_rows
    }
    output: list[dict[str, Any]] = []
    for order, current_id in enumerate(objective_ids):
        selected = [lookup[(target, current_id)] for target in targets]
        gains = [float(row["mean_gain_over_pair_off"]) for row in selected]
        output.append(
            {
                "objective_order": order,
                "objective_id": current_id,
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
                "mean_absolute_train_quantization_error": statistics.fmean(
                    float(row["mean_absolute_train_quantization_error"])
                    for row in selected
                ),
                "mean_subset_jaccard_vs_continuous": statistics.fmean(
                    float(row["mean_subset_jaccard_vs_continuous"])
                    for row in selected
                ),
                "mean_holdout_bedroc_gap_vs_continuous": statistics.fmean(
                    float(row["mean_holdout_bedroc_gap_vs_continuous"])
                    for row in selected
                ),
                "mean_target_selection_jaccard": statistics.fmean(
                    float(row["mean_fixed_k_selection_jaccard"])
                    for row in selected
                ),
            }
        )
    return output


def factorized_energy(record: dict[str, Any]) -> float:
    receptor_index = {
        value: index for index, value in enumerate(record["receptor_ids"])
    }
    selected = [
        receptor_index[value]
        for value in record["selected_subset"].split("+")
        if value
    ]
    selected_mask = sum(1 << index for index in selected)
    value = float(record["penalties"]["cardinality_penalty"]) * (
        len(selected) - int(record["reference_k"])
    ) ** 2
    for state in record["states"]:
        exposed = bool(int(state["incidence_hex"], 16) & selected_mask)
        if state["label"] == "active":
            value -= float(state["objective_weight"]) * int(exposed)
        else:
            value += float(state["objective_weight"]) * int(exposed)
    return float(value)


def run(
    config_path: Path, result_path: Path, root: Path, output_path: Path
) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path.resolve())
    result = read_json(result_path.resolve())
    if result.get("status") != "stage67_bedroc_rankbin_qubo_complete":
        raise ValueError("Stage67 source result did not complete")
    if checked(root, result["config"]).resolve() != config_path.resolve():
        raise ValueError("Stage67 result config differs")
    for value in config["implementation"].values():
        checked(root, value)
    for value in config["inputs"].values():
        checked(root, value)
    output_paths = {
        key: checked(root, value) for key, value in result["outputs"].items()
    }
    rows = read_csv(output_paths["fixed_k_metrics_csv"])
    observed_target = read_csv(output_paths["target_summary_csv"])
    observed_global = read_csv(output_paths["resolution_summary_csv"])
    model = read_json(output_paths["model_record_json"])
    targets = [str(value) for value in config["development"]["target_order"]]
    objective_ids = [CONTINUOUS_ID] + [
        f"rankbin_b{int(value)}" for value in config["development"]["bin_counts"]
    ]
    expected_count = len(targets) * 4 * len(objective_ids) * 6 * 2 + 96
    if len(rows) != expected_count:
        raise ValueError("Stage67 fixed-k row count differs")
    source_rows = read_csv(checked(root, config["inputs"]["stage66_fixed_k_metrics"]))
    source_pair_off = {
        (row["target_id"], row["outer_fold"], row["subset_size"]): row
        for row in source_rows
        if row["solver_id"] == SOLVER_PAIR_OFF
    }
    observed_pair_off = {
        (row["target_id"], row["outer_fold"], row["subset_size"]): row
        for row in rows
        if row["solver_id"] == SOLVER_PAIR_OFF
    }
    if set(source_pair_off) != set(observed_pair_off):
        raise ValueError("Stage67 pair-off key set differs")
    for key, source in source_pair_off.items():
        observed = observed_pair_off[key]
        if observed["selected_subset"] != source["selected_subset"]:
            raise ValueError(f"Stage67 pair-off subset differs: {key}")
        compare(
            observed["holdout_robust_bedroc"],
            source["holdout_robust_bedroc"],
            f"pair_off.{key}",
        )
    greedy = {
        (
            row["target_id"], row["outer_fold"], row["objective_id"], row["subset_size"]
        ): row
        for row in rows
        if row["solver_id"] == SOLVER_GREEDY
    }
    beam_rows = [row for row in rows if row["solver_id"] == SOLVER_BEAM]
    for row in beam_rows:
        key = (
            row["target_id"], row["outer_fold"], row["objective_id"], row["subset_size"]
        )
        if float(row["train_objective"]) + TOLERANCE < float(
            greedy[key]["train_objective"]
        ):
            raise ValueError(f"Stage67 beam search is inferior to greedy: {key}")
    expected_target = recompute_target_rows(rows, objective_ids, targets)
    target_lookup = {
        (row["target_id"], row["objective_id"]): row for row in observed_target
    }
    if len(target_lookup) != len(expected_target):
        raise ValueError("Stage67 target summary length differs")
    for expected in expected_target:
        observed = target_lookup[(expected["target_id"], expected["objective_id"])]
        for field, value in expected.items():
            compare(observed[field], value, f"target.{expected['target_id']}.{field}")
    expected_global = recompute_global_rows(expected_target, objective_ids, targets)
    global_lookup = {row["objective_id"]: row for row in observed_global}
    for expected in expected_global:
        observed = global_lookup[expected["objective_id"]]
        for field, value in expected.items():
            compare(observed[field], value, f"global.{expected['objective_id']}.{field}")
    maximum_residual = 0.0
    for target_id, record in model["targets"].items():
        core = {
            key: record[key]
            for key in (
                "target_id",
                "bin_count",
                "reference_k",
                "receptor_ids",
                "states",
                "penalties",
                "factorized_qubo",
            )
        }
        if canonical_sha256(core) != record["compact_model_sha256"]:
            raise ValueError(f"Stage67 compact model hash differs: {target_id}")
        energy = factorized_energy(record)
        if abs(energy - float(record["selected_factorized_energy"])) > 1e-8:
            raise ValueError(f"Stage67 factorized energy differs: {target_id}")
        residual = abs(energy + float(record["selected_objective"]))
        if residual > float(config["route_gate"]["maximum_factorized_qubo_energy_residual"]):
            raise ValueError(f"Stage67 factorized certificate differs: {target_id}")
        maximum_residual = max(maximum_residual, residual)
    continuous = global_lookup[CONTINUOUS_ID]
    reference_id = f"rankbin_b{int(config['qubo_encoding']['reference_bin_count'])}"
    rankbin = global_lookup[reference_id]
    for field in (
        "mean_target_gain_over_pair_off",
        "worst_target_gain_over_pair_off",
        "nonnegative_target_count_over_pair_off",
    ):
        compare(result["continuous_reference"][field], continuous[field], f"continuous.{field}")
        compare(result["rankbin_reference"][field], rankbin[field], f"rankbin.{field}")
    thresholds = config["route_gate"]

    def supported(row: dict[str, str]) -> bool:
        return (
            float(row["mean_target_gain_over_pair_off"])
            >= float(thresholds["minimum_mean_target_gain_over_pair_off"]) - TOLERANCE
            and float(row["worst_target_gain_over_pair_off"])
            >= float(thresholds["minimum_worst_target_gain_over_pair_off"]) - TOLERANCE
            and int(row["nonnegative_target_count_over_pair_off"])
            >= int(thresholds["minimum_nonnegative_target_count_over_pair_off"])
        )

    continuous_supported = supported(continuous)
    rankbin_frozen = (
        continuous_supported
        and supported(rankbin)
        and float(rankbin["mean_subset_jaccard_vs_continuous"])
        >= float(thresholds["minimum_mean_subset_jaccard_vs_continuous"])
        - TOLERANCE
        and float(rankbin["mean_absolute_train_quantization_error"])
        <= float(thresholds["maximum_mean_absolute_train_quantization_error"])
        + TOLERANCE
        and maximum_residual
        <= float(thresholds["maximum_factorized_qubo_energy_residual"])
        + TOLERANCE
    )
    if result["route_gate"]["continuous_objective_supported"] != continuous_supported:
        raise ValueError("Stage67 continuous route decision differs")
    if result["route_gate"]["rankbin_qubo_freeze_authorized"] != rankbin_frozen:
        raise ValueError("Stage67 rank-bin route decision differs")
    audit = {
        "schema_version": "1.0",
        "status": "stage67_bedroc_rankbin_qubo_independent_audit_ok",
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
            "resolution_summary": len(observed_global),
        },
        "pair_off_reproduction_cells_independently_verified": len(observed_pair_off),
        "same_objective_search_cells_independently_verified": len(beam_rows),
        "candidate_summaries_independently_recomputed": True,
        "factorized_qubo_models_independently_checked": len(model["targets"]),
        "maximum_selected_state_energy_residual": maximum_residual,
        "decision_gate_independently_recomputed": True,
        "continuous_objective_supported": continuous_supported,
        "rankbin_qubo_freeze_authorized": rankbin_frozen,
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
        raise ValueError("Stage67 data boundary differs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/stage67_bedroc_rankbin_qubo.json")
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("data/stage67_bedroc_rankbin_qubo_result.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage67_bedroc_rankbin_qubo_audit.json"),
    )
    args = parser.parse_args()
    run(args.config, args.result, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
