"""Independently audit Stage68 quality-plateau portfolio QUBO outputs."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import read_json, write_json  # noqa: F401 (deduped)
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
SOLVER_PAIR_OFF = "pair_off_baseline"
SOLVER_EXACT = "continuous_milp_certificate"
SOLVER_GREEDY = "same_constraint_direct_greedy"
SOLVER_SWAP = "same_constraint_greedy_swap"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()




def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))




def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def checked(root: Path, value: dict[str, Any], label: str) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage68 {label} identity differs: {path}")
    if "size_bytes" in value and path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage68 {label} size differs: {path}")
    return path


def close(observed: Any, expected: Any, label: str) -> None:
    if not math.isclose(
        float(observed), float(expected), rel_tol=0.0, abs_tol=TOLERANCE
    ):
        raise ValueError(f"Stage68 numeric value differs: {label}")


def subset_set(value: str) -> set[str]:
    output = {item for item in value.split("+") if item}
    if not output:
        raise ValueError("Stage68 subset is empty")
    return output


def recompute_target_rows(
    rows: list[dict[str, str]], targets: list[str], candidate_ids: list[str]
) -> list[dict[str, Any]]:
    pair_off = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): row
        for row in rows
        if row["solver_id"] == SOLVER_PAIR_OFF
    }
    output: list[dict[str, Any]] = []
    for target_id in targets:
        for candidate_id in candidate_ids:
            selected = [
                row
                for row in rows
                if row["target_id"] == target_id
                and row["candidate_id"] == candidate_id
                and row["solver_id"] == SOLVER_EXACT
            ]
            gains: list[float] = []
            reductions: list[float] = []
            for row in selected:
                baseline = pair_off[
                    (
                        target_id,
                        int(row["outer_fold"]),
                        int(row["subset_size"]),
                    )
                ]
                gains.append(
                    float(row["holdout_robust_bedroc"])
                    - float(baseline["holdout_robust_bedroc"])
                )
                reductions.append(
                    float(baseline["stable_redundancy_mean"])
                    - float(row["stable_redundancy_mean"])
                )
            multiplier = float(selected[0]["uncertainty_multiplier"])
            fixed_k_jaccards: list[float] = []
            for subset_size in range(2, 7):
                subsets = [
                    subset_set(row["selected_subset"])
                    for row in selected
                    if int(row["subset_size"]) == subset_size
                ]
                pairs = list(itertools.combinations(subsets, 2))
                fixed_k_jaccards.append(
                    statistics.fmean(
                        len(left & right) / len(left | right)
                        for left, right in pairs
                    )
                )
            output.append(
                {
                    "target_id": target_id,
                    "candidate_id": candidate_id,
                    "uncertainty_multiplier": multiplier,
                    "fixed_k_cell_count": len(selected),
                    "mean_holdout_robust_bedroc": statistics.fmean(
                        float(row["holdout_robust_bedroc"]) for row in selected
                    ),
                    "mean_gain_over_pair_off": statistics.fmean(gains),
                    "minimum_fold_k_gain_over_pair_off": min(gains),
                    "noninferior_fold_k_count_at_0p01": sum(
                        value >= -0.01 - TOLERANCE for value in gains
                    ),
                    "mean_stable_redundancy_reduction": statistics.fmean(
                        reductions
                    ),
                    "minimum_stable_redundancy_reduction": min(reductions),
                    "selection_difference_count_vs_pair_off": sum(
                        row["selected_subset"]
                        != pair_off[
                            (
                                target_id,
                                int(row["outer_fold"]),
                                int(row["subset_size"]),
                            )
                        ]["selected_subset"]
                        for row in selected
                    ),
                    "mean_fixed_k_selection_jaccard": statistics.fmean(
                        fixed_k_jaccards
                    ),
                }
            )
    return output


def compare_rows(
    observed: list[dict[str, str]],
    expected: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> None:
    observed_lookup = {
        tuple(row[key] for key in keys): row for row in observed
    }
    expected_lookup = {
        tuple(str(row[key]) for key in keys): row for row in expected
    }
    if set(observed_lookup) != set(expected_lookup):
        raise ValueError("Stage68 summary key set differs")
    for key, expected_row in expected_lookup.items():
        observed_row = observed_lookup[key]
        for field, value in expected_row.items():
            if field in keys:
                continue
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                close(observed_row[field], value, f"{key}:{field}")
            elif str(observed_row[field]) != str(value):
                raise ValueError(f"Stage68 value differs: {key}:{field}")


def audit_models(model: dict[str, Any]) -> dict[str, Any]:
    maximum_residual = 0.0
    maximum_variables = 0
    maximum_dynamic_range = 0.0
    for record in model["models"]:
        receptor_ids = [str(value) for value in record["receptor_ids"]]
        index = {value: position for position, value in enumerate(receptor_ids)}
        selected = tuple(
            sorted(index[value] for value in record["selected_subset"].split("+"))
        )
        deficits = [int(value) for value in record["integer_deficits"]]
        deficit = sum(deficits[value] for value in selected)
        slack = int(record["selected_slack_value"])
        maximum_deficit = int(record["maximum_integer_deficit"])
        if deficit + slack != maximum_deficit:
            raise ValueError("Stage68 model slack does not satisfy its equality")
        pair_values = iter(record["stable_redundancy_upper_triangle"])
        redundancy = 0.0
        selected_set = set(selected)
        for left, right in itertools.combinations(range(len(receptor_ids)), 2):
            value = float(next(pair_values))
            if left in selected_set and right in selected_set:
                redundancy += value
        try:
            next(pair_values)
            raise ValueError("Stage68 redundancy triangle contains excess values")
        except StopIteration:
            pass
        energy = (
            redundancy
            + float(model["cardinality_penalty"])
            * (len(selected) - int(record["reference_k"])) ** 2
            + float(model["quality_penalty"])
            * (deficit + slack - maximum_deficit) ** 2
        )
        close(energy, record["selected_factorized_energy"], "factorized energy")
        close(redundancy, record["selected_redundancy_sum"], "redundancy sum")
        residual = abs(energy - redundancy)
        close(residual, record["energy_residual"], "energy residual")
        maximum_residual = max(maximum_residual, residual)
        scale = record["qubo_scale"]
        maximum_variables = max(
            maximum_variables, int(scale["logical_variable_count"])
        )
        maximum_dynamic_range = max(
            maximum_dynamic_range, float(scale["coefficient_dynamic_range"])
        )
    return {
        "model_count": len(model["models"]),
        "maximum_factorized_energy_residual": maximum_residual,
        "maximum_logical_variable_count": maximum_variables,
        "maximum_coefficient_dynamic_range": maximum_dynamic_range,
    }


def run(
    config_path: Path,
    result_path: Path,
    root: Path,
    output_path: Path,
) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path.resolve())
    result = read_json(result_path.resolve())
    if result.get("status") != "stage68_quality_plateau_portfolio_qubo_complete":
        raise ValueError("Stage68 source result did not complete")
    if checked(root, result["config"], "config").resolve() != config_path.resolve():
        raise ValueError("Stage68 result config differs")
    for key, value in config["implementation"].items():
        checked(root, value, key)
    for key, value in config["inputs"].items():
        checked(root, value, key)
    outputs = {
        key: checked(root, value, key) for key, value in result["outputs"].items()
    }
    rows = read_csv(outputs["fixed_k_metrics_csv"])
    if len(rows) != 800:
        raise ValueError("Stage68 fixed-k metric count differs")
    counts = {
        solver: sum(row["solver_id"] == solver for row in rows)
        for solver in (SOLVER_PAIR_OFF, SOLVER_EXACT, SOLVER_GREEDY, SOLVER_SWAP)
    }
    if counts != {
        SOLVER_PAIR_OFF: 80,
        SOLVER_EXACT: 240,
        SOLVER_GREEDY: 240,
        SOLVER_SWAP: 240,
    }:
        raise ValueError("Stage68 solver row counts differ")
    if any(
        float(row["train_quality_margin"]) < -TOLERANCE
        for row in rows
    ):
        raise ValueError("Stage68 output violates a quality floor")
    exact = {
        (
            row["target_id"],
            row["outer_fold"],
            row["candidate_id"],
            row["subset_size"],
        ): row
        for row in rows
        if row["solver_id"] == SOLVER_EXACT
    }
    heuristic = [
        row
        for row in rows
        if row["solver_id"] in {SOLVER_GREEDY, SOLVER_SWAP}
    ]
    for row in heuristic:
        key = (
            row["target_id"],
            row["outer_fold"],
            row["candidate_id"],
            row["subset_size"],
        )
        if float(exact[key]["stable_redundancy_sum"]) > float(
            row["stable_redundancy_sum"]
        ) + 1e-8:
            raise ValueError("Stage68 MILP is worse than a heuristic")
    targets = [str(value) for value in config["development"]["target_order"]]
    multipliers = [
        float(value) for value in config["development"]["uncertainty_multipliers"]
    ]
    candidate_ids = [
        f"uncertainty_{str(value).replace('.', 'p')}x" for value in multipliers
    ]
    recomputed_targets = recompute_target_rows(rows, targets, candidate_ids)
    compare_rows(
        read_csv(outputs["target_summary_csv"]),
        recomputed_targets,
        ("target_id", "candidate_id"),
    )
    fidelity_rows = read_csv(outputs["qubo_fidelity_csv"])
    if len(fidelity_rows) != 80:
        raise ValueError("Stage68 QUBO fidelity cell count differs")
    jaccards: list[float] = []
    bedroc_gaps: list[float] = []
    quality_margins: list[float] = []
    for row in fidelity_rows:
        left = subset_set(row["continuous_subset"])
        right = subset_set(row["quantized_qubo_subset"])
        jaccard = len(left & right) / len(left | right)
        close(jaccard, row["subset_jaccard"], "QUBO subset Jaccard")
        if int(row["integer_deficit"]) > int(row["maximum_integer_deficit"]):
            raise ValueError("Stage68 integer quality inequality is violated")
        jaccards.append(jaccard)
        bedroc_gaps.append(
            float(row["quantized_minus_continuous_holdout_bedroc"])
        )
        quality_margins.append(float(row["actual_quality_floor_margin"]))
    model_audit = audit_models(read_json(outputs["model_record_json"]))
    fidelity = result["qubo_fidelity"]
    close(
        statistics.fmean(jaccards),
        fidelity["mean_subset_jaccard_vs_continuous"],
        "mean QUBO Jaccard",
    )
    close(
        statistics.fmean(bedroc_gaps),
        fidelity["mean_holdout_bedroc_gap_vs_continuous"],
        "mean QUBO BEDROC gap",
    )
    close(
        min(quality_margins),
        fidelity["minimum_actual_quality_floor_margin"],
        "minimum quality margin",
    )
    close(
        model_audit["maximum_factorized_energy_residual"],
        fidelity["maximum_factorized_energy_residual"],
        "maximum energy residual",
    )
    if int(model_audit["maximum_logical_variable_count"]) != int(
        fidelity["maximum_logical_variable_count"]
    ):
        raise ValueError("Stage68 maximum variable count differs")
    audit = {
        "schema_version": "1.0",
        "status": "stage68_quality_plateau_portfolio_qubo_independent_audit_ok",
        "source_result": descriptor(root, result_path),
        "fixed_k_rows_independently_checked": len(rows),
        "continuous_milp_cells_independently_checked": counts[SOLVER_EXACT],
        "heuristic_dominance_cells_independently_checked": len(heuristic),
        "qubo_fidelity_cells_independently_checked": len(fidelity_rows),
        "factorized_qubo_models_independently_checked": model_audit[
            "model_count"
        ],
        "maximum_factorized_energy_residual": model_audit[
            "maximum_factorized_energy_residual"
        ],
        "quality_plateau_qubo_freeze_authorized": result["route_gate"][
            "quality_plateau_qubo_freeze_authorized"
        ],
        "quantum_hardware_authorized": False,
        "data_boundary": result["data_boundary"],
    }
    write_json(output_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage68_quality_plateau_portfolio_qubo.json"),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("data/stage68_quality_plateau_portfolio_qubo_result.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage68_quality_plateau_portfolio_qubo_audit.json"),
    )
    args = parser.parse_args()
    run(args.config, args.result, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
