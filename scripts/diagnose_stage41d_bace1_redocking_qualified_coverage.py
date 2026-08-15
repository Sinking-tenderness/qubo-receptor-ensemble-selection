"""Audit structural coverage retained by the Stage 41c BACE1 pass pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verified(root: Path, descriptor: dict[str, object]) -> Path:
    path = root / str(descriptor["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"input hash differs: {descriptor['path']}")
    return path


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_distances(
    rows: list[dict[str, str]], expected_ids: set[str]
) -> dict[tuple[str, str], float]:
    distances: dict[tuple[str, str], float] = {}
    for row in rows:
        first = row["conformer_id_a"]
        second = row["conformer_id_b"]
        key = tuple(sorted((first, second)))
        if first == second or key in distances:
            raise ValueError("duplicate or self BACE1 structural distance")
        distances[key] = float(row["standardized_pocket_distance"])
    expected_pairs = math.comb(len(expected_ids), 2)
    observed_ids = {value for pair in distances for value in pair}
    if observed_ids != expected_ids or len(distances) != expected_pairs:
        raise ValueError("BACE1 structural distance matrix is incomplete")
    return distances


def distance(distances: dict[tuple[str, str], float], first: str, second: str) -> float:
    if first == second:
        return 0.0
    return distances[tuple(sorted((first, second)))]


def farthest_first_cover(
    candidates: list[str],
    population: list[str],
    count: int,
    distances: dict[tuple[str, str], float],
) -> tuple[list[str], float, float]:
    if count > len(candidates):
        raise ValueError("farthest-first count exceeds candidate count")
    first = min(
        candidates,
        key=lambda candidate: (
            max(distance(distances, candidate, value) for value in population),
            statistics.mean(distance(distances, candidate, value) for value in population),
            candidate,
        ),
    )
    selected = [first]
    while len(selected) < count:
        remaining = [value for value in candidates if value not in selected]
        addition = min(
            remaining,
            key=lambda candidate: (
                -min(distance(distances, candidate, chosen) for chosen in selected),
                candidate,
            ),
        )
        selected.append(addition)
    nearest = [
        min(distance(distances, value, chosen) for chosen in selected)
        for value in population
    ]
    return selected, max(nearest), statistics.mean(nearest)


def run(config_path: Path, root: Path) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = dict(config["implementation"])
    if file_sha256(Path(__file__)) != str(implementation["sha256"]).upper():
        raise ValueError("Stage 41d implementation hash differs")
    inputs = dict(config["inputs"])
    stage41a = read_json(verified(root, dict(inputs["stage41a_result"])))
    stage41c = read_json(verified(root, dict(inputs["stage41c_summary"])))
    gate_rows = read_csv(verified(root, dict(inputs["stage41c_gate_results"])))
    distance_rows = read_csv(verified(root, dict(inputs["structural_distances"])))
    if stage41a["status"] != "stage41a_bace1_large_pool_frozen":
        raise ValueError("Stage 41a pool is not frozen")
    if stage41c["status"] != "stage41c_bace1_large_pool_cognate_redocking_gate_failed":
        raise ValueError("Stage 41c failure status differs")
    if stage41c["technical_gate_pass"] is not False:
        raise ValueError("Stage 41c technical gate was retrospectively changed")
    expected_count = int(dict(config["expected"])["frozen_receptor_count"])
    if len(gate_rows) != expected_count:
        raise ValueError("Stage 41d gate ledger count differs")
    population = [row["conformer_id"] for row in gate_rows]
    if len(set(population)) != expected_count:
        raise ValueError("Stage 41d gate ledger IDs are not unique")
    passing = [row["conformer_id"] for row in gate_rows if truth(row["gate_pass"])]
    failing = [row["conformer_id"] for row in gate_rows if not truth(row["gate_pass"])]
    distances = build_distances(distance_rows, set(population))

    full_pairs = [
        distance(distances, first, second)
        for first, second in itertools.combinations(population, 2)
    ]
    passing_pairs = [
        distance(distances, first, second)
        for first, second in itertools.combinations(passing, 2)
    ]
    full_nearest = [
        min(distance(distances, value, other) for other in population if other != value)
        for value in population
    ]
    full_nearest_q95 = quantile(full_nearest, 0.95)
    failure_rows: list[dict[str, object]] = []
    for failed in failing:
        nearest = min(
            passing,
            key=lambda candidate: (distance(distances, failed, candidate), candidate),
        )
        nearest_distance = distance(distances, failed, nearest)
        failure_rows.append({
            "failed_conformer_id": failed,
            "nearest_passing_conformer_id": nearest,
            "nearest_passing_distance": nearest_distance,
            "distance_over_full_pool_q95_nearest_neighbor": nearest_distance / full_nearest_q95,
        })
    failure_rows.sort(key=lambda row: (-float(row["nearest_passing_distance"]), str(row["failed_conformer_id"])))

    budget_rows: list[dict[str, object]] = []
    for budget in [int(value) for value in dict(config["coverage"])["landmark_budgets"]]:
        full_centers, full_radius, full_mean = farthest_first_cover(
            population, population, budget, distances
        )
        pass_centers, pass_radius, pass_mean = farthest_first_cover(
            passing, population, budget, distances
        )
        budget_rows.append({
            "landmark_budget": budget,
            "full_pool_cover_radius": full_radius,
            "passing_pool_cover_radius": pass_radius,
            "radius_inflation": pass_radius / full_radius,
            "full_pool_mean_nearest_distance": full_mean,
            "passing_pool_mean_nearest_distance": pass_mean,
            "mean_distance_inflation": pass_mean / full_mean,
            "full_pool_centers": ";".join(full_centers),
            "passing_pool_centers": ";".join(pass_centers),
        })

    metrics = {
        "passing_receptor_count": len(passing),
        "failing_receptor_count": len(failing),
        "full_pool_pairwise_mean": statistics.mean(full_pairs),
        "passing_pool_pairwise_mean": statistics.mean(passing_pairs),
        "pairwise_mean_retention": statistics.mean(passing_pairs) / statistics.mean(full_pairs),
        "full_pool_pairwise_q95": quantile(full_pairs, 0.95),
        "passing_pool_pairwise_q95": quantile(passing_pairs, 0.95),
        "pairwise_q95_retention": quantile(passing_pairs, 0.95) / quantile(full_pairs, 0.95),
        "full_pool_diameter": max(full_pairs),
        "passing_pool_diameter": max(passing_pairs),
        "diameter_retention": max(passing_pairs) / max(full_pairs),
        "full_pool_q95_nearest_neighbor": full_nearest_q95,
        "maximum_failed_to_passing_distance": max(float(row["nearest_passing_distance"]) for row in failure_rows),
        "maximum_failed_to_passing_distance_ratio": max(
            float(row["distance_over_full_pool_q95_nearest_neighbor"])
            for row in failure_rows
        ),
        "maximum_landmark_radius_inflation": max(float(row["radius_inflation"]) for row in budget_rows),
    }
    criteria = dict(config["go_criteria"])
    checks = {
        "minimum_passing_receptor_count": len(passing) >= int(criteria["minimum_passing_receptor_count"]),
        "minimum_pairwise_q95_retention": metrics["pairwise_q95_retention"] >= float(criteria["minimum_pairwise_q95_retention"]),
        "minimum_diameter_retention": metrics["diameter_retention"] >= float(criteria["minimum_diameter_retention"]),
        "maximum_failed_distance_ratio": metrics["maximum_failed_to_passing_distance_ratio"] <= float(criteria["maximum_failed_distance_ratio"]),
        "maximum_landmark_radius_inflation": metrics["maximum_landmark_radius_inflation"] <= float(criteria["maximum_landmark_radius_inflation"]),
    }
    outputs = dict(config["outputs"])
    failure_path = root / str(outputs["failed_coverage_csv"])
    budget_path = root / str(outputs["landmark_coverage_csv"])
    result_path = root / str(outputs["result_json"])
    write_csv(failure_path, failure_rows)
    write_csv(budget_path, budget_rows)
    states = {str(k): math.comb(len(passing), k) for k in range(1, 7)}
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage41d_conditional_go_new_posthoc_development_route" if all(checks.values()) else "stage41d_no_go_insufficient_structural_coverage",
        "config": {"path": str(config_path.relative_to(root)).replace("\\", "/"), "sha256": file_sha256(config_path)},
        "evidence_timing": config["evidence_timing"],
        "metrics": metrics,
        "go_criteria": criteria,
        "criterion_checks": checks,
        "passing_receptor_ids": passing,
        "failing_receptor_ids": failing,
        "state_count_by_k": states,
        "total_state_count_k1_to_k6": sum(states.values()),
        "prospective_stage42_pair_count": len(passing) * int(dict(config["prospective_stage42"])["ligand_count"]) * int(dict(config["prospective_stage42"])["seed_count"]),
        "outputs": {
            "failed_coverage_csv": {"path": str(failure_path.relative_to(root)).replace("\\", "/"), "sha256": file_sha256(failure_path)},
            "landmark_coverage_csv": {"path": str(budget_path.relative_to(root)).replace("\\", "/"), "sha256": file_sha256(budget_path)},
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
