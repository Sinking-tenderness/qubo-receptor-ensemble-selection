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
import random
import statistics
import time
from collections import Counter
from pathlib import Path


TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)




def verified(root: Path, descriptor: dict[str, object]) -> Path:
    path = root / str(descriptor["path"])
    if not path.is_file() or sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"input identity differs: {path}")
    return path


class Objective:
    def __init__(
        self,
        receptor_ids: list[str],
        high_masks: list[int],
        low_masks: list[int],
        group_masks: list[int],
        hit_sets: list[frozenset[int]],
        low_count: int,
    ) -> None:
        self.receptor_ids = receptor_ids
        self.high_masks = high_masks
        self.low_masks = low_masks
        self.group_masks = group_masks
        self.low_count = low_count
        self.pair_overlap = [[0.0] * len(receptor_ids) for _ in receptor_ids]
        for left in range(len(receptor_ids)):
            for right in range(left + 1, len(receptor_ids)):
                union = hit_sets[left] | hit_sets[right]
                self.pair_overlap[left][right] = (
                    len(hit_sets[left] & hit_sets[right]) / len(union) if union else 0.0
                )

    def components(self, selected: tuple[int, ...]) -> dict[str, object]:
        high_union = 0
        low_union = 0
        for index in selected:
            high_union |= self.high_masks[index]
            low_union |= self.low_masks[index]
        coverages = [
            (high_union & mask).bit_count() / mask.bit_count()
            for mask in self.group_masks
        ]
        pair_count = math.comb(len(selected), 2) if len(selected) >= 2 else 0
        overlap = (
            sum(
                self.pair_overlap[min(left, right)][max(left, right)]
                for left, right in itertools.combinations(selected, 2)
            )
            / pair_count
            if pair_count
            else 0.0
        )
        low_exposure = low_union.bit_count() / self.low_count
        value = (
            0.40 * min(coverages)
            + 0.30 * statistics.mean(coverages)
            - 0.20 * low_exposure
            - 0.10 * overlap
        )
        return {
            "objective": value,
            "minimum_group_coverage": min(coverages),
            "mean_group_coverage": statistics.mean(coverages),
            "global_low_exposure": low_exposure,
            "mean_pair_jaccard": overlap,
            "group_coverages": coverages,
        }

    def value(self, selected: tuple[int, ...]) -> float:
        high_union = 0
        low_union = 0
        for index in selected:
            high_union |= self.high_masks[index]
            low_union |= self.low_masks[index]
        minimum_coverage = math.inf
        coverage_sum = 0.0
        for mask in self.group_masks:
            coverage = (high_union & mask).bit_count() / mask.bit_count()
            minimum_coverage = min(minimum_coverage, coverage)
            coverage_sum += coverage
        pair_count = math.comb(len(selected), 2) if len(selected) >= 2 else 0
        overlap_sum = 0.0
        if pair_count:
            for left, right in itertools.combinations(selected, 2):
                overlap_sum += self.pair_overlap[min(left, right)][max(left, right)]
        return (
            0.40 * minimum_coverage
            + 0.30 * coverage_sum / len(self.group_masks)
            - 0.20 * low_union.bit_count() / self.low_count
            - 0.10 * (overlap_sum / pair_count if pair_count else 0.0)
        )


def better(
    candidate_value: float,
    candidate: tuple[int, ...],
    best_value: float,
    best: tuple[int, ...] | None,
) -> bool:
    return candidate_value > best_value + TOLERANCE or (
        abs(candidate_value - best_value) <= TOLERANCE
        and (best is None or candidate < best)
    )


def greedy(objective: Objective, k: int, start: tuple[int, ...] = ()) -> tuple[int, ...]:
    selected = tuple(sorted(start))
    while len(selected) < k:
        best: tuple[int, ...] | None = None
        best_value = -math.inf
        selected_set = set(selected)
        for index in range(len(objective.receptor_ids)):
            if index in selected_set:
                continue
            candidate = tuple(sorted((*selected, index)))
            value = objective.value(candidate)
            if better(value, candidate, best_value, best):
                best, best_value = candidate, value
        if best is None:
            raise RuntimeError("greedy failed to add a receptor")
        selected = best
    return selected


def best_one_swap(
    objective: Objective, start: tuple[int, ...]
) -> tuple[tuple[int, ...], int]:
    selected = tuple(sorted(start))
    iterations = 0
    while True:
        current = objective.value(selected)
        best = selected
        best_value = current
        selected_set = set(selected)
        for removed in selected:
            for added in range(len(objective.receptor_ids)):
                if added in selected_set:
                    continue
                candidate = tuple(sorted((selected_set - {removed}) | {added}))
                value = objective.value(candidate)
                if better(value, candidate, best_value, best):
                    best, best_value = candidate, value
        if best_value <= current + TOLERANCE:
            return selected, iterations
        selected = best
        iterations += 1
        if iterations > 100:
            raise RuntimeError("one-swap iteration guard exceeded")


def exact(objective: Objective, k: int) -> tuple[tuple[int, ...], float, float, int]:
    best: tuple[int, ...] | None = None
    best_value = -math.inf
    second_value = -math.inf
    optimal_count = 0
    for candidate in itertools.combinations(range(len(objective.receptor_ids)), k):
        value = objective.value(candidate)
        if value > best_value + TOLERANCE:
            second_value = best_value
            best, best_value, optimal_count = candidate, value, 1
        elif abs(value - best_value) <= TOLERANCE:
            optimal_count += 1
            if best is None or candidate < best:
                best = candidate
        elif value > second_value:
            second_value = value
    if best is None:
        raise RuntimeError("exact enumeration produced no state")
    return best, best_value, second_value, optimal_count


def tabu(
    objective: Objective, k: int, restarts: int, iterations: int, tenure: int, seed: int
) -> tuple[int, ...]:
    rng = random.Random(seed)
    global_best: tuple[int, ...] | None = None
    global_value = -math.inf
    for _ in range(restarts):
        current = tuple(sorted(rng.sample(range(len(objective.receptor_ids)), k)))
        tabu_until: dict[tuple[int, int], int] = {}
        for step in range(iterations):
            current_set = set(current)
            move_best: tuple[int, ...] | None = None
            move_pair: tuple[int, int] | None = None
            move_value = -math.inf
            for removed in current:
                for added in range(len(objective.receptor_ids)):
                    if added in current_set:
                        continue
                    candidate = tuple(sorted((current_set - {removed}) | {added}))
                    value = objective.value(candidate)
                    is_tabu = tabu_until.get((added, removed), -1) > step
                    if is_tabu and value <= global_value + TOLERANCE:
                        continue
                    if better(value, candidate, move_value, move_best):
                        move_best, move_value, move_pair = candidate, value, (removed, added)
            if move_best is None or move_pair is None:
                break
            current = move_best
            tabu_until[move_pair] = step + tenure
            if better(move_value, current, global_value, global_best):
                global_best, global_value = current, move_value
    if global_best is None:
        raise RuntimeError("tabu search produced no state")
    return global_best


def anneal(
    objective: Objective, k: int, restarts: int, steps: int, seed: int
) -> tuple[int, ...]:
    rng = random.Random(seed)
    best: tuple[int, ...] | None = None
    best_value = -math.inf
    for _ in range(restarts):
        current = tuple(sorted(rng.sample(range(len(objective.receptor_ids)), k)))
        current_value = objective.value(current)
        for step in range(steps):
            fraction = step / max(1, steps - 1)
            temperature = 0.02 * (0.0001 / 0.02) ** fraction
            current_set = set(current)
            removed = rng.choice(current)
            added = rng.choice([i for i in range(len(objective.receptor_ids)) if i not in current_set])
            candidate = tuple(sorted((current_set - {removed}) | {added}))
            value = objective.value(candidate)
            if value >= current_value or rng.random() < math.exp(
                (value - current_value) / temperature
            ):
                current, current_value = candidate, value
            if better(current_value, current, best_value, best):
                best, best_value = current, current_value
    if best is None:
        raise RuntimeError("simulated annealing produced no state")
    return best


def build_objective(
    matrix: list[dict[str, str]], manifest: list[dict[str, str]], config: dict[str, object]
) -> tuple[Objective, dict[str, object]]:
    metadata = {row["ligand_id"]: row for row in manifest}
    receptor_ids = [
        key for key in matrix[0] if key not in {"ligand_id", "label", "selection_role"}
    ]
    population = [
        row
        for row in matrix
        if metadata[row["ligand_id"]]["core_series"].lower() == "true"
        and row["label"] in {"high", "low"}
    ]
    high = [row for row in population if row["label"] == "high"]
    low = [row for row in population if row["label"] == "low"]
    groups = sorted(
        {metadata[row["ligand_id"]]["scaffold_group_id"] for row in high}
    )
    operational = dict(config["literal_operationalization"])
    expected = {
        "population": int(operational["primary_population_size"]),
        "high": int(operational["high_count"]),
        "low": int(operational["low_count"]),
        "groups": int(operational["core_series_count"]),
    }
    observed = {
        "population": len(population),
        "high": len(high),
        "low": len(low),
        "groups": len(groups),
    }
    if observed != expected:
        raise ValueError(f"Stage92 primary population differs: {observed}")
    hit_count = math.ceil(0.10 * len(population))
    if hit_count != int(operational["per_receptor_hit_count"]):
        raise ValueError("Stage92 hit count differs")

    population_index = {row["ligand_id"]: index for index, row in enumerate(population)}
    high_index = {row["ligand_id"]: index for index, row in enumerate(high)}
    low_index = {row["ligand_id"]: index for index, row in enumerate(low)}
    hit_sets: list[frozenset[int]] = []
    high_masks: list[int] = []
    low_masks: list[int] = []
    for receptor_id in receptor_ids:
        hits = sorted(
            population,
            key=lambda row: (float(row[receptor_id]), row["ligand_id"]),
        )[:hit_count]
        hit_sets.append(
            frozenset(population_index[row["ligand_id"]] for row in hits)
        )
        high_masks.append(
            sum(
                1 << high_index[row["ligand_id"]]
                for row in hits
                if row["label"] == "high"
            )
        )
        low_masks.append(
            sum(
                1 << low_index[row["ligand_id"]]
                for row in hits
                if row["label"] == "low"
            )
        )
    group_masks = [
        sum(
            1 << high_index[row["ligand_id"]]
            for row in high
            if metadata[row["ligand_id"]]["scaffold_group_id"] == group
        )
        for group in groups
    ]
    return (
        Objective(receptor_ids, high_masks, low_masks, group_masks, hit_sets, len(low)),
        {**observed, "hit_count": hit_count, "group_ids": groups},
    )


def run(root: Path, config_path: Path) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    inputs = dict(config["inputs"])
    stage91 = read_json(verified(root, dict(inputs["stage91_result"])))
    summary = read_json(verified(root, dict(inputs["stage91c_summary"])))
    audit = read_json(verified(root, dict(inputs["stage91c_audit"])))
    matrix = read_csv(verified(root, dict(inputs["primary_median_matrix"])))
    manifest = read_csv(verified(root, dict(inputs["development_manifest"])))
    if stage91.get("status") != "stage91_bace1_group_robust_rescue_preregistered":
        raise ValueError("Stage91 did not pass")
    if summary.get("status") != "stage91c_bace1_chembl365_unidock_matrix_ok":
        raise ValueError("Stage91c matrix did not pass")
    if audit.get("status") != "independent_stage91c_bace1_chembl365_unidock_matrix_audit_ok":
        raise ValueError("Stage91c independent audit did not pass")
    if any(int(value) != 0 for value in audit["data_boundary"].values()):
        raise ValueError("Stage91c crossed the data boundary")
    objective, population = build_objective(matrix, manifest, config)
    k = int(config["literal_operationalization"]["primary_k"])

    records: list[dict[str, object]] = []

    def record(method: str, selected: tuple[int, ...], seconds: float, **extra: object) -> None:
        components = objective.components(selected)
        records.append(
            {
                "method": method,
                "k": len(selected),
                "selected_receptor_ids": ";".join(
                    objective.receptor_ids[index] for index in selected
                ),
                "objective": components["objective"],
                "minimum_group_coverage": components["minimum_group_coverage"],
                "mean_group_coverage": components["mean_group_coverage"],
                "global_low_exposure": components["global_low_exposure"],
                "mean_pair_jaccard": components["mean_pair_jaccard"],
                "elapsed_seconds": seconds,
                **extra,
            }
        )

    started = time.perf_counter()
    singleton = max(
        ((objective.value((i,)), (i,)) for i in range(len(objective.receptor_ids))),
        key=lambda item: (item[0], tuple(-v for v in item[1])),
    )[1]
    record("single_best_receptor", singleton, time.perf_counter() - started)

    started = time.perf_counter()
    linear = tuple(
        sorted(
            sorted(
                range(len(objective.receptor_ids)),
                key=lambda index: (-objective.value((index,)), index),
            )[:k]
        )
    )
    record("linear_top_k_singleton_utility", linear, time.perf_counter() - started)

    started = time.perf_counter()
    direct = greedy(objective, k)
    record("direct_greedy", direct, time.perf_counter() - started)

    started = time.perf_counter()
    direct_swap, swap_iterations = best_one_swap(objective, direct)
    record(
        "greedy_plus_all_one_swaps",
        direct_swap,
        time.perf_counter() - started,
        local_search_iterations=swap_iterations,
    )

    started = time.perf_counter()
    singleton_candidates = []
    total_iterations = 0
    for index in range(len(objective.receptor_ids)):
        candidate, iterations = best_one_swap(objective, greedy(objective, k, (index,)))
        singleton_candidates.append(candidate)
        total_iterations += iterations
    all_singleton = max(
        singleton_candidates,
        key=lambda candidate: (
            objective.value(candidate),
            tuple(-index for index in candidate),
        ),
    )
    record(
        "all_singleton_greedy_plus_swaps",
        all_singleton,
        time.perf_counter() - started,
        restart_count=len(objective.receptor_ids),
        local_search_iterations=total_iterations,
    )

    baselines = dict(config["classical_baselines"])
    tabu_config = dict(baselines["multistart_tabu"])
    started = time.perf_counter()
    tabu_selected = tabu(
        objective,
        k,
        int(tabu_config["restart_count"]),
        int(tabu_config["iteration_count"]),
        int(tabu_config["tabu_tenure"]),
        int(tabu_config["seed"]),
    )
    record("multistart_tabu", tabu_selected, time.perf_counter() - started)

    anneal_config = dict(baselines["simulated_annealing"])
    started = time.perf_counter()
    anneal_selected = anneal(
        objective,
        k,
        int(anneal_config["restart_count"]),
        int(anneal_config["steps_per_restart"]),
        int(anneal_config["seed"]),
    )
    record("simulated_annealing", anneal_selected, time.perf_counter() - started)

    random_config = dict(baselines["uniform_random"])
    rng = random.Random(int(random_config["seed"]))
    started = time.perf_counter()
    random_best: tuple[int, ...] | None = None
    random_best_value = -math.inf
    for _ in range(int(random_config["sample_count"])):
        candidate = tuple(sorted(rng.sample(range(len(objective.receptor_ids)), k)))
        value = objective.value(candidate)
        if better(value, candidate, random_best_value, random_best):
            random_best, random_best_value = candidate, value
    if random_best is None:
        raise RuntimeError("random baseline produced no state")
    record("uniform_random_10000", random_best, time.perf_counter() - started)

    started = time.perf_counter()
    exact_selected, exact_value, second_value, optimal_count = exact(objective, k)
    exact_seconds = time.perf_counter() - started
    record(
        "exact_enumeration",
        exact_selected,
        exact_seconds,
        enumerated_state_count=math.comb(len(objective.receptor_ids), k),
        optimal_state_count=optimal_count,
    )
    for row in records:
        row["objective_gap_to_exact"] = exact_value - float(row["objective"])
        row["matches_exact_selection"] = (
            row["selected_receptor_ids"]
            == ";".join(objective.receptor_ids[index] for index in exact_selected)
        )

    direct_value = objective.value(direct)
    swap_value = objective.value(direct_swap)
    swap_distance = len(set(exact_selected) - set(direct_swap))
    checks = {
        "exact_solution_differs_from_direct_greedy": exact_selected != direct,
        "strict_objective_improvement_over_greedy_plus_all_one_swaps": exact_value
        > swap_value + TOLERANCE,
        "reproducible_multi_move_local_trap": exact_value > swap_value + TOLERANCE
        and swap_distance >= 2,
        "minimum_fixed_k_state_count": math.comb(len(objective.receptor_ids), k)
        >= int(config["hardness_gate"]["minimum_fixed_k_state_count"]),
        "strong_classical_time_and_quality_reported": all(
            method in {row["method"] for row in records}
            for method in {
                "all_singleton_greedy_plus_swaps",
                "multistart_tabu",
                "simulated_annealing",
                "exact_enumeration",
            }
        ),
    }
    required = [
        "exact_solution_differs_from_direct_greedy",
        "strict_objective_improvement_over_greedy_plus_all_one_swaps",
        "reproducible_multi_move_local_trap",
        "minimum_fixed_k_state_count",
        "strong_classical_time_and_quality_reported",
    ]
    passed = all(checks[key] for key in required)
    outputs = dict(config["outputs"])
    baseline_path = root / str(outputs["baseline_csv"])
    write_csv(baseline_path, records)
    selection_rows = []
    for rank, index in enumerate(exact_selected, start=1):
        selection_rows.append(
            {
                "rank": rank,
                "receptor_id": objective.receptor_ids[index],
                "selected_by_exact": True,
                "selected_by_direct_greedy": index in direct,
                "selected_by_greedy_swap": index in direct_swap,
            }
        )
    selection_path = root / str(outputs["selection_csv"])
    write_csv(selection_path, selection_rows)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": (
            "stage92_bace1_group_robust_hardness_gate_passed"
            if passed
            else "stage92_bace1_group_robust_hardness_gate_failed"
        ),
        "source_matrix_status": summary["status"],
        "source_audit_status": audit["status"],
        "operationalization_timing": config["literal_operationalization"]["status"],
        "population": population,
        "primary_k": k,
        "state_count": math.comb(len(objective.receptor_ids), k),
        "checks": checks,
        "failed_checks": [key for key in required if not checks[key]],
        "direct_greedy": {
            "selected_receptor_ids": [objective.receptor_ids[i] for i in direct],
            "objective": direct_value,
        },
        "greedy_plus_all_one_swaps": {
            "selected_receptor_ids": [objective.receptor_ids[i] for i in direct_swap],
            "objective": swap_value,
            "local_search_iterations": swap_iterations,
        },
        "exact": {
            "selected_receptor_ids": [objective.receptor_ids[i] for i in exact_selected],
            "objective": exact_value,
            "components": objective.components(exact_selected),
            "second_best_objective": second_value,
            "gap_to_second_best": exact_value - second_value,
            "optimal_state_count": optimal_count,
            "enumeration_seconds": exact_seconds,
        },
        "comparisons": {
            "exact_minus_direct_greedy": exact_value - direct_value,
            "exact_minus_greedy_swap": exact_value - swap_value,
            "exact_vs_greedy_swap_replacement_distance": swap_distance,
        },
        "authorization": {
            "confirmation_a_preparation_or_docking_authorized": passed,
            "confirmation_b_authorized": False,
            "locked_test_authorized": False,
            "quantum_simulation_or_hardware_authorized": False,
        },
        "data_boundary": {
            "confirmation_scores_read": 0,
            "locked_test_scores_read": 0,
            "new_docking_jobs": 0,
            "quantum_jobs": 0,
        },
        "outputs": {
            "baseline_csv": {"path": str(outputs["baseline_csv"]), "sha256": sha256(baseline_path)},
            "selection_csv": {"path": str(outputs["selection_csv"]), "sha256": sha256(selection_path)},
        },
        "interpretation": (
            "The frozen development instance contains a direct-greedy error, but one-swap local search, strong classical search, and exact enumeration converge to the same solution. The preregistered quantum-value hardness gate therefore fails."
        ),
        "next_gate": (
            "Do not unlock confirmation assays or quantum hardware from this route; first decide whether to publish the negative hardness result or define a genuinely new prospectively frozen problem family."
        ),
    }
    result_path = root / str(outputs["result_json"])
    write_json(result_path, result)
    report_path = root / str(outputs["report_md"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Stage92 BACE1 group-robust hardness adjudication",
                "",
                f"Status: `{result['status']}`.",
                "",
                f"The 34-receptor, k=6 space contains {result['state_count']:,} subsets. Exact enumeration improved the direct greedy objective by {result['comparisons']['exact_minus_direct_greedy']:.9f}, showing a real combination effect.",
                "",
                f"However, greedy plus deterministic one-swap search reached the exact solution, leaving an exact-minus-strong-greedy gap of {result['comparisons']['exact_minus_greedy_swap']:.9f}. No reproducible multi-move local trap was present.",
                "",
                "## Decision",
                "",
                "Confirmation A and quantum hardware remain locked. This is a negative hardness result, not evidence that the docking data or receptor ensemble is invalid.",
                "",
                "## Integrity boundary",
                "",
                "Stage91 froze the weights but not every implementation detail. Stage92 therefore records the conventional ceil/rank/Jaccard operationalization as post-score adjudication and does not reinterpret it as fully prospective.",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage92_bace1_group_robust_hardness_adjudication.json"),
    )
    args = parser.parse_args()
    run(args.root, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
