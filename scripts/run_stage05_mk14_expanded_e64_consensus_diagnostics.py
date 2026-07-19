"""Run e64 diagnostics for every failed expanded-MAPK14 e32 consensus pair."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from .audit_stage05_expanded_e32_matrix import audit_consensus_rows
    from .batch_vina_docking import get_vina_version, read_vina_config
    from .prepare_receptor import file_sha256
    from .run_vina_warning_diagnostics import run_one, write_csv
except ImportError:
    from audit_stage05_expanded_e32_matrix import audit_consensus_rows
    from batch_vina_docking import get_vina_version, read_vina_config
    from prepare_receptor import file_sha256
    from run_vina_warning_diagnostics import run_one, write_csv


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def require_hash(path: Path, expected: object, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(expected).upper():
        raise ValueError(f"{name} SHA-256 differs")


def compare_protocols(e32_path: Path, e64_path: Path) -> dict[str, str]:
    e32 = read_vina_config(e32_path)
    e64 = read_vina_config(e64_path)
    if int(e32.get("exhaustiveness", "0")) != 32:
        raise ValueError("source protocol is not e32")
    if int(e64.get("exhaustiveness", "0")) != 64:
        raise ValueError("diagnostic protocol is not e64")
    keys = (set(e32) | set(e64)) - {"exhaustiveness"}
    differing = sorted(key for key in keys if e32.get(key) != e64.get(key))
    if differing:
        raise ValueError(
            "e32 and e64 configs differ outside exhaustiveness: " + ", ".join(differing)
        )
    if e64.get("num_modes") != "1" or int(e64.get("cpu", "0")) <= 0:
        raise ValueError("e64 diagnostics require num_modes=1 and positive CPU")
    return e64


def resolve_cases(
    flagged_rows: list[dict[str, str]],
    aggregate_rows: list[dict[str, str]],
    receptor_rows: list[dict[str, str]],
    ligand_rows: list[dict[str, str]],
    seed_replicates: list[object],
) -> list[dict[str, object]]:
    aggregate_by_key = {
        (row["ligand_id"], row["receptor_id"]): row for row in aggregate_rows
    }
    if len(aggregate_by_key) != len(aggregate_rows):
        raise ValueError("e32 aggregate contains duplicate pairs")
    receptors = {row["conformer_id"]: row for row in receptor_rows}
    ligands = {row["ligand_id"]: row for row in ligand_rows}
    ligand_indices = {row["ligand_id"]: index for index, row in enumerate(ligand_rows)}
    cases = []
    seen: set[tuple[str, str]] = set()
    for flagged in flagged_rows:
        key = (flagged["ligand_id"], flagged["receptor_id"])
        if key in seen:
            raise ValueError("e32 flagged table contains a duplicate pair")
        seen.add(key)
        if key not in aggregate_by_key or key[0] not in ligands or key[1] not in receptors:
            raise ValueError(f"flagged pair is absent from source inputs: {key}")
        ligand = ligands[key[0]]
        receptor = receptors[key[1]]
        if ligand.get("selection_role") != "development_train":
            raise ValueError(f"prohibited ligand role entered e64 diagnostics: {key[0]}")
        if ligand.get("pdbqt_status") != "ok" or receptor.get("status") != "ok":
            raise ValueError(f"flagged input preparation did not pass: {key}")
        ligand_path = Path(ligand["pdbqt_path"].replace("\\", "/"))
        receptor_path = Path(receptor["receptor_pdbqt"].replace("\\", "/"))
        require_hash(ligand_path, ligand["pdbqt_sha256"], f"{key[0]} ligand")
        require_hash(
            receptor_path,
            receptor["receptor_pdbqt_sha256"],
            f"{key[1]} receptor",
        )
        source = aggregate_by_key[key]
        actual_seeds = {}
        source_scores = {}
        for item in seed_replicates:
            if not isinstance(item, dict):
                raise ValueError("seed replicate must be an object")
            replicate_id = str(item["replicate_id"])
            actual_seeds[replicate_id] = int(item["base_seed"]) + ligand_indices[key[0]]
            source_scores[replicate_id] = float(
                source[f"{replicate_id}_representative_score"]
            )
        cases.append(
            {
                "case_id": f"{key[0]}__{key[1]}",
                "ligand_id": key[0],
                "label": ligand["label"],
                "selection_role": ligand["selection_role"],
                "ligand_path": ligand_path.as_posix(),
                "ligand_sha256": file_sha256(ligand_path),
                "receptor_id": key[1],
                "receptor_path": receptor_path.as_posix(),
                "receptor_sha256": file_sha256(receptor_path),
                "actual_seeds": actual_seeds,
                "source_e32_scores": source_scores,
            }
        )
    return sorted(cases, key=lambda row: str(row["case_id"]))


def summarize_case(
    case: dict[str, object],
    rows: list[dict[str, object]],
    gate: dict[str, object],
) -> dict[str, object]:
    actual_seeds = {str(key): int(value) for key, value in dict(case["actual_seeds"]).items()}
    source_by_id = {
        str(key): float(value) for key, value in dict(case["source_e32_scores"]).items()
    }
    successful = [row for row in rows if row["status"] == "ok"]
    by_seed = {int(row["seed"]): float(row["docking_score"]) for row in successful}
    e32_scores = [source_by_id[key] for key in actual_seeds]
    e64_scores = [by_seed[seed] for seed in actual_seeds.values() if seed in by_seed]
    complete = len(e64_scores) == len(actual_seeds)
    if complete:
        minimum = min(e64_scores)
        median = statistics.median(e64_scores)
        maximum = max(e64_scores)
        delta = float(gate["maximum_consensus_delta_kcal_per_mol"])
        consensus_count = sum(value <= minimum + delta for value in e64_scores)
        median_minimum = median - minimum
        checks = {
            "all_e64_scores_negative": all(value < 0.0 for value in e64_scores),
            "minimum_consensus_replicates_met": consensus_count
            >= int(gate["minimum_consensus_replicates_per_pair"]),
            "median_minus_minimum_within_threshold": median_minimum
            <= float(gate["maximum_allowed_median_minus_minimum_kcal_per_mol"]),
        }
        passed = all(checks.values())
    else:
        minimum = median = maximum = median_minimum = None
        consensus_count = 0
        checks = {"all_e64_runs_complete": False}
        passed = False
    e32_minimum = min(e32_scores)
    e32_median = statistics.median(e32_scores)
    if not complete:
        classification = "incomplete"
    elif passed:
        classification = "stable_two_of_three_consensus_at_e64"
    else:
        classification = "persistent_e64_search_instability"
    return {
        "case_id": case["case_id"],
        "ligand_id": case["ligand_id"],
        "receptor_id": case["receptor_id"],
        "selection_role": case["selection_role"],
        "expected_runs": len(actual_seeds),
        "successful_runs": len(successful),
        "e32_scores": e32_scores,
        "e32_minimum_score": e32_minimum,
        "e32_median_score": e32_median,
        "e64_scores": e64_scores,
        "e64_minimum_score": minimum,
        "e64_median_score": median,
        "e64_maximum_score": maximum,
        "e64_consensus_replicate_count": consensus_count,
        "e64_median_minus_minimum": median_minimum,
        "absolute_e32_e64_minimum_delta": (
            None if minimum is None else abs(e32_minimum - minimum)
        ),
        "absolute_e32_e64_median_delta": (
            None if median is None else abs(e32_median - median)
        ),
        "checks": checks,
        "e64_consensus_passed": passed,
        "diagnostic_classification": classification,
    }


def load_context(config_path: Path) -> dict[str, object]:
    config = load_json(config_path)
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    assert isinstance(inputs, dict)
    assert isinstance(hashes, dict)
    paths = {key: Path(str(value)) for key, value in inputs.items()}
    for key, path in paths.items():
        require_hash(path, hashes[key], key)
    amendment = load_json(paths["amendment"])
    admission_summary = load_json(paths["e32_admission_summary"])
    aggregate_summary = load_json(paths["e32_aggregation_summary"])
    if admission_summary.get("status") != "e32_matrix_admission_rejected":
        raise ValueError("source e32 matrix was not rejected")
    if int(admission_summary["audit"]["flagged_pair_count"]) != int(
        config["selection"]["expected_case_count"]
    ):
        raise ValueError("source e32 flagged count differs")
    if aggregate_summary.get("status") != "ok" or int(
        aggregate_summary.get("locked_test_manifest_rows", -1)
    ) != 0:
        raise ValueError("source e32 aggregation did not pass")
    seed_ids = [str(item["seed_id"]) for item in aggregate_summary["seed_evidence"]]
    protocol = amendment["uniform_e32_recomputation"]
    gate = amendment["e32_matrix_admission"]
    assert isinstance(protocol, dict)
    assert isinstance(gate, dict)
    aggregate_rows = read_csv(paths["e32_aggregated_scores"])
    recomputed_flags, _ = audit_consensus_rows(
        aggregate_rows,
        seed_ids,
        int(protocol["ligand_count"]),
        int(protocol["receptor_count"]),
        float(gate["maximum_consensus_delta_kcal_per_mol"]),
        int(gate["minimum_consensus_replicates_per_pair"]),
        float(gate["maximum_allowed_median_minus_minimum_kcal_per_mol"]),
        int(gate["maximum_allowed_nonnegative_median_score_pairs"]),
    )
    flagged_rows = read_csv(paths["e32_flagged_pairs"])
    recomputed_keys = {(row["ligand_id"], row["receptor_id"]) for row in recomputed_flags}
    flagged_keys = {(row["ligand_id"], row["receptor_id"]) for row in flagged_rows}
    if recomputed_keys != flagged_keys:
        raise ValueError("e64 selector differs from the frozen e32 audit")
    cases = resolve_cases(
        flagged_rows,
        aggregate_rows,
        read_csv(paths["receptor_manifest"]),
        read_csv(paths["ligand_manifest"]),
        list(config["seed_replicates"]),
    )
    e64_config = compare_protocols(paths["e32_config"], paths["e64_config"])
    execution = config["execution"]
    assert isinstance(execution, dict)
    workers = int(execution["workers"])
    if workers * int(e64_config["cpu"]) > int(execution["max_total_cpu"]):
        raise ValueError("e64 diagnostics exceed the CPU budget")
    expected_runs = sum(len(dict(case["actual_seeds"])) for case in cases)
    if expected_runs != int(execution["expected_vina_runs"]):
        raise ValueError("expected e64 run count differs")
    return {
        "config": config,
        "paths": paths,
        "cases": cases,
        "e64_config": e64_config,
        "workers": workers,
        "expected_runs": expected_runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    context = load_context(args.config)
    config = context["config"]
    paths = context["paths"]
    cases = context["cases"]
    e64_config = context["e64_config"]
    assert isinstance(config, dict)
    assert isinstance(paths, dict)
    assert isinstance(cases, list)
    assert isinstance(e64_config, dict)
    if args.audit_only:
        print(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "audit_only_ok",
                    "case_count": len(cases),
                    "expected_vina_runs": context["expected_runs"],
                    "validation_rows_read": 0,
                    "test_rows_read": 0,
                },
                indent=2,
            )
        )
        return 0
    outputs = config["outputs"]
    assert isinstance(outputs, dict)
    run_directory = Path(str(outputs["run_directory"]))
    if run_directory.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; use --overwrite: {run_directory}")
        shutil.rmtree(run_directory)
    run_directory.mkdir(parents=True)
    protocol = {
        "protocol_id": "e64",
        "expected_exhaustiveness": 64,
        "search_config": e64_config,
    }
    tasks = []
    for case in cases:
        actual_seeds = dict(case["actual_seeds"])
        source_scores = dict(case["source_e32_scores"])
        for replicate_id, seed in actual_seeds.items():
            tasks.append(
                (
                    {
                        "case_id": case["case_id"],
                        "ligand_id": case["ligand_id"],
                        "label": case["label"],
                        "selection_role": case["selection_role"],
                        "ligand_path": case["ligand_path"],
                        "ligand_sha256": case["ligand_sha256"],
                        "receptor_id": case["receptor_id"],
                        "receptor_path": case["receptor_path"],
                        "receptor_sha256": case["receptor_sha256"],
                        "replicate_id": replicate_id,
                        "source_score": float(source_scores[replicate_id]),
                        "source_seed": int(seed),
                    },
                    int(seed),
                )
            )
    raw_rows = []
    vina_version = get_vina_version(paths["vina_executable"])
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=int(context["workers"])) as executor:
        futures = {
            executor.submit(
                run_one,
                case,
                protocol,
                seed,
                paths["vina_executable"],
                run_directory,
                vina_version,
            ): (case["case_id"], seed)
            for case, seed in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            raw_rows.append(future.result())
            raw_rows.sort(key=lambda row: (str(row["case_id"]), int(row["seed"])))
            write_csv(Path(str(outputs["raw_runs_csv"])), raw_rows)
            print(f"completed {completed}/{len(tasks)}", flush=True)
    wall_seconds = time.perf_counter() - started
    gate = config["e64_consensus_gate"]
    assert isinstance(gate, dict)
    case_results = [
        summarize_case(
            case,
            [row for row in raw_rows if row["case_id"] == case["case_id"]],
            gate,
        )
        for case in cases
    ]
    write_csv(Path(str(outputs["case_summary_csv"])), case_results)
    successful = sum(row["status"] == "ok" for row in raw_rows)
    execution_ok = successful == len(tasks)
    all_consensus = execution_ok and all(
        row["e64_consensus_passed"] for row in case_results
    )
    if all_consensus:
        status = "e64_diagnostics_support_uniform_e64_recomputation"
    elif execution_ok:
        status = "e64_diagnostics_inconclusive"
    else:
        status = "partial_execution_failure"
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": status,
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "source_archive": config["source_archive"],
        "case_count": len(cases),
        "expected_vina_runs": len(tasks),
        "successful_vina_runs": successful,
        "failed_vina_runs": len(tasks) - successful,
        "protocol": e64_config,
        "e64_consensus_gate": gate,
        "wall_runtime_seconds": round(wall_seconds, 3),
        "case_results": case_results,
        "e64_consensus_passed_case_count": sum(
            bool(row["e64_consensus_passed"]) for row in case_results
        ),
        "all_cases_passed": all_consensus,
        "e32_matrix_cells_replaced": 0,
        "e32_matrix_authorized": False,
        "qubo_fitted": False,
        "enrichment_metrics_calculated": False,
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "outputs": {
            "raw_runs_csv": {
                "path": str(outputs["raw_runs_csv"]),
                "sha256": file_sha256(Path(str(outputs["raw_runs_csv"]))),
            },
            "case_summary_csv": {
                "path": str(outputs["case_summary_csv"]),
                "sha256": file_sha256(Path(str(outputs["case_summary_csv"]))),
            },
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    Path(str(outputs["summary_json"])).write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)
    return 0 if execution_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
