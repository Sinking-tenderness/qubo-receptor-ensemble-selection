"""Run paired-seed e32 diagnostics for every rejected expanded-matrix pair."""

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
    from .audit_stage05_e32_matrix_rescue import evaluate_rescue
    from .batch_vina_docking import get_vina_version, read_vina_config
    from .prepare_receptor import file_sha256
    from .run_vina_warning_diagnostics import run_one, write_csv
except ImportError:
    from audit_stage05_e32_matrix_rescue import evaluate_rescue
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


def portable_path(value: str) -> Path:
    return Path(value.replace("\\", "/"))


def seed_values(row: dict[str, str], replicate_ids: list[str]) -> list[float]:
    values = []
    for replicate_id in replicate_ids:
        value = float(row[f"{replicate_id}_representative_score"])
        if not math.isfinite(value):
            raise ValueError("aggregate contains a non-finite seed score")
        values.append(value)
    return values


def select_flagged_rows(
    rows: list[dict[str, str]],
    replicate_ids: list[str],
    maximum_nonnegative_pairs: int,
    maximum_seed_range: float,
) -> list[dict[str, str]]:
    if maximum_nonnegative_pairs != 0:
        raise ValueError("diagnostic selector currently requires a zero nonnegative limit")
    selected = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["ligand_id"], row["receptor_id"])
        if key in seen:
            raise ValueError(f"duplicate aggregate pair: {key}")
        seen.add(key)
        values = seed_values(row, replicate_ids)
        observed_range = max(values) - min(values)
        if not math.isclose(
            observed_range,
            float(row["seed_score_range"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"stored seed range differs: {key}")
        if any(value >= 0.0 for value in values) or observed_range > maximum_seed_range:
            selected.append(row)
    return selected


def compare_search_configs(e16_path: Path, e32_path: Path) -> dict[str, str]:
    e16 = read_vina_config(e16_path)
    e32 = read_vina_config(e32_path)
    if int(e16.get("exhaustiveness", "0")) != 16:
        raise ValueError("source protocol is not e16")
    if int(e32.get("exhaustiveness", "0")) != 32:
        raise ValueError("diagnostic protocol is not e32")
    keys = (set(e16) | set(e32)) - {"exhaustiveness"}
    differing = sorted(key for key in keys if e16.get(key) != e32.get(key))
    if differing:
        raise ValueError(
            "e16 and e32 configs differ outside exhaustiveness: " + ", ".join(differing)
        )
    if e32.get("num_modes") != "1" or int(e32.get("cpu", "0")) <= 0:
        raise ValueError("diagnostic protocol requires num_modes=1 and positive CPU")
    return e32


def resolve_cases(
    selected: list[dict[str, str]],
    expected_cases: list[object],
    receptor_rows: list[dict[str, str]],
    ligand_rows: list[dict[str, str]],
    replicate_records: list[object],
) -> list[dict[str, object]]:
    expected_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for item in expected_cases:
        if not isinstance(item, dict):
            raise ValueError("expected cases must contain objects")
        key = (str(item["ligand_id"]), str(item["receptor_id"]))
        if key in expected_by_key:
            raise ValueError("expected cases contain a duplicate pair")
        expected_by_key[key] = item
    selected_by_key = {
        (row["ligand_id"], row["receptor_id"]): row for row in selected
    }
    if set(selected_by_key) != set(expected_by_key):
        raise ValueError("automatically flagged pairs differ from frozen expected cases")

    receptors = {row["conformer_id"]: row for row in receptor_rows}
    ligands = {row["ligand_id"]: row for row in ligand_rows}
    ligand_indices = {row["ligand_id"]: index for index, row in enumerate(ligand_rows)}
    resolved: list[dict[str, object]] = []
    for key in sorted(expected_by_key):
        ligand_id, receptor_id = key
        expected = expected_by_key[key]
        source = selected_by_key[key]
        if receptor_id not in receptors or ligand_id not in ligands:
            raise ValueError(f"diagnostic pair is absent from a manifest: {key}")
        receptor = receptors[receptor_id]
        ligand = ligands[ligand_id]
        if receptor.get("status") != "ok" or ligand.get("pdbqt_status") != "ok":
            raise ValueError(f"diagnostic input preparation did not pass: {key}")
        if ligand.get("selection_role") != "development_train":
            raise ValueError(f"prohibited ligand role entered diagnostics: {ligand_id}")
        receptor_path = portable_path(receptor["receptor_pdbqt"])
        ligand_path = portable_path(ligand["pdbqt_path"])
        require_hash(
            receptor_path,
            expected["receptor_pdbqt_sha256"],
            f"{receptor_id} receptor",
        )
        require_hash(
            ligand_path,
            expected["ligand_pdbqt_sha256"],
            f"{ligand_id} ligand",
        )
        if file_sha256(receptor_path) != receptor["receptor_pdbqt_sha256"].upper():
            raise ValueError(f"receptor manifest hash differs: {receptor_id}")
        if file_sha256(ligand_path) != ligand["pdbqt_sha256"].upper():
            raise ValueError(f"ligand manifest hash differs: {ligand_id}")

        actual_seeds: dict[str, int] = {}
        source_scores: dict[str, float] = {}
        for item in replicate_records:
            if not isinstance(item, dict):
                raise ValueError("seed replicates must contain objects")
            replicate_id = str(item["replicate_id"])
            base_seed = int(item["base_seed"])
            actual_seeds[replicate_id] = base_seed + ligand_indices[ligand_id]
            source_scores[replicate_id] = float(
                source[f"{replicate_id}_representative_score"]
            )
        resolved.append(
            {
                "case_id": str(expected["case_id"]),
                "ligand_id": ligand_id,
                "label": ligand["label"],
                "selection_role": ligand["selection_role"],
                "ligand_path": ligand_path.as_posix(),
                "ligand_sha256": file_sha256(ligand_path),
                "receptor_id": receptor_id,
                "receptor_path": receptor_path.as_posix(),
                "receptor_sha256": file_sha256(receptor_path),
                "actual_seeds": actual_seeds,
                "source_e16_scores": source_scores,
                "source_e16_median_score": statistics.median(source_scores.values()),
                "source_e16_minimum_score": min(source_scores.values()),
                "source_e16_seed_range": max(source_scores.values())
                - min(source_scores.values()),
            }
        )
    return resolved


def summarize_case(
    case: dict[str, object],
    rows: list[dict[str, object]],
    threshold: float,
) -> dict[str, object]:
    actual_seeds = {str(key): int(value) for key, value in dict(case["actual_seeds"]).items()}
    source_scores_by_id = {
        str(key): float(value) for key, value in dict(case["source_e16_scores"]).items()
    }
    successful = [row for row in rows if row["status"] == "ok"]
    by_seed = {int(row["seed"]): float(row["docking_score"]) for row in successful}
    e16_scores = [source_scores_by_id[key] for key in actual_seeds]
    e32_scores = [by_seed[seed] for seed in actual_seeds.values() if seed in by_seed]
    complete = len(e32_scores) == len(actual_seeds)
    if complete:
        rescue = evaluate_rescue(e16_scores, e32_scores, threshold)
        all_negative = all(value < 0.0 for value in e32_scores)
        checks = {**dict(rescue["checks"]), "all_e32_scores_negative": all_negative}
        passed = all(checks.values())
    else:
        rescue = {
            "e16_scores": e16_scores,
            "e16_seed_range_kcal_per_mol": max(e16_scores) - min(e16_scores),
            "e16_median_score": statistics.median(e16_scores),
            "e16_minimum_score": min(e16_scores),
            "e32_scores": e32_scores,
            "e32_seed_range_kcal_per_mol": None,
            "e32_median_score": None,
            "e32_minimum_score": None,
            "absolute_e16_e32_median_delta_kcal_per_mol": None,
            "absolute_e16_e32_minimum_delta_kcal_per_mol": None,
            "acceptance_threshold_kcal_per_mol": threshold,
        }
        checks = {"complete_e32_runs": False}
        passed = False
    source_nonnegative = any(value >= 0.0 for value in e16_scores)
    if not complete:
        classification = "incomplete"
    elif passed and source_nonnegative:
        classification = "isolated_e16_nonnegative_search_failure_rescued"
    elif passed:
        classification = "e16_seed_range_failure_rescued"
    else:
        classification = "persistent_or_inconclusive_search_instability"
    return {
        "case_id": case["case_id"],
        "ligand_id": case["ligand_id"],
        "receptor_id": case["receptor_id"],
        "selection_role": case["selection_role"],
        "expected_runs": len(actual_seeds),
        "successful_runs": len(successful),
        **rescue,
        "checks": checks,
        "diagnostic_classification": classification,
        "rescue_passed": passed,
    }


def load_context(config_path: Path) -> dict[str, object]:
    config = load_json(config_path)
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    assert isinstance(inputs, dict)
    assert isinstance(hashes, dict)
    input_paths = {key: Path(str(value)) for key, value in inputs.items()}
    for key, path in input_paths.items():
        require_hash(path, hashes[key], key)

    preregistration = load_json(input_paths["preregistration"])
    admission_summary = load_json(input_paths["admission_summary"])
    aggregate_summary = load_json(input_paths["aggregation_summary"])
    if admission_summary.get("status") != "matrix_admission_rejected_pending_label_blind_diagnostics":
        raise ValueError("source matrix was not rejected into diagnostics")
    if aggregate_summary.get("status") != "ok":
        raise ValueError("source aggregation did not pass")
    if int(aggregate_summary.get("locked_test_manifest_rows", -1)) != 0:
        raise ValueError("source aggregation contains locked test rows")

    admission = preregistration["matrix_admission"]
    assert isinstance(admission, dict)
    replicate_records = config["seed_replicates"]
    if not isinstance(replicate_records, list) or len(replicate_records) != int(
        admission["required_seed_count"]
    ):
        raise ValueError("diagnostic seed replicate count differs")
    replicate_ids = [str(item["replicate_id"]) for item in replicate_records]
    aggregate_rows = read_csv(input_paths["aggregated_seed_scores"])
    selected = select_flagged_rows(
        aggregate_rows,
        replicate_ids,
        int(admission["maximum_allowed_nonnegative_score_pairs"]),
        float(admission["maximum_allowed_seed_score_range_kcal_per_mol"]),
    )
    flagged_rows = read_csv(input_paths["flagged_pairs"])
    selected_keys = {(row["ligand_id"], row["receptor_id"]) for row in selected}
    audit_keys = {(row["ligand_id"], row["receptor_id"]) for row in flagged_rows}
    if selected_keys != audit_keys:
        raise ValueError("diagnostic selector differs from independent audit flags")
    if len(selected) != int(config["selection"]["expected_case_count"]):
        raise ValueError("diagnostic case count differs")

    receptor_rows = read_csv(input_paths["receptor_manifest"])
    ligand_rows = read_csv(input_paths["ligand_manifest"])
    cases = resolve_cases(
        selected,
        list(config["expected_cases"]),
        receptor_rows,
        ligand_rows,
        replicate_records,
    )
    search_config = compare_search_configs(
        input_paths["e16_config"], input_paths["e32_config"]
    )
    execution = config["execution"]
    assert isinstance(execution, dict)
    workers = int(execution["workers"])
    cpu = int(search_config["cpu"])
    if workers <= 0 or workers * cpu > int(execution["max_total_cpu"]):
        raise ValueError("diagnostic execution exceeds its CPU budget")

    threshold_source = load_json(input_paths["threshold_source"])
    source_threshold = float(
        threshold_source["acceptance_thresholds"][
            "maximum_highest_protocol_seed_range_kcal_per_mol"
        ]
    )
    threshold = float(config["acceptance"]["maximum_score_delta_kcal_per_mol"])
    if not math.isclose(threshold, source_threshold, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("diagnostic acceptance threshold differs from frozen source")
    return {
        "config": config,
        "input_paths": input_paths,
        "cases": cases,
        "search_config": search_config,
        "workers": workers,
        "threshold": threshold,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    context = load_context(args.config)
    config = context["config"]
    input_paths = context["input_paths"]
    cases = context["cases"]
    search_config = context["search_config"]
    assert isinstance(config, dict)
    assert isinstance(input_paths, dict)
    assert isinstance(cases, list)
    assert isinstance(search_config, dict)
    if args.audit_only:
        expected_runs = sum(len(dict(case["actual_seeds"])) for case in cases)
        if expected_runs != int(config["execution"]["expected_vina_runs"]):
            raise ValueError("configured diagnostic run count differs")
        print(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "audit_only_ok",
                    "case_count": len(cases),
                    "expected_vina_runs": expected_runs,
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
        "protocol_id": "e32",
        "expected_exhaustiveness": 32,
        "search_config": search_config,
    }
    tasks: list[tuple[dict[str, object], int]] = []
    for case in cases:
        actual_seeds = dict(case["actual_seeds"])
        source_scores = dict(case["source_e16_scores"])
        for replicate_id, seed_value in actual_seeds.items():
            task_case = {
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
                "source_seed": int(seed_value),
            }
            tasks.append((task_case, int(seed_value)))

    vina_path = input_paths["vina_executable"]
    vina_version = get_vina_version(vina_path)
    raw_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=int(context["workers"])) as executor:
        futures = {
            executor.submit(
                run_one,
                case,
                protocol,
                seed,
                vina_path,
                run_directory,
                vina_version,
            ): (case["case_id"], seed)
            for case, seed in tasks
        }
        completed = 0
        for future in as_completed(futures):
            raw_rows.append(future.result())
            raw_rows.sort(key=lambda row: (str(row["case_id"]), int(row["seed"])))
            write_csv(Path(str(outputs["raw_runs_csv"])), raw_rows)
            completed += 1
            print(f"completed {completed}/{len(tasks)}", flush=True)
    wall_seconds = time.perf_counter() - started

    case_summaries = [
        summarize_case(
            case,
            [row for row in raw_rows if row["case_id"] == case["case_id"]],
            float(context["threshold"]),
        )
        for case in cases
    ]
    write_csv(Path(str(outputs["case_summary_csv"])), case_summaries)
    successful_runs = sum(row["status"] == "ok" for row in raw_rows)
    execution_ok = successful_runs == len(tasks)
    all_rescued = execution_ok and all(row["rescue_passed"] for row in case_summaries)
    if all_rescued:
        status = "matrix_admission_rescued"
    elif execution_ok:
        status = "matrix_admission_rejected_after_e32_diagnostics"
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
        "successful_vina_runs": successful_runs,
        "failed_vina_runs": len(tasks) - successful_runs,
        "protocol": search_config,
        "acceptance": config["acceptance"],
        "wall_runtime_seconds": round(wall_seconds, 3),
        "case_results": case_summaries,
        "all_cases_rescued": all_rescued,
        "original_matrix_cells_replaced": 0,
        "primary_e16_matrix_authorized": all_rescued,
        "sensitivity_e16_matrix_authorized": all_rescued,
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
    summary_path = Path(str(outputs["summary_json"]))
    summary_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)
    return 0 if execution_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
