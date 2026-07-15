"""Rerun selected multiseed aggregate warnings at higher exhaustiveness."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from .batch_vina_docking import get_vina_version
    from .prepare_receptor import file_sha256
    from .run_vina_warning_diagnostics import (
        artifact_paths,
        portable_manifest_path,
        read_csv,
        run_one,
        validate_protocol_configs,
        write_csv,
    )
except ImportError:
    from batch_vina_docking import get_vina_version
    from prepare_receptor import file_sha256
    from run_vina_warning_diagnostics import (
        artifact_paths,
        portable_manifest_path,
        read_csv,
        run_one,
        validate_protocol_configs,
        write_csv,
    )


REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "purpose",
    "inputs",
    "input_sha256",
    "search_protocols",
    "selection",
    "expected_cases",
    "seed_replicates",
    "execution",
    "acceptance_thresholds",
    "outputs",
    "interpretation_boundary",
}
REQUIRED_INPUT_KEYS = {
    "receptor_manifest",
    "ligand_manifest",
    "aggregate_scores",
    "vina_executable",
}
REQUIRED_PROTOCOL_KEYS = {
    "protocol_id",
    "config_path",
    "config_sha256",
    "expected_exhaustiveness",
}
REQUIRED_CASE_KEYS = {
    "case_id",
    "receptor_id",
    "receptor_pdbqt_sha256",
    "ligand_id",
    "ligand_pdbqt_sha256",
    "label",
}
REQUIRED_SEED_KEYS = {"replicate_id", "base_seed"}
REQUIRED_OUTPUT_KEYS = {
    "run_directory",
    "raw_runs_csv",
    "case_summary_csv",
    "summary_json",
}


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(config, dict):
        raise ValueError("aggregate warning diagnostic config must be an object")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"diagnostic config is missing keys: {', '.join(missing)}")
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    protocols = config["search_protocols"]
    selection = config["selection"]
    cases = config["expected_cases"]
    seeds = config["seed_replicates"]
    execution = config["execution"]
    thresholds = config["acceptance_thresholds"]
    outputs = config["outputs"]
    if not isinstance(inputs, dict) or not REQUIRED_INPUT_KEYS.issubset(inputs):
        raise ValueError("inputs is missing one or more required paths")
    if not isinstance(hashes, dict) or not REQUIRED_INPUT_KEYS.issubset(hashes):
        raise ValueError("input_sha256 is missing one or more required hashes")
    if (
        not isinstance(protocols, list)
        or len(protocols) != 2
        or any(
            not isinstance(protocol, dict)
            or not REQUIRED_PROTOCOL_KEYS.issubset(protocol)
            for protocol in protocols
        )
    ):
        raise ValueError("search_protocols must contain complete e32 and e64 objects")
    protocol_ids = [str(protocol["protocol_id"]) for protocol in protocols]
    if set(protocol_ids) != {"e32", "e64"} or len(protocol_ids) != len(
        set(protocol_ids)
    ):
        raise ValueError("search_protocols must contain e32 and e64 exactly once")
    if not isinstance(selection, dict):
        raise ValueError("selection must be an object")
    reasons = selection.get("warning_reasons_any")
    if not isinstance(reasons, list) or not reasons:
        raise ValueError("warning_reasons_any must be a non-empty list")
    if int(selection.get("expected_case_count", 0)) <= 0:
        raise ValueError("expected_case_count must be positive")
    if (
        not isinstance(cases, list)
        or len(cases) != int(selection["expected_case_count"])
        or any(
            not isinstance(case, dict) or not REQUIRED_CASE_KEYS.issubset(case)
            for case in cases
        )
    ):
        raise ValueError("expected_cases does not match expected_case_count")
    case_ids = [str(case["case_id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be unique")
    case_pairs = [
        (str(case["receptor_id"]), str(case["ligand_id"])) for case in cases
    ]
    if len(case_pairs) != len(set(case_pairs)):
        raise ValueError("expected receptor-ligand pairs must be unique")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or any(
            not isinstance(seed, dict) or not REQUIRED_SEED_KEYS.issubset(seed)
            for seed in seeds
        )
    ):
        raise ValueError("seed_replicates must contain complete seed objects")
    replicate_ids = [str(seed["replicate_id"]) for seed in seeds]
    base_seeds = [int(seed["base_seed"]) for seed in seeds]
    if len(replicate_ids) != len(set(replicate_ids)):
        raise ValueError("replicate IDs must be unique")
    if any(seed <= 0 for seed in base_seeds) or len(base_seeds) != len(set(base_seeds)):
        raise ValueError("base seeds must be unique and positive")
    if not isinstance(execution, dict):
        raise ValueError("execution must be an object")
    if int(execution.get("workers", 0)) <= 0 or int(execution.get("max_total_cpu", 0)) <= 0:
        raise ValueError("execution CPU settings must be positive")
    if not isinstance(thresholds, dict):
        raise ValueError("acceptance_thresholds must be an object")
    if int(thresholds.get("minimum_favorable_replicates", 0)) <= 0:
        raise ValueError("minimum_favorable_replicates must be positive")
    if int(thresholds["minimum_favorable_replicates"]) > len(seeds):
        raise ValueError("minimum_favorable_replicates exceeds replicate count")
    for key in (
        "maximum_seed_range_kcal_per_mol",
        "maximum_minimum_median_delta_kcal_per_mol",
    ):
        if float(thresholds.get(key, 0.0)) <= 0.0:
            raise ValueError(f"{key} must be positive")
    if not isinstance(thresholds.get("flag_nonnegative_median_score"), bool):
        raise ValueError("flag_nonnegative_median_score must be boolean")
    if not isinstance(outputs, dict) or not REQUIRED_OUTPUT_KEYS.issubset(outputs):
        raise ValueError("outputs is missing one or more required paths")
    return config


def select_warning_rows(
    rows: list[dict[str, str]], reasons_any: list[str]
) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        reasons = set(row.get("seed_stability_warning_reasons", "").split(";"))
        if reasons.intersection(reasons_any):
            selected.append(row)
    return selected


def resolve_cases(
    selected_rows: list[dict[str, str]],
    expected_cases: list[dict[str, object]],
    receptor_rows: list[dict[str, str]],
    ligand_rows: list[dict[str, str]],
    seed_replicates: list[dict[str, object]],
) -> list[dict[str, object]]:
    expected_keys = [
        (str(case["receptor_id"]), str(case["ligand_id"]))
        for case in expected_cases
    ]
    selected_keys = [
        (row["receptor_id"], row["ligand_id"]) for row in selected_rows
    ]
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError("expected cases contain duplicate receptor-ligand pairs")
    if len(selected_keys) != len(set(selected_keys)):
        raise ValueError("selected warnings contain duplicate receptor-ligand pairs")
    expected_by_key = {
        (str(case["receptor_id"]), str(case["ligand_id"])): case
        for case in expected_cases
    }
    selected_by_key = {
        (row["receptor_id"], row["ligand_id"]): row for row in selected_rows
    }
    if set(selected_by_key) != set(expected_by_key):
        raise ValueError("selected aggregate warning pairs differ from expected cases")
    receptor_ids = [row["conformer_id"] for row in receptor_rows]
    ligand_ids = [row["ligand_id"] for row in ligand_rows]
    if len(receptor_ids) != len(set(receptor_ids)):
        raise ValueError("receptor manifest contains duplicate conformer IDs")
    if len(ligand_ids) != len(set(ligand_ids)):
        raise ValueError("ligand manifest contains duplicate ligand IDs")
    receptors = {row["conformer_id"]: row for row in receptor_rows}
    ligands = {row["ligand_id"]: row for row in ligand_rows}
    ligand_indices = {row["ligand_id"]: index for index, row in enumerate(ligand_rows)}
    resolved: list[dict[str, object]] = []
    for key in sorted(expected_by_key):
        expected = expected_by_key[key]
        source = selected_by_key[key]
        receptor_id, ligand_id = key
        if receptor_id not in receptors or ligand_id not in ligands:
            raise ValueError(f"case is absent from an input manifest: {key}")
        receptor = receptors[receptor_id]
        ligand = ligands[ligand_id]
        if source["label"] != expected["label"] or ligand["label"] != expected["label"]:
            raise ValueError(f"case label differs: {key}")
        if receptor.get("preparation_status") != "ok" or ligand.get("pdbqt_status") != "ok":
            raise ValueError(f"case preparation did not pass: {key}")
        receptor_path = portable_manifest_path(receptor["receptor_pdbqt_path"])
        ligand_path = portable_manifest_path(ligand["pdbqt_path"])
        if not receptor_path.is_file() or not ligand_path.is_file():
            raise FileNotFoundError(receptor_path if not receptor_path.is_file() else ligand_path)
        receptor_hash = file_sha256(receptor_path)
        ligand_hash = file_sha256(ligand_path)
        if receptor_hash != str(expected["receptor_pdbqt_sha256"]).upper():
            raise ValueError(f"configured receptor PDBQT SHA-256 differs: {receptor_id}")
        if receptor_hash != receptor["receptor_pdbqt_sha256"].upper():
            raise ValueError(f"receptor PDBQT SHA-256 differs: {receptor_id}")
        if ligand_hash != str(expected["ligand_pdbqt_sha256"]).upper():
            raise ValueError(f"ligand PDBQT SHA-256 differs: {ligand_id}")
        source_scores: dict[str, float] = {}
        actual_seeds: dict[str, int] = {}
        index = ligand_indices[ligand_id]
        for replicate in seed_replicates:
            replicate_id = str(replicate["replicate_id"])
            base_seed = int(replicate["base_seed"])
            if int(float(source[f"base_seed_{replicate_id}"])) != base_seed:
                raise ValueError(f"aggregate base seed differs: {key} / {replicate_id}")
            source_scores[replicate_id] = float(source[f"score_{replicate_id}"])
            actual_seeds[replicate_id] = base_seed + index
        resolved.append(
            {
                "case_id": str(expected["case_id"]),
                "receptor_id": receptor_id,
                "receptor_path": receptor_path.as_posix(),
                "receptor_sha256": receptor_hash,
                "ligand_id": ligand_id,
                "label": str(expected["label"]),
                "ligand_path": ligand_path.as_posix(),
                "ligand_sha256": ligand_hash,
                "source_seed": next(iter(actual_seeds.values())),
                "source_e32_scores": source_scores,
                "actual_seeds": actual_seeds,
                "source_e32_minimum_score": float(source["minimum_score"]),
                "source_e32_median_score": float(source["median_score"]),
                "source_e32_seed_range": float(source["seed_range"]),
                "source_e32_favorable_replicate_count": int(
                    float(source["favorable_replicate_count"])
                ),
                "source_warning_reasons": source["seed_stability_warning_reasons"],
            }
        )
    return resolved


def summarize_case(
    case: dict[str, object],
    rows: list[dict[str, object]],
    thresholds: dict[str, object],
) -> dict[str, object]:
    actual_seeds = {
        str(replicate_id): int(seed)
        for replicate_id, seed in dict(case["actual_seeds"]).items()
    }
    expected_seeds = list(actual_seeds.values())
    successful = [row for row in rows if row["status"] == "ok"]
    by_seed = {int(row["seed"]): float(row["docking_score"]) for row in successful}
    scores = [by_seed[seed] for seed in expected_seeds if seed in by_seed]
    expected_count = len(expected_seeds)
    minimum = min(scores) if scores else None
    median = statistics.median(scores) if scores else None
    maximum = max(scores) if scores else None
    seed_range = maximum - minimum if minimum is not None and maximum is not None else None
    minimum_median_delta = median - minimum if minimum is not None and median is not None else None
    favorable_count = sum(score < 0.0 for score in scores)
    minimum_favorable = int(thresholds["minimum_favorable_replicates"])
    reasons: list[str] = []
    if len(scores) != expected_count:
        reasons.append("incomplete_e64_runs")
    if favorable_count < minimum_favorable:
        reasons.append("insufficient_e64_favorable_replicates")
    if bool(thresholds["flag_nonnegative_median_score"]) and (
        median is None or median >= 0.0
    ):
        reasons.append("nonnegative_e64_median_score")
    if seed_range is None or seed_range > float(
        thresholds["maximum_seed_range_kcal_per_mol"]
    ):
        reasons.append("e64_seed_range_exceeded")
    if minimum_median_delta is None or minimum_median_delta > float(
        thresholds["maximum_minimum_median_delta_kcal_per_mol"]
    ):
        reasons.append("e64_minimum_median_delta_exceeded")
    source_median = float(case["source_e32_median_score"])
    if len(scores) != expected_count:
        classification = "incomplete"
    elif (
        source_median >= 0.0
        and median is not None
        and median < 0.0
        and favorable_count >= minimum_favorable
    ):
        classification = "critical_pair_rescued_by_e64"
    elif median is None or median >= 0.0 or favorable_count < minimum_favorable:
        classification = "persistent_primary_failure"
    elif reasons:
        classification = "favorable_but_seed_variable_at_e64"
    else:
        classification = "stable_at_e64"
    summary = {
        "case_id": case["case_id"],
        "receptor_id": case["receptor_id"],
        "ligand_id": case["ligand_id"],
        "label": case["label"],
        "source_e32_minimum_score": case["source_e32_minimum_score"],
        "source_e32_median_score": source_median,
        "source_e32_seed_range": case["source_e32_seed_range"],
        "source_e32_favorable_replicate_count": case[
            "source_e32_favorable_replicate_count"
        ],
        "e64_expected_runs": expected_count,
        "e64_successful_runs": len(scores),
        "e64_minimum_score": "" if minimum is None else round(minimum, 6),
        "e64_median_score": "" if median is None else round(median, 6),
        "e64_maximum_score": "" if maximum is None else round(maximum, 6),
        "e64_seed_range": "" if seed_range is None else round(seed_range, 6),
        "e64_minimum_median_delta": (
            "" if minimum_median_delta is None else round(minimum_median_delta, 6)
        ),
        "e64_favorable_replicate_count": favorable_count,
        "diagnostic_classification": classification,
        "acceptance_pass": not reasons,
        "acceptance_failure_reasons": ";".join(reasons),
    }
    source_scores = dict(case["source_e32_scores"])
    for replicate_id, seed in actual_seeds.items():
        summary[f"actual_seed_{replicate_id}"] = seed
        summary[f"source_e32_score_{replicate_id}"] = float(
            source_scores[replicate_id]
        )
        summary[f"e64_score_{replicate_id}"] = (
            "" if seed not in by_seed else round(by_seed[seed], 6)
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = config["inputs"]
    expected_hashes = config["input_sha256"]
    protocols_spec = config["search_protocols"]
    selection = config["selection"]
    expected_cases = config["expected_cases"]
    seed_replicates = config["seed_replicates"]
    execution = config["execution"]
    thresholds = config["acceptance_thresholds"]
    outputs = config["outputs"]
    assert isinstance(inputs, dict)
    assert isinstance(expected_hashes, dict)
    assert isinstance(protocols_spec, list)
    assert isinstance(selection, dict)
    assert isinstance(expected_cases, list)
    assert isinstance(seed_replicates, list)
    assert isinstance(execution, dict)
    assert isinstance(thresholds, dict)
    assert isinstance(outputs, dict)

    input_paths = {key: Path(str(value)) for key, value in inputs.items()}
    for key, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != str(expected_hashes[key]).upper():
            raise ValueError(f"input SHA-256 differs for {key}")
    protocols = validate_protocol_configs(
        protocols_spec,
        int(execution["workers"]),
        int(execution["max_total_cpu"]),
    )
    e64 = next(
        protocol for protocol in protocols if str(protocol["protocol_id"]) == "e64"
    )
    aggregate_rows = read_csv(input_paths["aggregate_scores"])
    selected = select_warning_rows(
        aggregate_rows, [str(reason) for reason in selection["warning_reasons_any"]]
    )
    if len(selected) != int(selection["expected_case_count"]):
        raise ValueError("selected aggregate warning count differs")
    cases = resolve_cases(
        selected,
        expected_cases,
        read_csv(input_paths["receptor_manifest"]),
        read_csv(input_paths["ligand_manifest"]),
        seed_replicates,
    )

    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    run_directory = output_paths["run_directory"]
    core_outputs = [path for key, path in output_paths.items() if key != "run_directory"]
    tasks = [
        (case, int(seed))
        for case in cases
        for seed in dict(case["actual_seeds"]).values()
    ]
    artifacts = [
        path
        for case, seed in tasks
        for path in artifact_paths(
            str(case["case_id"]), str(e64["protocol_id"]), seed, run_directory
        )
    ]
    existing = [path for path in core_outputs + artifacts if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("diagnostic outputs exist; use --overwrite after review")
    if args.overwrite:
        for path in existing:
            path.unlink()
    run_directory.mkdir(parents=True, exist_ok=True)

    vina_version = get_vina_version(input_paths["vina_executable"])
    raw_rows: list[dict[str, object]] = []
    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=int(execution["workers"])) as executor:
        futures = {
            executor.submit(
                run_one,
                case,
                e64,
                seed,
                input_paths["vina_executable"],
                run_directory,
                vina_version,
            ): (case["case_id"], seed)
            for case, seed in tasks
        }
        completed_count = 0
        for future in as_completed(futures):
            raw_rows.append(future.result())
            completed_count += 1
            raw_rows.sort(
                key=lambda row: (str(row["case_id"]), int(row["seed"]))
            )
            write_csv(output_paths["raw_runs_csv"], raw_rows)
            print(f"completed {completed_count}/{len(tasks)}", flush=True)
    wall_seconds = time.perf_counter() - wall_started

    case_summaries = []
    for case in cases:
        case_summaries.append(
            summarize_case(
                case,
                [row for row in raw_rows if row["case_id"] == case["case_id"]],
                thresholds,
            )
        )
    write_csv(output_paths["case_summary_csv"], case_summaries)
    successful = [row for row in raw_rows if row["status"] == "ok"]
    execution_ok = len(successful) == len(tasks)
    classifications: dict[str, int] = {}
    for row in case_summaries:
        name = str(row["diagnostic_classification"])
        classifications[name] = classifications.get(name, 0) + 1
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": (
            "ok" if execution_ok and all(row["acceptance_pass"] for row in case_summaries)
            else "completed_with_e64_instability" if execution_ok
            else "partial_failure"
        ),
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "inputs": {
            key: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for key, path in input_paths.items()
        },
        "case_count": len(cases),
        "expected_run_count": len(tasks),
        "successful_run_count": len(successful),
        "failed_run_count": len(tasks) - len(successful),
        "protocol": e64["search_config"],
        "seed_replicates": seed_replicates,
        "acceptance_thresholds": thresholds,
        "diagnostic_classification_counts": classifications,
        "wall_runtime_seconds": round(wall_seconds, 3),
        "total_vina_runtime_seconds": round(
            sum(float(row["runtime_seconds"]) for row in raw_rows), 3
        ),
        "case_summaries": case_summaries,
        "outputs": {
            "raw_runs_csv": {
                "path": output_paths["raw_runs_csv"].as_posix(),
                "sha256": file_sha256(output_paths["raw_runs_csv"]),
            },
            "case_summary_csv": {
                "path": output_paths["case_summary_csv"].as_posix(),
                "sha256": file_sha256(output_paths["case_summary_csv"]),
            },
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    output_paths["summary_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if execution_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
