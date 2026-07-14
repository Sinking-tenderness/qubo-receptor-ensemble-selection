"""Diagnose flagged Vina receptor-ligand pairs across protocols and seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from .batch_vina_docking import (
        build_vina_command,
        get_vina_version,
        parse_vina_modes,
        read_vina_config,
        safe_filename,
    )
    from .prepare_receptor import file_sha256
except ImportError:
    from batch_vina_docking import (
        build_vina_command,
        get_vina_version,
        parse_vina_modes,
        read_vina_config,
        safe_filename,
    )
    from prepare_receptor import file_sha256


REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "purpose",
    "inputs",
    "input_sha256",
    "search_protocols",
    "cases",
    "seed_offsets",
    "execution",
    "acceptance_thresholds",
    "outputs",
    "interpretation_boundary",
}
REQUIRED_INPUT_KEYS = {
    "receptor_manifest",
    "ligand_manifest",
    "source_warning_table",
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
    "label",
    "ligand_pdbqt_sha256",
    "source_score",
    "source_seed",
}
REQUIRED_OUTPUT_KEYS = {
    "run_directory",
    "raw_runs_csv",
    "case_summary_csv",
    "summary_json",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def portable_manifest_path(value: str) -> Path:
    return Path(value.replace("\\", "/"))


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(config, dict):
        raise ValueError("warning diagnostics config must be a JSON object")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"warning diagnostics config is missing keys: {', '.join(missing)}")

    inputs = config["inputs"]
    hashes = config["input_sha256"]
    protocols = config["search_protocols"]
    cases = config["cases"]
    offsets = config["seed_offsets"]
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
        raise ValueError("search_protocols must contain exactly two complete protocols")
    protocol_ids = [str(protocol["protocol_id"]) for protocol in protocols]
    if len(protocol_ids) != len(set(protocol_ids)):
        raise ValueError("protocol IDs must be unique")
    exhaustiveness = [int(protocol["expected_exhaustiveness"]) for protocol in protocols]
    if any(value <= 0 for value in exhaustiveness) or len(set(exhaustiveness)) != 2:
        raise ValueError("protocol exhaustiveness values must be unique and positive")
    if (
        not isinstance(cases, list)
        or not cases
        or any(
            not isinstance(case, dict) or not REQUIRED_CASE_KEYS.issubset(case)
            for case in cases
        )
    ):
        raise ValueError("cases must be a non-empty list of complete case objects")
    case_ids = [str(case["case_id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be unique")
    if any(int(case["source_seed"]) <= 0 for case in cases):
        raise ValueError("source seeds must be positive")
    if (
        not isinstance(offsets, list)
        or not offsets
        or any(int(offset) < 0 for offset in offsets)
        or 0 not in [int(offset) for offset in offsets]
    ):
        raise ValueError("seed_offsets must be unique nonnegative integers including zero")
    if len(offsets) != len(set(int(offset) for offset in offsets)):
        raise ValueError("seed offsets must be unique")
    if not isinstance(execution, dict):
        raise ValueError("execution must be a JSON object")
    if int(execution.get("workers", 0)) <= 0 or int(execution.get("max_total_cpu", 0)) <= 0:
        raise ValueError("execution workers and max_total_cpu must be positive")
    if not isinstance(thresholds, dict):
        raise ValueError("acceptance_thresholds must be a JSON object")
    if not isinstance(thresholds.get("flag_nonnegative_high_protocol_scores"), bool):
        raise ValueError("flag_nonnegative_high_protocol_scores must be boolean")
    for key in (
        "source_reproduction_tolerance_kcal_per_mol",
        "maximum_high_protocol_seed_range_kcal_per_mol",
        "maximum_paired_protocol_delta_kcal_per_mol",
    ):
        if float(thresholds.get(key, -1.0)) < 0.0:
            raise ValueError(f"{key} must be nonnegative")
    if not isinstance(outputs, dict) or not REQUIRED_OUTPUT_KEYS.issubset(outputs):
        raise ValueError("outputs is missing one or more required paths")
    return config


def validate_protocol_configs(
    protocols: list[dict[str, object]], workers: int, max_total_cpu: int
) -> list[dict[str, object]]:
    resolved: list[dict[str, object]] = []
    for protocol in protocols:
        path = Path(str(protocol["config_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != str(protocol["config_sha256"]).upper():
            raise ValueError(f"search config SHA-256 differs: {path}")
        search_config = read_vina_config(path)
        expected = int(protocol["expected_exhaustiveness"])
        if int(search_config.get("exhaustiveness", "0")) != expected:
            raise ValueError(f"unexpected exhaustiveness for {protocol['protocol_id']}")
        resolved.append({
            **protocol,
            "path": path,
            "search_config": search_config,
        })

    resolved.sort(key=lambda row: int(row["expected_exhaustiveness"]))
    first_config = resolved[0]["search_config"]
    second_config = resolved[1]["search_config"]
    assert isinstance(first_config, dict)
    assert isinstance(second_config, dict)
    fixed_keys = sorted((set(first_config) | set(second_config)) - {"exhaustiveness"})
    differing = [key for key in fixed_keys if first_config.get(key) != second_config.get(key)]
    if differing:
        raise ValueError(f"search configs differ outside exhaustiveness: {', '.join(differing)}")
    cpu = int(first_config.get("cpu", "0"))
    if cpu <= 0 or workers * cpu > max_total_cpu:
        raise ValueError("workers times Vina CPU exceeds the declared CPU budget")
    if first_config.get("num_modes") != "1":
        raise ValueError("warning diagnostics require num_modes=1")
    return resolved


def resolve_cases(
    requested: list[dict[str, object]],
    receptor_rows: list[dict[str, str]],
    ligand_rows: list[dict[str, str]],
    warning_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    receptors = {row["conformer_id"]: row for row in receptor_rows}
    ligands = {row["ligand_id"]: row for row in ligand_rows}
    warnings = {(row["receptor_id"], row["ligand_id"]): row for row in warning_rows}
    requested_keys = {
        (str(case["receptor_id"]), str(case["ligand_id"])) for case in requested
    }
    if set(warnings) != requested_keys:
        raise ValueError("source warning table pairs differ from configured cases")

    resolved: list[dict[str, object]] = []
    for case in requested:
        receptor_id = str(case["receptor_id"])
        ligand_id = str(case["ligand_id"])
        if receptor_id not in receptors:
            raise ValueError(f"requested receptor is absent from manifest: {receptor_id}")
        if ligand_id not in ligands:
            raise ValueError(f"requested ligand is absent from manifest: {ligand_id}")
        receptor = receptors[receptor_id]
        ligand = ligands[ligand_id]
        warning = warnings[(receptor_id, ligand_id)]
        if receptor.get("preparation_status") != "ok":
            raise ValueError(f"receptor preparation did not pass: {receptor_id}")
        if ligand.get("pdbqt_status") != "ok" or ligand.get("label") != case["label"]:
            raise ValueError(f"ligand label or PDBQT status differs: {ligand_id}")
        if warning.get("label") != case["label"]:
            raise ValueError(f"source warning label differs: {ligand_id}")
        source_score = float(case["source_score"])
        if abs(float(warning["representative_score"]) - source_score) > 1e-6:
            raise ValueError(f"source warning score differs: {case['case_id']}")

        receptor_path = portable_manifest_path(receptor["receptor_pdbqt_path"])
        ligand_path = portable_manifest_path(ligand["pdbqt_path"])
        if not receptor_path.is_file() or not ligand_path.is_file():
            raise FileNotFoundError(receptor_path if not receptor_path.is_file() else ligand_path)
        receptor_hash = file_sha256(receptor_path)
        ligand_hash = file_sha256(ligand_path)
        if receptor_hash != str(case["receptor_pdbqt_sha256"]).upper():
            raise ValueError(f"receptor PDBQT SHA-256 differs: {receptor_id}")
        if receptor_hash != receptor["receptor_pdbqt_sha256"].upper():
            raise ValueError(f"receptor manifest SHA-256 differs: {receptor_id}")
        if ligand_hash != str(case["ligand_pdbqt_sha256"]).upper():
            raise ValueError(f"ligand PDBQT SHA-256 differs: {ligand_id}")

        resolved.append({
            "case_id": str(case["case_id"]),
            "receptor_id": receptor_id,
            "receptor_path": receptor_path.as_posix(),
            "receptor_sha256": receptor_hash,
            "ligand_id": ligand_id,
            "label": str(case["label"]),
            "ligand_path": ligand_path.as_posix(),
            "ligand_sha256": ligand_hash,
            "source_score": source_score,
            "source_seed": int(case["source_seed"]),
            "source_ligand_median_score": float(warning["ligand_median_score"]),
            "source_delta_from_ligand_median": float(warning["delta_from_ligand_median"]),
            "source_warning_reasons": warning["search_quality_warning_reasons"],
        })
    return resolved


def artifact_paths(
    case_id: str, protocol_id: str, seed: int, run_directory: Path
) -> tuple[Path, Path]:
    stem = f"{safe_filename(case_id)}_{safe_filename(protocol_id)}_seed{seed}"
    return (
        run_directory / "poses" / f"{stem}.pdbqt",
        run_directory / "logs" / f"{stem}.log",
    )


def run_one(
    case: dict[str, object],
    protocol: dict[str, object],
    seed: int,
    vina_executable: Path,
    run_directory: Path,
    vina_version: str,
) -> dict[str, object]:
    protocol_id = str(protocol["protocol_id"])
    search_config = protocol["search_config"]
    assert isinstance(search_config, dict)
    pose_path, log_path = artifact_paths(
        str(case["case_id"]), protocol_id, seed, run_directory
    )
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_vina_command(
        vina_executable,
        Path(str(case["receptor_path"])),
        Path(str(case["ligand_path"])),
        pose_path,
        search_config,
        seed,
    )
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    runtime = time.perf_counter() - started
    combined_log = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    log_path.write_text(combined_log, encoding="utf-8")
    modes = parse_vina_modes(completed.stdout)
    ok = completed.returncode == 0 and pose_path.is_file() and bool(modes)
    score = float(modes[0]["docking_score"]) if ok else ""
    return {
        **case,
        "protocol_id": protocol_id,
        "exhaustiveness": int(search_config["exhaustiveness"]),
        "cpu": int(search_config["cpu"]),
        "seed": seed,
        "seed_offset": seed - int(case["source_seed"]),
        "status": "ok" if ok else "failed",
        "return_code": completed.returncode,
        "docking_score": score,
        "runtime_seconds": round(runtime, 3),
        "vina_version": vina_version,
        "pose_path": pose_path.as_posix() if pose_path.is_file() else "",
        "pose_sha256": file_sha256(pose_path) if pose_path.is_file() else "",
        "log_path": log_path.as_posix(),
        "log_sha256": file_sha256(log_path),
        "message": "vina_ok" if ok else combined_log[-500:],
    }


def summarize_case(
    rows: list[dict[str, object]],
    protocols: list[dict[str, object]],
    seeds: list[int],
    thresholds: dict[str, object],
) -> dict[str, object]:
    first = rows[0]
    low_id = str(protocols[0]["protocol_id"])
    high_id = str(protocols[1]["protocol_id"])
    successful = [row for row in rows if row["status"] == "ok"]
    by_protocol_seed = {
        (str(row["protocol_id"]), int(row["seed"])): float(row["docking_score"])
        for row in successful
    }
    low_scores = [
        by_protocol_seed[(low_id, seed)]
        for seed in seeds
        if (low_id, seed) in by_protocol_seed
    ]
    high_scores = [
        by_protocol_seed[(high_id, seed)]
        for seed in seeds
        if (high_id, seed) in by_protocol_seed
    ]
    paired_deltas = [
        abs(by_protocol_seed[(low_id, seed)] - by_protocol_seed[(high_id, seed)])
        for seed in seeds
        if (low_id, seed) in by_protocol_seed and (high_id, seed) in by_protocol_seed
    ]
    source_key = (low_id, int(first["source_seed"]))
    source_reproduction_delta = (
        abs(by_protocol_seed[source_key] - float(first["source_score"]))
        if source_key in by_protocol_seed
        else None
    )
    low_range = max(low_scores) - min(low_scores) if low_scores else None
    high_range = max(high_scores) - min(high_scores) if high_scores else None
    expected_runs = len(seeds) * len(protocols)

    reasons: list[str] = []
    if len(successful) != expected_runs or len(paired_deltas) != len(seeds):
        reasons.append("incomplete_runs")
    if (
        source_reproduction_delta is None
        or source_reproduction_delta
        > float(thresholds["source_reproduction_tolerance_kcal_per_mol"])
    ):
        reasons.append("source_score_not_reproduced")
    if bool(thresholds["flag_nonnegative_high_protocol_scores"]) and any(
        score >= 0.0 for score in high_scores
    ):
        reasons.append("high_protocol_nonnegative_score")
    if (
        high_range is None
        or high_range
        > float(thresholds["maximum_high_protocol_seed_range_kcal_per_mol"])
    ):
        reasons.append("high_protocol_seed_range_exceeded")
    if (
        not paired_deltas
        or max(paired_deltas)
        > float(thresholds["maximum_paired_protocol_delta_kcal_per_mol"])
    ):
        reasons.append("paired_protocol_delta_exceeded")

    all_scores = [float(row["docking_score"]) for row in successful]
    source_positive_rescued = float(first["source_score"]) >= 0.0 and any(
        score < 0.0 for score in all_scores
    )
    persistent_nonnegative = (
        len(all_scores) == expected_runs and all(score >= 0.0 for score in all_scores)
    )
    if len(successful) != expected_runs:
        classification = "incomplete"
    elif persistent_nonnegative:
        classification = "persistent_unfavorable"
    elif source_positive_rescued:
        classification = "search_instability_confirmed"
    elif reasons:
        classification = "inconclusive_instability"
    else:
        classification = "stable"

    return {
        "case_id": first["case_id"],
        "receptor_id": first["receptor_id"],
        "ligand_id": first["ligand_id"],
        "label": first["label"],
        "source_score": first["source_score"],
        "source_seed": first["source_seed"],
        "source_reproduction_delta": (
            round(source_reproduction_delta, 6)
            if source_reproduction_delta is not None
            else ""
        ),
        "expected_runs": expected_runs,
        "successful_runs": len(successful),
        "lower_protocol_id": low_id,
        "lower_protocol_mean_score": (
            round(statistics.fmean(low_scores), 6) if low_scores else ""
        ),
        "lower_protocol_minimum_score": round(min(low_scores), 6) if low_scores else "",
        "lower_protocol_maximum_score": round(max(low_scores), 6) if low_scores else "",
        "lower_protocol_seed_range": round(low_range, 6) if low_range is not None else "",
        "higher_protocol_id": high_id,
        "higher_protocol_mean_score": (
            round(statistics.fmean(high_scores), 6) if high_scores else ""
        ),
        "higher_protocol_minimum_score": round(min(high_scores), 6) if high_scores else "",
        "higher_protocol_maximum_score": round(max(high_scores), 6) if high_scores else "",
        "higher_protocol_seed_range": round(high_range, 6) if high_range is not None else "",
        "maximum_absolute_paired_protocol_delta": (
            round(max(paired_deltas), 6) if paired_deltas else ""
        ),
        "nonnegative_score_count": sum(score >= 0.0 for score in all_scores),
        "source_positive_rescued": source_positive_rescued,
        "persistent_nonnegative": persistent_nonnegative,
        "diagnostic_classification": classification,
        "acceptance_pass": not reasons,
        "acceptance_failure_reasons": ";".join(reasons),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = config["inputs"]
    expected_hashes = config["input_sha256"]
    protocol_specs = config["search_protocols"]
    cases = config["cases"]
    offsets = [int(offset) for offset in config["seed_offsets"]]
    execution = config["execution"]
    thresholds = config["acceptance_thresholds"]
    outputs = config["outputs"]
    assert isinstance(inputs, dict)
    assert isinstance(expected_hashes, dict)
    assert isinstance(protocol_specs, list)
    assert isinstance(cases, list)
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
        protocol_specs,
        int(execution["workers"]),
        int(execution["max_total_cpu"]),
    )
    resolved_cases = resolve_cases(
        cases,
        read_csv(input_paths["receptor_manifest"]),
        read_csv(input_paths["ligand_manifest"]),
        read_csv(input_paths["source_warning_table"]),
    )
    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    run_directory = output_paths["run_directory"]
    core_outputs = [path for key, path in output_paths.items() if key != "run_directory"]
    tasks = [
        (case, protocol, int(case["source_seed"]) + offset)
        for case in resolved_cases
        for offset in offsets
        for protocol in protocols
    ]
    run_artifacts = [
        path
        for case, protocol, seed in tasks
        for path in artifact_paths(
            str(case["case_id"]), str(protocol["protocol_id"]), seed, run_directory
        )
    ]
    existing = [path for path in core_outputs + run_artifacts if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("warning diagnostic outputs exist; use --overwrite after review")
    if args.overwrite:
        for path in existing:
            path.unlink()
    run_directory.mkdir(parents=True, exist_ok=True)

    vina_version = get_vina_version(input_paths["vina_executable"])
    expected_run_count = len(tasks)
    raw_rows: list[dict[str, object]] = []
    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=int(execution["workers"])) as executor:
        futures = {
            executor.submit(
                run_one,
                case,
                protocol,
                seed,
                input_paths["vina_executable"],
                run_directory,
                vina_version,
            ): (str(case["case_id"]), str(protocol["protocol_id"]), seed)
            for case, protocol, seed in tasks
        }
        completed_count = 0
        for future in as_completed(futures):
            raw_rows.append(future.result())
            completed_count += 1
            raw_rows.sort(
                key=lambda row: (
                    str(row["case_id"]),
                    int(row["seed"]),
                    int(row["exhaustiveness"]),
                )
            )
            write_csv(output_paths["raw_runs_csv"], raw_rows)
            print(f"completed {completed_count}/{expected_run_count}", flush=True)
    wall_seconds = time.perf_counter() - wall_started

    case_summaries: list[dict[str, object]] = []
    for case in resolved_cases:
        seeds = [int(case["source_seed"]) + offset for offset in offsets]
        case_summaries.append(
            summarize_case(
                [row for row in raw_rows if row["case_id"] == case["case_id"]],
                protocols,
                seeds,
                thresholds,
            )
        )
    write_csv(output_paths["case_summary_csv"], case_summaries)
    successful_runs = [row for row in raw_rows if row["status"] == "ok"]
    execution_ok = len(successful_runs) == expected_run_count
    passing_cases = [row for row in case_summaries if row["acceptance_pass"]]
    classifications: dict[str, int] = {}
    for row in case_summaries:
        key = str(row["diagnostic_classification"])
        classifications[key] = classifications.get(key, 0) + 1
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": (
            "ok" if execution_ok and len(passing_cases) == len(case_summaries)
            else "completed_with_search_instability" if execution_ok
            else "partial_failure"
        ),
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "inputs": {
            key: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for key, path in input_paths.items()
        },
        "protocols": {
            str(protocol["protocol_id"]): protocol["search_config"]
            for protocol in protocols
        },
        "case_count": len(resolved_cases),
        "seed_offsets": offsets,
        "expected_run_count": expected_run_count,
        "successful_run_count": len(successful_runs),
        "failed_run_count": expected_run_count - len(successful_runs),
        "passing_case_count": len(passing_cases),
        "diagnostic_classification_counts": classifications,
        "acceptance_thresholds": thresholds,
        "total_vina_runtime_seconds": round(
            sum(float(row["runtime_seconds"]) for row in raw_rows), 3
        ),
        "wall_runtime_seconds": round(wall_seconds, 3),
        "score_range_kcal_per_mol": {
            "minimum": min(float(row["docking_score"]) for row in successful_runs),
            "maximum": max(float(row["docking_score"]) for row in successful_runs),
        } if successful_runs else None,
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
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if execution_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
