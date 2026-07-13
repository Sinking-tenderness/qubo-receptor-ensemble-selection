"""Compare paired Vina exhaustiveness settings across fixed cases and seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import time
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
    "cases",
    "seeds",
    "execution",
    "acceptance_thresholds",
    "outputs",
    "interpretation_boundary",
}
REQUIRED_INPUT_KEYS = {
    "receptor_manifest",
    "ligand_manifest",
    "vina_executable",
    "exhaustiveness4_config",
    "exhaustiveness8_config",
}
REQUIRED_OUTPUT_KEYS = {
    "run_directory",
    "raw_runs_csv",
    "case_summary_csv",
    "summary_json",
}
REQUIRED_CASE_KEYS = {
    "case_id",
    "receptor_id",
    "receptor_pdbqt_sha256",
    "ligand_id",
    "label",
    "ligand_pdbqt_sha256",
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


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(config, dict):
        raise ValueError("search robustness config must be a JSON object")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"search robustness config is missing keys: {', '.join(missing)}")
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    outputs = config["outputs"]
    cases = config["cases"]
    seeds = config["seeds"]
    execution = config["execution"]
    thresholds = config["acceptance_thresholds"]
    if not isinstance(inputs, dict) or not REQUIRED_INPUT_KEYS.issubset(inputs):
        raise ValueError("inputs is missing one or more required paths")
    if not isinstance(hashes, dict) or not REQUIRED_INPUT_KEYS.issubset(hashes):
        raise ValueError("input_sha256 is missing one or more required hashes")
    if not isinstance(outputs, dict) or not REQUIRED_OUTPUT_KEYS.issubset(outputs):
        raise ValueError("outputs is missing one or more required paths")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")
    if any(not isinstance(case, dict) or not REQUIRED_CASE_KEYS.issubset(case) for case in cases):
        raise ValueError("each case is missing one or more required fields")
    case_ids = [str(case["case_id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be unique")
    if not isinstance(seeds, list) or not seeds or any(int(seed) <= 0 for seed in seeds):
        raise ValueError("seeds must be a non-empty list of positive integers")
    if len(seeds) != len(set(int(seed) for seed in seeds)):
        raise ValueError("seeds must be unique")
    if not isinstance(execution, dict) or int(execution.get("max_total_cpu", 0)) <= 0:
        raise ValueError("execution.max_total_cpu must be positive")
    if not isinstance(thresholds, dict):
        raise ValueError("acceptance_thresholds must be a JSON object")
    if not isinstance(thresholds.get("flag_nonnegative_scores"), bool):
        raise ValueError("flag_nonnegative_scores must be a JSON boolean")
    for key in (
        "maximum_e4_seed_range_kcal_per_mol",
        "maximum_paired_e4_e8_delta_kcal_per_mol",
    ):
        if float(thresholds.get(key, 0.0)) <= 0.0:
            raise ValueError(f"{key} must be positive")
    return config


def validate_search_configs(
    e4: dict[str, str], e8: dict[str, str], max_total_cpu: int
) -> None:
    if e4.get("exhaustiveness") != "4" or e8.get("exhaustiveness") != "8":
        raise ValueError("search configs must specify exhaustiveness 4 and 8")
    fixed_keys = (
        "center_x",
        "center_y",
        "center_z",
        "size_x",
        "size_y",
        "size_z",
        "num_modes",
        "cpu",
    )
    differing = [key for key in fixed_keys if e4.get(key) != e8.get(key)]
    if differing:
        raise ValueError(f"search configs differ outside exhaustiveness: {', '.join(differing)}")
    if int(e4.get("cpu", "0")) <= 0 or int(e4["cpu"]) > max_total_cpu:
        raise ValueError("Vina CPU setting exceeds the declared CPU budget")
    if e4.get("num_modes") != "1":
        raise ValueError("search robustness pilot requires num_modes=1")


def resolve_cases(
    requested: list[dict[str, object]],
    receptor_rows: list[dict[str, str]],
    ligand_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    receptors = {row["conformer_id"]: row for row in receptor_rows}
    ligands = {row["ligand_id"]: row for row in ligand_rows}
    resolved: list[dict[str, str]] = []
    for case in requested:
        receptor_id = str(case["receptor_id"])
        ligand_id = str(case["ligand_id"])
        if receptor_id not in receptors:
            raise ValueError(f"requested receptor is absent from manifest: {receptor_id}")
        if ligand_id not in ligands:
            raise ValueError(f"requested ligand is absent from manifest: {ligand_id}")
        receptor = receptors[receptor_id]
        ligand = ligands[ligand_id]
        if receptor.get("preparation_status") != "ok":
            raise ValueError(f"receptor preparation did not pass: {receptor_id}")
        if ligand.get("pdbqt_status") != "ok" or ligand.get("label") != case["label"]:
            raise ValueError(f"ligand label or PDBQT status differs: {ligand_id}")
        receptor_path = Path(receptor["receptor_pdbqt_path"])
        ligand_path = Path(ligand["pdbqt_path"])
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
            "receptor_path": str(receptor_path),
            "receptor_sha256": receptor_hash,
            "ligand_id": ligand_id,
            "label": ligand["label"],
            "ligand_path": str(ligand_path),
            "ligand_sha256": ligand_hash,
        })
    return resolved


def summarize_case(
    rows: list[dict[str, object]],
    seeds: list[int],
    flag_nonnegative_scores: bool,
    maximum_e4_seed_range: float,
    maximum_paired_delta: float,
) -> dict[str, object]:
    first = rows[0]
    expected_runs = len(seeds) * 2
    successful = [row for row in rows if row["status"] == "ok"]
    by_protocol_seed = {
        (str(row["protocol"]), int(row["seed"])): float(row["docking_score"])
        for row in successful
    }
    e4_scores = [by_protocol_seed[("e4", seed)] for seed in seeds if ("e4", seed) in by_protocol_seed]
    e8_scores = [by_protocol_seed[("e8", seed)] for seed in seeds if ("e8", seed) in by_protocol_seed]
    paired_deltas = [
        abs(by_protocol_seed[("e4", seed)] - by_protocol_seed[("e8", seed)])
        for seed in seeds
        if ("e4", seed) in by_protocol_seed and ("e8", seed) in by_protocol_seed
    ]
    e4_range = max(e4_scores) - min(e4_scores) if e4_scores else None
    e8_range = max(e8_scores) - min(e8_scores) if e8_scores else None
    reasons: list[str] = []
    if len(successful) != expected_runs or len(paired_deltas) != len(seeds):
        reasons.append("incomplete_runs")
    if flag_nonnegative_scores and any(
        float(row["docking_score"]) >= 0.0 for row in successful
    ):
        reasons.append("nonnegative_score")
    if e4_range is None or e4_range > maximum_e4_seed_range:
        reasons.append("e4_seed_range_exceeded")
    if not paired_deltas or max(paired_deltas) > maximum_paired_delta:
        reasons.append("paired_e4_e8_delta_exceeded")
    return {
        "case_id": first["case_id"],
        "receptor_id": first["receptor_id"],
        "ligand_id": first["ligand_id"],
        "label": first["label"],
        "expected_runs": expected_runs,
        "successful_runs": len(successful),
        "e4_mean_score": round(statistics.fmean(e4_scores), 6) if e4_scores else "",
        "e4_seed_range": round(e4_range, 6) if e4_range is not None else "",
        "e8_mean_score": round(statistics.fmean(e8_scores), 6) if e8_scores else "",
        "e8_seed_range": round(e8_range, 6) if e8_range is not None else "",
        "mean_absolute_paired_delta": (
            round(statistics.fmean(paired_deltas), 6) if paired_deltas else ""
        ),
        "maximum_absolute_paired_delta": round(max(paired_deltas), 6) if paired_deltas else "",
        "nonnegative_score_count": sum(
            1 for row in successful if float(row["docking_score"]) >= 0.0
        ),
        "acceptance_pass": not reasons,
        "acceptance_failure_reasons": ";".join(reasons),
    }


def run_one(
    case: dict[str, str],
    protocol: str,
    search_config: dict[str, str],
    seed: int,
    vina_executable: Path,
    run_directory: Path,
    vina_version: str,
) -> dict[str, object]:
    pose_path, log_path = artifact_paths(case["case_id"], protocol, seed, run_directory)
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_vina_command(
        vina_executable,
        Path(case["receptor_path"]),
        Path(case["ligand_path"]),
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
        "protocol": protocol,
        "exhaustiveness": int(search_config["exhaustiveness"]),
        "cpu": int(search_config["cpu"]),
        "seed": seed,
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


def artifact_paths(
    case_id: str, protocol: str, seed: int, run_directory: Path
) -> tuple[Path, Path]:
    stem = f"{safe_filename(case_id)}_{protocol}_seed{seed}"
    return (
        run_directory / "poses" / f"{stem}.pdbqt",
        run_directory / "logs" / f"{stem}.log",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = config["inputs"]
    expected_hashes = config["input_sha256"]
    outputs = config["outputs"]
    execution = config["execution"]
    thresholds = config["acceptance_thresholds"]
    cases = config["cases"]
    seeds = [int(seed) for seed in config["seeds"]]
    assert isinstance(inputs, dict)
    assert isinstance(expected_hashes, dict)
    assert isinstance(outputs, dict)
    assert isinstance(execution, dict)
    assert isinstance(thresholds, dict)
    assert isinstance(cases, list)

    input_paths = {key: Path(str(value)) for key, value in inputs.items()}
    for key, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != str(expected_hashes[key]).upper():
            raise ValueError(f"input SHA-256 differs for {key}")

    e4_config = read_vina_config(input_paths["exhaustiveness4_config"])
    e8_config = read_vina_config(input_paths["exhaustiveness8_config"])
    validate_search_configs(e4_config, e8_config, int(execution["max_total_cpu"]))
    resolved_cases = resolve_cases(
        cases,
        read_csv(input_paths["receptor_manifest"]),
        read_csv(input_paths["ligand_manifest"]),
    )

    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    run_directory = output_paths["run_directory"]
    core_outputs = [path for key, path in output_paths.items() if key != "run_directory"]
    run_artifacts = [
        path
        for case in resolved_cases
        for seed in seeds
        for protocol in ("e4", "e8")
        for path in artifact_paths(case["case_id"], protocol, seed, run_directory)
    ]
    existing = [path for path in core_outputs + run_artifacts if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("search robustness outputs exist; use --overwrite after review")
    if args.overwrite:
        for path in existing:
            path.unlink()
    run_directory.mkdir(parents=True, exist_ok=True)

    vina_version = get_vina_version(input_paths["vina_executable"])
    raw_rows: list[dict[str, object]] = []
    protocols = (("e4", e4_config), ("e8", e8_config))
    expected_run_count = len(resolved_cases) * len(seeds) * len(protocols)
    wall_started = time.perf_counter()
    run_index = 0
    for case in resolved_cases:
        for seed in seeds:
            for protocol, search_config in protocols:
                run_index += 1
                print(
                    f"run {run_index}/{expected_run_count}: "
                    f"{case['case_id']} {protocol} seed={seed}",
                    flush=True,
                )
                raw_rows.append(
                    run_one(
                        case,
                        protocol,
                        search_config,
                        seed,
                        input_paths["vina_executable"],
                        run_directory,
                        vina_version,
                    )
                )
                write_csv(output_paths["raw_runs_csv"], raw_rows)
    wall_seconds = time.perf_counter() - wall_started

    case_summaries = [
        summarize_case(
            [row for row in raw_rows if row["case_id"] == case["case_id"]],
            seeds,
            bool(thresholds["flag_nonnegative_scores"]),
            float(thresholds["maximum_e4_seed_range_kcal_per_mol"]),
            float(thresholds["maximum_paired_e4_e8_delta_kcal_per_mol"]),
        )
        for case in resolved_cases
    ]
    write_csv(output_paths["case_summary_csv"], case_summaries)
    successful_runs = [row for row in raw_rows if row["status"] == "ok"]
    passing_cases = [row for row in case_summaries if row["acceptance_pass"]]
    execution_ok = len(successful_runs) == expected_run_count
    acceptance_ok = len(passing_cases) == len(case_summaries)
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": (
            "ok" if execution_ok and acceptance_ok
            else "completed_with_search_instability" if execution_ok
            else "partial_failure"
        ),
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "inputs": {
            key: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for key, path in input_paths.items()
        },
        "case_count": len(resolved_cases),
        "seeds": seeds,
        "protocols": {
            "e4": e4_config,
            "e8": e8_config,
        },
        "expected_run_count": expected_run_count,
        "successful_run_count": len(successful_runs),
        "failed_run_count": expected_run_count - len(successful_runs),
        "passing_case_count": len(passing_cases),
        "failed_acceptance_case_ids": [
            row["case_id"] for row in case_summaries if not row["acceptance_pass"]
        ],
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
