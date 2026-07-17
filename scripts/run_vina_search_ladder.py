"""Run a resumable paired Vina exhaustiveness ladder on fixed high-risk cases."""

from __future__ import annotations

import argparse
import csv
import json
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
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
    required = {
        "schema_version",
        "experiment_id",
        "purpose",
        "inputs",
        "input_sha256",
        "protocols",
        "cases",
        "seeds",
        "execution",
        "acceptance_thresholds",
        "outputs",
        "interpretation_boundary",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"search ladder config is missing keys: {sorted(missing)}")
    return config


def protocol_configs(
    protocols: list[dict[str, object]], max_total_cpu: int
) -> list[dict[str, object]]:
    loaded: list[dict[str, object]] = []
    for protocol in protocols:
        protocol_id = str(protocol["protocol_id"])
        path = Path(str(protocol["config_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != str(protocol["config_sha256"]).upper():
            raise ValueError(f"search config hash differs: {protocol_id}")
        values = read_vina_config(path)
        exhaustiveness = int(values["exhaustiveness"])
        if exhaustiveness != int(protocol["exhaustiveness"]):
            raise ValueError(f"exhaustiveness differs for {protocol_id}")
        if int(values.get("cpu", "0")) <= 0 or int(values["cpu"]) > max_total_cpu:
            raise ValueError(f"invalid CPU setting for {protocol_id}")
        if values["num_modes"] != "1":
            raise ValueError("search ladder requires num_modes=1")
        loaded.append({**protocol, "path": path, "values": values})
    loaded.sort(key=lambda item: int(item["exhaustiveness"]))
    exhaustiveness_values = [int(item["exhaustiveness"]) for item in loaded]
    if len(loaded) < 2 or len(exhaustiveness_values) != len(set(exhaustiveness_values)):
        raise ValueError("protocols need at least two unique exhaustiveness values")
    fixed_keys = ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z", "cpu", "num_modes")
    reference = loaded[0]["values"]
    assert isinstance(reference, dict)
    for item in loaded[1:]:
        values = item["values"]
        assert isinstance(values, dict)
        differing = [key for key in fixed_keys if values.get(key) != reference.get(key)]
        if differing:
            raise ValueError(f"protocols differ outside exhaustiveness: {differing}")
    return loaded


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
        if receptor_id not in receptors or ligand_id not in ligands:
            raise ValueError(f"case input is absent: {receptor_id}/{ligand_id}")
        receptor = receptors[receptor_id]
        ligand = ligands[ligand_id]
        receptor_status = receptor.get("preparation_status", receptor.get("status", ""))
        receptor_path_value = receptor.get(
            "receptor_pdbqt_path", receptor.get("receptor_pdbqt", "")
        )
        if receptor_status != "ok" or ligand.get("pdbqt_status") != "ok":
            raise ValueError(f"case preparation did not pass: {receptor_id}/{ligand_id}")
        if ligand.get("label") != str(case["label"]):
            raise ValueError(f"case label differs: {ligand_id}")
        receptor_path = Path(receptor_path_value.replace("\\", "/"))
        ligand_path = Path(ligand["pdbqt_path"].replace("\\", "/"))
        for path in (receptor_path, ligand_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        receptor_hash = file_sha256(receptor_path)
        ligand_hash = file_sha256(ligand_path)
        if receptor_hash != str(case["receptor_pdbqt_sha256"]).upper():
            raise ValueError(f"case receptor hash differs: {receptor_id}")
        if ligand_hash != str(case["ligand_pdbqt_sha256"]).upper():
            raise ValueError(f"case ligand hash differs: {ligand_id}")
        resolved.append(
            {
                "case_id": str(case["case_id"]),
                "case_reason": str(case["case_reason"]),
                "receptor_id": receptor_id,
                "receptor_path": receptor_path.as_posix(),
                "receptor_sha256": receptor_hash,
                "ligand_id": ligand_id,
                "ligand_path": ligand_path.as_posix(),
                "ligand_sha256": ligand_hash,
                "label": ligand["label"],
            }
        )
    return resolved


def artifact_paths(run_directory: Path, case_id: str, protocol_id: str, seed: int) -> tuple[Path, Path]:
    stem = f"{safe_filename(case_id)}_{safe_filename(protocol_id)}_seed{seed}"
    return run_directory / "poses" / f"{stem}.pdbqt", run_directory / "logs" / f"{stem}.log"


def run_one(
    case: dict[str, str],
    protocol: dict[str, object],
    seed: int,
    vina_executable: Path,
    run_directory: Path,
    vina_version: str,
) -> dict[str, object]:
    protocol_id = str(protocol["protocol_id"])
    values = protocol["values"]
    assert isinstance(values, dict)
    pose_path, log_path = artifact_paths(run_directory, case["case_id"], protocol_id, seed)
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_vina_command(
        vina_executable,
        Path(case["receptor_path"]),
        Path(case["ligand_path"]),
        pose_path,
        values,
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
    return {
        **case,
        "protocol_id": protocol_id,
        "exhaustiveness": int(protocol["exhaustiveness"]),
        "seed": seed,
        "status": "ok" if ok else "failed",
        "return_code": completed.returncode,
        "docking_score": float(modes[0]["docking_score"]) if ok else "",
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
    protocol_ids: list[str],
    seeds: list[int],
    maximum_high_pair_delta: float,
    maximum_highest_seed_range: float | None = None,
) -> dict[str, object]:
    first = rows[0]
    scores = {
        (str(row["protocol_id"]), int(row["seed"])): float(row["docking_score"])
        for row in rows
        if row["status"] == "ok"
    }
    expected = len(protocol_ids) * len(seeds)
    high_pair_deltas = [
        abs(scores[(protocol_ids[-2], seed)] - scores[(protocol_ids[-1], seed)])
        for seed in seeds
        if (protocol_ids[-2], seed) in scores and (protocol_ids[-1], seed) in scores
    ]
    low_high_deltas = [
        abs(scores[(protocol_ids[0], seed)] - scores[(protocol_ids[-1], seed)])
        for seed in seeds
        if (protocol_ids[0], seed) in scores and (protocol_ids[-1], seed) in scores
    ]
    reasons: list[str] = []
    if len(scores) != expected:
        reasons.append("incomplete_runs")
    if not high_pair_deltas or max(high_pair_deltas) > maximum_high_pair_delta:
        reasons.append("highest_pair_delta_exceeded")
    protocol_statistics: dict[str, float | str] = {}
    for protocol_id in protocol_ids:
        values = [scores[(protocol_id, seed)] for seed in seeds if (protocol_id, seed) in scores]
        protocol_statistics[f"{protocol_id}_mean_score"] = (
            round(sum(values) / len(values), 6) if values else ""
        )
        protocol_statistics[f"{protocol_id}_seed_range"] = (
            round(max(values) - min(values), 6) if values else ""
        )
    highest_values = [
        scores[(protocol_ids[-1], seed)]
        for seed in seeds
        if (protocol_ids[-1], seed) in scores
    ]
    highest_seed_range = max(highest_values) - min(highest_values) if highest_values else None
    if (
        len(seeds) > 1
        and maximum_highest_seed_range is not None
        and (highest_seed_range is None or highest_seed_range > maximum_highest_seed_range)
    ):
        reasons.append("highest_protocol_seed_range_exceeded")
    return {
        "case_id": first["case_id"],
        "receptor_id": first["receptor_id"],
        "ligand_id": first["ligand_id"],
        "label": first["label"],
        "expected_runs": expected,
        "successful_runs": len(scores),
        **protocol_statistics,
        "maximum_absolute_highest_pair_delta": (
            round(max(high_pair_deltas), 6) if high_pair_deltas else ""
        ),
        "maximum_absolute_lowest_highest_delta": (
            round(max(low_high_deltas), 6) if low_high_deltas else ""
        ),
        "penultimate_protocol_stable": not reasons,
        "stability_failure_reasons": ";".join(reasons),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    config = load_config(args.config)
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    execution = config["execution"]
    outputs = config["outputs"]
    thresholds = config["acceptance_thresholds"]
    assert isinstance(inputs, dict)
    assert isinstance(hashes, dict)
    assert isinstance(execution, dict)
    assert isinstance(outputs, dict)
    assert isinstance(thresholds, dict)

    input_paths = {key: Path(str(value)) for key, value in inputs.items()}
    for key, path in input_paths.items():
        if not path.is_file() or file_sha256(path) != str(hashes[key]).upper():
            raise ValueError(f"input missing or hash differs: {key}")
    max_total_cpu = int(execution["max_total_cpu"])
    workers = int(execution["workers"])
    protocols = protocol_configs(config["protocols"], max_total_cpu)
    cpu_per_process = int(protocols[0]["values"]["cpu"])
    if workers < 1 or workers * cpu_per_process > max_total_cpu:
        raise ValueError("workers times protocol CPU exceeds maximum total CPU")
    seeds = [int(seed) for seed in config["seeds"]]
    cases = resolve_cases(
        config["cases"],
        read_csv(input_paths["receptor_manifest"]),
        read_csv(input_paths["ligand_manifest"]),
    )
    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    run_directory = output_paths["run_directory"]
    raw_path = output_paths["raw_runs_csv"]
    summary_path = output_paths["summary_json"]
    case_path = output_paths["case_summary_csv"]
    if args.overwrite:
        for path in (raw_path, summary_path, case_path):
            if path.exists():
                path.unlink()
    elif not args.resume and any(path.exists() for path in (raw_path, summary_path, case_path)):
        raise FileExistsError("search ladder outputs exist; use --resume or --overwrite")

    existing_rows: list[dict[str, object]] = []
    if args.resume and raw_path.is_file():
        existing_rows = [dict(row) for row in read_csv(raw_path) if row.get("status") == "ok"]
    completed_keys = {
        (str(row["case_id"]), str(row["protocol_id"]), int(row["seed"]))
        for row in existing_rows
    }
    tasks = [
        (case, protocol, seed)
        for case in cases
        for seed in seeds
        for protocol in protocols
        if (case["case_id"], str(protocol["protocol_id"]), seed) not in completed_keys
    ]
    vina_version = get_vina_version(input_paths["vina_executable"])
    rows = existing_rows

    def save() -> None:
        rows.sort(key=lambda row: (str(row["case_id"]), int(row["seed"]), int(row["exhaustiveness"])))
        write_csv(raw_path, rows)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_one,
                case,
                protocol,
                seed,
                input_paths["vina_executable"],
                run_directory,
                vina_version,
            ): (case["case_id"], protocol["protocol_id"], seed)
            for case, protocol, seed in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            key = (result["case_id"], result["protocol_id"], int(result["seed"]))
            rows = [
                row
                for row in rows
                if (row["case_id"], row["protocol_id"], int(row["seed"])) != key
            ]
            rows.append(result)
            save()

    protocol_ids = [str(protocol["protocol_id"]) for protocol in protocols]
    case_summaries = [
        summarize_case(
            [row for row in rows if row["case_id"] == case["case_id"]],
            protocol_ids,
            seeds,
            float(thresholds["maximum_highest_pair_delta_kcal_per_mol"]),
            (
                float(thresholds["maximum_highest_protocol_seed_range_kcal_per_mol"])
                if "maximum_highest_protocol_seed_range_kcal_per_mol" in thresholds
                else None
            ),
        )
        for case in cases
    ]
    write_csv(case_path, case_summaries)
    expected_runs = len(cases) * len(protocols) * len(seeds)
    successful_runs = sum(row["status"] == "ok" for row in rows)
    all_stable = all(row["penultimate_protocol_stable"] for row in case_summaries)
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": (
            "ok" if successful_runs == expected_runs and all_stable
            else "completed_with_search_instability" if successful_runs == expected_runs
            else "partial_failure"
        ),
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "case_count": len(cases),
        "protocols": [
            {"protocol_id": item["protocol_id"], "exhaustiveness": item["exhaustiveness"]}
            for item in protocols
        ],
        "seeds": seeds,
        "expected_run_count": expected_runs,
        "successful_run_count": successful_runs,
        "failed_run_count": expected_runs - successful_runs,
        "penultimate_protocol": protocol_ids[-2],
        "highest_protocol": protocol_ids[-1],
        "stable_case_count": sum(row["penultimate_protocol_stable"] for row in case_summaries),
        "provisional_minimum_exhaustiveness": (
            int(protocols[-2]["exhaustiveness"]) if all_stable else int(protocols[-1]["exhaustiveness"])
        ),
        "single_seed_boundary": len(seeds) == 1,
        "case_summaries": case_summaries,
        "outputs": {
            "raw_runs_csv": {"path": raw_path.as_posix(), "sha256": file_sha256(raw_path)},
            "case_summary_csv": {"path": case_path.as_posix(), "sha256": file_sha256(case_path)},
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0 if successful_runs == expected_runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
