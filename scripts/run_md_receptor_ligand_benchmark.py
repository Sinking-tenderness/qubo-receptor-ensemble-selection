"""Run a resumable ligand benchmark across a prepared receptor manifest."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

try:
    from .batch_vina_docking import safe_filename
    from .build_score_matrix import build_wide_matrix, select_representative_scores
    from .prepare_receptor import file_sha256
except ImportError:
    from batch_vina_docking import safe_filename
    from build_score_matrix import build_wide_matrix, select_representative_scores
    from prepare_receptor import file_sha256


REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "purpose",
    "inputs",
    "input_sha256",
    "expected_receptor_count",
    "expected_ligand_count",
    "expected_label_counts",
    "docking",
    "search_quality_warnings",
    "outputs",
    "interpretation_boundary",
}
REQUIRED_INPUT_KEYS = {
    "receptor_manifest",
    "ligand_manifest",
    "vina_executable",
    "vina_config",
    "parallel_runner",
    "score_matrix_module",
}
REQUIRED_OUTPUT_KEYS = {
    "run_directory",
    "audited_ligand_manifest_csv",
    "receptor_run_manifest_csv",
    "combined_raw_scores_csv",
    "representative_long_csv",
    "score_matrix_csv",
    "search_warnings_csv",
    "summary_json",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def portable_manifest_path(value: str) -> Path:
    """Interpret repository-relative manifest paths on Windows or POSIX."""
    return Path(value.replace("\\", "/"))


def write_csv(path: Path, rows: list[dict[str, object] | dict[str, str]]) -> None:
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
        raise ValueError("MD receptor benchmark config must be a JSON object")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"benchmark config is missing keys: {', '.join(missing)}")
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    outputs = config["outputs"]
    docking = config["docking"]
    warnings = config["search_quality_warnings"]
    label_counts = config["expected_label_counts"]
    if not isinstance(inputs, dict) or not REQUIRED_INPUT_KEYS.issubset(inputs):
        raise ValueError("inputs is missing one or more benchmark paths")
    if not isinstance(hashes, dict) or not REQUIRED_INPUT_KEYS.issubset(hashes):
        raise ValueError("input_sha256 is missing one or more required hashes")
    if not isinstance(outputs, dict) or not REQUIRED_OUTPUT_KEYS.issubset(outputs):
        raise ValueError("outputs is missing one or more benchmark paths")
    if not isinstance(docking, dict):
        raise ValueError("docking must be a JSON object")
    if int(docking.get("workers", 0)) <= 0 or int(docking.get("max_total_cpu", 0)) <= 0:
        raise ValueError("workers and max_total_cpu must be positive")
    if docking.get("representative_method") not in {"pose_rank_1", "min_score"}:
        raise ValueError("unsupported representative_method")
    if int(docking.get("base_seed", 0)) <= 0:
        raise ValueError("base_seed must be positive")
    if not isinstance(warnings, dict):
        raise ValueError("search_quality_warnings must be a JSON object")
    if not isinstance(warnings.get("flag_nonnegative_scores"), bool):
        raise ValueError("flag_nonnegative_scores must be a JSON boolean")
    if float(warnings.get("maximum_delta_from_ligand_median_kcal_per_mol", 0.0)) <= 0:
        raise ValueError("ligand-median warning threshold must be positive")
    if not isinstance(label_counts, dict) or not label_counts:
        raise ValueError("expected_label_counts must be a non-empty JSON object")
    if any(int(value) <= 0 for value in label_counts.values()):
        raise ValueError("expected label counts must be positive")
    if int(config["expected_receptor_count"]) <= 0 or int(config["expected_ligand_count"]) <= 0:
        raise ValueError("expected receptor and ligand counts must be positive")
    return config


def audit_ligands(
    rows: list[dict[str, str]],
    expected_count: int,
    expected_label_counts: dict[str, object],
) -> list[dict[str, str]]:
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} ligands, got {len(rows)}")
    ligand_ids = [row["ligand_id"] for row in rows]
    if len(ligand_ids) != len(set(ligand_ids)):
        raise ValueError("ligand manifest contains duplicate ligand IDs")
    observed_labels: dict[str, int] = {}
    audited: list[dict[str, str]] = []
    for row in rows:
        ligand_id = row["ligand_id"]
        if row.get("pdbqt_status") != "ok":
            raise ValueError(f"ligand PDBQT preparation did not pass: {ligand_id}")
        path = portable_manifest_path(row["pdbqt_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = file_sha256(path)
        recorded_hash = row.get("pdbqt_sha256", "").strip().upper()
        if recorded_hash and recorded_hash != actual_hash:
            raise ValueError(f"ligand PDBQT SHA-256 differs for {ligand_id}")
        label = row["label"]
        observed_labels[label] = observed_labels.get(label, 0) + 1
        audited.append({
            **row,
            "pdbqt_path": path.as_posix(),
            "pdbqt_sha256": actual_hash,
        })
    normalized_expected = {key: int(value) for key, value in expected_label_counts.items()}
    if observed_labels != normalized_expected:
        raise ValueError(
            f"ligand label counts differ: expected {normalized_expected}, got {observed_labels}"
        )
    return audited


def score_table_is_complete(
    path: Path, expected_ligand_ids: set[str]
) -> bool:
    if not path.is_file():
        return False
    rows = read_csv(path)
    ok_ids = {row["ligand_id"] for row in rows if row.get("status") == "ok"}
    failed_ids = {row["ligand_id"] for row in rows if row.get("status") == "failed"}
    return ok_ids == expected_ligand_ids and not failed_ids


def annotate_search_warnings(
    rows: list[dict[str, object]],
    flag_nonnegative_scores: bool,
    maximum_delta_from_ligand_median: float,
) -> list[dict[str, object]]:
    scores_by_ligand: dict[str, list[float]] = {}
    for row in rows:
        if row["status"] == "ok" and row["representative_score"] != "":
            scores_by_ligand.setdefault(str(row["ligand_id"]), []).append(
                float(row["representative_score"])
            )
    medians = {
        ligand_id: statistics.median(scores)
        for ligand_id, scores in scores_by_ligand.items()
    }
    output: list[dict[str, object]] = []
    for row in rows:
        reasons: list[str] = []
        median_score: float | str = ""
        delta: float | str = ""
        if row["status"] == "ok" and row["representative_score"] != "":
            score = float(row["representative_score"])
            median_score = medians[str(row["ligand_id"])]
            delta = score - float(median_score)
            if flag_nonnegative_scores and score >= 0.0:
                reasons.append("nonnegative_vina_score")
            if float(delta) > maximum_delta_from_ligand_median:
                reasons.append("large_unfavorable_delta_from_ligand_median")
        output.append({
            **row,
            "ligand_median_score": "" if median_score == "" else round(float(median_score), 6),
            "delta_from_ligand_median": "" if delta == "" else round(float(delta), 6),
            "search_quality_warning": bool(reasons),
            "search_quality_warning_reasons": ";".join(reasons),
        })
    return output


def clean_expected_artifacts(
    receptor_rows: list[dict[str, str]],
    ligand_rows: list[dict[str, str]],
    run_directory: Path,
    core_outputs: list[Path],
) -> None:
    for path in core_outputs:
        if path.is_file():
            path.unlink()
    for receptor in receptor_rows:
        receptor_id = receptor["conformer_id"]
        receptor_directory = run_directory / "receptors" / receptor_id
        for path in (
            receptor_directory / "scores.csv",
            receptor_directory / "scores.checkpoint.csv",
        ):
            if path.is_file():
                path.unlink()
        for ligand in ligand_rows:
            stem = safe_filename(ligand["ligand_id"])
            for path in (
                receptor_directory / "poses" / f"{stem}_docked.pdbqt",
                receptor_directory / "logs" / f"{stem}_vina.log",
            ):
                if path.is_file():
                    path.unlink()


def receptor_provenance_fields(row: dict[str, str]) -> dict[str, str]:
    """Normalize optional provenance fields across MD and non-MD manifests."""
    return {
        "source_type": row.get("source_type", "unspecified"),
        "temporal_support_role": row.get("temporal_support_role", "not_applicable"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="verify all input hashes and manifests without creating outputs or running Vina",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = config["inputs"]
    expected_hashes = config["input_sha256"]
    outputs = config["outputs"]
    docking = config["docking"]
    warning_config = config["search_quality_warnings"]
    expected_label_counts = config["expected_label_counts"]
    assert isinstance(inputs, dict)
    assert isinstance(expected_hashes, dict)
    assert isinstance(outputs, dict)
    assert isinstance(docking, dict)
    assert isinstance(warning_config, dict)
    assert isinstance(expected_label_counts, dict)

    input_paths = {key: Path(str(value)) for key, value in inputs.items()}
    for key, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != str(expected_hashes[key]).upper():
            raise ValueError(f"input SHA-256 differs for {key}")

    receptor_rows = read_csv(input_paths["receptor_manifest"])
    expected_receptors = int(config["expected_receptor_count"])
    if len(receptor_rows) != expected_receptors:
        raise ValueError(f"expected {expected_receptors} receptors, got {len(receptor_rows)}")
    for row in receptor_rows:
        receptor_id = row["conformer_id"]
        preparation_status = row.get("preparation_status", row.get("status", ""))
        if preparation_status != "ok":
            raise ValueError(f"receptor preparation did not pass: {receptor_id}")
        path_value = row.get("receptor_pdbqt_path", row.get("receptor_pdbqt", ""))
        path = portable_manifest_path(path_value)
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != row["receptor_pdbqt_sha256"].upper():
            raise ValueError(f"receptor PDBQT SHA-256 differs: {receptor_id}")
        row["receptor_pdbqt_path"] = path.as_posix()

    ligand_rows = audit_ligands(
        read_csv(input_paths["ligand_manifest"]),
        int(config["expected_ligand_count"]),
        expected_label_counts,
    )
    expected_ligand_ids = {row["ligand_id"] for row in ligand_rows}
    if args.audit_only:
        audit_summary = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "status": "audit_only_ok",
            "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
            "inputs": {
                key: {"path": path.as_posix(), "sha256": file_sha256(path)}
                for key, path in input_paths.items()
            },
            "receptor_count": len(receptor_rows),
            "receptor_ids": [row["conformer_id"] for row in receptor_rows],
            "ligand_count": len(ligand_rows),
            "label_counts": {key: int(value) for key, value in expected_label_counts.items()},
            "expected_receptor_ligand_pairs": len(receptor_rows) * len(ligand_rows),
            "locked_test_manifest_rows": sum(
                row.get("benchmark_split") == "test" for row in ligand_rows
            ),
            "outputs_created": 0,
            "vina_jobs_started": 0,
        }
        print(json.dumps(audit_summary, indent=2, sort_keys=True), flush=True)
        return 0
    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    run_directory = output_paths["run_directory"]
    core_outputs = [path for key, path in output_paths.items() if key != "run_directory"]
    existing_core = [path for path in core_outputs if path.exists()]
    if existing_core and not args.resume and not args.overwrite:
        raise FileExistsError("benchmark outputs exist; use --resume or --overwrite")
    if args.overwrite:
        clean_expected_artifacts(receptor_rows, ligand_rows, run_directory, core_outputs)
    run_directory.mkdir(parents=True, exist_ok=True)
    write_csv(output_paths["audited_ligand_manifest_csv"], ligand_rows)

    workers = int(docking["workers"])
    max_total_cpu = int(docking["max_total_cpu"])
    base_seed = int(docking["base_seed"])
    run_rows: list[dict[str, object]] = []
    benchmark_started = time.perf_counter()
    for index, receptor in enumerate(receptor_rows, start=1):
        receptor_id = receptor["conformer_id"]
        receptor_directory = run_directory / "receptors" / receptor_id
        score_table = receptor_directory / "scores.csv"
        checkpoint_table = receptor_directory / "scores.checkpoint.csv"
        if args.resume and score_table_is_complete(score_table, expected_ligand_ids):
            print(f"reusing receptor {index}/{len(receptor_rows)}: {receptor_id}", flush=True)
            status = "ok"
            return_code = 0
            runtime = 0.0
            message = "complete_score_table_reused"
        else:
            command = [
                sys.executable,
                str(input_paths["parallel_runner"]),
                "--manifest", str(output_paths["audited_ligand_manifest_csv"]),
                "--vina-exe", str(input_paths["vina_executable"]),
                "--receptor", receptor["receptor_pdbqt_path"],
                "--receptor-id", receptor_id,
                "--config", str(input_paths["vina_config"]),
                "--output-dir", str(receptor_directory / "poses"),
                "--log-dir", str(receptor_directory / "logs"),
                "--score-table", str(score_table),
                "--checkpoint-table", str(checkpoint_table),
                "--workers", str(workers),
                "--max-total-cpu", str(max_total_cpu),
                "--base-seed", str(base_seed),
            ]
            if args.resume:
                command.append("--resume")
            print(f"docking receptor {index}/{len(receptor_rows)}: {receptor_id}", flush=True)
            started = time.perf_counter()
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            runtime = time.perf_counter() - started
            complete = score_table_is_complete(score_table, expected_ligand_ids)
            status = "ok" if completed.returncode == 0 and complete else "failed"
            return_code = completed.returncode
            message = (
                "vina_benchmark_ok" if status == "ok"
                else (completed.stderr or completed.stdout)[-500:]
            )
        run_rows.append({
            "receptor_id": receptor_id,
            **receptor_provenance_fields(receptor),
            "receptor_pdbqt_path": receptor["receptor_pdbqt_path"],
            "receptor_pdbqt_sha256": receptor["receptor_pdbqt_sha256"],
            "status": status,
            "return_code": return_code,
            "runtime_seconds": round(runtime, 3),
            "score_table_path": score_table.as_posix() if score_table.is_file() else "",
            "score_table_sha256": file_sha256(score_table) if score_table.is_file() else "",
            "message": message,
        })
        write_csv(output_paths["receptor_run_manifest_csv"], run_rows)

    raw_rows: list[dict[str, str]] = []
    for run in run_rows:
        if run["score_table_path"]:
            raw_rows.extend(read_csv(Path(str(run["score_table_path"]))))
    if not raw_rows:
        raise RuntimeError("benchmark produced no score rows")
    write_csv(output_paths["combined_raw_scores_csv"], raw_rows)
    representative = select_representative_scores(
        raw_rows, str(docking["representative_method"])
    )
    representative = annotate_search_warnings(
        representative,
        bool(warning_config["flag_nonnegative_scores"]),
        float(warning_config["maximum_delta_from_ligand_median_kcal_per_mol"]),
    )
    matrix = build_wide_matrix(representative)
    write_csv(output_paths["representative_long_csv"], representative)
    write_csv(output_paths["score_matrix_csv"], matrix)
    warning_rows = [row for row in representative if row["search_quality_warning"]]
    if warning_rows:
        write_csv(output_paths["search_warnings_csv"], warning_rows)
    elif output_paths["search_warnings_csv"].exists():
        output_paths["search_warnings_csv"].unlink()

    failed_runs = [row for row in run_rows if row["status"] != "ok"]
    failed_pairs = [row for row in representative if row["status"] != "ok"]
    successful_scores = [
        float(row["representative_score"])
        for row in representative
        if row["status"] == "ok"
    ]
    expected_pairs = len(receptor_rows) * len(ligand_rows)
    execution_ok = not failed_runs and not failed_pairs and len(representative) == expected_pairs
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": (
            "ok_with_search_warning" if execution_ok and warning_rows
            else "ok" if execution_ok
            else "partial_failure"
        ),
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "inputs": {
            key: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for key, path in input_paths.items()
        },
        "receptor_count": len(receptor_rows),
        "ligand_count": len(ligand_rows),
        "label_counts": {key: int(value) for key, value in expected_label_counts.items()},
        "expected_receptor_ligand_pairs": expected_pairs,
        "observed_receptor_ligand_pairs": len(representative),
        "successful_receptor_ligand_pairs": len(representative) - len(failed_pairs),
        "failed_receptor_ligand_pairs": len(failed_pairs),
        "failed_receptor_runs": [row["receptor_id"] for row in failed_runs],
        "search_quality_warning_count": len(warning_rows),
        "search_quality_warning_thresholds": warning_config,
        "search_quality_warning_reason_counts": {
            reason: sum(
                reason in str(row["search_quality_warning_reasons"]).split(";")
                for row in warning_rows
            )
            for reason in (
                "nonnegative_vina_score",
                "large_unfavorable_delta_from_ligand_median",
            )
        },
        "docking_parameters": {
            "workers": workers,
            "max_total_cpu": max_total_cpu,
            "base_seed": base_seed,
            "representative_method": docking["representative_method"],
            "config_path": input_paths["vina_config"].as_posix(),
        },
        "score_range_kcal_per_mol": {
            "minimum": min(successful_scores),
            "maximum": max(successful_scores),
        } if successful_scores else None,
        "measured_wall_runtime_seconds": round(time.perf_counter() - benchmark_started, 3),
        "outputs": {
            key: {
                "path": path.as_posix(),
                "sha256": file_sha256(path),
            }
            for key, path in output_paths.items()
            if key not in {"run_directory", "summary_json"} and path.is_file()
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
