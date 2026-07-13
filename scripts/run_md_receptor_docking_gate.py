"""Run a fixed four-ligand Vina gate across prepared MD medoid receptors."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    from .build_score_matrix import (
        build_wide_matrix,
        select_representative_scores,
    )
    from .prepare_receptor import file_sha256
except ImportError:
    from build_score_matrix import build_wide_matrix, select_representative_scores
    from prepare_receptor import file_sha256


REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "purpose",
    "inputs",
    "input_sha256",
    "expected_receptor_count",
    "ligands",
    "docking",
    "search_quality_warnings",
    "outputs",
    "interpretation_boundary",
}
REQUIRED_INPUT_KEYS = {
    "receptor_preparation_summary",
    "receptor_preparation_manifest",
    "ligand_manifest",
    "parent_af2_smoke_json",
    "vina_executable",
    "vina_config",
    "parallel_runner",
    "score_matrix_module",
}
REQUIRED_OUTPUT_KEYS = {
    "run_directory",
    "selected_ligand_manifest_csv",
    "receptor_run_manifest_csv",
    "combined_raw_scores_csv",
    "representative_long_csv",
    "score_matrix_csv",
    "summary_json",
}


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(config, dict):
        raise ValueError("MD docking gate config must be a JSON object")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"MD docking gate config is missing keys: {', '.join(missing)}")
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    outputs = config["outputs"]
    docking = config["docking"]
    warnings = config["search_quality_warnings"]
    if not isinstance(inputs, dict) or not REQUIRED_INPUT_KEYS.issubset(inputs):
        raise ValueError("inputs is missing one or more docking gate paths")
    if not isinstance(hashes, dict) or not REQUIRED_INPUT_KEYS.issubset(hashes):
        raise ValueError("input_sha256 is missing one or more required hashes")
    if not isinstance(outputs, dict) or not REQUIRED_OUTPUT_KEYS.issubset(outputs):
        raise ValueError("outputs is missing one or more docking gate paths")
    if not isinstance(docking, dict):
        raise ValueError("docking must be a JSON object")
    if not isinstance(warnings, dict):
        raise ValueError("search_quality_warnings must be a JSON object")
    if not isinstance(warnings.get("flag_nonnegative_scores"), bool):
        raise ValueError("flag_nonnegative_scores must be a JSON boolean")
    if float(warnings.get("maximum_delta_from_parent_af2_kcal_per_mol", 0.0)) <= 0.0:
        raise ValueError("maximum delta warning threshold must be positive")
    if int(docking.get("workers", 0)) <= 0 or int(docking.get("max_total_cpu", 0)) <= 0:
        raise ValueError("workers and max_total_cpu must be positive")
    if docking.get("representative_method") not in {"pose_rank_1", "min_score"}:
        raise ValueError("unsupported representative_method")
    ligands = config["ligands"]
    if not isinstance(ligands, list) or not ligands:
        raise ValueError("ligands must be a non-empty list")
    ligand_ids = [row.get("ligand_id") for row in ligands if isinstance(row, dict)]
    if len(ligand_ids) != len(ligands) or len(set(ligand_ids)) != len(ligand_ids):
        raise ValueError("ligand IDs must be present and unique")
    return config


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


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


def select_fixed_ligands(
    manifest_rows: list[dict[str, str]], requested: list[dict[str, object]]
) -> list[dict[str, str]]:
    by_id = {row["ligand_id"]: row for row in manifest_rows}
    selected: list[dict[str, str]] = []
    for request in requested:
        ligand_id = str(request["ligand_id"])
        if ligand_id not in by_id:
            raise ValueError(f"requested ligand is absent from manifest: {ligand_id}")
        row = dict(by_id[ligand_id])
        if row.get("label") != request["label"] or row.get("pdbqt_status") != "ok":
            raise ValueError(f"label or PDBQT status differs for {ligand_id}")
        path = Path(row["pdbqt_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = file_sha256(path)
        if actual_hash != str(request["pdbqt_sha256"]).upper():
            raise ValueError(f"PDBQT SHA-256 differs for {ligand_id}")
        row["pdbqt_sha256"] = actual_hash
        selected.append(row)
    return selected


def add_parent_comparison(
    rows: list[dict[str, object]], parent_scores: dict[str, float]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        ligand_id = str(row["ligand_id"])
        if ligand_id not in parent_scores:
            raise ValueError(f"parent AF2 score is missing for {ligand_id}")
        parent_score = float(parent_scores[ligand_id])
        representative = row["representative_score"]
        delta = "" if representative == "" else float(representative) - parent_score
        output.append({
            **row,
            "parent_af2_score": parent_score,
            "delta_from_parent_af2": "" if delta == "" else round(float(delta), 6),
        })
    return output


def annotate_search_warnings(
    rows: list[dict[str, object]],
    flag_nonnegative_scores: bool,
    maximum_delta_from_parent: float,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        reasons: list[str] = []
        if row["status"] == "ok" and row["representative_score"] != "":
            score = float(row["representative_score"])
            delta = float(row["delta_from_parent_af2"])
            if flag_nonnegative_scores and score >= 0.0:
                reasons.append("nonnegative_vina_score")
            if delta > maximum_delta_from_parent:
                reasons.append("large_unfavorable_delta_from_parent")
        output.append({
            **row,
            "search_quality_warning": bool(reasons),
            "search_quality_warning_reasons": ";".join(reasons),
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = config["inputs"]
    expected_hashes = config["input_sha256"]
    outputs = config["outputs"]
    docking = config["docking"]
    warning_config = config["search_quality_warnings"]
    requested_ligands = config["ligands"]
    assert isinstance(inputs, dict)
    assert isinstance(expected_hashes, dict)
    assert isinstance(outputs, dict)
    assert isinstance(docking, dict)
    assert isinstance(warning_config, dict)
    assert isinstance(requested_ligands, list)

    input_paths = {key: Path(str(value)) for key, value in inputs.items()}
    for path in input_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    for key, expected in expected_hashes.items():
        if file_sha256(input_paths[key]) != str(expected).upper():
            raise ValueError(f"input SHA-256 differs for {key}")

    preparation_summary = json.loads(
        input_paths["receptor_preparation_summary"].read_text(encoding="ascii")
    )
    if preparation_summary.get("status") != "ok":
        raise ValueError("receptor preparation summary does not have status=ok")
    receptor_rows = read_csv(input_paths["receptor_preparation_manifest"])
    expected_receptors = int(config["expected_receptor_count"])
    if len(receptor_rows) != expected_receptors:
        raise ValueError(f"expected {expected_receptors} receptors, got {len(receptor_rows)}")
    for row in receptor_rows:
        if row.get("preparation_status") != "ok":
            raise ValueError("receptor preparation manifest contains a failed receptor")
        path = Path(row["receptor_pdbqt_path"])
        if not path.is_file() or file_sha256(path) != row["receptor_pdbqt_sha256"].upper():
            raise ValueError(f"prepared receptor file or hash differs for {row['conformer_id']}")

    ligand_rows = select_fixed_ligands(
        read_csv(input_paths["ligand_manifest"]), requested_ligands
    )
    parent_smoke = json.loads(input_paths["parent_af2_smoke_json"].read_text(encoding="ascii"))
    parent_scores = {
        str(row["ligand_id"]): float(row["docking_score"])
        for row in parent_smoke["selected_ligands"]
    }
    if list(parent_scores) != [row["ligand_id"] for row in requested_ligands]:
        raise ValueError("parent AF2 ligand order differs from the docking gate")

    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    run_directory = output_paths["run_directory"]
    core_files = [path for key, path in output_paths.items() if key != "run_directory"]
    existing = [path for path in core_files if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("docking gate outputs exist; use --overwrite after review")
    if args.overwrite:
        for path in existing:
            path.unlink()
    run_directory.mkdir(parents=True, exist_ok=True)
    write_csv(output_paths["selected_ligand_manifest_csv"], ligand_rows)

    workers = int(docking["workers"])
    max_total_cpu = int(docking["max_total_cpu"])
    base_seed = int(docking["base_seed"])
    run_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, str]] = []
    for index, receptor in enumerate(receptor_rows, start=1):
        receptor_id = receptor["conformer_id"]
        receptor_directory = run_directory / "receptors" / receptor_id
        pose_directory = receptor_directory / "poses"
        log_directory = receptor_directory / "logs"
        score_table = receptor_directory / "scores.csv"
        checkpoint_table = receptor_directory / "scores.checkpoint.csv"
        if args.overwrite:
            for path in (score_table, checkpoint_table):
                if path.exists():
                    path.unlink()
            for directory in (pose_directory, log_directory):
                if directory.exists():
                    for path in directory.iterdir():
                        if path.is_file():
                            path.unlink()
        command = [
            sys.executable,
            str(input_paths["parallel_runner"]),
            "--manifest", str(output_paths["selected_ligand_manifest_csv"]),
            "--vina-exe", str(input_paths["vina_executable"]),
            "--receptor", receptor["receptor_pdbqt_path"],
            "--receptor-id", receptor_id,
            "--config", str(input_paths["vina_config"]),
            "--output-dir", str(pose_directory),
            "--log-dir", str(log_directory),
            "--score-table", str(score_table),
            "--checkpoint-table", str(checkpoint_table),
            "--workers", str(workers),
            "--max-total-cpu", str(max_total_cpu),
            "--base-seed", str(base_seed),
        ]
        print(f"docking receptor {index}/{len(receptor_rows)}: {receptor_id}", flush=True)
        started = time.perf_counter()
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        runtime = time.perf_counter() - started
        status = "ok" if completed.returncode == 0 and score_table.is_file() else "failed"
        score_hash = file_sha256(score_table) if score_table.is_file() else ""
        run_rows.append({
            "receptor_id": receptor_id,
            "temporal_support_role": receptor["temporal_support_role"],
            "receptor_pdbqt_path": receptor["receptor_pdbqt_path"],
            "receptor_pdbqt_sha256": receptor["receptor_pdbqt_sha256"],
            "status": status,
            "return_code": completed.returncode,
            "runtime_seconds": round(runtime, 3),
            "score_table_path": score_table.as_posix() if score_table.is_file() else "",
            "score_table_sha256": score_hash,
            "message": "vina_gate_ok" if status == "ok" else (completed.stderr or completed.stdout)[-500:],
        })
        if score_table.is_file():
            raw_rows.extend(read_csv(score_table))

    write_csv(output_paths["receptor_run_manifest_csv"], run_rows)
    if not raw_rows:
        raise RuntimeError("docking gate produced no score rows")
    write_csv(output_paths["combined_raw_scores_csv"], raw_rows)
    representative = select_representative_scores(
        raw_rows, str(docking["representative_method"])
    )
    representative = add_parent_comparison(representative, parent_scores)
    representative = annotate_search_warnings(
        representative,
        bool(warning_config["flag_nonnegative_scores"]),
        float(warning_config["maximum_delta_from_parent_af2_kcal_per_mol"]),
    )
    matrix = build_wide_matrix(representative)
    write_csv(output_paths["representative_long_csv"], representative)
    write_csv(output_paths["score_matrix_csv"], matrix)

    failed_runs = [row for row in run_rows if row["status"] != "ok"]
    failed_pairs = [row for row in representative if row["status"] != "ok"]
    search_warning_pairs = [
        row for row in representative if bool(row["search_quality_warning"])
    ]
    successful_scores = [
        float(row["representative_score"])
        for row in representative
        if row["status"] == "ok"
    ]
    delta_by_receptor: dict[str, list[float]] = {}
    for row in representative:
        if row["delta_from_parent_af2"] != "":
            delta_by_receptor.setdefault(str(row["receptor_id"]), []).append(
                float(row["delta_from_parent_af2"])
            )
    execution_ok = not failed_runs and not failed_pairs
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": (
            "ok_with_search_warning" if execution_ok and search_warning_pairs
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
        "expected_receptor_ligand_pairs": len(receptor_rows) * len(ligand_rows),
        "successful_receptor_ligand_pairs": len(representative) - len(failed_pairs),
        "failed_receptor_ligand_pairs": len(failed_pairs),
        "failed_receptor_runs": [row["receptor_id"] for row in failed_runs],
        "search_quality_warning_count": len(search_warning_pairs),
        "search_quality_warning_pairs": [
            {
                "receptor_id": row["receptor_id"],
                "ligand_id": row["ligand_id"],
                "representative_score": row["representative_score"],
                "parent_af2_score": row["parent_af2_score"],
                "delta_from_parent_af2": row["delta_from_parent_af2"],
                "reasons": row["search_quality_warning_reasons"],
            }
            for row in search_warning_pairs
        ],
        "vina_version": next(
            (row["software_version"] for row in raw_rows if row.get("software_version")), ""
        ),
        "docking_parameters": {
            "workers": workers,
            "max_total_cpu": max_total_cpu,
            "base_seed": base_seed,
            "representative_method": docking["representative_method"],
            "config_path": input_paths["vina_config"].as_posix(),
        },
        "search_quality_warning_thresholds": warning_config,
        "score_range_kcal_per_mol": {
            "minimum": min(successful_scores) if successful_scores else None,
            "maximum": max(successful_scores) if successful_scores else None,
        },
        "mean_score_delta_from_parent_af2_by_receptor": {
            receptor_id: round(sum(values) / len(values), 6)
            for receptor_id, values in sorted(delta_by_receptor.items())
        },
        "outputs": {
            key: {
                "path": path.as_posix(),
                "sha256": file_sha256(path) if path.is_file() else "",
            }
            for key, path in output_paths.items()
            if key not in {"run_directory", "summary_json"}
        },
        "run_directory": run_directory.as_posix(),
        "interpretation_note": config["interpretation_boundary"],
    }
    output_paths["summary_json"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True))
    return 0 if execution_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
