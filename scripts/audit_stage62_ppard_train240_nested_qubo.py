"""Fully recompute and audit the frozen Stage62 PPARD Train-240 analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts import run_stage62_ppard_train240_nested_qubo as runner


def checked(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or runner.sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage62 output identity differs: {path}")
    if path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage62 output size differs: {path}")
    return path


def expected_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in rows
    ]


def run(config_path: Path, result_path: Path, root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    result_path = result_path.resolve()
    config = runner.read_json(config_path)
    result = runner.read_json(result_path)
    if result.get("status") != "stage62_ppard_train240_frozen_nested_qubo_complete":
        raise ValueError("Stage62 source analysis did not complete")
    if checked(root, result["config"]).resolve() != config_path:
        raise ValueError("Stage62 result config differs")
    auditor_descriptor = dict(config["implementation"])["full_recomputation_auditor"]
    auditor_path = root / str(auditor_descriptor["path"])
    if auditor_path.resolve() != Path(__file__).resolve() or runner.sha256(
        auditor_path
    ) != str(auditor_descriptor["sha256"]).upper():
        raise ValueError("Stage62 recomputation auditor identity differs")
    output_paths = {
        key: checked(root, value) for key, value in dict(result["outputs"]).items()
    }

    recomputed = runner.compute_analysis(config, root)
    fingerprint = runner.canonical_sha256(
        {key: value for key, value in recomputed.items() if key != "merged_score_rows"}
    )
    if fingerprint != result["analysis_payload_sha256"]:
        raise ValueError("Stage62 full recomputation fingerprint differs")
    for key in ("input_audit", "final_choice", "performance", "decision"):
        result_key = "final_k_selection" if key == "final_choice" else key
        if recomputed[key] != result[result_key]:
            raise ValueError(f"Stage62 recomputed {key} differs")
    if runner.read_json(output_paths["model_record_json"]) != recomputed["model_record"]:
        raise ValueError("Stage62 model record differs")

    csv_checks = {
        "merged_scores_csv": recomputed["merged_score_rows"],
        "inner_k_metrics_csv": recomputed["inner_rows"],
        "inner_k_selection_csv": recomputed["inner_selection_rows"],
        "outer_k_metrics_csv": recomputed["outer_k_rows"],
        "nested_outer_metrics_csv": recomputed["nested_rows"],
        "objective_gap_cells_csv": recomputed["gap_rows"],
        "final_method_metrics_csv": recomputed["final_rows"],
    }
    row_counts: dict[str, int] = {}
    for key, expected in csv_checks.items():
        observed = runner.read_csv(output_paths[key])
        if observed != expected_csv_rows(expected):
            raise ValueError(f"Stage62 recomputed CSV differs: {key}")
        row_counts[key] = len(observed)

    audit = {
        "schema_version": "1.0",
        "status": "stage62_ppard_train240_full_recomputation_audit_ok",
        "source_result": runner.descriptor(root, result_path),
        "config": runner.descriptor(root, config_path),
        "analysis_payload_sha256": fingerprint,
        "all_output_hashes_exact": True,
        "all_csv_rows_exact": True,
        "model_record_exact": True,
        "row_counts": row_counts,
        "application_support_decision_exact": True,
        "optimization_novelty_decision_exact": True,
        "data_boundary": result["data_boundary"],
        "interpretation_boundary": result["interpretation_boundary"],
    }
    output_path = output_path if output_path.is_absolute() else root / output_path
    runner.write_json(output_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.result, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
