"""Independently audit targeted MAPK14 e64 consensus diagnostics."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

try:
    from .audit_stage05_expanded_e32_diagnostics import (
        LOG_SCORE,
        POSE_SCORE,
        fixed_order_rmsd,
        single_score,
    )
    from .prepare_receptor import file_sha256
    from .run_stage05_mk14_expanded_e64_consensus_diagnostics import (
        load_context,
        read_csv,
        summarize_case,
    )
except ImportError:
    from audit_stage05_expanded_e32_diagnostics import (
        LOG_SCORE,
        POSE_SCORE,
        fixed_order_rmsd,
        single_score,
    )
    from prepare_receptor import file_sha256
    from run_stage05_mk14_expanded_e64_consensus_diagnostics import (
        load_context,
        read_csv,
        summarize_case,
    )


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def require_hash(path: Path, expected: object, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(expected).upper():
        raise ValueError(f"{name} SHA-256 differs")


def compare_case(expected: dict[str, object], observed: dict[str, object]) -> None:
    for field in (
        "case_id",
        "ligand_id",
        "receptor_id",
        "successful_runs",
        "e64_consensus_replicate_count",
        "e64_consensus_passed",
        "diagnostic_classification",
    ):
        if expected[field] != observed[field]:
            raise ValueError(f"recomputed e64 case differs: {expected['case_id']} / {field}")
    for field in (
        "e64_minimum_score",
        "e64_median_score",
        "e64_maximum_score",
        "e64_median_minus_minimum",
        "absolute_e32_e64_minimum_delta",
        "absolute_e32_e64_median_delta",
    ):
        if not math.isclose(
            float(expected[field]),
            float(observed[field]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"recomputed e64 value differs: {expected['case_id']} / {field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    expected = config["expected"]
    outputs = config["outputs"]
    assert isinstance(inputs, dict)
    assert isinstance(hashes, dict)
    assert isinstance(expected, dict)
    assert isinstance(outputs, dict)
    paths = {key: Path(str(value)) for key, value in inputs.items()}
    for key, path in paths.items():
        require_hash(path, hashes[key], key)
    require_hash(
        args.source_archive,
        config["source_archive"]["sha256"],
        "source archive",
    )
    context = load_context(paths["diagnostic_config"])
    cases = context["cases"]
    assert isinstance(cases, list)
    summary = load_json(paths["diagnostic_summary"])
    if summary.get("status") != expected["summary_status"]:
        raise ValueError("e64 diagnostic status differs")
    for field, expected_field in (
        ("case_count", "case_count"),
        ("expected_vina_runs", "run_count"),
        ("successful_vina_runs", "successful_run_count"),
        ("e64_consensus_passed_case_count", "consensus_passed_case_count"),
    ):
        if int(summary[field]) != int(expected[expected_field]):
            raise ValueError(f"e64 diagnostic {field} differs")
    if bool(summary.get("all_cases_passed")):
        raise ValueError("e64 diagnostics unexpectedly passed every case")
    if int(summary.get("e32_matrix_cells_replaced", -1)) != 0:
        raise ValueError("e64 diagnostics replaced an e32 matrix cell")

    raw_rows = read_csv(paths["raw_runs"])
    if len(raw_rows) != int(expected["run_count"]):
        raise ValueError("e64 raw run count differs")
    seen: set[tuple[str, int]] = set()
    for row in raw_rows:
        key = (row["case_id"], int(row["seed"]))
        if key in seen:
            raise ValueError(f"duplicate e64 run: {key}")
        seen.add(key)
        if (
            row["status"] != "ok"
            or int(row["return_code"]) != 0
            or row["protocol_id"] != "e64"
            or int(row["exhaustiveness"]) != 64
        ):
            raise ValueError(f"e64 execution did not pass: {key}")
        pose_path = Path(row["pose_path"])
        log_path = Path(row["log_path"])
        require_hash(pose_path, row["pose_sha256"], f"{key} pose")
        require_hash(log_path, row["log_sha256"], f"{key} log")
        pose_text = pose_path.read_text(encoding="ascii", errors="replace")
        log_text = log_path.read_text(encoding="ascii", errors="replace")
        if (
            "AutoDock Vina v1.2.7" not in log_text
            or "Exhaustiveness: 64" not in log_text
            or "Grid size  : X 22 Y 24 Z 32" not in log_text
        ):
            raise ValueError(f"e64 log protocol differs: {key}")
        score = float(row["docking_score"])
        if not math.isclose(
            single_score(POSE_SCORE, pose_text, pose_path),
            score,
            rel_tol=0.0,
            abs_tol=0.005000001,
        ):
            raise ValueError(f"e64 pose score differs: {key}")
        if not math.isclose(
            single_score(LOG_SCORE, log_text, log_path),
            score,
            rel_tol=0.0,
            abs_tol=0.005000001,
        ):
            raise ValueError(f"e64 log score differs: {key}")

    remote_by_case = {
        str(row["case_id"]): row for row in summary["case_results"]
    }
    gate = dict(context["config"])["e64_consensus_gate"]
    assert isinstance(gate, dict)
    recomputed = []
    for case in cases:
        result = summarize_case(
            case,
            [row for row in raw_rows if row["case_id"] == case["case_id"]],
            gate,
        )
        compare_case(remote_by_case[str(case["case_id"])], result)
        recomputed.append(result)
    passed = sum(bool(row["e64_consensus_passed"]) for row in recomputed)
    persistent = len(recomputed) - passed
    if passed != int(expected["consensus_passed_case_count"]) or persistent != int(
        expected["persistent_case_count"]
    ):
        raise ValueError("e64 recomputed case counts differ")

    persistent_geometry = {}
    for row in recomputed:
        if row["e64_consensus_passed"]:
            continue
        pose_paths = [
            Path(raw["pose_path"])
            for raw in raw_rows
            if raw["case_id"] == row["case_id"]
        ]
        rmsds = [
            fixed_order_rmsd(first, second)
            for first, second in itertools.combinations(pose_paths, 2)
        ]
        persistent_geometry[str(row["case_id"])] = {
            "maximum_fixed_order_rmsd_angstrom": max(rmsds),
            "absolute_e32_e64_median_delta_kcal_per_mol": row[
                "absolute_e32_e64_median_delta"
            ],
        }
    median_deltas = [float(row["absolute_e32_e64_median_delta"]) for row in recomputed]
    persistent_median_deltas = [
        float(row["absolute_e32_e64_median_delta"])
        for row in recomputed
        if not row["e64_consensus_passed"]
    ]

    output_path = Path(str(outputs["summary_json"]))
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; use --overwrite: {output_path}")
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "independent_e64_audit_ok_uniform_e64_not_supported",
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "source_archive": {
            "filename": args.source_archive.name,
            "sha256": file_sha256(args.source_archive),
        },
        "run_count": len(raw_rows),
        "successful_run_count": len(raw_rows),
        "case_count": len(recomputed),
        "consensus_passed_case_count": passed,
        "persistent_case_count": persistent,
        "persistent_case_ids": [
            row["case_id"] for row in recomputed if not row["e64_consensus_passed"]
        ],
        "cross_protocol_median_delta_kcal_per_mol": {
            "maximum_all_cases": max(median_deltas),
            "count_above_0_5": sum(value > 0.5 for value in median_deltas),
            "maximum_persistent_cases": max(persistent_median_deltas),
        },
        "persistent_case_pose_geometry": persistent_geometry,
        "complete_uniform_e64_recomputation_supported": False,
        "e32_matrix_cells_replaced": 0,
        "qubo_fitted": False,
        "enrichment_metrics_calculated": False,
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "recommended_action": "stop brute-force exhaustiveness escalation and preregister an uncertainty-aware train-only receptor-selection gate using the unchanged uniform e32 seed matrices",
        "interpretation_note": config["interpretation_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
