"""Independently audit targeted MAPK14 e32 matrix diagnostics."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from pathlib import Path

try:
    from .prepare_receptor import file_sha256
    from .run_stage05_mk14_expanded_matrix_diagnostics import (
        load_context,
        summarize_case,
    )
except ImportError:
    from prepare_receptor import file_sha256
    from run_stage05_mk14_expanded_matrix_diagnostics import (
        load_context,
        summarize_case,
    )


POSE_SCORE = re.compile(r"^REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)", re.MULTILINE)
LOG_SCORE = re.compile(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+", re.MULTILINE)


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


def single_score(pattern: re.Pattern[str], text: str, path: Path) -> float:
    values = [float(match.group(1)) for match in pattern.finditer(text)]
    if len(values) != 1:
        raise ValueError(f"expected one top score in {path}, got {len(values)}")
    return values[0]


def pdbqt_heavy_coordinates(path: Path) -> list[tuple[str, float, float, float]]:
    atoms = []
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line.split()[-1] == "HD":
            continue
        atoms.append(
            (
                line[12:16].strip(),
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        )
    if not atoms:
        raise ValueError(f"pose contains no heavy atoms: {path}")
    return atoms


def fixed_order_rmsd(first: Path, second: Path) -> float:
    first_atoms = pdbqt_heavy_coordinates(first)
    second_atoms = pdbqt_heavy_coordinates(second)
    if [row[0] for row in first_atoms] != [row[0] for row in second_atoms]:
        raise ValueError("pose atom order differs")
    return math.sqrt(
        sum(
            (left[1] - right[1]) ** 2
            + (left[2] - right[2]) ** 2
            + (left[3] - right[3]) ** 2
            for left, right in zip(first_atoms, second_atoms, strict=True)
        )
        / len(first_atoms)
    )


def compare_case_results(
    expected: dict[str, object], observed: dict[str, object]
) -> None:
    exact_fields = (
        "case_id",
        "ligand_id",
        "receptor_id",
        "successful_runs",
        "diagnostic_classification",
        "rescue_passed",
    )
    for field in exact_fields:
        if expected[field] != observed[field]:
            raise ValueError(f"recomputed case field differs: {expected['case_id']} / {field}")
    numeric_fields = (
        "e32_seed_range_kcal_per_mol",
        "e32_median_score",
        "e32_minimum_score",
        "absolute_e16_e32_median_delta_kcal_per_mol",
        "absolute_e16_e32_minimum_delta_kcal_per_mol",
    )
    for field in numeric_fields:
        if not math.isclose(
            float(expected[field]),
            float(observed[field]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"recomputed case value differs: {expected['case_id']} / {field}")


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
    input_paths = {key: Path(str(value)) for key, value in inputs.items()}
    for key, path in input_paths.items():
        require_hash(path, hashes[key], key)
    require_hash(
        args.source_archive,
        config["source_archive"]["sha256"],
        "source archive",
    )

    context = load_context(input_paths["diagnostic_config"])
    cases = context["cases"]
    assert isinstance(cases, list)
    summary = load_json(input_paths["diagnostic_summary"])
    if summary.get("status") != expected["summary_status"]:
        raise ValueError("diagnostic summary status differs")
    for field, expected_field in (
        ("case_count", "case_count"),
        ("expected_vina_runs", "run_count"),
        ("successful_vina_runs", "successful_run_count"),
    ):
        if int(summary[field]) != int(expected[expected_field]):
            raise ValueError(f"diagnostic summary {field} differs")
    if bool(summary.get("all_cases_rescued")):
        raise ValueError("diagnostic summary unexpectedly rescued all cases")
    if int(summary.get("original_matrix_cells_replaced", -1)) != 0:
        raise ValueError("diagnostic replaced an original matrix cell")

    raw_rows = read_csv(input_paths["raw_runs"])
    if len(raw_rows) != int(expected["run_count"]):
        raise ValueError("raw diagnostic row count differs")
    seen: set[tuple[str, int]] = set()
    for row in raw_rows:
        key = (row["case_id"], int(row["seed"]))
        if key in seen:
            raise ValueError(f"duplicate diagnostic run: {key}")
        seen.add(key)
        if (
            row["status"] != "ok"
            or int(row["return_code"]) != 0
            or row["protocol_id"] != "e32"
            or int(row["exhaustiveness"]) != 32
        ):
            raise ValueError(f"diagnostic execution did not pass: {key}")
        pose_path = Path(row["pose_path"])
        log_path = Path(row["log_path"])
        require_hash(pose_path, row["pose_sha256"], f"{key} pose")
        require_hash(log_path, row["log_sha256"], f"{key} log")
        pose_text = pose_path.read_text(encoding="ascii", errors="replace")
        log_text = log_path.read_text(encoding="ascii", errors="replace")
        if (
            "AutoDock Vina v1.2.7" not in log_text
            or "Exhaustiveness: 32" not in log_text
            or "Grid size  : X 22 Y 24 Z 32" not in log_text
        ):
            raise ValueError(f"diagnostic log protocol differs: {key}")
        raw_score = float(row["docking_score"])
        if not math.isclose(
            single_score(POSE_SCORE, pose_text, pose_path),
            raw_score,
            rel_tol=0.0,
            abs_tol=0.005000001,
        ):
            raise ValueError(f"pose score differs: {key}")
        if not math.isclose(
            single_score(LOG_SCORE, log_text, log_path),
            raw_score,
            rel_tol=0.0,
            abs_tol=0.005000001,
        ):
            raise ValueError(f"log score differs: {key}")

    remote_cases = {
        str(row["case_id"]): row for row in summary["case_results"]
    }
    recomputed_cases = []
    for case in cases:
        result = summarize_case(
            case,
            [row for row in raw_rows if row["case_id"] == case["case_id"]],
            float(context["threshold"]),
        )
        compare_case_results(remote_cases[str(case["case_id"])], result)
        recomputed_cases.append(result)
    rescued = sum(bool(row["rescue_passed"]) for row in recomputed_cases)
    unresolved = len(recomputed_cases) - rescued
    if rescued != int(expected["rescued_case_count"]) or unresolved != int(
        expected["unresolved_case_count"]
    ):
        raise ValueError("recomputed rescue counts differ")

    pose_separation: dict[str, object] = {}
    for case_id in ("L000348_2BAJ_seed_range", "L000348_3KQ7_seed_range"):
        paths = [
            Path(row["pose_path"])
            for row in raw_rows
            if row["case_id"] == case_id
        ]
        pairwise = [
            fixed_order_rmsd(first, second)
            for first, second in itertools.combinations(paths, 2)
        ]
        pose_separation[case_id] = {
            "heavy_atom_count": len(pdbqt_heavy_coordinates(paths[0])),
            "maximum_fixed_order_rmsd_angstrom": max(pairwise),
            "interpretation": (
                "same_pose_basin"
                if max(pairwise) <= 2.0
                else "distinct_pose_basins"
            ),
        }

    output_path = Path(str(outputs["summary_json"]))
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; use --overwrite: {output_path}")
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "independent_e32_diagnostic_audit_ok_matrix_rejected",
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "source_archive": {
            "filename": args.source_archive.name,
            "sha256": file_sha256(args.source_archive),
        },
        "run_count": len(raw_rows),
        "successful_run_count": len(raw_rows),
        "case_count": len(recomputed_cases),
        "rescued_case_count": rescued,
        "unresolved_case_count": unresolved,
        "unresolved_case_ids": [
            row["case_id"] for row in recomputed_cases if not row["rescue_passed"]
        ],
        "pose_separation": pose_separation,
        "original_matrix_cells_replaced": 0,
        "e16_primary_matrix_authorized": False,
        "e16_sensitivity_matrix_authorized": False,
        "qubo_fitted": False,
        "enrichment_metrics_calculated": False,
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "next_action": "recompute the complete eight-receptor train matrix under one uniform stronger search protocol before QUBO fitting",
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
