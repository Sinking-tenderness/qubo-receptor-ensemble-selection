"""Independently recompute and audit the expanded MAPK14 redocking gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

try:
    from .evaluate_redocking_rmsd import calculate_pose_rmsds, parse_vina_affinities
except ImportError:
    from evaluate_redocking_rmsd import calculate_pose_rmsds, parse_vina_affinities


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def checked_record(record: dict[str, object]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(record["sha256"]).upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def run_audit(config_path: Path) -> dict[str, object]:
    config = read_json(config_path)
    summary_path = checked_record(config["inputs"]["summary"])
    results_path = checked_record(config["inputs"]["redocking_results"])
    summary = read_json(summary_path)
    if summary.get("status") != "expanded_redocking_gate_ok":
        raise ValueError("source redocking gate did not pass")
    boundary = summary.get("data_boundary")
    if not isinstance(boundary, dict) or any(int(value) != 0 for value in boundary.values()):
        raise ValueError("redocking gate crossed a frozen data boundary")

    expected = config["expected"]
    rows = read_csv(results_path)
    if len(rows) != int(expected["case_count"]):
        raise ValueError("redocking case count differs")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("redocking case IDs are not unique")
    threshold = float(expected["maximum_top_ranked_rmsd_angstrom"])
    tolerance = float(expected["numeric_comparison_tolerance"])

    recomputed: list[dict[str, object]] = []
    for row in rows:
        reference_sdf = Path(row["reference_sdf"])
        docked_pdbqt = Path(row["docked_pdbqt"])
        if file_sha256(reference_sdf) != row["reference_sdf_sha256"]:
            raise ValueError(f"reference SDF hash differs: {row['case_id']}")
        if file_sha256(docked_pdbqt) != row["docked_pdbqt_sha256"]:
            raise ValueError(f"docked PDBQT hash differs: {row['case_id']}")
        affinities = parse_vina_affinities(
            docked_pdbqt.read_text(encoding="ascii", errors="replace")
        )
        rmsds = calculate_pose_rmsds(reference_sdf, docked_pdbqt)
        if len(affinities) != len(rmsds) or len(rmsds) != int(row["pose_count"]):
            raise ValueError(f"pose count differs: {row['case_id']}")
        if abs(affinities[0] - float(row["top_ranked_affinity_kcal_per_mol"])) > tolerance:
            raise ValueError(f"top affinity differs: {row['case_id']}")
        if abs(rmsds[0] - float(row["top_ranked_rmsd_angstrom"])) > tolerance:
            raise ValueError(f"top RMSD differs: {row['case_id']}")
        if rmsds[0] > threshold or row["top_ranked_pose_success"].lower() != "true":
            raise ValueError(f"top-ranked pose failed: {row['case_id']}")
        recomputed.append(
            {
                "case_id": row["case_id"],
                "conformer_id": row["conformer_id"],
                "pose_count": len(rmsds),
                "top_ranked_affinity_kcal_per_mol": affinities[0],
                "top_ranked_rmsd_angstrom": round(rmsds[0], 6),
                "best_rmsd_angstrom": round(min(rmsds), 6),
                "top_ranked_pose_success": True,
            }
        )

    box_cases = list(summary["case_preparation"]) + list(
        summary["existing_case_revalidation_inputs"]
    )
    if len(box_cases) != len(rows):
        raise ValueError("box-audit case count differs")
    minimum_margin = min(
        float(case["box_audit"]["minimum_margin_angstrom"])
        for case in box_cases
    )
    required_margin = float(summary["common_box"]["minimum_crystal_pose_margin_angstrom"])
    if minimum_margin < required_margin:
        raise ValueError("a crystal pose is too close to a box face")

    result = {
        "schema_version": "1.0",
        "audit_id": config["audit_id"],
        "status": "independent_expanded_redocking_audit_ok",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_summary": {
            "path": summary_path.as_posix(),
            "sha256": file_sha256(summary_path),
        },
        "redocking_results": {
            "path": results_path.as_posix(),
            "sha256": file_sha256(results_path),
        },
        "case_count": len(rows),
        "minimum_crystal_pose_box_margin_angstrom": minimum_margin,
        "required_crystal_pose_box_margin_angstrom": required_margin,
        "maximum_allowed_top_ranked_rmsd_angstrom": threshold,
        "maximum_observed_top_ranked_rmsd_angstrom": max(
            float(row["top_ranked_rmsd_angstrom"]) for row in recomputed
        ),
        "recomputed_cases": recomputed,
        "data_boundary": boundary,
        "interpretation_boundary": "This audit verifies file identity, common-box margins, and redocking RMSD reproduction. It does not validate affinity ranking or enrichment.",
    }
    output_path = Path(str(config["output_json"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_audit(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
