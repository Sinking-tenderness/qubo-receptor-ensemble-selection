"""Independently audit Stage 08b replacement redocking and final receptor pool."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

try:
    from scripts.evaluate_redocking_rmsd import (
        calculate_pose_rmsds,
        parse_vina_affinities,
    )
    from scripts.prepare_receptor import file_sha256
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.evaluate_redocking_rmsd import (
        calculate_pose_rmsds,
        parse_vina_affinities,
    )
    from scripts.prepare_receptor import file_sha256


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def rooted(root: Path, value: str) -> Path:
    path = (root / value.replace("\\", "/")).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path leaves audit root: {value}") from error
    return path


def checked_output(root: Path, descriptor: dict[str, object]) -> Path:
    path = rooted(root, str(descriptor["path"]))
    if not path.is_file() or file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"output identity differs: {path}")
    return path


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    raise ValueError(f"invalid Boolean value: {value}")


def run_audit(config_path: Path, root: Path) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    outputs = dict(config["outputs"])
    summary_path = rooted(root, str(outputs["summary_json"]))
    summary = read_json(summary_path)
    if summary.get("status") != "stage08b_expanded16_replacement_redocking_gate_ok":
        raise ValueError("Stage 08b replacement gate did not pass")
    if str(summary["config"]["sha256"]).upper() != file_sha256(config_path):
        raise ValueError("Stage 08b source config hash differs")
    boundary = dict(summary["data_boundary"])
    if any(int(value) != 0 for value in boundary.values()):
        raise ValueError("Stage 08b source crossed a data boundary")

    source_outputs = dict(summary["outputs"])
    results_path = checked_output(root, dict(source_outputs["redocking_results_csv"]))
    final_manifest_path = checked_output(
        root, dict(source_outputs["final_receptor_manifest_csv"])
    )
    rows = read_csv(results_path)
    final_rows = read_csv(final_manifest_path)
    expected = dict(config["expected"])
    replacement_ids = [str(value) for value in expected["replacement_receptor_ids"]]
    if len(rows) != int(expected["redocking_pair_count"]):
        raise ValueError("replacement redocking pair count differs")
    if len(final_rows) != int(expected["final_receptor_count"]):
        raise ValueError("final receptor count differs")
    if len({row["conformer_id"] for row in final_rows}) != len(final_rows):
        raise ValueError("final receptor manifest contains duplicate IDs")
    final_ids = [row["conformer_id"] for row in final_rows]
    excluded = [str(value) for value in expected["excluded_receptor_ids"]]
    if set(excluded).intersection(final_ids):
        raise ValueError("a permanently excluded receptor entered the final manifest")
    if not set(replacement_ids).issubset(final_ids):
        raise ValueError("a replacement receptor is absent from the final manifest")
    for row in final_rows:
        receptor = rooted(root, row["receptor_pdbqt"])
        if row["status"] != "ok" or file_sha256(receptor) != row[
            "receptor_pdbqt_sha256"
        ].upper():
            raise ValueError(f"final receptor identity differs: {row['conformer_id']}")

    seeds = [str(row["seed_id"]) for row in dict(config["inputs"])["seeds"]]
    keys = {(row["conformer_id"], row["seed_id"]) for row in rows}
    expected_keys = {(receptor_id, seed_id) for receptor_id in replacement_ids for seed_id in seeds}
    if keys != expected_keys or len(keys) != len(rows):
        raise ValueError("replacement receptor/seed grid differs")

    threshold = float(dict(config["redocking_gate"])["maximum_rmsd_angstrom"])
    recomputed: list[dict[str, object]] = []
    for row in rows:
        reference = rooted(root, row["reference_sdf"])
        docked = rooted(root, row["docked_pdbqt"])
        if file_sha256(reference) != row["reference_sdf_sha256"].upper():
            raise ValueError("replacement reference hash differs")
        if file_sha256(docked) != row["docked_pdbqt_sha256"].upper():
            raise ValueError("replacement pose hash differs")
        affinities = parse_vina_affinities(
            docked.read_text(encoding="ascii", errors="replace")
        )
        rmsds = calculate_pose_rmsds(reference, docked)
        if len(affinities) != 1 or len(rmsds) != 1:
            raise ValueError("replacement output must contain exactly one pose")
        if abs(affinities[0] - float(row["top_ranked_affinity_kcal_per_mol"])) > 1e-9:
            raise ValueError("replacement affinity differs")
        if abs(rmsds[0] - float(row["top_ranked_rmsd_angstrom"])) > 1e-6:
            raise ValueError("replacement RMSD differs")
        success = rmsds[0] <= threshold
        if success != parse_bool(row["top_ranked_pose_success"]):
            raise ValueError("replacement success flag differs")
        if int(row["unresolved_warning_event_count"]) != 0:
            raise ValueError("replacement result contains an unresolved warning")
        if int(row["pose_integrity_failure_count"]) != 0:
            raise ValueError("replacement result contains a pose-integrity failure")
        recomputed.append(
            {
                "conformer_id": row["conformer_id"],
                "seed_id": row["seed_id"],
                "rmsd_angstrom": rmsds[0],
                "success": success,
            }
        )

    minimum_successes = int(
        dict(config["redocking_gate"])["minimum_successful_seeds_per_receptor"]
    )
    receptor_checks: list[dict[str, object]] = []
    for receptor_id in replacement_ids:
        receptor_rows = [row for row in recomputed if row["conformer_id"] == receptor_id]
        rmsds = [float(row["rmsd_angstrom"]) for row in receptor_rows]
        successes = sum(bool(row["success"]) for row in receptor_rows)
        median_rmsd = statistics.median(rmsds)
        passed = successes >= minimum_successes and median_rmsd <= threshold
        receptor_checks.append(
            {
                "conformer_id": receptor_id,
                "successful_seed_count": successes,
                "median_rmsd_angstrom": median_rmsd,
                "maximum_rmsd_angstrom": max(rmsds),
                "gate_pass": passed,
            }
        )
    if not all(bool(row["gate_pass"]) for row in receptor_checks):
        raise ValueError("independent replacement gate failed")

    result = {
        "schema_version": "1.0",
        "audit_id": "stage08b-mk14-expanded16-replacement-redocking-independent-audit-v1",
        "status": "independent_stage08b_replacement_redocking_audit_ok",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_summary": {
            "path": summary_path.relative_to(root).as_posix(),
            "sha256": file_sha256(summary_path),
        },
        "independently_recomputed_pair_count": len(recomputed),
        "replacement_gate_checks": receptor_checks,
        "final_receptor_count": len(final_rows),
        "final_receptor_ids": final_ids,
        "permanently_excluded_receptor_ids": excluded,
        "unresolved_warning_event_count": 0,
        "pose_integrity_failure_count": 0,
        "data_boundary": {str(key): int(value) for key, value in boundary.items()},
        "next_gate": "a Train-696 x 16 receptors x 3 seeds Uni-Dock production bundle may now be preregistered",
        "decision_boundary": "This audit verifies receptor admission and final input identity only. It does not establish enrichment, QUBO benefit, or quantum advantage.",
    }
    output_path = rooted(root, str(outputs["audit_json"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run_audit(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
