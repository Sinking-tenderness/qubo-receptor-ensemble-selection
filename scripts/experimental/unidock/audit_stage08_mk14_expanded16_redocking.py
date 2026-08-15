"""Independently audit Stage 08 three-seed Uni-Dock redocking results."""

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
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"output SHA-256 differs: {path}")
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
    if summary.get("status") != "expanded16_unidock_redocking_gate_ok":
        raise ValueError("Stage 08 redocking source gate did not pass")
    source_config = dict(summary["config"])
    if str(source_config["sha256"]).upper() != file_sha256(config_path):
        raise ValueError("Stage 08 redocking source config hash differs")
    boundary = summary.get("data_boundary")
    if not isinstance(boundary, dict) or any(int(value) != 0 for value in boundary.values()):
        raise ValueError("Stage 08 redocking source crossed a data boundary")

    source_outputs = dict(summary["outputs"])
    results_path = checked_output(root, dict(source_outputs["redocking_results_csv"]))
    combined_path = checked_output(
        root, dict(source_outputs["combined_receptor_manifest_csv"])
    )
    results = read_csv(results_path)
    combined = read_csv(combined_path)
    expected = dict(config["expected"])
    if len(results) != int(expected["redocking_pair_count"]):
        raise ValueError("redocking result count differs")
    if len(combined) != int(expected["final_receptor_count"]):
        raise ValueError("combined receptor manifest count differs")
    if len({row["conformer_id"] for row in combined}) != len(combined):
        raise ValueError("combined receptor manifest contains duplicate IDs")
    for row in combined:
        receptor_path = rooted(root, row["receptor_pdbqt"])
        if row["status"] != "ok" or file_sha256(receptor_path) != row[
            "receptor_pdbqt_sha256"
        ].upper():
            raise ValueError(f"combined receptor input differs: {row['conformer_id']}")

    expected_ids = [str(value) for value in expected["new_receptor_ids"]]
    seeds = [str(row["seed_id"]) for row in dict(config["inputs"])["seeds"]]
    expected_keys = {(receptor_id, seed_id) for receptor_id in expected_ids for seed_id in seeds}
    observed_keys = {(row["conformer_id"], row["seed_id"]) for row in results}
    if observed_keys != expected_keys or len(results) != len(observed_keys):
        raise ValueError("Stage 08 redocking receptor/seed grid differs")

    threshold = float(dict(config["redocking_gate"])["maximum_rmsd_angstrom"])
    independently_recomputed: list[dict[str, object]] = []
    for row in results:
        reference = rooted(root, row["reference_sdf"])
        docked = rooted(root, row["docked_pdbqt"])
        if file_sha256(reference) != row["reference_sdf_sha256"].upper():
            raise ValueError("redocking reference SDF hash differs")
        if file_sha256(docked) != row["docked_pdbqt_sha256"].upper():
            raise ValueError("redocking pose hash differs")
        affinities = parse_vina_affinities(
            docked.read_text(encoding="ascii", errors="replace")
        )
        rmsds = calculate_pose_rmsds(reference, docked)
        if len(affinities) != 1 or len(rmsds) != 1:
            raise ValueError("Stage 08 redocking output must contain one pose")
        if abs(affinities[0] - float(row["top_ranked_affinity_kcal_per_mol"])) > 1e-9:
            raise ValueError("independently parsed affinity differs")
        if abs(rmsds[0] - float(row["top_ranked_rmsd_angstrom"])) > 1e-6:
            raise ValueError("independently recomputed RMSD differs")
        success = rmsds[0] <= threshold
        if success != parse_bool(row["top_ranked_pose_success"]):
            raise ValueError("independently recomputed redocking pass flag differs")
        if int(row["unresolved_warning_event_count"]) != 0:
            raise ValueError("source results contain an unresolved warning")
        if int(row["pose_integrity_failure_count"]) != 0:
            raise ValueError("source results contain a pose-integrity failure")
        independently_recomputed.append(
            {
                "conformer_id": row["conformer_id"],
                "seed_id": row["seed_id"],
                "affinity_kcal_per_mol": affinities[0],
                "rmsd_angstrom": rmsds[0],
                "success": success,
            }
        )

    minimum_successes = int(
        dict(config["redocking_gate"])["minimum_successful_seeds_per_receptor"]
    )
    receptor_checks: list[dict[str, object]] = []
    for receptor_id in expected_ids:
        rows = [
            row
            for row in independently_recomputed
            if row["conformer_id"] == receptor_id
        ]
        rmsds = [float(row["rmsd_angstrom"]) for row in rows]
        successes = sum(bool(row["success"]) for row in rows)
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
        raise ValueError("independently reconstructed receptor gate failed")

    result = {
        "schema_version": "1.0",
        "audit_id": "stage08-mk14-expanded16-unidock113-redocking-independent-audit-v1",
        "status": "independent_expanded16_unidock_redocking_audit_ok",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_summary": {
            "path": summary_path.relative_to(root).as_posix(),
            "sha256": file_sha256(summary_path),
        },
        "independently_recomputed_pair_count": len(independently_recomputed),
        "receptor_gate_checks": receptor_checks,
        "combined_receptor_count": len(combined),
        "unresolved_warning_event_count": 0,
        "pose_integrity_failure_count": 0,
        "data_boundary": {str(key): int(value) for key, value in boundary.items()},
        "next_gate": "a separate preregistered Train-696 x 16 receptors x 3 seeds Uni-Dock production bundle may now be created",
        "decision_boundary": "This independent audit verifies Stage 08 preparation, pose files, RMSD reproduction, and receptor admission only. It does not establish enrichment, QUBO benefit, or quantum advantage.",
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
