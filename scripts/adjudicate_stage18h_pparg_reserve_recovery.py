"""Independently adjudicate the Stage 18h PPARG reserve-recovery archives."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import file_sha256, read_csv  # noqa: F401 (deduped)
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path




def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))




def truth(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def assert_hash(path: Path, expected: object, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(expected).upper():
        raise ValueError(f"SHA-256 differs for {label}: {path}")


def assert_float(left: object, right: object, label: str) -> None:
    if abs(float(left) - float(right)) > 1e-9:
        raise ValueError(f"numeric value differs for {label}: {left} != {right}")


def compare_core_to_diagnostics(core_root: Path, diagnostics_root: Path) -> int:
    compared = 0
    for core_path in sorted(path for path in core_root.rglob("*") if path.is_file()):
        relative = core_path.relative_to(core_root)
        diagnostics_path = diagnostics_root / relative
        if not diagnostics_path.is_file():
            raise FileNotFoundError(f"diagnostics archive lacks core member: {relative}")
        if file_sha256(core_path) != file_sha256(diagnostics_path):
            raise ValueError(f"core and diagnostics members differ: {relative}")
        compared += 1
    return compared


def adjudicate(
    root: Path,
    core_root: Path,
    diagnostics_root: Path,
    core_archive: Path,
    diagnostics_archive: Path,
) -> dict[str, object]:
    root = root.resolve()
    core_root = core_root.resolve()
    diagnostics_root = diagnostics_root.resolve()
    compared_file_count = compare_core_to_diagnostics(core_root, diagnostics_root)

    relative_config = Path("configs/stage18h_pparg_posthoc_reserve_redocking.json")
    relative_summary = Path("data/stage18h_pparg_posthoc_reserve_redocking_summary.json")
    relative_results = Path("data/processed/stage18h_pparg_posthoc_reserve_redocking_results.csv")
    relative_gate = Path("data/processed/stage18h_pparg_posthoc_reserve_gate_results.csv")
    config_path = core_root / relative_config
    summary_path = core_root / relative_summary
    results_path = core_root / relative_results
    gate_path = core_root / relative_gate
    local_config = root / relative_config
    if file_sha256(config_path) != file_sha256(local_config):
        raise ValueError("executed Stage 18h config differs from the frozen local config")

    config = read_json(config_path)
    summary = read_json(summary_path)
    rows = read_csv(results_path)
    gate_rows = read_csv(gate_path)
    if summary["status"] != "stage18h_pparg_posthoc_reserve_recovery_gate_ok":
        raise ValueError("Stage 18h remote summary did not pass")
    if summary["experiment_class"] != "posthoc_exploratory_reserve_recovery":
        raise ValueError("Stage 18h experiment class differs")
    if summary["stage18e_confirmatory_gate"] != "closed_failed_14_of_24":
        raise ValueError("Stage 18e failed boundary was not preserved")
    if any(int(value) != 0 for value in dict(summary["data_boundary"]).values()):
        raise ValueError("Stage 18h summary crossed a protected data boundary")
    assert_hash(results_path, summary["outputs"]["redocking_results_csv"]["sha256"], "results table")
    assert_hash(gate_path, summary["outputs"]["receptor_gate_results_csv"]["sha256"], "gate table")

    expected = dict(config["expected"])
    receptor_ids = [str(value) for value in expected["receptor_ids"]]
    seed_records = list(dict(config["inputs"])["seeds"])
    seed_ids = [str(value["seed_id"]) for value in seed_records]
    seed_map = {str(value["seed_id"]): int(value["base_seed"]) for value in seed_records}
    expected_pairs = {(receptor_id, seed_id) for receptor_id in receptor_ids for seed_id in seed_ids}
    observed_pairs = {(row["conformer_id"], row["seed_id"]) for row in rows}
    if len(rows) != int(expected["redocking_pair_count"]) or observed_pairs != expected_pairs:
        raise ValueError("Stage 18h receptor-seed grid is incomplete")
    if any(row["status"] != "ok" for row in rows):
        raise ValueError("Stage 18h contains a failed result row")

    batch_count = 0
    pose_count = 0
    for row in rows:
        receptor_id = row["conformer_id"]
        seed_id = row["seed_id"]
        if int(row["base_seed"]) != seed_map[seed_id]:
            raise ValueError(f"base seed differs: {receptor_id}/{seed_id}")
        if int(row["known_warning_event_count"]) != 0:
            raise ValueError(f"known warning event remains: {receptor_id}/{seed_id}")
        if int(row["unresolved_warning_event_count"]) != 0:
            raise ValueError(f"unresolved warning event remains: {receptor_id}/{seed_id}")
        if int(row["pose_integrity_failure_count"]) != 0:
            raise ValueError(f"pose-integrity failure remains: {receptor_id}/{seed_id}")

        batch_path = core_root / row["batch_summary"]
        evaluation_path = core_root / row["evaluation_summary"]
        pose_path = diagnostics_root / row["docked_pdbqt"]
        assert_hash(batch_path, row["batch_summary_sha256"], "batch summary")
        assert_hash(evaluation_path, row["evaluation_summary_sha256"], "RMSD summary")
        assert_hash(pose_path, row["docked_pdbqt_sha256"], "docked pose")
        batch = read_json(batch_path)
        evaluation = read_json(evaluation_path)
        if batch["status"] != "ok" or batch["pose_integrity"]["failure_count"] != 0:
            raise ValueError(f"batch audit differs: {receptor_id}/{seed_id}")
        if batch["warning_adjudication"]["unresolved_warning_event_count"] != 0:
            raise ValueError(f"batch warning audit differs: {receptor_id}/{seed_id}")
        assert_hash(core_root / str(batch["log_path"]), batch["log_sha256"], "Uni-Dock log")
        assert_hash(core_root / str(batch["scores_path"]), batch["scores_sha256"], "batch scores")
        if evaluation["status"] != "ok" or int(evaluation["pose_count"]) != 1:
            raise ValueError(f"RMSD evaluation differs: {receptor_id}/{seed_id}")
        if evaluation["docked_pdbqt"]["sha256"] != row["docked_pdbqt_sha256"]:
            raise ValueError(f"RMSD pose identity differs: {receptor_id}/{seed_id}")
        assert_float(evaluation["top_ranked_affinity_kcal_per_mol"], row["top_ranked_affinity_kcal_per_mol"], "affinity")
        assert_float(evaluation["top_ranked_rmsd_angstrom"], row["top_ranked_rmsd_angstrom"], "RMSD")
        if truth(evaluation["top_ranked_pose_success"]) != truth(row["top_ranked_pose_success"]):
            raise ValueError(f"RMSD success differs: {receptor_id}/{seed_id}")
        rmsd_pose_table = evaluation_path.parent / "poses.csv"
        assert_hash(rmsd_pose_table, evaluation["pose_table"]["sha256"], "RMSD pose table")
        batch_count += 1
        pose_count += int(evaluation["pose_count"])

    gate_by_id = {row["conformer_id"]: row for row in gate_rows}
    if set(gate_by_id) != set(receptor_ids):
        raise ValueError("Stage 18h gate receptor set differs")
    threshold = float(config["redocking_gate"]["maximum_rmsd_angstrom"])
    minimum_successes = int(config["redocking_gate"]["minimum_successful_seeds_per_receptor"])
    recomputed_gate: list[dict[str, object]] = []
    for receptor_id in receptor_ids:
        selected = [row for row in rows if row["conformer_id"] == receptor_id]
        rmsds = [float(row["top_ranked_rmsd_angstrom"]) for row in selected]
        successes = sum(truth(row["top_ranked_pose_success"]) for row in selected)
        median_rmsd = statistics.median(rmsds)
        gate_pass = successes >= minimum_successes and median_rmsd <= threshold
        recorded = gate_by_id[receptor_id]
        if int(recorded["successful_seed_count"]) != successes or truth(recorded["gate_pass"]) != gate_pass:
            raise ValueError(f"recomputed gate differs: {receptor_id}")
        assert_float(recorded["median_top_ranked_rmsd_angstrom"], median_rmsd, "median RMSD")
        assert_float(recorded["maximum_top_ranked_rmsd_angstrom"], max(rmsds), "maximum RMSD")
        recomputed_gate.append({
            "conformer_id": receptor_id,
            "successful_seed_count": successes,
            "median_top_ranked_rmsd_angstrom": median_rmsd,
            "maximum_top_ranked_rmsd_angstrom": max(rmsds),
            "gate_pass": gate_pass,
        })

    passing = [row["conformer_id"] for row in recomputed_gate if row["gate_pass"]]
    failed = [row["conformer_id"] for row in recomputed_gate if not row["gate_pass"]]
    minimum_passing = int(config["redocking_gate"]["minimum_new_passing_receptor_count"])
    if len(passing) < minimum_passing:
        raise ValueError("independent Stage 18h gate did not pass")
    if len(passing) != int(summary["passed_reserve_receptor_count"]):
        raise ValueError("summary passing-reserve count differs")
    if int(summary["unresolved_warning_event_count"]) != 0 or int(summary["pose_integrity_failure_count"]) != 0:
        raise ValueError("summary contains unresolved technical failures")

    return {
        "schema_version": "1.0",
        "adjudication_id": "stage18h-pparg-posthoc-reserve-recovery-adjudication-20260801-v1",
        "status": "stage18h_pparg_posthoc_reserve_recovery_independently_adjudicated_ok",
        "archive_identity": {
            "core": {"path": core_archive.as_posix(), "sha256": file_sha256(core_archive), "size_bytes": core_archive.stat().st_size},
            "diagnostics": {"path": diagnostics_archive.as_posix(), "sha256": file_sha256(diagnostics_archive), "size_bytes": diagnostics_archive.stat().st_size},
        },
        "cross_archive_audit": {"core_files_matched_in_diagnostics": compared_file_count},
        "completion_audit": {
            "expected_pair_count": int(expected["redocking_pair_count"]),
            "observed_pair_count": len(rows),
            "unique_receptor_seed_pair_count": len(observed_pairs),
            "batch_summary_count": batch_count,
            "docked_pose_count": pose_count,
            "known_warning_event_count": 0,
            "unresolved_warning_event_count": 0,
            "pose_integrity_failure_count": 0,
        },
        "frozen_gate": {
            "maximum_rmsd_angstrom": threshold,
            "minimum_successful_seeds_per_receptor": minimum_successes,
            "minimum_new_passing_receptor_count": minimum_passing,
        },
        "recomputed_receptor_gate_results": recomputed_gate,
        "outcome": {
            "passing_reserve_receptor_count": len(passing),
            "failed_reserve_receptor_count": len(failed),
            "passing_reserve_receptors": passing,
            "failed_reserve_receptors": failed,
            "exploratory_recovery_gate_pass": True,
        },
        "data_boundary": {
            "ligand_labels_read": 0,
            "benchmark_docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "next_gate": "select two receptors from the seven passers by frozen structural distance to form an exploratory final 16",
        "interpretation_boundary": "Stage 18e remains a failed confirmatory gate. This successful Stage 18h result authorizes only a post-hoc exploratory PPARG final-16 pool.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--core-archive", type=Path, required=True)
    parser.add_argument("--diagnostics-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = adjudicate(
        args.root,
        args.core_root,
        args.diagnostics_root,
        args.core_archive.resolve(),
        args.diagnostics_archive.resolve(),
    )
    output = args.output if args.output.is_absolute() else args.root.resolve() / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
