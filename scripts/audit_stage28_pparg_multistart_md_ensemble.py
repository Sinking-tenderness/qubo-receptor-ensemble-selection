"""Audit the completed Stage28 PPARG multi-start MD feature pool."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import (
    descriptor,
    file_sha256,
    read_csv,
    read_json,
    rooted,
    write_json,
)


def audit(config_path: Path, root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    starts = read_csv(rooted(root, config["runtime"]["start_manifest"]))
    summary_path = rooted(root, config["outputs"]["ensemble_summary_json"])
    summary = read_json(summary_path)
    if summary.get("status") != "stage28_pparg_multistart_md_ensemble_complete":
        raise ValueError("unexpected Stage28 ensemble status")
    if summary["config"]["sha256"] != file_sha256(config_path):
        raise ValueError("Stage28 config hash differs")
    expected_starts = int(config["target"]["starting_structure_count"])
    expected_frames = int(config["sampling"]["expected_total_frames"])
    if len(starts) != expected_starts or int(summary["start_count"]) != expected_starts:
        raise ValueError("Stage28 start count differs")
    qc_records = []
    frame_total = 0
    for row in starts:
        for key in ("system_manifest", "equilibration_manifest", "production_manifest", "trajectory_qc_summary"):
            path = rooted(root, row[key])
            value = read_json(path)
            if value.get("status") != "ok":
                raise ValueError(f"{row['conformer_id']}: {key} is incomplete")
        qc_path = rooted(root, row["trajectory_qc_summary"])
        qc = read_json(qc_path)
        count = int(qc["frame_count"])
        if count != int(row["expected_frame_count"]):
            raise ValueError(f"{row['conformer_id']}: QC frame count differs")
        frame_total += count
        qc_records.append(descriptor(root, qc_path))
    if frame_total != expected_frames or int(summary["frame_count"]) != expected_frames:
        raise ValueError("Stage28 total frame count differs")
    feature_path = rooted(root, config["outputs"]["feature_archive_npz"])
    distance_path = rooted(root, config["outputs"]["distance_archive_npz"])
    frame_manifest_path = rooted(root, config["outputs"]["frame_manifest_csv"])
    frames = read_csv(frame_manifest_path)
    if len(frames) != expected_frames or len({row["frame_id"] for row in frames}) != expected_frames:
        raise ValueError("Stage28 frame manifest differs")
    with np.load(feature_path, allow_pickle=False) as archive:
        frame_ids = archive["frame_ids"]
        raw = archive["raw_features"]
        scaled = archive["standardized_features"]
        kept = archive["kept_feature_mask"]
        means = archive["feature_means"]
        sds = archive["feature_standard_deviations"]
    with np.load(distance_path, allow_pickle=False) as archive:
        distance_frame_ids = archive["frame_ids"]
        distances = archive["condensed_distances"]
        metric = str(archive["metric"])
    if raw.shape[0] != expected_frames or scaled.shape[0] != expected_frames:
        raise ValueError("Stage28 feature row count differs")
    if scaled.shape[1] != int(np.count_nonzero(kept)) or raw.shape[1] != len(kept):
        raise ValueError("Stage28 variable feature dimensions differ")
    if len(means) != raw.shape[1] or len(sds) != raw.shape[1]:
        raise ValueError("Stage28 feature normalization dimensions differ")
    if not np.array_equal(frame_ids, distance_frame_ids):
        raise ValueError("Stage28 feature and distance frame IDs differ")
    if list(frame_ids.astype(str)) != [row["frame_id"] for row in frames]:
        raise ValueError("Stage28 archive frame order differs")
    expected_distances = expected_frames * (expected_frames - 1) // 2
    if len(distances) != expected_distances or metric != "rms_standardized_euclidean":
        raise ValueError("Stage28 distance archive dimensions or metric differ")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(scaled)) or not np.all(np.isfinite(distances)):
        raise ValueError("Stage28 archive contains non-finite values")
    from scipy.spatial.distance import squareform
    square = squareform(distances)
    sample_pairs = [(0, expected_frames - 1), (17, 503), (199, 801), (400, 999)]
    for first, second in sample_pairs:
        expected = float(np.linalg.norm(scaled[first].astype(float) - scaled[second].astype(float)) / math.sqrt(scaled.shape[1]))
        if not math.isclose(float(square[first, second]), expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Stage28 sampled pairwise distance differs")
    if any(int(value) != 0 for value in summary["data_boundary"].values()):
        raise ValueError("nonzero Stage28 data boundary")
    passed = (
        len(starts) >= int(config["gate"]["minimum_distinct_start_count"])
        and frame_total == int(config["gate"]["required_total_frames"])
        and scaled.shape[1] > 0
    )
    result = {
        "schema_version": "1.0",
        "status": "stage28_pparg_multistart_md_ensemble_audit_ok",
        "config": descriptor(root, config_path),
        "ensemble_summary": descriptor(root, summary_path),
        "coverage": {
            "start_count": len(starts),
            "frame_count": expected_frames,
            "raw_feature_count": raw.shape[1],
            "variable_feature_count": scaled.shape[1],
            "pairwise_distance_count": len(distances),
            "sampled_pairwise_distances_recomputed": len(sample_pairs),
        },
        "inputs_verified": {"trajectory_qc_summaries": qc_records},
        "decision": {
            "stage28_md_pool_gate_passed": passed,
            "stage29_solver_scaling_authorized": passed,
            "new_docking_jobs_authorized": False,
            "quantum_hardware_authorized": False,
        },
        "checks": {
            "all_run_manifests_status_ok": True,
            "all_frame_counts_verified": True,
            "feature_and_distance_archives_verified": True,
            "data_boundary_zero": True,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(rooted(root, output_path.as_posix()), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage28_pparg_multistart_md_ensemble.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/stage28_pparg_multistart_md_ensemble_audit.json"))
    args = parser.parse_args()
    audit(args.config, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
