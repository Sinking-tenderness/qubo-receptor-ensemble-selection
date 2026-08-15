"""Aggregate Stage28 PPARG trajectories into one compressed pocket-state pool."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cluster_md_pocket_frames import (
    BACKBONE_ATOM_NAMES,
    is_heavy_atom,
    pairwise_point_distances_angstrom,
    standardize_features,
)
from scripts.run_stage21_structure_aware_qubo import (
    descriptor,
    file_sha256,
    read_csv,
    read_json,
    rooted,
    write_csv,
    write_json,
)


def pocket_features(
    trajectory: Any, chain_id: str, pocket_numbers: list[int]
) -> tuple[np.ndarray, list[str]]:
    pocket_set = set(pocket_numbers)
    residues = [
        residue for residue in trajectory.topology.residues
        if residue.is_protein
        and residue.chain.chain_id == chain_id
        and residue.resSeq in pocket_set
    ]
    by_number: dict[int, Any] = {}
    for residue in residues:
        if residue.resSeq in by_number:
            raise ValueError(f"duplicate pocket residue {residue.resSeq}")
        by_number[residue.resSeq] = residue
    missing = sorted(pocket_set - set(by_number))
    if missing:
        raise ValueError(f"missing pocket residues: {missing}")
    ca_indices = []
    sidechain_groups = []
    residue_names = []
    for number in pocket_numbers:
        residue = by_number[number]
        atoms = list(residue.atoms)
        ca = [atom for atom in atoms if atom.name == "CA"]
        if len(ca) != 1:
            raise ValueError(f"pocket residue {number} lacks one CA")
        ca_indices.append(ca[0].index)
        sidechain = [
            atom.index for atom in atoms
            if atom.name not in BACKBONE_ATOM_NAMES and is_heavy_atom(atom)
        ]
        sidechain_groups.append(sidechain or [ca[0].index])
        residue_names.append(residue.name)
    ca_points = trajectory.xyz[:, np.asarray(ca_indices, dtype=int), :]
    centroids = np.stack(
        [trajectory.xyz[:, group, :].mean(axis=1) for group in sidechain_groups],
        axis=1,
    )
    first, second = np.triu_indices(len(pocket_numbers), k=1)
    ca_values = pairwise_point_distances_angstrom(ca_points)
    centroid_values = pairwise_point_distances_angstrom(centroids)
    names = [
        f"ca_{pocket_numbers[i]}_{residue_names[i]}__{pocket_numbers[j]}_{residue_names[j]}_angstrom"
        for i, j in zip(first, second)
    ] + [
        f"sidechain_centroid_{pocket_numbers[i]}_{residue_names[i]}__{pocket_numbers[j]}_{residue_names[j]}_angstrom"
        for i, j in zip(first, second)
    ]
    return np.concatenate((ca_values, centroid_values), axis=1), names


def collect(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    manifest_path = rooted(root, config["runtime"]["start_manifest"])
    starts = read_csv(manifest_path)
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items() if key in {"frame_manifest_csv", "feature_archive_npz", "distance_archive_npz", "ensemble_summary_json"}}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"ensemble outputs exist: {existing}")
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    import mdtraj as md
    from scipy.spatial.distance import pdist
    all_features = []
    feature_names: list[str] | None = None
    frame_rows = []
    source_records = []
    expected_per_start = int(config["sampling"]["expected_frames_per_start"])
    chain_id = str(config["target"]["protein_chain_id"])
    pocket_numbers = [int(value) for value in config["target"]["pocket_residue_numbers"]]
    global_index = 0
    for row in starts:
        qc_path = rooted(root, row["trajectory_qc_summary"])
        qc = read_json(qc_path)
        if qc.get("status") != "ok" or int(qc.get("frame_count", -1)) != expected_per_start:
            raise ValueError(f"{row['conformer_id']}: trajectory QC is incomplete")
        top_path = rooted(root, row["aligned_protein_pdb"])
        dcd_path = rooted(root, row["aligned_protein_dcd"])
        trajectory = md.load_dcd(str(dcd_path), top=str(top_path))
        if trajectory.n_frames != expected_per_start:
            raise ValueError(f"{row['conformer_id']}: aligned DCD frame count differs")
        values, names = pocket_features(trajectory, chain_id, pocket_numbers)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(f"{row['conformer_id']}: pocket feature names differ across starts")
        all_features.append(values)
        interval = float(config["dynamics"]["production_frame_interval_ps"])
        for local_index in range(trajectory.n_frames):
            frame_rows.append({
                "frame_id": f"PPARG_MD_{global_index:05d}",
                "global_frame_index": global_index,
                "start_index": int(row["start_index"]),
                "conformer_id": row["conformer_id"],
                "local_frame_index": local_index,
                "time_ps": round((local_index + 1) * interval, 4),
                "aligned_protein_dcd": row["aligned_protein_dcd"],
                "aligned_protein_pdb": row["aligned_protein_pdb"],
            })
            global_index += 1
        source_records.append({
            "conformer_id": row["conformer_id"],
            "qc_summary": descriptor(root, qc_path),
            "aligned_protein_pdb": descriptor(root, top_path),
            "aligned_protein_dcd": descriptor(root, dcd_path),
            "frame_count": trajectory.n_frames,
        })
    if feature_names is None:
        raise ValueError("no trajectory features were collected")
    raw = np.concatenate(all_features, axis=0).astype(np.float32)
    expected_total = int(config["sampling"]["expected_total_frames"])
    if raw.shape[0] != expected_total or len(frame_rows) != expected_total:
        raise ValueError("total Stage28 frame count differs")
    scaled, kept, means, standard_deviations = standardize_features(
        raw.astype(float), float(config["sampling"]["minimum_feature_sd_angstrom"])
    )
    scaled = scaled.astype(np.float32)
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(scaled)):
        raise ValueError("non-finite Stage28 feature value")
    distances = (
        pdist(scaled.astype(float), metric="euclidean") / math.sqrt(scaled.shape[1])
    ).astype(np.float32)
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("invalid Stage28 pairwise distance")
    write_csv(outputs["frame_manifest_csv"], frame_rows)
    np.savez_compressed(
        outputs["feature_archive_npz"],
        frame_ids=np.asarray([row["frame_id"] for row in frame_rows]),
        feature_names=np.asarray(feature_names),
        kept_feature_mask=kept,
        raw_features=raw,
        standardized_features=scaled,
        feature_means=means.astype(np.float32),
        feature_standard_deviations=standard_deviations.astype(np.float32),
    )
    np.savez_compressed(
        outputs["distance_archive_npz"],
        frame_ids=np.asarray([row["frame_id"] for row in frame_rows]),
        condensed_distances=distances,
        metric=np.asarray("rms_standardized_euclidean"),
    )
    summary = {
        "schema_version": "1.0",
        "status": "stage28_pparg_multistart_md_ensemble_complete",
        "config": descriptor(root, config_path),
        "start_manifest": descriptor(root, manifest_path),
        "start_count": len(starts),
        "frame_count": raw.shape[0],
        "raw_feature_count": raw.shape[1],
        "variable_feature_count": scaled.shape[1],
        "condensed_distance_count": len(distances),
        "distance_minimum": float(distances.min()),
        "distance_maximum": float(distances.max()),
        "sources": source_records,
        "outputs": {key: descriptor(root, path) for key, path in outputs.items() if key != "ensemble_summary_json"},
        "data_boundary": {"docking_scores_read": 0, "ligand_labels_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0},
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["ensemble_summary_json"], summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage28_pparg_multistart_md_ensemble.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    collect(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
