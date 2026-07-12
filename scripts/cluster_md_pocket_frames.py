"""Cluster aligned MD frames from invariant pocket geometry and export medoids."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform


BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O", "OXT"}
REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "trajectory_qc_experiment_id",
    "purpose",
    "conformer_id_prefix",
    "inputs",
    "protein_chain_id",
    "frame_interval_ps",
    "expected_frame_count",
    "pocket_residue_numbers",
    "features",
    "candidate_cluster_counts",
    "selected_cluster_count",
    "outputs",
    "interpretation_boundary",
}
REQUIRED_INPUT_KEYS = {
    "trajectory_qc_summary",
    "aligned_protein_pdb",
    "aligned_protein_dcd",
}
REQUIRED_OUTPUT_KEYS = {
    "feature_matrix_csv",
    "cluster_diagnostics_csv",
    "frame_assignments_csv",
    "medoid_manifest_csv",
    "summary_json",
    "medoid_directory",
}


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(config, dict):
        raise ValueError("pocket clustering configuration must be a JSON object")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"pocket clustering configuration is missing keys: {', '.join(missing)}")
    inputs = config["inputs"]
    outputs = config["outputs"]
    features = config["features"]
    if not isinstance(inputs, dict) or not REQUIRED_INPUT_KEYS.issubset(inputs):
        raise ValueError("inputs is missing one or more required trajectory paths")
    if not isinstance(outputs, dict) or not REQUIRED_OUTPUT_KEYS.issubset(outputs):
        raise ValueError("outputs is missing one or more required clustering paths")
    if not isinstance(features, dict):
        raise ValueError("features must be a JSON object")
    flag_names = (
        "include_ca_pairwise_distances",
        "include_sidechain_heavy_centroid_pairwise_distances",
    )
    if any(not isinstance(features.get(name), bool) for name in flag_names):
        raise ValueError("pocket distance feature flags must be JSON booleans")
    feature_flags = tuple(bool(features[name]) for name in flag_names)
    if not any(feature_flags):
        raise ValueError("at least one pocket distance feature block must be enabled")
    if float(features.get("minimum_feature_sd_angstrom", 0.0)) < 0.0:
        raise ValueError("minimum_feature_sd_angstrom must be non-negative")
    expected_frames = int(config["expected_frame_count"])
    if expected_frames < 3 or float(config["frame_interval_ps"]) <= 0.0:
        raise ValueError("expected_frame_count and frame_interval_ps must be positive")
    pocket = config["pocket_residue_numbers"]
    if (
        not isinstance(pocket, list)
        or len(pocket) < 3
        or any(not isinstance(value, int) or value <= 0 for value in pocket)
        or len(set(pocket)) != len(pocket)
    ):
        raise ValueError("pocket_residue_numbers must contain at least three unique positive integers")
    candidates = config["candidate_cluster_counts"]
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(not isinstance(value, int) or not 2 <= value < expected_frames for value in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        raise ValueError("candidate_cluster_counts must be unique integers between 2 and frame_count-1")
    selected = int(config["selected_cluster_count"])
    if selected not in candidates:
        raise ValueError("selected_cluster_count must occur in candidate_cluster_counts")
    if not str(config["protein_chain_id"]):
        raise ValueError("protein_chain_id must be non-empty")
    if not str(config["conformer_id_prefix"]):
        raise ValueError("conformer_id_prefix must be non-empty")
    return config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def pairwise_point_distances_angstrom(points_nm: np.ndarray) -> np.ndarray:
    if points_nm.ndim != 3 or points_nm.shape[1] < 2 or points_nm.shape[2] != 3:
        raise ValueError("point coordinates must have shape (frames, points, 3)")
    if not np.all(np.isfinite(points_nm)):
        raise ValueError("point coordinates contain non-finite values")
    first, second = np.triu_indices(points_nm.shape[1], k=1)
    differences = points_nm[:, first, :] - points_nm[:, second, :]
    return np.linalg.norm(differences, axis=2) * 10.0


def standardize_features(
    values: np.ndarray, minimum_sd: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("feature matrix must have at least two rows and one column")
    if not np.all(np.isfinite(values)):
        raise ValueError("feature matrix contains non-finite values")
    means = values.mean(axis=0)
    standard_deviations = values.std(axis=0, ddof=0)
    kept = (standard_deviations >= minimum_sd) & (standard_deviations > 0.0)
    if not np.any(kept):
        raise ValueError("all pocket geometry features were removed by the SD threshold")
    scaled = (values[:, kept] - means[kept]) / standard_deviations[kept]
    return scaled, kept, means, standard_deviations


def medoid_indices(labels: np.ndarray, distances: np.ndarray) -> dict[int, int]:
    if labels.ndim != 1 or distances.shape != (len(labels), len(labels)):
        raise ValueError("cluster labels and distance matrix dimensions differ")
    medoids: dict[int, int] = {}
    for label in sorted(set(int(value) for value in labels)):
        indices = np.flatnonzero(labels == label)
        within_sums = distances[np.ix_(indices, indices)].sum(axis=1)
        medoids[label] = int(indices[int(np.argmin(within_sums))])
    return medoids


def relabel_by_medoid_time(labels: np.ndarray, distances: np.ndarray) -> np.ndarray:
    original_medoids = medoid_indices(labels, distances)
    old_labels = sorted(original_medoids, key=lambda label: original_medoids[label])
    mapping = {old: new for new, old in enumerate(old_labels)}
    return np.array([mapping[int(label)] for label in labels], dtype=int)


def silhouette_from_distances(labels: np.ndarray, distances: np.ndarray) -> float:
    unique = sorted(set(int(value) for value in labels))
    if len(unique) < 2 or distances.shape != (len(labels), len(labels)):
        raise ValueError("silhouette requires at least two clusters and a square distance matrix")
    values = np.zeros(len(labels), dtype=float)
    for index, label_value in enumerate(labels):
        same = np.flatnonzero(labels == label_value)
        same = same[same != index]
        if not len(same):
            values[index] = 0.0
            continue
        within = float(distances[index, same].mean())
        nearest_other = min(
            float(distances[index, labels == other].mean())
            for other in unique
            if other != int(label_value)
        )
        denominator = max(within, nearest_other)
        values[index] = 0.0 if denominator == 0.0 else (nearest_other - within) / denominator
    return float(values.mean())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def is_heavy_atom(atom: object) -> bool:
    element = getattr(atom, "element", None)
    if element is not None:
        return str(element.symbol).upper() != "H"
    return not str(getattr(atom, "name", "")).upper().startswith("H")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = config["inputs"]
    outputs = config["outputs"]
    feature_config = config["features"]
    assert isinstance(inputs, dict)
    assert isinstance(outputs, dict)
    assert isinstance(feature_config, dict)

    input_paths = {key: Path(str(value)) for key, value in inputs.items()}
    for path in input_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_files = {
        key: Path(str(outputs[key])) for key in REQUIRED_OUTPUT_KEYS if key != "medoid_directory"
    }
    medoid_directory = Path(str(outputs["medoid_directory"]))
    existing = [path for path in output_files.values() if path.exists()]
    existing_medoids = list(medoid_directory.glob("*.pdb")) if medoid_directory.exists() else []
    if (existing or existing_medoids) and not args.overwrite:
        raise FileExistsError("clustering outputs exist; use --overwrite after review")
    if args.overwrite:
        for path in [*existing, *existing_medoids]:
            path.unlink()
    for path in output_files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    medoid_directory.mkdir(parents=True, exist_ok=True)

    qc_summary = json.loads(input_paths["trajectory_qc_summary"].read_text(encoding="ascii"))
    if qc_summary.get("status") != "ok":
        raise ValueError("trajectory QC summary does not have status=ok")
    if qc_summary.get("experiment_id") != config["trajectory_qc_experiment_id"]:
        raise ValueError("trajectory QC experiment ID differs from clustering configuration")
    expected_frames = int(config["expected_frame_count"])
    if int(qc_summary.get("frame_count", -1)) != expected_frames:
        raise ValueError("trajectory QC frame count differs from clustering configuration")
    if float(qc_summary.get("frame_interval_ps", -1.0)) != float(config["frame_interval_ps"]):
        raise ValueError("trajectory QC frame interval differs from clustering configuration")
    qc_outputs = qc_summary.get("outputs")
    if not isinstance(qc_outputs, dict):
        raise ValueError("trajectory QC summary does not contain an outputs object")
    for key in ("aligned_protein_pdb", "aligned_protein_dcd"):
        if Path(str(qc_outputs.get(key, ""))).as_posix() != input_paths[key].as_posix():
            raise ValueError(f"clustering {key} differs from the trajectory QC output")

    import mdtraj as md

    trajectory = md.load_dcd(
        str(input_paths["aligned_protein_dcd"]),
        top=str(input_paths["aligned_protein_pdb"]),
    )
    if trajectory.n_frames != expected_frames:
        raise ValueError(
            f"aligned trajectory frame count mismatch: expected {expected_frames}, got {trajectory.n_frames}"
        )
    chain_id = str(config["protein_chain_id"])
    pocket_numbers = [int(value) for value in config["pocket_residue_numbers"]]
    pocket_set = set(pocket_numbers)
    residues = [
        residue for residue in trajectory.topology.residues
        if residue.is_protein
        and residue.chain.chain_id == chain_id
        and residue.resSeq in pocket_set
    ]
    residue_counts = {
        number: sum(residue.resSeq == number for residue in residues)
        for number in pocket_numbers
    }
    invalid_counts = {number: count for number, count in residue_counts.items() if count != 1}
    if invalid_counts:
        raise ValueError(f"each pocket residue must occur exactly once in chain {chain_id}: {invalid_counts}")
    residue_by_number = {residue.resSeq: residue for residue in residues}
    ordered_residues = [residue_by_number[number] for number in pocket_numbers]

    ca_indices: list[int] = []
    sidechain_groups: list[list[int]] = []
    sidechain_heavy_counts: list[int] = []
    residue_names: list[str] = []
    for residue in ordered_residues:
        atoms = list(residue.atoms)
        ca_atoms = [atom for atom in atoms if atom.name == "CA"]
        if len(ca_atoms) != 1:
            raise ValueError(f"residue {residue.resSeq} does not have exactly one CA atom")
        ca_indices.append(ca_atoms[0].index)
        sidechain_heavy = [
            atom.index for atom in atoms
            if atom.name not in BACKBONE_ATOM_NAMES and is_heavy_atom(atom)
        ]
        sidechain_groups.append(sidechain_heavy or [ca_atoms[0].index])
        sidechain_heavy_counts.append(len(sidechain_heavy))
        residue_names.append(residue.name)

    ca_points = trajectory.xyz[:, np.array(ca_indices, dtype=int), :]
    sidechain_centroids = np.stack(
        [trajectory.xyz[:, group, :].mean(axis=1) for group in sidechain_groups],
        axis=1,
    )
    first, second = np.triu_indices(len(pocket_numbers), k=1)
    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    if bool(feature_config.get("include_ca_pairwise_distances")):
        feature_blocks.append(pairwise_point_distances_angstrom(ca_points))
        feature_names.extend(
            f"ca_distance_{pocket_numbers[i]}_{pocket_numbers[j]}_angstrom"
            for i, j in zip(first, second)
        )
    if bool(feature_config.get("include_sidechain_heavy_centroid_pairwise_distances")):
        feature_blocks.append(pairwise_point_distances_angstrom(sidechain_centroids))
        feature_names.extend(
            f"sidechain_centroid_distance_{pocket_numbers[i]}_{pocket_numbers[j]}_angstrom"
            for i, j in zip(first, second)
        )
    features = np.concatenate(feature_blocks, axis=1)
    if features.shape[1] != len(feature_names):
        raise RuntimeError("pocket feature name and matrix dimensions differ")
    minimum_sd = float(feature_config.get("minimum_feature_sd_angstrom", 0.0))
    scaled, kept_mask, _feature_means, _feature_sds = standardize_features(features, minimum_sd)
    dropped_feature_names = [name for name, keep in zip(feature_names, kept_mask) if not keep]

    condensed = pdist(scaled, metric="euclidean")
    frame_distances = squareform(condensed)
    hierarchy = linkage(scaled, method="ward", metric="euclidean", optimal_ordering=True)
    diagnostics: list[dict[str, object]] = []
    labels_by_count: dict[int, np.ndarray] = {}
    for requested_count in sorted(int(value) for value in config["candidate_cluster_counts"]):
        labels = fcluster(hierarchy, t=requested_count, criterion="maxclust").astype(int) - 1
        labels = relabel_by_medoid_time(labels, frame_distances)
        actual_count = len(set(int(value) for value in labels))
        if actual_count != requested_count:
            raise RuntimeError(
                f"Ward clustering requested {requested_count} clusters but returned {actual_count}"
            )
        labels_by_count[requested_count] = labels
        sizes = [int(np.sum(labels == label)) for label in sorted(set(labels.tolist()))]
        medoids = medoid_indices(labels, frame_distances)
        diagnostics.append({
            "requested_cluster_count": requested_count,
            "actual_cluster_count": actual_count,
            "silhouette_score": round(silhouette_from_distances(labels, frame_distances), 6),
            "smallest_cluster_size": min(sizes),
            "largest_cluster_size": max(sizes),
            "singleton_cluster_count": sum(size == 1 for size in sizes),
            "medoid_frame_indices_json": json.dumps(
                [medoids[label] for label in sorted(medoids)], separators=(",", ":")
            ),
        })

    selected_count = int(config["selected_cluster_count"])
    selected_labels = labels_by_count[selected_count]
    selected_medoids = medoid_indices(selected_labels, frame_distances)
    frame_interval_ps = float(config["frame_interval_ps"])
    assignment_rows: list[dict[str, object]] = []
    for frame_index, label in enumerate(selected_labels):
        cluster_id = int(label)
        member_indices = np.flatnonzero(selected_labels == cluster_id)
        medoid_index = selected_medoids[cluster_id]
        assignment_rows.append({
            "frame_index": frame_index,
            "frame_number": frame_index + 1,
            "time_ps": round((frame_index + 1) * frame_interval_ps, 4),
            "cluster_id": cluster_id,
            "cluster_size": len(member_indices),
            "is_cluster_medoid": frame_index == medoid_index,
            "medoid_frame_index": medoid_index,
            "distance_to_medoid_standardized_euclidean": round(
                float(frame_distances[frame_index, medoid_index]), 6
            ),
            "mean_distance_to_cluster_members_standardized_euclidean": round(
                float(frame_distances[frame_index, member_indices].mean()), 6
            ),
        })

    feature_rows: list[dict[str, object]] = []
    for frame_index in range(trajectory.n_frames):
        row: dict[str, object] = {
            "frame_index": frame_index,
            "frame_number": frame_index + 1,
            "time_ps": round((frame_index + 1) * frame_interval_ps, 4),
        }
        row.update({
            name: round(float(value), 6)
            for name, value in zip(feature_names, features[frame_index])
        })
        feature_rows.append(row)

    medoid_rows: list[dict[str, object]] = []
    for cluster_id in sorted(selected_medoids):
        frame_index = selected_medoids[cluster_id]
        frame_number = frame_index + 1
        time_ps = frame_number * frame_interval_ps
        conformer_id = (
            f"{config['conformer_id_prefix']}_C{cluster_id:02d}_F{frame_index:03d}"
        )
        pdb_path = medoid_directory / f"{conformer_id}_{time_ps:07.1f}ps.pdb"
        trajectory[frame_index].save_pdb(str(pdb_path), force_overwrite=True)
        cluster_size = int(np.sum(selected_labels == cluster_id))
        medoid_rows.append({
            "conformer_id": conformer_id,
            "source_type": "md_cluster_medoid",
            "clustering_experiment_id": config["experiment_id"],
            "cluster_id": cluster_id,
            "cluster_size": cluster_size,
            "frame_index": frame_index,
            "frame_number": frame_number,
            "time_ps": round(time_ps, 4),
            "pdb_path": pdb_path.as_posix(),
            "pdb_sha256": sha256(pdb_path),
            "preparation_status": "raw_md_medoid_requires_1AQ1_alignment_and_standard_receptor_preparation",
        })

    write_csv(output_files["feature_matrix_csv"], feature_rows)
    write_csv(output_files["cluster_diagnostics_csv"], diagnostics)
    write_csv(output_files["frame_assignments_csv"], assignment_rows)
    write_csv(output_files["medoid_manifest_csv"], medoid_rows)
    selected_diagnostic = next(
        row for row in diagnostics if int(row["requested_cluster_count"]) == selected_count
    )
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "trajectory_qc_experiment_id": config["trajectory_qc_experiment_id"],
        "config": {"path": args.config.as_posix(), "sha256": sha256(args.config)},
        "software_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "mdtraj": md.__version__,
        },
        "inputs": {
            key: {"path": path.as_posix(), "sha256": sha256(path)}
            for key, path in input_paths.items()
        },
        "frame_count": trajectory.n_frames,
        "frame_interval_ps": frame_interval_ps,
        "protein_atom_count": trajectory.n_atoms,
        "protein_chain_id": chain_id,
        "pocket_residue_count": len(pocket_numbers),
        "pocket_residues": [
            {
                "residue_number": number,
                "residue_name": name,
                "sidechain_heavy_atom_count": sidechain_count,
            }
            for number, name, sidechain_count in zip(
                pocket_numbers, residue_names, sidechain_heavy_counts
            )
        ],
        "raw_feature_count": len(feature_names),
        "retained_feature_count": int(kept_mask.sum()),
        "dropped_low_variance_feature_count": int((~kept_mask).sum()),
        "minimum_feature_sd_angstrom": minimum_sd,
        "dropped_low_variance_feature_names": dropped_feature_names,
        "feature_standardization": (
            f"per-feature population mean and SD over the {trajectory.n_frames} structural frames"
        ),
        "clustering_method": "Ward agglomerative clustering on standardized invariant pocket-distance features",
        "candidate_cluster_counts": sorted(int(value) for value in config["candidate_cluster_counts"]),
        "selected_cluster_count": selected_count,
        "selected_silhouette_score": selected_diagnostic["silhouette_score"],
        "selected_smallest_cluster_size": selected_diagnostic["smallest_cluster_size"],
        "selected_largest_cluster_size": selected_diagnostic["largest_cluster_size"],
        "selected_singleton_cluster_count": selected_diagnostic["singleton_cluster_count"],
        "medoids": medoid_rows,
        "outputs": {
            **{key: path.as_posix() for key, path in output_files.items()},
            "medoid_directory": medoid_directory.as_posix(),
        },
        "output_sha256": {
            key: sha256(path)
            for key, path in output_files.items()
            if key != "summary_json"
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    output_files["summary_json"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
