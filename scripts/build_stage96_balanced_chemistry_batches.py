from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fingerprint_distances(smiles: list[str]) -> tuple[np.ndarray, list[object]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    fingerprints = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"RDKit failed to parse SMILES: {value}")
        fingerprints.append(generator.GetFingerprint(molecule))
    count = len(fingerprints)
    distance = np.zeros((count, count), dtype=float)
    for index, fingerprint in enumerate(fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint, fingerprints[index + 1 :]
        )
        for right, similarity in enumerate(similarities, start=index + 1):
            distance[index, right] = distance[right, index] = 1.0 - float(similarity)
    return distance, fingerprints


def initial_medoids(distance: np.ndarray, count: int) -> list[int]:
    first = int(np.argmax(distance.mean(axis=1)))
    medoids = [first]
    while len(medoids) < count:
        minimum = distance[:, medoids].min(axis=1)
        minimum[medoids] = -1.0
        medoids.append(int(np.argmax(minimum)))
    return medoids


def balanced_assignment(
    distance: np.ndarray, medoids: list[int], capacities: list[int]
) -> np.ndarray:
    slots = [cluster for cluster, capacity in enumerate(capacities) for _ in range(capacity)]
    cost = distance[:, np.asarray([medoids[cluster] for cluster in slots])]
    rows, columns = linear_sum_assignment(cost)
    labels = np.empty(distance.shape[0], dtype=int)
    labels[rows] = np.asarray([slots[column] for column in columns], dtype=int)
    return labels


def balanced_medoids(distance: np.ndarray, count: int) -> tuple[np.ndarray, list[int]]:
    population = distance.shape[0]
    capacities = [population // count + int(i < population % count) for i in range(count)]
    medoids = initial_medoids(distance, count)
    labels = balanced_assignment(distance, medoids, capacities)
    for _ in range(20):
        updated = []
        for cluster in range(count):
            members = np.flatnonzero(labels == cluster)
            totals = distance[np.ix_(members, members)].sum(axis=1)
            updated.append(int(members[int(np.argmin(totals))]))
        new_labels = balanced_assignment(distance, updated, capacities)
        if updated == medoids and np.array_equal(new_labels, labels):
            break
        medoids, labels = updated, new_labels
    return labels, medoids


def build(root: Path, config_path: Path) -> dict[str, object]:
    root = root.resolve()
    config = json.loads(config_path.read_text(encoding="ascii"))
    output_rows = []
    summaries = {}
    for target_id, target in config["targets"].items():
        manifest_path = root / target["ligand_manifest"]["path"]
        if sha256(manifest_path) != target["ligand_manifest"]["sha256"]:
            raise ValueError(f"{target_id} ligand manifest hash differs")
        rows = sorted(read_csv(manifest_path), key=lambda row: row["ligand_id"])
        if len(rows) != int(target["expected_ligands"]):
            raise ValueError(f"{target_id} ligand count differs")
        smiles = [row.get("canonical_smiles") or row["smiles"] for row in rows]
        distance, fingerprints = fingerprint_distances(smiles)
        cluster_count = int(target["cluster_count"])
        labels, medoids = balanced_medoids(distance, cluster_count)
        members = {cluster: np.flatnonzero(labels == cluster).tolist() for cluster in range(cluster_count)}
        ordering = sorted(
            range(cluster_count),
            key=lambda cluster: tuple(rows[index]["ligand_id"] for index in members[cluster]),
        )
        canonical = {old: new for new, old in enumerate(ordering)}
        sizes = []
        centroid_similarities = {}
        width = max(2, len(str(cluster_count - 1)))
        for old_cluster in ordering:
            new_cluster = canonical[old_cluster]
            batch_id = f"{target_id}_CB{new_cluster:0{width}d}"
            indices = members[old_cluster]
            sizes.append(len(indices))
            medoid_index = medoids[old_cluster]
            centroid_similarities[batch_id] = {
                f"{target_id}_CB{canonical[other]:0{width}d}": float(
                    DataStructs.TanimotoSimilarity(
                        fingerprints[medoid_index], fingerprints[medoids[other]]
                    )
                )
                for other in ordering
            }
            for index in indices:
                output_rows.append(
                    {
                        "target_id": target_id,
                        "ligand_id": rows[index]["ligand_id"],
                        "chemistry_batch_id": batch_id,
                        "batch_size": len(indices),
                        "is_medoid": index == medoid_index,
                        "canonical_smiles": smiles[index],
                    }
                )
        summaries[target_id] = {
            "ligand_count": len(rows),
            "cluster_count": cluster_count,
            "minimum_batch_size": min(sizes),
            "maximum_batch_size": max(sizes),
            "capacity_difference": max(sizes) - min(sizes),
            "centroid_tanimoto_similarity": centroid_similarities,
        }
    output_path = root / config["outputs"]["cluster_manifest"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(output_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(sorted(output_rows, key=lambda row: (row["target_id"], row["chemistry_batch_id"], row["ligand_id"])))
    summary = {
        "schema_version": "1.0",
        "status": "stage96_balanced_chemistry_batches_structure_only_ok",
        "rdkit_version": rdBase.rdkitVersion,
        "labels_read": 0,
        "docking_score_rows_read": 0,
        "algorithm": config["chemistry_batch_freeze"]["algorithm"],
        "targets": summaries,
        "output_manifest": {
            "path": config["outputs"]["cluster_manifest"],
            "sha256": sha256(output_path),
        },
    }
    summary_path = root / config["outputs"]["cluster_summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage96_multitarget_adaptive_docking_replay.json"),
    )
    args = parser.parse_args()
    build(args.root, args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
