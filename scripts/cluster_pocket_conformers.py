"""Cluster conformers from pocket geometry features and select cluster medoids."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("pocket feature matrix is empty")
    return rows


def geometry_columns(rows: list[dict[str, str]]) -> list[str]:
    excluded = {"conformer_id", "source_type", "pdb_path", "chain"}
    columns = [
        name for name in rows[0]
        if name not in excluded and not name.endswith("__present")
    ]
    if not columns:
        raise ValueError("matrix has no continuous geometry columns")
    return columns


def numeric_matrix(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    values = np.full((len(rows), len(columns)), np.nan, dtype=float)
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            if row[column] != "":
                values[row_index, column_index] = float(row[column])
    medians = np.nanmedian(values, axis=0)
    if np.isnan(medians).any():
        raise ValueError("a geometry feature is missing for every conformer")
    missing = np.isnan(values)
    values[missing] = np.take(medians, np.where(missing)[1])
    return values


def medoids(labels: np.ndarray, distances: np.ndarray) -> dict[int, int]:
    output: dict[int, int] = {}
    for label in sorted(set(int(value) for value in labels)):
        indices = np.where(labels == label)[0]
        within = distances[np.ix_(indices, indices)].sum(axis=1)
        output[label] = int(indices[int(np.argmin(within))])
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-matrix", type=Path, required=True)
    parser.add_argument("--n-clusters", type=int, required=True)
    parser.add_argument("--assignment-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_csv(args.feature_matrix)
    if not 1 <= args.n_clusters <= len(rows):
        raise ValueError("n-clusters must be between 1 and the conformer count")
    columns = geometry_columns(rows)
    matrix = numeric_matrix(rows, columns)
    scaled = StandardScaler().fit_transform(matrix)
    distance = pairwise_distances(scaled, metric="euclidean")
    labels = AgglomerativeClustering(n_clusters=args.n_clusters, linkage="ward").fit_predict(scaled)
    cluster_medoids = medoids(labels, distance)
    output_rows = []
    for index, row in enumerate(rows):
        label = int(labels[index])
        output_rows.append({
            "conformer_id": row["conformer_id"],
            "source_type": row["source_type"],
            "cluster_id": label,
            "is_cluster_medoid": index == cluster_medoids[label],
            "mean_distance_to_cluster_members": round(float(distance[index, labels == label].mean()), 6),
        })
    args.assignment_output.parent.mkdir(parents=True, exist_ok=True)
    with args.assignment_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "conformer_count": len(rows),
        "n_clusters": args.n_clusters,
        "geometry_feature_count": len(columns),
        "cluster_medoids": {
            str(label): rows[index]["conformer_id"] for label, index in cluster_medoids.items()
        },
        "method": "median imputation, standard scaling, Ward agglomerative clustering, Euclidean medoid",
        "interpretation_note": "This is a structure-only clustering baseline. It does not optimize virtual-screening performance.",
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
