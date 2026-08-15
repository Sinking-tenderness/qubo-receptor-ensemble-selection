from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform


CLUSTER_COUNTS = (6, 12, 24, 48)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def canonical_labels(raw_labels: np.ndarray, ligand_ids: list[str]) -> list[str]:
    members: dict[int, list[str]] = {}
    for label, ligand_id in zip(raw_labels.tolist(), ligand_ids):
        members.setdefault(int(label), []).append(ligand_id)
    ordered = sorted(members, key=lambda label: tuple(sorted(members[label])))
    mapping = {label: rank for rank, label in enumerate(ordered)}
    width = max(2, len(str(len(ordered) - 1)))
    return [f"S{mapping[int(label)]:0{width}d}" for label in raw_labels]


def build(input_path: Path, output_path: Path, summary_path: Path) -> dict[str, object]:
    rows = read_csv(input_path)
    active = sorted((row for row in rows if row["label"] == "active"), key=lambda row: row["ligand_id"])
    if len(rows) != 160 or len(active) != 80:
        raise ValueError("Stage95 expects the frozen 80-active/80-decoy panel")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    fingerprints = []
    for row in active:
        molecule = Chem.MolFromSmiles(row["canonical_smiles"])
        if molecule is None:
            raise ValueError(f"RDKit failed to parse {row['ligand_id']}")
        fingerprints.append(generator.GetFingerprint(molecule))
    distance = np.zeros((len(active), len(active)), dtype=float)
    for index, fingerprint in enumerate(fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints[index + 1 :])
        for offset, similarity in enumerate(similarities, start=index + 1):
            distance[index, offset] = distance[offset, index] = 1.0 - float(similarity)
    hierarchy = linkage(squareform(distance, checks=True), method="complete", optimal_ordering=True)
    ligand_ids = [row["ligand_id"] for row in active]
    columns: dict[int, list[str]] = {}
    cluster_stats = {}
    for count in CLUSTER_COUNTS:
        raw = cut_tree(hierarchy, n_clusters=[count]).reshape(-1)
        columns[count] = canonical_labels(raw, ligand_ids)
        sizes: dict[str, int] = {}
        for label in columns[count]:
            sizes[label] = sizes.get(label, 0) + 1
        if len(sizes) != count:
            raise ValueError(f"requested {count} clusters but obtained {len(sizes)}")
        cluster_stats[str(count)] = {
            "cluster_count": count,
            "minimum_size": min(sizes.values()),
            "maximum_size": max(sizes.values()),
            "singleton_count": sum(size == 1 for size in sizes.values()),
            "sizes": sizes,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ligand_id", "label", "canonical_smiles"] + [f"series_{count}" for count in CLUSTER_COUNTS]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(active):
            writer.writerow(
                {
                    "ligand_id": row["ligand_id"],
                    "label": row["label"],
                    "canonical_smiles": row["canonical_smiles"],
                    **{f"series_{count}": columns[count][index] for count in CLUSTER_COUNTS},
                }
            )
    summary = {
        "schema_version": "1.0",
        "status": "stage95_pparg_active_series_structure_only_ok",
        "input_manifest": {"path": input_path.as_posix(), "sha256": sha256(input_path)},
        "output_manifest": {"path": output_path.as_posix(), "sha256": sha256(output_path)},
        "active_ligand_count": len(active),
        "decoy_ligand_count_used_for_clustering": 0,
        "docking_score_rows_read": 0,
        "rdkit_version": rdBase.rdkitVersion,
        "fingerprint": "Morgan radius 2, 2048 bits, chirality disabled",
        "distance": "one minus Tanimoto similarity",
        "linkage": "complete",
        "cluster_statistics": cluster_stats,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/stage32_pparg_train160_ligand_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/stage95_pparg_active_series_manifest.csv"))
    parser.add_argument("--summary", type=Path, default=Path("data/stage95_pparg_active_series_summary.json"))
    args = parser.parse_args()
    build(args.input, args.output, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
