"""Small self-contained utilities shared by the Stage32b workflow."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCENARIOS = ("primary", "sensitivity", "seed0", "seed1", "seed2")
SEEDS = ("seed0", "seed1", "seed2")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {"path": path.resolve().relative_to(root.resolve()).as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def vectorized_bedroc(scores: np.ndarray, labels: np.ndarray, alpha: float) -> np.ndarray:
    if scores.ndim != 2 or labels.ndim != 1 or scores.shape[0] != len(labels):
        raise ValueError("BEDROC array dimensions differ")
    total = len(labels)
    active_total = int(labels.sum())
    if total == 0 or active_total == 0 or active_total == total:
        raise ValueError("BEDROC requires both active and decoy rows")
    order = np.argsort(scores, axis=0, kind="stable")
    ranked_labels = labels[order]
    weights = np.exp(-alpha * np.arange(1, total + 1, dtype=float) / total)
    random_expected = active_total * float(weights.mean())
    observed_rie = np.sum(ranked_labels * weights[:, None], axis=0) / random_expected
    maximum_rie = float(weights[:active_total].sum()) / random_expected
    minimum_rie = float(weights[-active_total:].sum()) / random_expected
    return np.asarray((observed_rie - minimum_rie) / (maximum_rie - minimum_rie), dtype=float)


def load_score_matrices(median_path: Path, minimum_path: Path, scores_path: Path, ligands: list[dict[str, str]], receptor_ids: list[str]) -> dict[str, np.ndarray]:
    ligand_index = {row["ligand_id"]: index for index, row in enumerate(ligands)}
    receptor_index = {value: index for index, value in enumerate(receptor_ids)}
    matrices = {scenario: np.full((len(ligands), len(receptor_ids)), np.nan) for scenario in SCENARIOS}
    for scenario, path in (("primary", median_path), ("sensitivity", minimum_path)):
        for row in read_csv(path):
            matrices[scenario][ligand_index[row["ligand_id"]]] = [float(row[value]) for value in receptor_ids]
    seen = set()
    for row in read_csv(scores_path):
        key = (row["seed_id"], row["ligand_id"], row["receptor_id"])
        if key in seen:
            raise ValueError(f"duplicate score key: {key}")
        seen.add(key)
        matrices[row["seed_id"]][ligand_index[row["ligand_id"]], receptor_index[row["receptor_id"]]] = float(row["gpu_score"])
    expected = 3 * len(ligands) * len(receptor_ids)
    if len(seen) != expected or any(not np.all(np.isfinite(matrix)) for matrix in matrices.values()):
        raise ValueError("score matrix coverage differs")
    return matrices


def normalize_from_train(matrix: np.ndarray) -> np.ndarray:
    output = np.empty_like(matrix, dtype=float)
    denominator = matrix.shape[0] + 1.0
    for receptor in range(matrix.shape[1]):
        frozen = np.sort(matrix[:, receptor], kind="stable")
        left = np.searchsorted(frozen, matrix[:, receptor], side="left")
        right = np.searchsorted(frozen, matrix[:, receptor], side="right")
        output[:, receptor] = (left + 0.5 * (right - left) + 0.5) / denominator
    return output
