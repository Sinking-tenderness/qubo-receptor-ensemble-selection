"""Diagnose co-crystal contact-state diversity without training a new selector."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_pdb_heavy_atoms(path: Path) -> tuple[list[tuple[str, float, float, float]], set[str]]:
    atoms: list[tuple[str, float, float, float]] = []
    residues: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        atom_name = line[12:16].strip().upper()
        if element == "H" or (not element and atom_name.startswith("H")):
            continue
        residue = f"{line[21:22].strip() or '_'}:{line[17:20].strip()}:{line[22:26].strip()}"
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError as error:
            raise ValueError(f"invalid PDB coordinate in {path}: {line!r}") from error
        atoms.append((residue, x, y, z))
        residues.add(residue)
    if not atoms:
        raise ValueError(f"no heavy atoms parsed from {path}")
    return atoms, residues


def parse_sdf_heavy_atoms(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 5:
        raise ValueError(f"invalid SDF: {path}")
    try:
        atom_count = int(lines[3][0:3])
    except ValueError as error:
        raise ValueError(f"invalid SDF counts line: {path}") from error
    coordinates = []
    for line in lines[4 : 4 + atom_count]:
        element = line[31:34].strip().upper()
        if element == "H":
            continue
        try:
            coordinates.append((float(line[0:10]), float(line[10:20]), float(line[20:30])))
        except ValueError as error:
            raise ValueError(f"invalid SDF coordinate in {path}: {line!r}") from error
    values = np.asarray(coordinates, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"no finite heavy-atom coordinates in {path}")
    return values


def contact_set(protein_atoms: list[tuple[str, float, float, float]], ligand_xyz: np.ndarray, cutoff: float) -> set[str]:
    protein_xyz = np.asarray([[x, y, z] for _, x, y, z in protein_atoms], dtype=float)
    squared = np.sum((protein_xyz[:, None, :] - ligand_xyz[None, :, :]) ** 2, axis=2)
    mask = np.any(squared <= cutoff**2, axis=1)
    return {protein_atoms[index][0] for index in np.flatnonzero(mask)}


def jaccard_distance(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        raise ValueError("contact sets cannot both be empty")
    return 1.0 - len(left & right) / len(union)


def rank_percentile(values: list[float], value: float) -> float:
    return sum(item <= value for item in values) / len(values)


def find_preparation_dir(root: Path, preparation_root: str, receptor_id: str) -> Path:
    pdb_id = receptor_id.split("_")[1]
    candidates = sorted((root / preparation_root).glob(f"{pdb_id}_*"))
    if len(candidates) != 1:
        raise ValueError(f"expected one preparation directory for {receptor_id}, found {candidates}")
    return candidates[0]


def source_files(root: Path, spec: dict[str, Any], receptor_id: str) -> tuple[Path, Path]:
    directory = find_preparation_dir(root, str(spec["preparation_root"]), receptor_id)
    prefix = directory.name
    protein = directory / "receptor" / f"{receptor_id}_protein_only.pdb"
    ligand = directory / f"{prefix}_common_frame.sdf"
    if not protein.is_file() or not ligand.is_file():
        raise ValueError(f"missing co-crystal inputs for {receptor_id}")
    return protein, ligand


def load_pair_distances(root: Path, path: str) -> dict[frozenset[str], float]:
    rows = read_csv(root / path)
    distance_column = next(column for column in rows[0] if column.startswith("standardized_"))
    result: dict[frozenset[str], float] = {}
    for row in rows:
        result[frozenset((row["conformer_id_a"], row["conformer_id_b"]))] = float(row[distance_column])
    return result


def load_stage102_fold_rows(root: Path) -> list[dict[str, str]]:
    rows = read_csv(root / "analysis/stage102a_received_20260813/analysis/fold_metrics.csv")
    selected = [row for row in rows if row["method"] in {"single", "fixed_k2"}]
    if len(selected) != 20:
        raise ValueError(f"expected 20 Stage102A single/fixed-k2 rows, found {len(selected)}")
    return selected


def load_pair_outcomes(root: Path, target: str, receptors: list[str]) -> dict[frozenset[str], list[float]]:
    rows = load_stage102_fold_rows(root)
    singles = {(row["target_id"], row["outer_fold"]): row for row in rows if row["method"] == "single"}
    pairs = [row for row in rows if row["target_id"] == target and row["method"] == "fixed_k2"]
    outcomes: dict[frozenset[str], list[float]] = defaultdict(list)
    for row in pairs:
        baseline = singles[(target, row["outer_fold"])]
        subset = row["selected_receptors"].split("+")
        if len(subset) != 2 or any(value not in receptors for value in subset):
            raise ValueError(f"invalid fixed-k2 subset: {row['selected_receptors']}")
        outcomes[frozenset(subset)].append(float(row["gain_over_single"]))
    return outcomes


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    left_rank = np.argsort(np.argsort(np.asarray(left), kind="stable"), kind="stable").astype(float)
    right_rank = np.argsort(np.argsort(np.asarray(right), kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def run(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    contact_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    target_records: dict[str, dict[str, Any]] = {}
    for target, spec in config["inputs"].items():
        manifest = read_csv(root / spec["receptor_manifest"])
        receptors = [row["conformer_id"] for row in manifest if row["stage102a_gate_pass"] == "True"]
        if len(receptors) != int(spec["expected_receptor_count"]):
            raise ValueError(f"{target}: unexpected receptor count {len(receptors)}")
        cutoff = float(spec["contact_cutoff_angstrom"])
        contacts: dict[str, set[str]] = {}
        heavy_atom_counts: dict[str, int] = {}
        protein_hashes: dict[str, str] = {}
        ligand_hashes: dict[str, str] = {}
        for receptor in receptors:
            protein_path, ligand_path = source_files(root, spec, receptor)
            protein_atoms, _ = parse_pdb_heavy_atoms(protein_path)
            ligand_xyz = parse_sdf_heavy_atoms(ligand_path)
            contacts[receptor] = contact_set(protein_atoms, ligand_xyz, cutoff)
            heavy_atom_counts[receptor] = int(ligand_xyz.shape[0])
            protein_hashes[receptor] = sha256(protein_path)
            ligand_hashes[receptor] = sha256(ligand_path)
            contact_rows.append(
                {
                    "target_id": target,
                    "receptor_id": receptor,
                    "contact_cutoff_angstrom": cutoff,
                    "cocrystal_ligand_heavy_atom_count": heavy_atom_counts[receptor],
                    "protein_contact_residue_count": len(contacts[receptor]),
                    "contact_residues": "|".join(sorted(contacts[receptor])),
                    "protein_pdb_sha256": protein_hashes[receptor],
                    "ligand_sdf_sha256": ligand_hashes[receptor],
                }
            )
        distances = load_pair_distances(root, str(spec["pairwise_structural_distances"]))
        outcomes = load_pair_outcomes(root, target, receptors)
        distance_values = list(distances.values())
        target_pair_rows = []
        for left, right in itertools.combinations(receptors, 2):
            key = frozenset((left, right))
            if key not in distances:
                raise ValueError(f"{target}: missing structural distance for {left}/{right}")
            gains = outcomes.get(key, [])
            row = {
                "target_id": target,
                "receptor_i": left,
                "receptor_j": right,
                "structural_distance": distances[key],
                "structural_distance_percentile_all_pool": rank_percentile(distance_values, distances[key]),
                "contact_jaccard_distance": jaccard_distance(contacts[left], contacts[right]),
                "contact_intersection_count": len(contacts[left] & contacts[right]),
                "contact_union_count": len(contacts[left] | contacts[right]),
                "cocrystal_ligand_heavy_atom_count_difference": abs(heavy_atom_counts[left] - heavy_atom_counts[right]),
                "stage102a_fixed_k2_selection_count": len(gains),
                "stage102a_fixed_k2_mean_outer_gain": float(np.mean(gains)) if gains else None,
                "stage102a_fixed_k2_min_outer_gain": float(np.min(gains)) if gains else None,
                "stage102a_fixed_k2_all_gains": "|".join(f"{value:.12g}" for value in gains),
            }
            pair_rows.append(row)
            target_pair_rows.append(row)
        selected = [row for row in target_pair_rows if row["stage102a_fixed_k2_selection_count"] > 0]
        gains = [float(row["stage102a_fixed_k2_mean_outer_gain"]) for row in selected]
        contacts_values = [float(row["contact_jaccard_distance"]) for row in selected]
        structure_values = [float(row["structural_distance"]) for row in selected]
        positive = [row for row in selected if float(row["stage102a_fixed_k2_mean_outer_gain"]) > 0.0]
        target_rows.append(
            {
                "target_id": target,
                "receptor_count": len(receptors),
                "all_pair_count": len(target_pair_rows),
                "fixed_k2_observed_pair_count": len(selected),
                "positive_observed_pair_count": len(positive),
                "maximum_observed_pair_gain": float(max(gains)) if gains else None,
                "maximum_observed_pair_contact_jaccard_distance": float(max(contacts_values)) if contacts_values else None,
                "mean_observed_pair_contact_jaccard_distance": float(np.mean(contacts_values)) if contacts_values else None,
                "spearman_observed_pair_gain_vs_contact_distance": spearman(contacts_values, gains),
                "spearman_observed_pair_gain_vs_structural_distance": spearman(structure_values, gains),
            }
        )
        target_records[target] = {
            "receptors": receptors,
            "protein_hashes": protein_hashes,
            "ligand_hashes": ligand_hashes,
            "pair_count": len(target_pair_rows),
        }
    fa10_pair = next(
        row
        for row in pair_rows
        if row["target_id"] == "FA10"
        and {row["receptor_i"], row["receptor_j"]} == {"FA10_2PHB_aligned", "FA10_5K0H_aligned"}
    )
    egfr_recurrent = next(
        row
        for row in pair_rows
        if row["target_id"] == "EGFR"
        and {row["receptor_i"], row["receptor_j"]} == {"EGFR_3W32_aligned", "EGFR_3POZ_aligned"}
    )
    feasibility = {
        "fa10_positive_pair_has_all_positive_observed_gains": bool(
            fa10_pair["stage102a_fixed_k2_selection_count"] == 5
            and float(fa10_pair["stage102a_fixed_k2_min_outer_gain"]) > 0.0
        ),
        "fa10_positive_pair_is_not_selected_for_extreme_structural_distance": bool(
            float(fa10_pair["structural_distance_percentile_all_pool"]) < 0.8
        ),
        "contact_state_signal_is_not_authorized_as_predictor": True,
        "reason": "Only two targets and a small number of observed fixed-k2 pair identities are available. This stage can identify a plausible external feature family, not estimate a deployable threshold or QUBO coefficient.",
    }
    outputs = config["outputs"]
    write_csv(root / outputs["contact_csv"], contact_rows)
    write_csv(root / outputs["pair_csv"], pair_rows)
    write_csv(root / outputs["target_csv"], target_rows)
    result = {
        "schema_version": "1.0",
        "status": "stage106_cocrystal_contact_state_audit_complete",
        "evidence_status": "posthoc two-target feasibility audit; no independent confirmation and no trained selection rule.",
        "target_records": target_records,
        "focal_pairs": {"FA10_positive": fa10_pair, "EGFR_recurrent_negative": egfr_recurrent},
        "target_summary": target_rows,
        "feasibility": feasibility,
        "data_boundary": config["data_boundary"],
        "decision": {
            "new_target_protocol_authorized": False,
            "parp1_released": False,
            "quantum_hardware_authorized": False,
            "next_action": "Do not fit a contact-diversity threshold or retune a QUBO on EGFR/FA10. Use the contact-state record only to specify an independently reviewed untouched-target protocol, including a predeclared structural-state rule and a classical baseline."
        },
    }
    result_path = root / outputs["result_json"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    report = [
        "# Stage106 共晶配体接触状态审计",
        "",
        "本阶段只使用已冻结的受体蛋白坐标、其共晶配体坐标、预计算口袋结构距离，以及 Stage102A 已消费的折外结果。接触描述符在对接前可得；活性标签和 Uni-Dock 分数不参与描述符构建。",
        "",
        "| Target | Receptors | All pairs | Observed fixed-k2 pairs | Positive observed pairs | Max pair gain | Mean contact Jaccard distance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    report.extend(
        f"| {row['target_id']} | {row['receptor_count']} | {row['all_pair_count']} | {row['fixed_k2_observed_pair_count']} | {row['positive_observed_pair_count']} | {row['maximum_observed_pair_gain']:.6f} | {row['mean_observed_pair_contact_jaccard_distance']:.6f} |"
        for row in target_rows
    )
    report.extend(
        [
            "",
            "## 两个预先指定的对照",
            "",
            f"- FA10 正例 `2PHB + 5K0H`：五个外层折均选择，最小外层增益 `{float(fa10_pair['stage102a_fixed_k2_min_outer_gain']):+.6f}`；结构距离分位数 `{float(fa10_pair['structural_distance_percentile_all_pool']):.3f}`，接触 Jaccard 距离 `{float(fa10_pair['contact_jaccard_distance']):.3f}`。",
            f"- EGFR 反例 `3W32 + 3POZ`：出现 `{egfr_recurrent['stage102a_fixed_k2_selection_count']}` 折，平均外层增益 `{float(egfr_recurrent['stage102a_fixed_k2_mean_outer_gain']):+.6f}`；结构距离分位数 `{float(egfr_recurrent['structural_distance_percentile_all_pool']):.3f}`，接触 Jaccard 距离 `{float(egfr_recurrent['contact_jaccard_distance']):.3f}`。",
            "",
            "## 判断",
            "",
            "FA10 的正组合并非仅因几何距离最大而被解释，故简单 RMSD/max-min 规则不足。共晶接触状态提供了一个对接前可获得、具有生物学可解释性的候选信息源；但本阶段只有两个靶点，不能从中学习阈值或系数，也不能授权新对接、PARP1 或量子硬件。",
            "",
        ]
    )
    report_path = root / outputs["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage106_cocrystal_contact_state_audit.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    result = run(root, config)
    print(json.dumps({"status": result["status"], "focal_pairs": result["focal_pairs"], "decision": result["decision"]}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
