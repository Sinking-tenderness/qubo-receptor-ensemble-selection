"""Diagnose structural coverage and cognate-ligand bias after the Stage51 gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from scipy.stats import mannwhitneyu, spearmanr


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def truth(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def verified_inputs(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key, descriptor in dict(config["inputs"]).items():
        path = root / str(descriptor["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != str(descriptor["sha256"]).upper():
            raise ValueError(f"Stage51b input hash differs: {key}")
        paths[key] = path
    return paths


def verify_implementation(root: Path, config: dict[str, Any]) -> None:
    descriptor = dict(config["implementation"])["analysis_script"]
    path = root / str(descriptor["path"])
    if path.resolve() != Path(__file__).resolve():
        raise ValueError("Stage51b analysis implementation path differs")
    if sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError("Stage51b analysis implementation hash differs")


def molecule_from_sdf(path: Path) -> Chem.Mol:
    block = path.read_text(encoding="utf-8", errors="replace").split("$$$$", 1)[0]
    molecule = Chem.MolFromMolBlock(block, removeHs=False, sanitize=True)
    if molecule is None:
        raise ValueError(f"RDKit could not parse {path}")
    return molecule


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [1.0] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        index = order[rank_index]
        rank = rank_index + 1
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    recalls = []
    for label in (False, True):
        mask = labels == label
        if not mask.any():
            raise ValueError("balanced accuracy requires both classes")
        recalls.append(float(np.mean(predictions[mask] == labels[mask])))
    return float(np.mean(recalls))


def nearest_neighbor_indices(matrix: np.ndarray, ids: list[str]) -> np.ndarray:
    indices = []
    for index in range(len(ids)):
        candidates = [
            (float(matrix[index, other]), ids[other], other)
            for other in range(len(ids))
            if other != index
        ]
        indices.append(min(candidates)[2])
    return np.asarray(indices, dtype=int)


def permutation_knn(
    labels: np.ndarray,
    neighbor_indices: np.ndarray,
    count: int,
    seed: int,
) -> dict[str, Any]:
    observed = balanced_accuracy(labels, labels[neighbor_indices])
    rng = np.random.default_rng(seed)
    null = np.empty(count, dtype=float)
    for index in range(count):
        permuted = rng.permutation(labels)
        null[index] = balanced_accuracy(permuted, permuted[neighbor_indices])
    return {
        "balanced_accuracy": observed,
        "permutation_p": float((1 + np.sum(null >= observed)) / (count + 1)),
        "null_mean": float(np.mean(null)),
        "null_standard_deviation": float(np.std(null)),
    }


def coverage_metrics(
    matrix: np.ndarray,
    selected: list[int],
    random_draws: int,
    seed: int,
) -> dict[str, Any]:
    selected_array = np.asarray(selected, dtype=int)
    nearest = np.min(matrix[:, selected_array], axis=1)
    observed_mean = float(np.mean(nearest))
    observed_radius = float(np.max(nearest))
    rng = np.random.default_rng(seed)
    random_means = np.empty(random_draws, dtype=float)
    random_radii = np.empty(random_draws, dtype=float)
    for index in range(random_draws):
        subset = rng.choice(len(matrix), size=len(selected), replace=False)
        distances = np.min(matrix[:, subset], axis=1)
        random_means[index] = float(np.mean(distances))
        random_radii[index] = float(np.max(distances))
    return {
        "selected_count": len(selected),
        "mean_nearest_distance": observed_mean,
        "coverage_radius": observed_radius,
        "mean_distance_random_percentile": float(
            (1 + np.sum(random_means <= observed_mean)) / (random_draws + 1)
        ),
        "coverage_radius_random_percentile": float(
            (1 + np.sum(random_radii <= observed_radius)) / (random_draws + 1)
        ),
        "random_mean_distance_median": float(np.median(random_means)),
        "random_coverage_radius_median": float(np.median(random_radii)),
    }


def build_structure_matrix(
    rows: list[dict[str, str]], ids: list[str]
) -> np.ndarray:
    index = {conformer_id: position for position, conformer_id in enumerate(ids)}
    matrix = np.zeros((len(ids), len(ids)), dtype=float)
    observed: set[tuple[int, int]] = set()
    for row in rows:
        first = row["conformer_id_a"]
        second = row["conformer_id_b"]
        if first not in index or second not in index:
            continue
        left, right = index[first], index[second]
        value = float(row["standardized_pocket_distance"])
        matrix[left, right] = matrix[right, left] = value
        observed.add(tuple(sorted((left, right))))
    expected = len(ids) * (len(ids) - 1) // 2
    if len(observed) != expected:
        raise ValueError(f"structural distance grid differs: {len(observed)} != {expected}")
    return matrix


def chemical_clusters(
    fingerprints: list[Any], ids: list[str], similarity_threshold: float
) -> tuple[list[str], list[list[int]]]:
    condensed = []
    for row in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[row], fingerprints[:row]
        )
        condensed.extend(1.0 - value for value in similarities)
    raw_clusters = Butina.ClusterData(
        condensed,
        len(fingerprints),
        1.0 - similarity_threshold,
        isDistData=True,
        reordering=True,
    )
    clusters = sorted(
        (sorted(cluster, key=lambda index: ids[index]) for cluster in raw_clusters),
        key=lambda cluster: (-len(cluster), [ids[index] for index in cluster]),
    )
    labels = [""] * len(ids)
    for cluster_index, cluster in enumerate(clusters, start=1):
        cluster_id = f"chemotype_{cluster_index:02d}"
        for member in cluster:
            labels[member] = cluster_id
    if any(not value for value in labels):
        raise ValueError("chemotype cluster assignment is incomplete")
    return labels, clusters


def run(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    verify_implementation(root, config)
    paths = verified_inputs(root, config)

    if any(int(value) != 0 for value in dict(config["data_boundary"]).values()):
        raise ValueError("Stage51b data boundary is not closed")
    stage51 = read_json(paths["stage51_summary"])
    prior_audit = read_json(paths["stage51_independent_audit"])
    if stage51["technical_gate_pass"] is not False:
        raise ValueError("Stage51b requires the frozen Stage51 gate failure")
    if prior_audit["status"] != "stage51_ppara_large_pool_gate_failure_independently_confirmed":
        raise ValueError("Stage51 independent failure audit is missing")

    gate_rows = read_csv(paths["stage51_receptor_gate_results"])
    prepared = [row for row in gate_rows if int(row["seed_count"]) == 3]
    ids = [row["conformer_id"] for row in prepared]
    expected = dict(config["expected"])
    if len(prepared) != int(expected["prepared_receptor_count"]):
        raise ValueError("Stage51b prepared receptor count differs")
    labels = np.asarray([truth(row["gate_pass"]) for row in prepared], dtype=bool)
    stable = np.asarray(
        [int(row["successful_seed_count"]) == 3 for row in prepared], dtype=bool
    )
    median_rmsd = np.asarray(
        [float(row["median_top_ranked_rmsd_angstrom"]) for row in prepared]
    )
    if int(labels.sum()) != int(expected["passing_receptor_count"]):
        raise ValueError("Stage51b passing receptor count differs")
    if int(stable.sum()) != int(expected["stable_three_of_three_receptor_count"]):
        raise ValueError("Stage51b stable receptor count differs")

    redocking_rows = read_csv(paths["stage51_redocking_results"])
    if len(redocking_rows) != int(expected["redocking_result_count"]):
        raise ValueError("Stage51b redocking result count differs")
    if any(row["status"] != "ok" for row in redocking_rows):
        raise ValueError("Stage51b encountered a non-ok redocking row")

    cases = {
        row["conformer_id"]: row for row in read_csv(paths["stage50_case_manifest"])
    }
    if set(ids) != set(cases):
        raise ValueError("Stage51b gate and cognate-case identities differ")
    coordinate_rows = {
        row["conformer_id"]: row
        for row in read_csv(paths["stage49b_coordinate_pool"])
    }
    if not set(ids).issubset(coordinate_rows):
        raise ValueError("Stage51b coordinate metadata are incomplete")

    chemistry = dict(config["ligand_chemistry"])
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(chemistry["morgan_radius"]),
        fpSize=int(chemistry["fingerprint_size"]),
    )
    molecules: list[Chem.Mol] = []
    fingerprints = []
    descriptors: list[dict[str, float]] = []
    scaffolds: list[str] = []
    for conformer_id in ids:
        case = cases[conformer_id]
        sdf = root / case["reference_sdf"]
        if sha256(sdf) != case["reference_sdf_sha256"].upper():
            raise ValueError(f"Stage51b reference SDF hash differs: {conformer_id}")
        molecule = molecule_from_sdf(sdf)
        molecules.append(molecule)
        fingerprints.append(generator.GetFingerprint(molecule))
        scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
        scaffolds.append(Chem.MolToSmiles(scaffold, isomericSmiles=False))
        descriptors.append(
            {
                "heavy_atom_count": float(molecule.GetNumHeavyAtoms()),
                "rotatable_bond_count": float(Lipinski.NumRotatableBonds(molecule)),
                "molecular_weight": float(Descriptors.MolWt(molecule)),
                "logp": float(Crippen.MolLogP(molecule)),
                "tpsa": float(Descriptors.TPSA(molecule)),
                "fraction_csp3": float(Descriptors.FractionCSP3(molecule)),
            }
        )

    cluster_labels, clusters = chemical_clusters(
        fingerprints,
        ids,
        float(chemistry["butina_tanimoto_similarity_threshold"]),
    )
    chemical_distance = np.zeros((len(ids), len(ids)), dtype=float)
    for row in range(len(ids)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[row], fingerprints
        )
        chemical_distance[row, :] = 1.0 - np.asarray(similarities, dtype=float)
    structure_matrix = build_structure_matrix(
        read_csv(paths["stage49b_structural_distances"]), ids
    )

    association = dict(config["association_tests"])
    permutation_count = int(association["permutation_count"])
    permutation_seed = int(association["permutation_seed"])
    structure_neighbors = nearest_neighbor_indices(structure_matrix, ids)
    chemistry_neighbors = nearest_neighbor_indices(chemical_distance, ids)
    structure_knn = permutation_knn(
        labels, structure_neighbors, permutation_count, permutation_seed
    )
    chemistry_knn = permutation_knn(
        labels, chemistry_neighbors, permutation_count, permutation_seed + 1
    )
    observed_difference = (
        chemistry_knn["balanced_accuracy"] - structure_knn["balanced_accuracy"]
    )
    rng = np.random.default_rng(permutation_seed + 2)
    null_differences = np.empty(permutation_count, dtype=float)
    for permutation_index in range(permutation_count):
        permuted = rng.permutation(labels)
        null_differences[permutation_index] = balanced_accuracy(
            permuted, permuted[chemistry_neighbors]
        ) - balanced_accuracy(permuted, permuted[structure_neighbors])
    difference_p = float(
        (1 + np.sum(null_differences >= observed_difference))
        / (permutation_count + 1)
    )
    minimum_difference = float(
        association["dominant_driver_minimum_balanced_accuracy_difference"]
    )
    maximum_p = float(association["dominant_driver_maximum_permutation_p"])
    if (
        observed_difference >= minimum_difference
        and chemistry_knn["permutation_p"] <= maximum_p
        and difference_p <= maximum_p
    ):
        dominant_driver = "cognate_ligand_chemistry"
    elif (
        observed_difference <= -minimum_difference
        and structure_knn["permutation_p"] <= maximum_p
        and float((1 + np.sum(null_differences <= observed_difference)) / (permutation_count + 1))
        <= maximum_p
    ):
        dominant_driver = "receptor_structure"
    else:
        dominant_driver = "mixed_or_unresolved"

    descriptor_tests = []
    raw_p_values = []
    for name in chemistry["descriptor_names"]:
        passing = np.asarray(
            [row[name] for row, label in zip(descriptors, labels) if label],
            dtype=float,
        )
        failing = np.asarray(
            [row[name] for row, label in zip(descriptors, labels) if not label],
            dtype=float,
        )
        statistic, p_value = mannwhitneyu(passing, failing, alternative="two-sided")
        rho, rho_p = spearmanr(
            [row[name] for row in descriptors], median_rmsd
        )
        raw_p_values.append(float(p_value))
        descriptor_tests.append(
            {
                "descriptor": name,
                "passing_median": float(np.median(passing)),
                "failing_median": float(np.median(failing)),
                "mann_whitney_u": float(statistic),
                "mann_whitney_p": float(p_value),
                "rank_biserial_effect": float(
                    2.0 * statistic / (len(passing) * len(failing)) - 1.0
                ),
                "spearman_rho_with_median_rmsd": float(rho),
                "spearman_p": float(rho_p),
            }
        )
    for row, adjusted in zip(descriptor_tests, benjamini_hochberg(raw_p_values)):
        row["benjamini_hochberg_q"] = adjusted

    structure_config = dict(config["structure"])
    passing_indices = np.flatnonzero(labels).tolist()
    stable_indices = np.flatnonzero(stable).tolist()
    passing_coverage = coverage_metrics(
        structure_matrix,
        passing_indices,
        int(structure_config["random_subset_draw_count"]),
        int(structure_config["random_seed"]),
    )
    stable_coverage = coverage_metrics(
        structure_matrix,
        stable_indices,
        int(structure_config["random_subset_draw_count"]),
        int(structure_config["random_seed"]) + 1,
    )

    cluster_rows: list[dict[str, Any]] = []
    for cluster_index, cluster in enumerate(clusters, start=1):
        members = [ids[index] for index in cluster]
        passing_members = [ids[index] for index in cluster if labels[index]]
        cluster_rows.append(
            {
                "chemotype_cluster_id": f"chemotype_{cluster_index:02d}",
                "member_count": len(cluster),
                "passing_member_count": len(passing_members),
                "passing_fraction": len(passing_members) / len(cluster),
                "member_ids": ";".join(members),
                "passing_member_ids": ";".join(passing_members),
                "murcko_scaffolds": ";".join(sorted({scaffolds[index] for index in cluster})),
            }
        )
    passing_cluster_count = len(
        {cluster_labels[index] for index in range(len(ids)) if labels[index]}
    )

    diagnostic_rows: list[dict[str, Any]] = []
    for index, conformer_id in enumerate(ids):
        other_passing = [value for value in passing_indices if value != index]
        nearest_structure = min(
            (structure_matrix[index, value], ids[value]) for value in other_passing
        )
        nearest_chemistry = min(
            (chemical_distance[index, value], ids[value]) for value in other_passing
        )
        diagnostic_rows.append(
            {
                "conformer_id": conformer_id,
                "pdb_id": cases[conformer_id]["pdb_id"],
                "pdb_family_prefix": cases[conformer_id]["pdb_id"][
                    : int(chemistry["family_prefix_length"])
                ],
                "gate_pass": bool(labels[index]),
                "stable_three_of_three": bool(stable[index]),
                "successful_seed_count": int(prepared[index]["successful_seed_count"]),
                "median_top_ranked_rmsd_angstrom": median_rmsd[index],
                "chemotype_cluster_id": cluster_labels[index],
                "murcko_scaffold": scaffolds[index],
                **descriptors[index],
                "resolution_angstrom": float(
                    coordinate_rows[conformer_id]["resolution_angstrom"]
                ),
                "nearest_other_passing_structure_id": nearest_structure[1],
                "nearest_other_passing_structure_distance": float(nearest_structure[0]),
                "nearest_other_passing_chemistry_id": nearest_chemistry[1],
                "nearest_other_passing_chemical_distance": float(nearest_chemistry[0]),
            }
        )

    family_groups: dict[str, list[int]] = defaultdict(list)
    for index, case in enumerate(cases[conformer_id] for conformer_id in ids):
        family_groups[case["pdb_id"][: int(chemistry["family_prefix_length"])]].append(index)
    family_summary = [
        {
            "pdb_family_prefix": family,
            "prepared_count": len(indices),
            "passing_count": int(sum(labels[index] for index in indices)),
            "passing_fraction": float(np.mean(labels[indices])),
            "member_ids": [ids[index] for index in indices],
        }
        for family, indices in sorted(
            family_groups.items(), key=lambda item: (-len(item[1]), item[0])
        )
        if len(indices) >= 2
    ]

    exploratory = dict(config["exploratory_branch_gate"])
    conditions = {
        "stable_receptor_count": int(stable.sum())
        >= int(exploratory["minimum_stable_three_of_three_receptors"]),
        "structural_coverage": passing_coverage[
            "coverage_radius_random_percentile"
        ]
        <= float(exploratory["maximum_structural_coverage_radius_random_percentile"]),
        "chemotype_cluster_count": passing_cluster_count
        >= int(exploratory["minimum_passing_chemotype_cluster_count"]),
    }
    exploratory_candidate = all(conditions.values())
    outputs = dict(config["outputs"])
    diagnostic_path = root / outputs["diagnostic_table_csv"]
    cluster_path = root / outputs["chemotype_cluster_csv"]
    result_path = root / outputs["result_json"]
    audit_path = root / outputs["audit_json"]
    write_csv(diagnostic_path, diagnostic_rows)
    write_csv(cluster_path, cluster_rows)

    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage51b_ppara_redocking_bias_diagnostic_ok",
        "analysis_class": config["analysis_class"],
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": sha256(config_path),
        },
        "stage51_frozen_outcome": {
            "technical_gate_pass": False,
            "frozen_receptor_count": int(expected["frozen_receptor_count"]),
            "prepared_receptor_count": len(ids),
            "passing_receptor_count": int(labels.sum()),
            "stable_three_of_three_receptor_count": int(stable.sum()),
            "confirmatory_status_changed": False,
        },
        "nearest_neighbor_association": {
            "receptor_structure": structure_knn,
            "cognate_ligand_chemistry": chemistry_knn,
            "chemistry_minus_structure_balanced_accuracy": observed_difference,
            "one_sided_difference_permutation_p": difference_p,
            "dominant_driver": dominant_driver,
        },
        "structural_coverage": {
            "passing_twenty": passing_coverage,
            "stable_eighteen": stable_coverage,
        },
        "ligand_descriptor_tests": descriptor_tests,
        "chemotype_clustering": {
            "similarity_threshold": chemistry[
                "butina_tanimoto_similarity_threshold"
            ],
            "total_cluster_count": len(clusters),
            "passing_cluster_count": passing_cluster_count,
            "cluster_rows": cluster_rows,
        },
        "pdb_family_summary": family_summary,
        "decision": {
            "stage51_confirmatory_gate_remains_failed": True,
            "confirmatory_development_panel_docking_authorized": False,
            "exploratory_twenty_receptor_branch_candidate": exploratory_candidate,
            "exploratory_gate_conditions": conditions,
            "recommended_next_step": (
                "preregister a separate post-hoc exploratory development-panel branch using the frozen passing pool; preserve Stage51 as failed"
                if exploratory_candidate
                else "stop PPARA development docking and move to a new target with a revised label-independent technical qualification design"
            ),
        },
        "data_boundary": config["data_boundary"],
        "interpretation_boundary": config["interpretation_boundary"],
        "outputs": {
            "diagnostic_table_csv": {
                "path": diagnostic_path.relative_to(root).as_posix(),
                "sha256": sha256(diagnostic_path),
            },
            "chemotype_cluster_csv": {
                "path": cluster_path.relative_to(root).as_posix(),
                "sha256": sha256(cluster_path),
            },
        },
    }
    write_json(result_path, result)
    audit = {
        "schema_version": "1.0",
        "status": "stage51b_ppara_redocking_bias_diagnostic_audit_ok",
        "config": result["config"],
        "input_hashes_verified": True,
        "prepared_receptor_count": len(ids),
        "redocking_result_count": len(redocking_rows),
        "structural_pair_count": len(ids) * (len(ids) - 1) // 2,
        "reference_sdf_count": len(molecules),
        "complete_chemotype_assignment": len(cluster_labels) == len(ids),
        "stage51_confirmatory_gate_remains_failed": True,
        "confirmatory_status_changed": False,
        "new_docking_jobs": 0,
        "activity_labels_read": 0,
        "fresh_validation_rows_read": 0,
        "locked_test_rows_read": 0,
        "outputs": {
            "result_json": {
                "path": result_path.relative_to(root).as_posix(),
                "sha256": sha256(result_path),
            },
            **result["outputs"],
        },
    }
    write_json(audit_path, audit)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
