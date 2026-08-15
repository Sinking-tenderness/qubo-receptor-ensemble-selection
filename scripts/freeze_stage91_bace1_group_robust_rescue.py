import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path):
    with path.open(encoding="ascii", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def scaffold_id(scaffold_smiles):
    token = hashlib.sha256(scaffold_smiles.encode("ascii")).hexdigest()[:12].upper()
    return f"BACE1_SCF_{token}"


def label_for(value):
    if value >= 6.5:
        return "high"
    if value <= 5.5:
        return "low"
    return "gray"


def run(root, config_path):
    config = read_json(root / config_path)
    source_freeze = read_json(root / config["inputs"]["stage90a_source_freeze"])
    activities = read_csv(root / config["inputs"]["stage90_normalized_activities"])
    receptors = read_csv(root / config["inputs"]["receptor_manifest"])
    if source_freeze["status"] != "stage90a_bace1_rescue_source_freeze_passed":
        raise ValueError("Stage90a source freeze did not pass")

    roles = {
        row["assay_chembl_id"]: row for row in source_freeze["frozen_assays"]
    }
    selected_rows = [
        row
        for row in activities
        if row["target_id"] == "BACE1" and row["assay_chembl_id"] in roles
    ]
    grouped = defaultdict(list)
    for row in selected_rows:
        grouped[(row["assay_chembl_id"], row["molecule_chembl_id"])].append(row)

    deduplicated = []
    for (assay_id, molecule_id), repeats in sorted(grouped.items()):
        first = repeats[0]
        pchembl = statistics.median(float(row["pchembl_value"]) for row in repeats)
        role = roles[assay_id]["role"]
        deduplicated.append(
            {
                "ligand_id": f"BACE1_{role}_{molecule_id}",
                "target_id": "BACE1",
                "role": role,
                "assay_chembl_id": assay_id,
                "document_chembl_id": first["document_chembl_id"],
                "molecule_chembl_id": molecule_id,
                "canonical_smiles": first["canonical_smiles"],
                "scaffold_smiles": first["scaffold_smiles"],
                "scaffold_group_id": scaffold_id(first["scaffold_smiles"]),
                "pchembl_value": pchembl,
                "potency_label": label_for(pchembl),
                "core_series": False,
                "docking_authorized": role == "development",
                "score_access_locked": role != "development",
            }
        )

    assay_scaffolds = defaultdict(Counter)
    for row in deduplicated:
        assay_scaffolds[row["assay_chembl_id"]][row["scaffold_group_id"]] += 1
    minimum_series = config["ligand_rules"]["minimum_core_series_size"]
    for row in deduplicated:
        row["core_series"] = (
            assay_scaffolds[row["assay_chembl_id"]][row["scaffold_group_id"]]
            >= minimum_series
        )

    series_rows = []
    series_members = defaultdict(list)
    for row in deduplicated:
        series_members[
            (row["role"], row["assay_chembl_id"], row["scaffold_group_id"])
        ].append(row)
    for (role, assay_id, group_id), members in series_members.items():
        labels = Counter(row["potency_label"] for row in members)
        values = [float(row["pchembl_value"]) for row in members]
        series_rows.append(
            {
                "role": role,
                "assay_chembl_id": assay_id,
                "scaffold_group_id": group_id,
                "scaffold_smiles": members[0]["scaffold_smiles"],
                "molecule_count": len(members),
                "high_count": labels["high"],
                "low_count": labels["low"],
                "gray_count": labels["gray"],
                "minimum_pchembl": min(values),
                "maximum_pchembl": max(values),
                "core_series": len(members) >= minimum_series,
            }
        )
    series_rows.sort(
        key=lambda row: (
            ["development", "confirmation_a", "confirmation_b", "locked_test"].index(
                row["role"]
            ),
            not row["core_series"],
            -row["molecule_count"],
            row["scaffold_group_id"],
        )
    )

    development = [row for row in deduplicated if row["role"] == "development"]
    development_core = [row for row in development if row["core_series"]]
    core_summaries = [
        row
        for row in series_rows
        if row["role"] == "development" and row["core_series"]
    ]
    role_summary = {}
    for role in ("development", "confirmation_a", "confirmation_b", "locked_test"):
        rows = [row for row in deduplicated if row["role"] == role]
        labels = Counter(row["potency_label"] for row in rows)
        role_summary[role] = {
            "molecule_count": len(rows),
            "high_count": labels["high"],
            "low_count": labels["low"],
            "gray_count": labels["gray"],
            "core_series_count": len(
                {
                    row["scaffold_group_id"] for row in rows if row["core_series"]
                }
            ),
        }

    checks = {
        "receptor_count_is_34": len(receptors) == 34,
        "four_roles_present": set(role_summary)
        == {"development", "confirmation_a", "confirmation_b", "locked_test"},
        "development_molecule_count_is_365": len(development) == 365,
        "development_core_series_count_is_6": len(core_summaries) == 6,
        "development_core_molecule_count_is_258": len(development_core) == 258,
        "every_development_core_series_has_high_and_low": all(
            row["high_count"] > 0 and row["low_count"] > 0 for row in core_summaries
        ),
        "only_development_docking_is_authorized": all(
            row["docking_authorized"] == (row["role"] == "development")
            for row in deduplicated
        ),
        "all_nondevelopment_scores_are_locked": all(
            row["score_access_locked"] == (row["role"] != "development")
            for row in deduplicated
        ),
        "primary_k6_state_count_exceeds_one_million": source_freeze[
            "large_pool_certificate"
        ]["k6_states"]
        >= config["development_hardness_gate"]["minimum_fixed_k_state_count"],
    }
    passed = all(checks.values())

    manifest_path = root / config["outputs"]["ligand_manifest_csv"]
    series_path = root / config["outputs"]["series_summary_csv"]
    write_csv(manifest_path, deduplicated)
    write_csv(series_path, series_rows)

    result = {
        "schema_version": "1.0",
        "status": (
            "stage91_bace1_group_robust_rescue_preregistered"
            if passed
            else "stage91_bace1_group_robust_rescue_preregistration_failed"
        ),
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "role_summary": role_summary,
        "development_core_series": core_summaries,
        "receptor_count": len(receptors),
        "primary_k": config["primary_selection_problem"]["cardinality_k"],
        "primary_k_state_count": source_freeze["large_pool_certificate"]["k6_states"],
        "frozen_objective": config["primary_selection_problem"],
        "development_hardness_gate": config["development_hardness_gate"],
        "authorization": {
            "development_ligand_input_preparation_bundle_authorized": passed,
            "development_docking_authorized": False,
            "confirmation_or_test_preparation_authorized": False,
            "quantum_simulation_or_hardware_authorized": False,
        },
        "data_boundary": {
            "new_docking_jobs": 0,
            "confirmation_docking_scores_read": 0,
            "locked_test_docking_scores_read": 0,
            "quantum_hardware_jobs": 0,
        },
        "outputs": {
            "ligand_manifest_csv": {
                "path": config["outputs"]["ligand_manifest_csv"],
                "sha256": sha256(manifest_path),
            },
            "series_summary_csv": {
                "path": config["outputs"]["series_summary_csv"],
                "sha256": sha256(series_path),
            },
        },
    }
    result_path = root / config["outputs"]["result_json"]
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )

    report_path = root / config["outputs"]["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage91 BACE1 group-robust rescue preregistration",
        "",
        f"Status: `{result['status']}`.",
        "",
        "The rescue now tests whether a six-receptor subset can protect the worst-served medicinal-chemistry series rather than merely maximize average ligand ranking.",
        "",
        "## Frozen data roles",
        "",
        "| Role | Molecules | High | Low | Gray | Core series | Score status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for role, summary in role_summary.items():
        lines.append(
            f"| {role} | {summary['molecule_count']} | {summary['high_count']} | "
            f"{summary['low_count']} | {summary['gray_count']} | "
            f"{summary['core_series_count']} | "
            f"{'development preparation only' if role == 'development' else 'locked'} |"
        )
    lines.extend(
        [
            "",
            "## Frozen objective",
            "",
            f"`{config['primary_selection_problem']['normalized_objective']}`",
            "",
            "Primary k is 6 over 34 receptors (1,344,904 subsets). Coefficients cannot be tuned after docking. k=4 and k=8 are sensitivity analyses only.",
            "",
            "## Release gate",
            "",
            "Development must show a strict certified improvement over greedy plus all one-swaps and a reproducible multi-move trap. Only then may confirmation A be prepared and docked. Confirmation B, locked test, and quantum execution remain sequentially blocked.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage91_bace1_group_robust_rescue_preregistration.json"),
    )
    args = parser.parse_args()
    run(args.root.resolve(), args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
