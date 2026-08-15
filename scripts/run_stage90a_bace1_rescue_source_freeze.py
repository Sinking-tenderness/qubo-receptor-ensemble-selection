import argparse
import csv
import hashlib
import json
from collections import defaultdict
from itertools import combinations
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


def as_bool(value):
    return str(value).lower() == "true"


def run(root, config_path):
    config = read_json(root / config_path)
    inputs = config["inputs"]
    stage90 = read_json(root / inputs["stage90_result"])
    stage41d = read_json(root / inputs["stage41d_large_pool_certificate"])
    candidates = read_csv(root / inputs["stage90_assay_candidates"])
    activities = read_csv(root / inputs["stage90_normalized_activities"])
    receptor_rows = read_csv(root / inputs["receptor_manifest"])

    rules = config["selection_rules"]
    eligible = [
        row
        for row in candidates
        if row["target_id"] == rules["target_id"]
        and row["standard_type"] == rules["standard_type"]
        and as_bool(row["analog_series_gate_passed"])
        and as_bool(row["potency_contrast_gate_passed"])
    ]
    if len(eligible) < 4:
        raise ValueError("fewer than four eligible BACE1 IC50 assays")

    development = sorted(
        eligible,
        key=lambda row: (
            -int(row["molecules_in_repeat_series"]),
            -int(row["unique_molecule_count"]),
            row["assay_chembl_id"],
        ),
    )[0]
    remaining = [row for row in eligible if row is not development]
    confirmation_a = sorted(
        remaining,
        key=lambda row: (-int(row["unique_molecule_count"]), row["assay_chembl_id"]),
    )[0]
    remaining = [row for row in remaining if row is not confirmation_a]
    confirmation_b = sorted(
        remaining,
        key=lambda row: (-int(row["low_potency_count"]), row["assay_chembl_id"]),
    )[0]
    remaining = [row for row in remaining if row is not confirmation_b]
    locked_test = sorted(
        remaining,
        key=lambda row: (-int(row["unique_molecule_count"]), row["assay_chembl_id"]),
    )[0]

    selected = [
        ("development", development),
        ("confirmation_a", confirmation_a),
        ("confirmation_b", confirmation_b),
        ("locked_test", locked_test),
    ]
    by_assay = defaultdict(list)
    for row in activities:
        if row["target_id"] == rules["target_id"]:
            by_assay[row["assay_chembl_id"]].append(row)

    role_rows = []
    for role, candidate in selected:
        assay_rows = by_assay[candidate["assay_chembl_id"]]
        documents = sorted({row["document_chembl_id"] for row in assay_rows})
        if len(documents) != 1:
            raise ValueError(f"assay {candidate['assay_chembl_id']} is not document-unique")
        role_rows.append(
            {
                "role": role,
                "target_id": candidate["target_id"],
                "assay_chembl_id": candidate["assay_chembl_id"],
                "document_chembl_id": documents[0],
                "standard_type": candidate["standard_type"],
                "unique_molecule_count": int(candidate["unique_molecule_count"]),
                "repeat_series_count": int(candidate["repeat_series_count"]),
                "molecules_in_repeat_series": int(
                    candidate["molecules_in_repeat_series"]
                ),
                "high_potency_count": int(candidate["high_potency_count"]),
                "low_potency_count": int(candidate["low_potency_count"]),
                "pchembl_range": float(candidate["pchembl_range"]),
                "docking_scores_locked": role != "development",
            }
        )

    overlap_rows = []
    for (left_role, left), (right_role, right) in combinations(selected, 2):
        left_rows = by_assay[left["assay_chembl_id"]]
        right_rows = by_assay[right["assay_chembl_id"]]
        left_molecules = {row["molecule_chembl_id"] for row in left_rows}
        right_molecules = {row["molecule_chembl_id"] for row in right_rows}
        left_scaffolds = {row["scaffold_smiles"] for row in left_rows}
        right_scaffolds = {row["scaffold_smiles"] for row in right_rows}
        left_documents = {row["document_chembl_id"] for row in left_rows}
        right_documents = {row["document_chembl_id"] for row in right_rows}
        overlap_rows.append(
            {
                "left_role": left_role,
                "right_role": right_role,
                "left_assay": left["assay_chembl_id"],
                "right_assay": right["assay_chembl_id"],
                "shared_molecule_count": len(left_molecules & right_molecules),
                "shared_scaffold_count": len(left_scaffolds & right_scaffolds),
                "shared_document_count": len(left_documents & right_documents),
            }
        )

    override = config["large_pool_override"]
    checks = {
        "stage90_remains_failed": stage90["status"]
        == "stage90_no_full_rescue_intake_candidate",
        "four_assays_frozen": len(role_rows) == 4,
        "all_assays_distinct": len({row["assay_chembl_id"] for row in role_rows}) == 4,
        "all_documents_distinct": len({row["document_chembl_id"] for row in role_rows})
        == 4,
        "all_pairwise_molecule_disjoint": all(
            row["shared_molecule_count"] == 0 for row in overlap_rows
        ),
        "all_pairwise_scaffold_disjoint": all(
            row["shared_scaffold_count"] == 0 for row in overlap_rows
        ),
        "all_pairwise_document_disjoint": all(
            row["shared_document_count"] == 0 for row in overlap_rows
        ),
        "stage41d_predates_stage90": stage41d["experiment_id"].startswith("stage41d-"),
        "stage41d_status_allows_development": stage41d["status"]
        == "stage41d_conditional_go_new_posthoc_development_route",
        "stage41d_receptor_count_passes": len(stage41d["passing_receptor_ids"])
        >= override["minimum_receptor_count"],
        "manifest_matches_stage41d_pool": len(receptor_rows)
        == len(stage41d["passing_receptor_ids"]),
        "large_total_state_space": stage41d["total_state_count_k1_to_k6"]
        >= override["minimum_stage41d_total_states_k1_to_k6"],
        "large_k6_state_space": int(stage41d["state_count_by_k"]["6"])
        >= override["minimum_stage41d_k6_states"],
    }

    roles_path = root / config["outputs"]["assay_roles_csv"]
    overlaps_path = root / config["outputs"]["overlap_matrix_csv"]
    write_csv(roles_path, role_rows)
    write_csv(overlaps_path, overlap_rows)

    passed = all(checks.values())
    result = {
        "schema_version": "1.0",
        "status": (
            "stage90a_bace1_rescue_source_freeze_passed"
            if passed
            else "stage90a_bace1_rescue_source_freeze_failed"
        ),
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "frozen_assays": role_rows,
        "large_pool_certificate": {
            "receptor_count": len(receptor_rows),
            "total_states_k1_to_k6": stage41d["total_state_count_k1_to_k6"],
            "k6_states": int(stage41d["state_count_by_k"]["6"]),
            "evidence_timing": "Stage41d certificate existed before Stage90 ChEMBL intake.",
        },
        "adjudication": (
            "Stage90 remains a failed preregistered 50-receptor intake gate. A separate prospective "
            "BACE1 route is nevertheless justified by the earlier Stage41d certificate: the 34 "
            "redocking-qualified receptors already passed label-independent structural coverage "
            "criteria and define more than one million k<=6 subsets. No Stage90 threshold is changed."
        ),
        "authorization": {
            "stage91_preregistration_and_development_ligand_freeze_authorized": passed,
            "confirmation_or_test_docking_authorized": False,
            "new_docking_jobs_authorized": 0,
            "quantum_hardware_jobs_authorized": 0,
        },
        "data_boundary": {
            "fresh_validation_rows_read": 0,
            "locked_test_docking_scores_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "outputs": {
            "assay_roles_csv": {
                "path": config["outputs"]["assay_roles_csv"],
                "sha256": sha256(roles_path),
            },
            "overlap_matrix_csv": {
                "path": config["outputs"]["overlap_matrix_csv"],
                "sha256": sha256(overlaps_path),
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
        "# Stage90a BACE1 rescue source freeze",
        "",
        f"Status: `{result['status']}`.",
        "",
        "Stage90 remains failed; its 50-receptor threshold was not changed. The rescue route instead uses the independent, pre-existing Stage41d certificate for 34 redocking-qualified receptors and 1,676,115 k=1..6 states.",
        "",
        "| Role | Assay | Document | Molecules | Repeat series | Series molecules | High | Low | Locked |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in role_rows:
        lines.append(
            "| {role} | {assay_chembl_id} | {document_chembl_id} | {unique_molecule_count} | "
            "{repeat_series_count} | {molecules_in_repeat_series} | {high_potency_count} | "
            "{low_potency_count} | {docking_scores_locked} |".format(**row)
        )
    lines.extend(
        [
            "",
            "All six assay pairs have zero molecule, Bemis-Murcko scaffold, and document overlap.",
            "",
            "## Decision",
            "",
            "Stage91 may freeze the development ligands and preregister the group-balanced large-pool comparison. Confirmation and locked-test docking remain prohibited. No quantum hardware route is reopened.",
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
        default=Path("configs/stage90a_bace1_rescue_source_freeze.json"),
    )
    args = parser.parse_args()
    run(args.root.resolve(), args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
