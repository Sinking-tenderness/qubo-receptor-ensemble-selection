import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urljoin

import requests
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def read_json(path):
    return json.loads(path.read_text(encoding="ascii"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def request_json(session, url, params, source):
    error = None
    for attempt in range(source["maximum_retries"]):
        try:
            response = session.get(
                url,
                params=params,
                timeout=source["request_timeout_seconds"],
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            error = exc
            if attempt + 1 < source["maximum_retries"]:
                time.sleep(2**attempt)
    raise RuntimeError(f"ChEMBL request failed after retries: {url}: {error}")


def fetch_pages(session, endpoint, params, collection_key, source):
    url = f"{source['base_url'].rstrip('/')}/{endpoint}.json"
    query = dict(params)
    query["limit"] = source["page_limit"]
    rows = []
    total_count = None
    while url:
        payload = request_json(session, url, query, source)
        rows.extend(payload[collection_key])
        page_meta = payload["page_meta"]
        total_count = page_meta["total_count"]
        next_path = page_meta.get("next")
        url = urljoin(source["base_url"], next_path) if next_path else None
        query = None
    if total_count is not None and len(rows) != total_count:
        raise ValueError(
            f"ChEMBL pagination mismatch for {endpoint}: {len(rows)} != {total_count}"
        )
    return rows


def scaffold_for(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None, None
    Chem.RemoveStereochemistry(molecule)
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    Chem.RemoveStereochemistry(scaffold)
    if scaffold.GetNumAtoms() == 0:
        scaffold = molecule
    scaffold_smiles = Chem.MolToSmiles(
        scaffold, canonical=True, isomericSmiles=False
    )
    return canonical, scaffold_smiles


def receptor_count(path):
    with path.open(encoding="ascii", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def normalize_target(session, target, config, root):
    source = config["source"]
    filters = config["activity_filters"]
    target_payload = request_json(
        session,
        f"{source['base_url'].rstrip('/')}/target/{target['target_chembl_id']}.json",
        None,
        source,
    )
    if target_payload["organism"] != filters["target_organism"]:
        raise ValueError(f"Unexpected target organism for {target['target_id']}")
    if target_payload["target_type"] != "SINGLE PROTEIN":
        raise ValueError(f"Unexpected target type for {target['target_id']}")
    if target_payload["pref_name"] != target["expected_name"]:
        raise ValueError(f"Unexpected target name for {target['target_id']}")

    assays = fetch_pages(
        session,
        "assay",
        {"target_chembl_id": target["target_chembl_id"]},
        "assays",
        source,
    )
    assay_map = {row["assay_chembl_id"]: row for row in assays}
    activities = fetch_pages(
        session,
        "activity",
        {
            "target_chembl_id": target["target_chembl_id"],
            "pchembl_value__isnull": "false",
        },
        "activities",
        source,
    )

    accepted = []
    rejected = Counter()
    for row in activities:
        assay = assay_map.get(row.get("assay_chembl_id"))
        if assay is None:
            rejected["missing_assay_metadata"] += 1
            continue
        if row.get("target_organism") != filters["target_organism"]:
            rejected["nonhuman_activity"] += 1
            continue
        if assay.get("assay_type") not in filters["allowed_assay_types"]:
            rejected["assay_type"] += 1
            continue
        if int(assay.get("confidence_score") or 0) < filters[
            "minimum_assay_confidence_score"
        ]:
            rejected["confidence_score"] += 1
            continue
        if row.get("standard_type") not in filters["allowed_standard_types"]:
            rejected["standard_type"] += 1
            continue
        if row.get("standard_relation") != filters["required_standard_relation"]:
            rejected["standard_relation"] += 1
            continue
        if filters["exclude_potential_duplicates"] and int(
            row.get("potential_duplicate") or 0
        ):
            rejected["potential_duplicate"] += 1
            continue
        if filters["exclude_data_validity_comments"] and row.get(
            "data_validity_comment"
        ):
            rejected["data_validity_comment"] += 1
            continue
        if not row.get("canonical_smiles") or not row.get("molecule_chembl_id"):
            rejected["missing_structure_or_molecule"] += 1
            continue
        try:
            pchembl = float(row["pchembl_value"])
        except (TypeError, ValueError):
            rejected["invalid_pchembl"] += 1
            continue
        canonical, scaffold = scaffold_for(row["canonical_smiles"])
        if canonical is None:
            rejected["rdkit_parse"] += 1
            continue
        accepted.append(
            {
                "target_id": target["target_id"],
                "target_chembl_id": target["target_chembl_id"],
                "assay_chembl_id": row["assay_chembl_id"],
                "assay_type": assay["assay_type"],
                "assay_confidence_score": int(assay["confidence_score"]),
                "document_chembl_id": row.get("document_chembl_id") or "",
                "standard_type": row["standard_type"],
                "molecule_chembl_id": row["molecule_chembl_id"],
                "canonical_smiles": canonical,
                "scaffold_smiles": scaffold,
                "pchembl_value": pchembl,
            }
        )
    return target_payload, activities, assays, accepted, dict(rejected)


def summarize_groups(rows, target_counts, gate):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["target_id"], row["assay_chembl_id"], row["standard_type"])].append(
            row
        )

    summaries = []
    for (target_id, assay_id, standard_type), group in grouped.items():
        by_molecule = defaultdict(list)
        for row in group:
            by_molecule[row["molecule_chembl_id"]].append(row)
        molecules = []
        for molecule_id, repeats in by_molecule.items():
            first = repeats[0]
            molecules.append(
                {
                    **first,
                    "molecule_chembl_id": molecule_id,
                    "pchembl_value": statistics.median(
                        item["pchembl_value"] for item in repeats
                    ),
                }
            )
        scaffolds = Counter(row["scaffold_smiles"] for row in molecules)
        repeat_sizes = sorted(
            (count for count in scaffolds.values() if count >= gate["minimum_series_size"]),
            reverse=True,
        )
        values = [row["pchembl_value"] for row in molecules]
        high = sum(value >= gate["high_potency_pchembl_threshold"] for value in values)
        low = sum(value <= gate["low_potency_pchembl_threshold"] for value in values)
        series_gate = (
            len(molecules) >= gate["minimum_unique_molecules_per_assay_endpoint"]
            and len(repeat_sizes) >= gate["minimum_repeat_series_count"]
            and sum(repeat_sizes) >= gate["minimum_molecules_in_repeat_series"]
        )
        potency_gate = (
            high >= gate["minimum_high_potency_molecules"]
            and low >= gate["minimum_low_potency_molecules"]
            and max(values) - min(values) >= gate["minimum_pchembl_range"]
        )
        pool_gate = target_counts[target_id] >= gate["minimum_receptor_pool_size"]
        summaries.append(
            {
                "target_id": target_id,
                "assay_chembl_id": assay_id,
                "standard_type": standard_type,
                "unique_molecule_count": len(molecules),
                "scaffold_count": len(scaffolds),
                "largest_scaffold_size": max(scaffolds.values()),
                "repeat_series_count": len(repeat_sizes),
                "molecules_in_repeat_series": sum(repeat_sizes),
                "high_potency_count": high,
                "low_potency_count": low,
                "minimum_pchembl": min(values),
                "maximum_pchembl": max(values),
                "pchembl_range": max(values) - min(values),
                "receptor_pool_size": target_counts[target_id],
                "analog_series_gate_passed": series_gate,
                "potency_contrast_gate_passed": potency_gate,
                "receptor_pool_gate_passed": pool_gate,
                "full_rescue_intake_gate_passed": series_gate
                and potency_gate
                and pool_gate,
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            not row["full_rescue_intake_gate_passed"],
            not row["analog_series_gate_passed"],
            -row["molecules_in_repeat_series"],
            -row["unique_molecule_count"],
            row["target_id"],
            row["assay_chembl_id"],
        ),
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="ascii", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def run(root, config_path):
    config = read_json(root / config_path)
    source = config["source"]
    session = requests.Session()
    session.headers.update({"User-Agent": "qubo-receptor-ensemble-stage90/1.0"})
    status = request_json(
        session, f"{source['base_url'].rstrip('/')}/status.json", None, source
    )

    all_rows = []
    target_audits = []
    target_counts = {}
    for target in config["targets"]:
        manifest = root / target["receptor_manifest"]
        target_counts[target["target_id"]] = receptor_count(manifest)
        metadata, raw_activities, assays, accepted, rejected = normalize_target(
            session, target, config, root
        )
        all_rows.extend(accepted)
        target_audits.append(
            {
                "target_id": target["target_id"],
                "target_chembl_id": target["target_chembl_id"],
                "pref_name": metadata["pref_name"],
                "raw_assay_count": len(assays),
                "raw_pchembl_activity_count": len(raw_activities),
                "accepted_activity_count": len(accepted),
                "rejected_activity_counts": rejected,
                "receptor_pool_size": target_counts[target["target_id"]],
                "receptor_manifest": {
                    "path": target["receptor_manifest"],
                    "sha256": sha256(manifest),
                },
            }
        )

    candidates = summarize_groups(all_rows, target_counts, config["intake_gate"])
    normalized_path = root / config["outputs"]["normalized_activities_csv"]
    candidates_path = root / config["outputs"]["assay_candidates_csv"]
    write_csv(normalized_path, all_rows)
    write_csv(candidates_path, candidates)

    full = [row for row in candidates if row["full_rescue_intake_gate_passed"]]
    analog = [row for row in candidates if row["analog_series_gate_passed"]]
    result = {
        "schema_version": "1.0",
        "status": (
            "stage90_chembl_analog_series_intake_passed"
            if full
            else "stage90_no_full_rescue_intake_candidate"
        ),
        "chembl_status": status,
        "target_audits": target_audits,
        "summary": {
            "target_count": len(config["targets"]),
            "accepted_activity_count": len(all_rows),
            "assay_endpoint_count": len(candidates),
            "analog_series_candidate_count": len(analog),
            "full_rescue_candidate_count": len(full),
            "best_candidate": candidates[0] if candidates else None,
        },
        "authorization": {
            "stage91_ligand_freeze_authorized": bool(full),
            "new_docking_jobs_authorized": 0,
            "quantum_hardware_jobs_authorized": 0,
        },
        "interpretation_boundary": (
            "Stage90 is a public-data intake audit. Passing identifies a source suitable for "
            "preregistration and ligand curation; it does not validate a QUBO, authorize docking, "
            "establish biological activity, or demonstrate classical hardness or quantum advantage."
        ),
        "data_boundary": {
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "outputs": {
            "normalized_activities_csv": {
                "path": config["outputs"]["normalized_activities_csv"],
                "sha256": sha256(normalized_path),
            },
            "assay_candidates_csv": {
                "path": config["outputs"]["assay_candidates_csv"],
                "sha256": sha256(candidates_path),
            },
        },
    }
    result_path = root / config["outputs"]["result_json"]
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )

    report_path = root / config["outputs"]["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    top = candidates[:10]
    lines = [
        "# Stage90 ChEMBL analog-series intake",
        "",
        f"Status: `{result['status']}`.",
        "",
        f"Accepted activities: `{len(all_rows)}` across `{len(candidates)}` single-assay endpoints.",
        f"Analog-series candidates: `{len(analog)}`; full rescue candidates: `{len(full)}`.",
        "",
        "| Target | Assay | Type | Molecules | Repeat series | Series molecules | High | Low | Receptors | Full gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in top:
        lines.append(
            "| {target_id} | {assay_chembl_id} | {standard_type} | {unique_molecule_count} | "
            "{repeat_series_count} | {molecules_in_repeat_series} | {high_potency_count} | "
            "{low_potency_count} | {receptor_pool_size} | {full_rescue_intake_gate_passed} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "At least one full intake candidate passed. Freeze the highest-ranked source in Stage91 before any ligand preparation or docking."
                if full
                else "No source passed all preregistered chemistry, potency, and receptor-pool gates. Do not start docking or quantum work."
            ),
            "",
            "This intake used public ChEMBL records only. It read no protected validation/test rows and launched no docking or hardware job.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage90_chembl_analog_series_intake.json"),
    )
    args = parser.parse_args()
    run(args.root.resolve(), args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
