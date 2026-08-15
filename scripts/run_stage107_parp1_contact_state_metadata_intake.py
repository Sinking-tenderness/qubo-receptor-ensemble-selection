"""Run a current, outcome-unseen PARP1 metadata intake for Stage107."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require_hash(root: Path, record: dict[str, str]) -> None:
    path = root / record["path"]
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"input hash mismatch: {record['path']}")


def write_csv_allow_empty(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path, config_path: Path) -> dict[str, Any]:
    from scripts.discover_mk14_rcsb_receptor_candidates import GRAPHQL_QUERY
    from scripts.discover_stage13_external_target_rcsb_candidates import (
        apply_title_exclusions,
        normalize_entry,
        request_json,
        search_payload,
        write_csv,
        write_json,
    )

    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    for key in ("reused_discovery_logic", "preregistration", "source_audit"):
        require_hash(root, config[key])
    stage106 = root / config["rationale"]["stage106_contact_state_audit"]
    stage106_record = json.loads(stage106.read_text(encoding="utf-8"))
    if stage106_record["status"] != "stage106_cocrystal_contact_state_audit_complete":
        raise ValueError("Stage106 contact-state audit is unavailable")
    if any(value != 0 for value in config["data_boundary"].values()):
        raise ValueError("Stage107 configuration must begin with an empty protected-data boundary")

    outputs = config["outputs"]
    preregistration = json.loads((root / config["preregistration"]["path"]).read_text(encoding="utf-8"))
    target = preregistration["target"]
    rcsb = config["rcsb"]
    request_kwargs = {
        "timeout_seconds": float(rcsb["request_timeout_seconds"]),
        "maximum_retries": int(rcsb["maximum_retries"]),
        "retry_backoff_seconds": float(rcsb["retry_backoff_seconds"]),
    }
    payload = search_payload(target["uniprot_accession"], config["discovery_rules"]["experimental_method"])
    response = request_json(str(rcsb["search_endpoint"]), payload, **request_kwargs)
    identifiers = sorted(
        str(item["identifier"]).upper()
        for item in response.get("result_set") or []
        if isinstance(item, dict) and item.get("identifier")
    )
    if not identifiers or len(identifiers) != int(response.get("total_count", -1)):
        raise ValueError("RCSB search identifiers differ")
    entries = []
    for start in range(0, len(identifiers), int(rcsb["metadata_chunk_size"])):
        response = request_json(
            str(rcsb["graphql_endpoint"]),
            {"query": GRAPHQL_QUERY, "variables": {"ids": identifiers[start : start + int(rcsb["metadata_chunk_size"])]}},
            **request_kwargs,
        )
        if response.get("errors") or not isinstance(response.get("data"), dict):
            raise ValueError("RCSB GraphQL response differs")
        entries.extend(item for item in response["data"].get("entries") or [] if isinstance(item, dict))
    if {str(item.get("rcsb_id", "")).upper() for item in entries} != set(identifiers):
        raise ValueError("RCSB metadata IDs differ")
    rules = config["discovery_rules"]
    eligibility = rules["metadata_eligibility"]
    reference_id = str(target["dude_reference_structure"]).upper()
    rows = [
        apply_title_exclusions(
            normalize_entry(item, target["uniprot_accession"], eligibility, {reference_id}),
            [str(value) for value in eligibility["excluded_title_patterns"]],
        )
        for item in entries
    ]
    for row in rows:
        row["is_dude_reference_structure"] = row["pdb_id"] == reference_id
    rows.sort(key=lambda row: str(row["pdb_id"]))
    eligible = [row for row in rows if row["status"] == "metadata_eligible"]
    reference_row = next(row for row in rows if row["pdb_id"] == reference_id)
    reference_eligible = reference_row["status"] == "metadata_eligible"
    structural_count_passes = len(eligible) >= int(rules["minimum_metadata_eligible_count"])
    gate_passes = structural_count_passes and reference_eligible
    search_snapshot = {
        "schema_version": "1.0",
        "retrieved_on": config["retrieved_on"],
        "endpoint": rcsb["search_endpoint"],
        "request": payload,
        "total_count": len(identifiers),
        "identifiers": identifiers,
    }
    metadata_snapshot = {
        "schema_version": "1.0",
        "retrieved_on": config["retrieved_on"],
        "endpoint": rcsb["graphql_endpoint"],
        "graphql_query": GRAPHQL_QUERY,
        "entry_count": len(entries),
        "entries": entries,
    }
    write_json(root / outputs["search_snapshot_json"], search_snapshot)
    write_json(root / outputs["metadata_snapshot_json"], metadata_snapshot)
    write_csv(root / outputs["candidate_metadata_csv"], rows)
    write_csv_allow_empty(root / outputs["eligible_candidates_csv"], eligible, list(rows[0]))
    status = "stage107_parp1_contact_state_metadata_intake_complete" if gate_passes else "stage107_parp1_contact_state_metadata_intake_no_go"
    result = {
        "schema_version": "1.0",
        "status": status,
        "evidence_status": "posthoc mechanism-guided, outcome-unseen structural feasibility intake; it is not an efficacy result or a trained selector.",
        "config": {"path": config_path.as_posix(), "sha256": sha256(root / config_path)},
        "parent_stage106_status": stage106_record["status"],
        "query": {"uniprot_accession": target["uniprot_accession"], "experimental_method": rules["experimental_method"], "search_result_count": len(identifiers)},
        "counts": {
            "metadata_entry_count": len(rows),
            "metadata_eligible_count": len(eligible),
            "new_metadata_eligible_count": sum(not row["is_dude_reference_structure"] for row in eligible),
            "reference_eligible": reference_eligible,
            "structural_count_passes": structural_count_passes,
            "gate_passes": gate_passes,
        },
        "reference_metadata_record": reference_row,
        "eligible_pdb_ids": [str(row["pdb_id"]) for row in eligible],
        "data_boundary": config["data_boundary"],
        "decision": {
            "coordinate_structural_audit_authorized": False,
            "ligand_preparation_authorized": False,
            "redocking_authorized": False,
            "production_docking_authorized": False,
            "parp1_fresh_validation_released": False,
            "parp1_locked_test_released": False,
            "quantum_hardware_authorized": False,
            "next_action": "Stop the legacy PARP1 branch before coordinate download because its 3L3M reference does not meet the frozen mutation and coverage rules. A future protocol, if any, must select and freeze a different wild-type reference using a separately audited public-metadata survey."
        },
        "interpretation": "The legacy reference fails because RCSB records V762A and 0.345 reference coverage. This is a No-Go under the frozen rules, not a reason to loosen them after seeing current metadata."
    }
    result_path = root / outputs["summary_json"]
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    report = [
        "# Stage107 PARP1 Contact-State Metadata Intake",
        "",
        "This is a current public RCSB metadata snapshot. It is posthoc mechanism-guided by Stage106 but remains outcome-unseen for PARP1: no PARP1 benchmark docking, validation, or test outcome was read.",
        "",
        f"- P09874 X-ray entries: `{result['query']['search_result_count']}`.",
        f"- Metadata-eligible entries: `{result['counts']['metadata_eligible_count']}`.",
        f"- Eligible entries excluding the 3L3M reference: `{result['counts']['new_metadata_eligible_count']}`.",
        f"- 3L3M reference eligible: `{result['counts']['reference_eligible']}`.",
        "",
        "## Decision",
        "",
        f"The legacy reference gate: `{'PASS' if gate_passes else 'NO-GO'}`. 3L3M has exclusion reasons `{reference_row['exclusion_reasons']}`. No coordinate download is authorized. Ligand preparation, Uni-Dock, training, validation, test evaluation, QUBO fitting, and quantum hardware remain locked.",
        ""
    ]
    report_path = root / outputs["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage107_parp1_contact_state_metadata_intake.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    result = run(root, args.config)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
