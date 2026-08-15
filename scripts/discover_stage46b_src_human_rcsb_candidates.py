"""Discover human SRC candidates after the frozen ortholog-reference correction."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.discover_mk14_rcsb_receptor_candidates import (
    GRAPHQL_QUERY,
    normalize_entry,
    request_json,
    search_payload,
)
from scripts.discover_stage13_external_target_rcsb_candidates import (
    apply_title_exclusions,
    file_sha256,
    read_json,
    write_csv,
    write_json,
)


def verified(root: Path, record: dict[str, Any]) -> Path:
    path = root / record["path"]
    if not path.is_file() or file_sha256(path) != record["sha256"].upper():
        raise ValueError(f"Stage46b input identity differs: {path}")
    return path


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage46b implementation path differs")
    source_path = verified(root, config["source_audit"])
    amendment_path = verified(root, config["identity_amendment"])
    source = read_json(source_path)
    amendment = read_json(amendment_path)
    if source.get("status") != "stage46_src_source_audit_ok":
        raise ValueError("Stage46 source audit did not pass")
    if amendment.get("amendment_id") != "stage46-src-human-ortholog-reference-amendment01-20260804-v1":
        raise ValueError("Stage46 identity amendment differs")
    accession = amendment["identity_correction"]["prospective_target_uniprot_accession"]
    ortholog_reference_id = amendment["identity_correction"]["dude_reference_pdb_id"]

    outputs = {key: root / value for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage46b outputs exist; pass --overwrite")
    rcsb = config["rcsb"]
    request_kwargs = {
        "timeout_seconds": float(rcsb["request_timeout_seconds"]),
        "maximum_retries": int(rcsb["maximum_retries"]),
        "retry_backoff_seconds": float(rcsb["retry_backoff_seconds"]),
    }
    method = config["discovery_rules"]["experimental_method"]
    payload = search_payload(accession, method)
    search_response = request_json(rcsb["search_endpoint"], payload, **request_kwargs)
    identifiers = sorted(
        row["identifier"].upper()
        for row in search_response.get("result_set", [])
        if isinstance(row, dict) and row.get("identifier")
    )
    if not identifiers:
        raise ValueError("human SRC RCSB search returned no structures")

    entries: list[dict[str, Any]] = []
    chunk_size = int(rcsb["metadata_chunk_size"])
    for start in range(0, len(identifiers), chunk_size):
        response = request_json(
            rcsb["graphql_endpoint"],
            {"query": GRAPHQL_QUERY, "variables": {"ids": identifiers[start : start + chunk_size]}},
            **request_kwargs,
        )
        if response.get("errors"):
            raise ValueError(f"Stage46b RCSB GraphQL errors: {response['errors']}")
        data = response.get("data")
        if not isinstance(data, dict):
            raise ValueError("Stage46b RCSB GraphQL data are missing")
        entries.extend(row for row in data.get("entries") or [] if isinstance(row, dict))
    entries.sort(key=lambda row: row.get("rcsb_id", ""))
    if {row.get("rcsb_id", "").upper() for row in entries} != set(identifiers):
        raise ValueError("human SRC metadata IDs differ from search response")

    eligibility = config["discovery_rules"]["metadata_eligibility"]
    rows = [
        apply_title_exclusions(
            normalize_entry(entry, accession, eligibility, set()),
            eligibility["excluded_title_patterns"],
        )
        for entry in entries
    ]
    rows.sort(key=lambda row: row["pdb_id"])
    eligible = [row for row in rows if row["status"] == "metadata_eligible"]
    minimum_count = int(config["discovery_rules"]["minimum_metadata_eligible_count"])
    if len(eligible) < minimum_count:
        raise ValueError(f"too few human SRC metadata-eligible structures: {len(eligible)} < {minimum_count}")
    if ortholog_reference_id in {row["pdb_id"].upper() for row in rows}:
        raise ValueError("ortholog 3EL8 unexpectedly entered the P12931 query")
    reference = min(
        eligible,
        key=lambda row: (
            float(row["resolution_angstrom"]),
            str(row["initial_release_date"]),
            str(row["pdb_id"]),
        ),
    )
    reference_id = reference["pdb_id"]
    for row in rows:
        row["is_selected_human_reference"] = row["pdb_id"] == reference_id

    write_json(
        outputs["search_snapshot_json"],
        {
            "schema_version": "1.0",
            "retrieved_on": config["retrieved_on"],
            "endpoint": rcsb["search_endpoint"],
            "request": payload,
            "total_count": len(identifiers),
            "identifiers": identifiers,
        },
    )
    write_json(
        outputs["metadata_snapshot_json"],
        {
            "schema_version": "1.0",
            "retrieved_on": config["retrieved_on"],
            "endpoint": rcsb["graphql_endpoint"],
            "graphql_query": GRAPHQL_QUERY,
            "entry_count": len(entries),
            "entries": entries,
        },
    )
    write_csv(outputs["candidate_metadata_csv"], rows)
    write_csv(outputs["eligible_candidates_csv"], eligible)
    reason_counts = Counter(
        reason
        for row in rows
        for reason in str(row["exclusion_reasons"]).split(";")
        if reason
    )
    resolutions = [float(row["resolution_angstrom"]) for row in eligible]
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage46b_src_human_metadata_discovery_ok",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "source_audit": {"path": source_path.relative_to(root).as_posix(), "sha256": file_sha256(source_path)},
        "identity_amendment": {"path": amendment_path.relative_to(root).as_posix(), "sha256": file_sha256(amendment_path)},
        "query": {
            "uniprot_accession": accession,
            "experimental_method": method,
            "search_result_count": len(identifiers),
            "ortholog_reference_excluded": ortholog_reference_id,
        },
        "counts": {
            "metadata_entry_count": len(rows),
            "metadata_eligible_count": len(eligible),
            "metadata_excluded_count": len(rows) - len(eligible),
        },
        "selected_human_reference": {
            "pdb_id": reference_id,
            "resolution_angstrom": float(reference["resolution_angstrom"]),
            "initial_release_date": reference["initial_release_date"],
            "selection_rule": amendment["human_reference_selection_rule"],
        },
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "eligible_resolution_angstrom": {
            "minimum": min(resolutions),
            "median": statistics.median(resolutions),
            "maximum": max(resolutions),
        },
        "eligible_pdb_ids": [row["pdb_id"] for row in eligible],
        "data_boundary": {
            "ligand_labels_read": 0,
            "docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
        },
        "decision": {
            "metadata_gate_passed": True,
            "coordinate_audit_authorized": True,
            "production_docking_authorized": False,
        },
        "outputs": {
            key: {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)}
            for key, path in outputs.items()
            if key not in {"summary_json", "report_md"}
        },
        "next_gate": "download the frozen human P12931 candidate coordinates and apply alignment, pocket-completeness, covalency, and preparation-readiness gates",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["summary_json"], summary)
    report = [
        "# Stage46b human SRC RCSB candidate discovery",
        "",
        f"- P12931 X-ray entries: {len(identifiers)}",
        f"- Metadata-eligible entries: {len(eligible)}",
        f"- Deterministic human reference: {reference_id}",
        "",
        "3EL8 remains an ortholog pocket template and is not a human receptor candidate.",
        "No docking or held-out outcome was read.",
        "",
    ]
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report), encoding="ascii")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else args.root / args.config
    run(config_path, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
