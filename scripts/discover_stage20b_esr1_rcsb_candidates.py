"""Discover label-independent RCSB receptor candidates for ESR1."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

try:
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
except ModuleNotFoundError:
    from discover_mk14_rcsb_receptor_candidates import (
        GRAPHQL_QUERY,
        normalize_entry,
        request_json,
        search_payload,
    )
    from discover_stage13_external_target_rcsb_candidates import (
        apply_title_exclusions,
        file_sha256,
        read_json,
        write_csv,
        write_json,
    )


def verified(root: Path, record: dict[str, object]) -> Path:
    path = root / str(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(record["sha256"]).upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return path


def write_report(path: Path, summary: dict[str, object]) -> None:
    counts = dict(summary["counts"])
    lines = [
        "# Stage 20b ESR1 RCSB Candidate Discovery",
        "",
        "- RCSB P03372 X-ray entries: " + str(summary["query"]["search_result_count"]),
        "- Metadata-eligible entries: " + str(counts["metadata_eligible_count"]),
        "- Eligible new entries: " + str(counts["new_metadata_eligible_count"]),
        "- 1SJ0 reference eligible: " + str(counts["reference_eligible"]).lower(),
        "",
        "The ESR1-specific coverage rule evaluates the approximately 250-residue ligand-binding domain rather than requiring half of the 595-residue full-length protein.",
        "No ESR1 docking score, fresh-validation row, test row, or PPARG Stage 19c enrichment row was read.",
        "Coordinate auditing, preparation, and three-seed cognate redocking remain required.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    if file_sha256(Path(__file__)) != str(config["implementation"]["sha256"]).upper():
        raise ValueError("Stage 20b implementation SHA-256 differs")
    preregistration_path = verified(root, dict(config["preregistration"]))
    preregistration = read_json(preregistration_path)
    if (
        preregistration["preregistration_id"]
        != "stage20-esr1-independent-exploratory-20260801-v1"
    ):
        raise ValueError("ESR1 preregistration differs")
    source_audit_path = verified(root, dict(config["source_audit"]))
    if (
        read_json(source_audit_path)["status"]
        != "stage20a_esr1_source_and_active_allocation_ok"
    ):
        raise ValueError("Stage 20a ESR1 source audit did not pass")
    protected = preregistration["data_boundary"]
    if any(
        int(protected[key]) != 0
        for key in (
            "ESR1_benchmark_docking_scores_read",
            "ESR1_fresh_validation_rows_read",
            "ESR1_test_rows_read",
            "PPARG_stage19c_enrichment_rows_read",
        )
    ):
        raise ValueError("Stage 20b crossed a protected data boundary")

    outputs = {key: root / str(value) for key, value in dict(config["outputs"]).items()}
    if any(path.exists() for path in outputs.values()) and not overwrite:
        raise FileExistsError("Stage 20b outputs exist; pass --overwrite")
    rcsb = dict(config["rcsb"])
    request_kwargs = {
        "timeout_seconds": float(rcsb["request_timeout_seconds"]),
        "maximum_retries": int(rcsb["maximum_retries"]),
        "retry_backoff_seconds": float(rcsb["retry_backoff_seconds"]),
    }
    target = dict(preregistration["target"])
    accession = str(target["uniprot_accession"])
    method = str(config["discovery_rules"]["experimental_method"])
    payload = search_payload(accession, method)
    search_response = request_json(str(rcsb["search_endpoint"]), payload, **request_kwargs)
    identifiers = sorted(
        str(row["identifier"]).upper()
        for row in search_response.get("result_set", [])
        if isinstance(row, dict) and row.get("identifier")
    )
    if not identifiers:
        raise ValueError("ESR1 RCSB search returned no structures")
    entries: list[dict[str, object]] = []
    chunk_size = int(rcsb["metadata_chunk_size"])
    for start in range(0, len(identifiers), chunk_size):
        response = request_json(
            str(rcsb["graphql_endpoint"]),
            {
                "query": GRAPHQL_QUERY,
                "variables": {"ids": identifiers[start : start + chunk_size]},
            },
            **request_kwargs,
        )
        if response.get("errors"):
            raise ValueError(f"Stage 20b RCSB GraphQL errors: {response['errors']}")
        data = response.get("data")
        if not isinstance(data, dict):
            raise ValueError("Stage 20b RCSB GraphQL data are missing")
        entries.extend(row for row in data.get("entries") or [] if isinstance(row, dict))
    entries.sort(key=lambda row: str(row.get("rcsb_id", "")))
    if {str(row.get("rcsb_id", "")).upper() for row in entries} != set(identifiers):
        raise ValueError("ESR1 RCSB metadata IDs differ from the search response")

    eligibility = dict(config["discovery_rules"]["metadata_eligibility"])
    reference_id = str(target["dude_reference_structure"]).upper()
    rows = [
        apply_title_exclusions(
            normalize_entry(entry, accession, eligibility, {reference_id}),
            [str(value) for value in eligibility["excluded_title_patterns"]],
        )
        for entry in entries
    ]
    rows.sort(key=lambda row: str(row["pdb_id"]))
    eligible = [row for row in rows if row["status"] == "metadata_eligible"]
    eligible_new = [
        row for row in eligible if str(row["pdb_id"]).upper() != reference_id
    ]
    reference_rows = [row for row in rows if str(row["pdb_id"]).upper() == reference_id]
    if len(reference_rows) != 1 or reference_rows[0]["status"] != "metadata_eligible":
        raise ValueError("ESR1 DUD-E reference is not metadata eligible")
    if len(eligible) < int(config["discovery_rules"]["minimum_metadata_eligible_count"]):
        raise ValueError("too few ESR1 metadata-eligible structures")

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
        "status": "stage20b_esr1_metadata_discovery_ok",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "preregistration": {
            "path": preregistration_path.relative_to(root).as_posix(),
            "sha256": file_sha256(preregistration_path),
        },
        "source_audit": {
            "path": source_audit_path.relative_to(root).as_posix(),
            "sha256": file_sha256(source_audit_path),
        },
        "query": {
            "uniprot_accession": accession,
            "experimental_method": method,
            "search_result_count": len(identifiers),
        },
        "counts": {
            "metadata_entry_count": len(rows),
            "metadata_eligible_count": len(eligible),
            "new_metadata_eligible_count": len(eligible_new),
            "metadata_excluded_count": len(rows) - len(eligible),
            "reference_eligible": True,
        },
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "eligible_resolution_angstrom": {
            "minimum": min(resolutions),
            "median": statistics.median(resolutions),
            "maximum": max(resolutions),
        },
        "eligible_pdb_ids": [str(row["pdb_id"]) for row in eligible],
        "data_boundary": {
            "ligand_labels_read": 0,
            "docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "pparg_stage19c_enrichment_rows_read": 0,
        },
        "outputs": {
            key: {
                "path": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
            }
            for key, path in outputs.items()
            if key not in {"summary_json", "report_md"}
        },
        "next_gate": "coordinate download, chain audit, proper-rotation alignment, pocket completeness, pocket-proximal ligand check, and deterministic structural max-min selection",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["summary_json"], summary)
    write_report(outputs["report_md"], summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
