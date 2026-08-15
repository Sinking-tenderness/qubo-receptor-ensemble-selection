"""Discover label-independent RCSB receptor candidates for DPP4."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
from collections import Counter
from pathlib import Path

from scripts.discover_mk14_rcsb_receptor_candidates import GRAPHQL_QUERY
from scripts.discover_stage13_external_target_rcsb_candidates import (
    apply_title_exclusions,
    file_sha256,
    normalize_entry,
    read_json,
    request_json,
    search_payload,
    write_csv,
    write_json,
)


def verified(path: Path, expected_sha256: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != expected_sha256.upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return path


def write_report(path: Path, summary: dict[str, object]) -> None:
    counts = dict(summary["counts"])
    lines = [
        "# Stage 16b DPP4 RCSB Candidate Discovery",
        "",
        "## Result",
        "",
        f"- RCSB P27487 X-ray entries: {summary['query']['search_result_count']}",
        f"- Metadata-eligible entries: {counts['metadata_eligible_count']}",
        f"- Eligible new entries: {counts['new_metadata_eligible_count']}",
        f"- 2I78 reference eligible: {str(counts['reference_eligible']).lower()}",
        "",
        "No DPP4 benchmark docking score, fresh-validation row, or test row was read.",
        "Metadata eligibility authorizes coordinate auditing only; catalytic-region",
        "completeness, noncovalency, preparation readiness, and cognate redocking",
        "remain mandatory.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = dict(config["implementation"])
    if file_sha256(Path(__file__)) != str(implementation["sha256"]).upper():
        raise ValueError("Stage 16b implementation SHA-256 differs")
    for dependency in implementation["dependencies"]:
        verified(root / str(dependency["path"]), str(dependency["sha256"]))

    prereg_record = dict(config["preregistration"])
    prereg_path = verified(
        root / str(prereg_record["path"]), str(prereg_record["sha256"])
    )
    preregistration = read_json(prereg_path)
    if preregistration["preregistration_id"] != "stage16-dpp4-replacement-exploratory-20260731-v1":
        raise ValueError("Stage 16 DPP4 preregistration differs")

    source_config_record = dict(config["source_configuration"])
    source_config_path = verified(
        root / str(source_config_record["path"]),
        str(source_config_record["sha256"]),
    )
    source_configuration = read_json(source_config_path)
    source_record = dict(config["source_audit"])
    source_path = verified(
        root / str(source_record["path"]), str(source_record["sha256"])
    )
    if read_json(source_path)["status"] != "stage16a_dpp4_source_audit_ok":
        raise ValueError("Stage 16a DPP4 source audit did not pass")

    expected_runtime = {
        key: str(value) for key, value in dict(config["runtime"]).items()
    }
    runtime = {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": platform.python_version(),
    }
    if runtime != expected_runtime:
        raise RuntimeError(f"Stage 16b runtime differs: {runtime} != {expected_runtime}")

    boundary = dict(preregistration["data_boundary"])
    if any(
        int(boundary[key]) != 0
        for key in (
            "DPP4_benchmark_docking_scores_read",
            "DPP4_fresh_validation_rows_read",
            "DPP4_test_rows_read",
        )
    ):
        raise ValueError("Stage 16b structural discovery boundary differs")

    target = dict(source_configuration["target"])
    discovery = dict(source_configuration["rcsb_discovery"])
    eligibility = dict(discovery["metadata_eligibility"])
    if target["uniprot_accession"] != discovery["uniprot_accession"]:
        raise ValueError("Stage 16b target accession differs")
    outputs = {
        key: root / str(value) for key, value in dict(config["outputs"]).items()
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Stage 16b outputs exist; pass --overwrite")

    rcsb = dict(config["rcsb"])
    request_kwargs = {
        "timeout_seconds": float(rcsb["request_timeout_seconds"]),
        "maximum_retries": int(rcsb["maximum_retries"]),
        "retry_backoff_seconds": float(rcsb["retry_backoff_seconds"]),
    }
    accession = str(discovery["uniprot_accession"])
    method = str(discovery["experimental_method"])
    payload = search_payload(accession, method)
    response = request_json(str(rcsb["search_endpoint"]), payload, **request_kwargs)
    identifiers = sorted(
        str(value["identifier"]).upper()
        for value in response.get("result_set") or []
        if isinstance(value, dict) and value.get("identifier")
    )
    if len(identifiers) != int(response.get("total_count", -1)):
        raise ValueError("Stage 16b RCSB identifiers are incomplete")
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("Stage 16b RCSB identifiers are invalid")

    entries: list[dict[str, object]] = []
    chunk_size = int(rcsb["metadata_chunk_size"])
    for start in range(0, len(identifiers), chunk_size):
        chunk = identifiers[start : start + chunk_size]
        metadata = request_json(
            str(rcsb["graphql_endpoint"]),
            {"query": GRAPHQL_QUERY, "variables": {"ids": chunk}},
            **request_kwargs,
        )
        if metadata.get("errors"):
            raise ValueError(f"Stage 16b RCSB GraphQL errors: {metadata['errors']}")
        data = metadata.get("data")
        if not isinstance(data, dict):
            raise ValueError("Stage 16b RCSB GraphQL data are missing")
        entries.extend(
            value for value in data.get("entries") or [] if isinstance(value, dict)
        )
    entries.sort(key=lambda value: str(value.get("rcsb_id", "")))
    if {str(value.get("rcsb_id", "")).upper() for value in entries} != set(
        identifiers
    ):
        raise ValueError("Stage 16b RCSB metadata IDs differ")

    reference_id = str(target["dude_reference_structure"]).upper()
    patterns = [str(value) for value in eligibility["excluded_title_patterns"]]
    rows = [
        apply_title_exclusions(
            normalize_entry(entry, accession, eligibility, {reference_id}), patterns
        )
        for entry in entries
    ]
    for row in rows:
        row["is_dude_reference_structure"] = row["pdb_id"] == reference_id
    rows.sort(key=lambda value: str(value["pdb_id"]))
    eligible = [row for row in rows if row["status"] == "metadata_eligible"]
    eligible_new = [row for row in eligible if not row["is_dude_reference_structure"]]
    reference_rows = [row for row in rows if row["pdb_id"] == reference_id]
    if len(reference_rows) != 1 or reference_rows[0]["status"] != "metadata_eligible":
        raise ValueError("Stage 16b 2I78 reference is not metadata-eligible")
    minimum = int(
        dict(source_configuration["receptor_pool_plan"])[
            "metadata_minimum_eligible_count_including_reference"
        ]
    )
    if len(eligible) < minimum:
        raise ValueError("too few Stage 16b metadata-eligible structures")

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
    write_json(outputs["search_snapshot_json"], search_snapshot)
    write_json(outputs["metadata_snapshot_json"], metadata_snapshot)
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
        "status": "stage16b_dpp4_metadata_discovery_ok",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "preregistration": {
            "path": prereg_path.relative_to(root).as_posix(),
            "sha256": file_sha256(prereg_path),
        },
        "source_audit": {
            "path": source_path.relative_to(root).as_posix(),
            "sha256": file_sha256(source_path),
        },
        "runtime": runtime,
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
        },
        "outputs": {
            key: {
                "path": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
            }
            for key, path in outputs.items()
            if key not in {"summary_json", "report_md"}
        },
        "next_gate": "download and structurally audit DPP4 coordinates in the 2I78 chain-B KIQ frame",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["summary_json"], summary)
    write_report(outputs["report_md"], summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
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
