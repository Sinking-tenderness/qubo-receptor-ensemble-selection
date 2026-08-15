"""Discover label-independent RCSB receptor candidates for PARP1."""

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
        "# Stage 17b PARP1 RCSB Candidate Discovery",
        "",
        "- RCSB P09874 X-ray entries: " + str(summary["query"]["search_result_count"]),
        "- Metadata-eligible entries: " + str(counts["metadata_eligible_count"]),
        "- Eligible new entries: " + str(counts["new_metadata_eligible_count"]),
        "- 3L3M reference eligible: " + str(counts["reference_eligible"]).lower(),
        "",
        "No PARP1 benchmark docking score, fresh-validation row, or test row was read.",
        "Coordinate auditing, preparation, and three-seed cognate redocking remain required.",
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
        raise ValueError("Stage 17b implementation SHA-256 differs")
    for dependency in implementation["dependencies"]:
        verified(root, dict(dependency))

    preregistration_path = verified(root, dict(config["preregistration"]))
    preregistration = read_json(preregistration_path)
    if preregistration["preregistration_id"] != "stage17-parp1-replacement-exploratory-20260801-v1":
        raise ValueError("PARP1 preregistration differs")
    source_path = verified(root, dict(config["source_audit"]))
    if read_json(source_path)["status"] != "stage17a_parp1_source_and_active_allocation_ok":
        raise ValueError("Stage 17a PARP1 source audit did not pass")

    runtime = {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": platform.python_version(),
    }
    expected_runtime = {key: str(value) for key, value in dict(config["runtime"]).items()}
    if runtime != expected_runtime:
        raise RuntimeError(f"Stage 17b runtime differs: {runtime} != {expected_runtime}")
    if any(
        int(preregistration["data_boundary"][key]) != 0
        for key in (
            "PARP1_benchmark_docking_scores_read",
            "PARP1_fresh_validation_rows_read",
            "PARP1_test_rows_read",
        )
    ):
        raise ValueError("Stage 17b crossed a protected data boundary")

    target = dict(preregistration["target"])
    rules = dict(config["discovery_rules"])
    eligibility = dict(rules["metadata_eligibility"])
    outputs = {key: root / str(value) for key, value in dict(config["outputs"]).items()}
    if any(path.exists() for path in outputs.values()) and not overwrite:
        raise FileExistsError("Stage 17b outputs exist; pass --overwrite")

    rcsb = dict(config["rcsb"])
    request_kwargs = {
        "timeout_seconds": float(rcsb["request_timeout_seconds"]),
        "maximum_retries": int(rcsb["maximum_retries"]),
        "retry_backoff_seconds": float(rcsb["retry_backoff_seconds"]),
    }
    accession = str(target["uniprot_accession"])
    method = str(rules["experimental_method"])
    payload = search_payload(accession, method)
    response = request_json(str(rcsb["search_endpoint"]), payload, **request_kwargs)
    identifiers = sorted(
        str(value["identifier"]).upper()
        for value in response.get("result_set") or []
        if isinstance(value, dict) and value.get("identifier")
    )
    if len(identifiers) != int(response.get("total_count", -1)) or not identifiers:
        raise ValueError("Stage 17b RCSB identifiers differ")

    entries: list[dict[str, object]] = []
    for start in range(0, len(identifiers), int(rcsb["metadata_chunk_size"])):
        metadata = request_json(
            str(rcsb["graphql_endpoint"]),
            {"query": GRAPHQL_QUERY, "variables": {"ids": identifiers[start : start + int(rcsb["metadata_chunk_size"])]}},
            **request_kwargs,
        )
        if metadata.get("errors") or not isinstance(metadata.get("data"), dict):
            raise ValueError("Stage 17b RCSB GraphQL response differs")
        entries.extend(
            value
            for value in metadata["data"].get("entries") or []
            if isinstance(value, dict)
        )
    if {str(value.get("rcsb_id", "")).upper() for value in entries} != set(identifiers):
        raise ValueError("Stage 17b RCSB metadata IDs differ")

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
    rows.sort(key=lambda row: str(row["pdb_id"]))
    eligible = [row for row in rows if row["status"] == "metadata_eligible"]
    reference_rows = [row for row in rows if row["pdb_id"] == reference_id]
    if len(reference_rows) != 1 or reference_rows[0]["status"] != "metadata_eligible":
        raise ValueError("3L3M reference is not metadata eligible")
    if len(eligible) < int(rules["minimum_metadata_eligible_count"]):
        raise ValueError("too few PARP1 metadata-eligible structures")

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
        "status": "stage17b_parp1_metadata_discovery_ok",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "runtime": runtime,
        "query": {"uniprot_accession": accession, "experimental_method": method, "search_result_count": len(identifiers)},
        "counts": {
            "metadata_entry_count": len(rows),
            "metadata_eligible_count": len(eligible),
            "new_metadata_eligible_count": sum(not row["is_dude_reference_structure"] for row in eligible),
            "metadata_excluded_count": len(rows) - len(eligible),
            "reference_eligible": True,
        },
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "eligible_resolution_angstrom": {"minimum": min(resolutions), "median": statistics.median(resolutions), "maximum": max(resolutions)},
        "eligible_pdb_ids": [str(row["pdb_id"]) for row in eligible],
        "data_boundary": {"ligand_labels_read": 0, "docking_scores_read": 0, "fresh_validation_rows_read": 0, "test_rows_read": 0},
        "outputs": {key: {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)} for key, path in outputs.items() if key not in {"summary_json", "report_md"}},
        "next_gate": "download and structurally audit PARP1 coordinates in the 3L3M chain-A A92 frame",
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
