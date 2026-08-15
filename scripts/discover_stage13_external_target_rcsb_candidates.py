"""Discover RCSB receptor candidates for a preregistered external target."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import statistics
from collections import Counter
from pathlib import Path

try:
    from .discover_mk14_rcsb_receptor_candidates import (
        GRAPHQL_QUERY,
        normalize_entry,
        request_json,
        search_payload,
    )
    from .prepare_receptor import file_sha256
except ImportError:
    from discover_mk14_rcsb_receptor_candidates import (
        GRAPHQL_QUERY,
        normalize_entry,
        request_json,
        search_payload,
    )
    from prepare_receptor import file_sha256


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verified(path: Path, expected_sha256: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != expected_sha256.upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return path


def apply_title_exclusions(
    row: dict[str, object], patterns: list[str]
) -> dict[str, object]:
    output = dict(row)
    matched = [
        pattern
        for pattern in patterns
        if re.search(pattern, str(row["title"]), flags=re.IGNORECASE)
    ]
    output["excluded_title_patterns"] = ";".join(matched)
    if matched:
        reasons = [
            value for value in str(output["exclusion_reasons"]).split(";") if value
        ]
        reasons.append("excluded_title_pattern")
        output["exclusion_reasons"] = ";".join(sorted(set(reasons)))
        output["status"] = "metadata_excluded"
    return output


def load_config(path: Path) -> dict[str, object]:
    config = read_json(path)
    required = {
        "schema_version",
        "experiment_id",
        "purpose",
        "implementation",
        "preregistration",
        "runtime",
        "rcsb",
        "outputs",
        "interpretation_boundary",
    }
    if not required.issubset(config):
        raise ValueError("Stage 13 discovery config is incomplete")
    return config


def write_report(path: Path, summary: dict[str, object]) -> None:
    counts = dict(summary["counts"])
    lines = [
        "# Stage 13 EGFR RCSB Candidate Discovery",
        "",
        "## Result",
        "",
        f"- RCSB X-ray entries: {summary['query']['search_result_count']}",
        f"- Metadata-eligible entries: {counts['metadata_eligible_count']}",
        f"- Eligible new entries: {counts['new_metadata_eligible_count']}",
        f"- Reference structure eligible: {counts['reference_eligible']}",
        "",
        "No ligand label, docking score, MAPK14 Stage 11 row, or test row was read.",
        "Metadata eligibility does not make a receptor docking-ready; coordinate, "
        "ATP-pocket, covalency, preparation, and redocking gates remain mandatory.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, overwrite: bool) -> dict[str, object]:
    config = load_config(config_path)
    implementation = dict(config["implementation"])
    script_path = verified(
        Path(str(implementation["path"])), str(implementation["sha256"])
    )
    if script_path.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 13 implementation path differs")
    preregistration_record = dict(config["preregistration"])
    preregistration_path = verified(
        Path(str(preregistration_record["path"])),
        str(preregistration_record["sha256"]),
    )
    preregistration = read_json(preregistration_path)
    target = dict(preregistration["target"])
    discovery = dict(preregistration["rcsb_discovery"])
    eligibility = dict(discovery["metadata_eligibility"])
    if target["uniprot_accession"] != discovery["uniprot_accession"]:
        raise ValueError("Stage 13 target accession differs")
    boundary = dict(preregistration["data_boundary"])
    if (
        boundary["labels_allowed_during_receptor_discovery_or_structural_selection"]
        is not False
        or boundary[
            "docking_scores_allowed_during_receptor_discovery_or_structural_selection"
        ]
        is not False
        or int(boundary["MAPK14_stage11_rows_permitted"]) != 0
    ):
        raise ValueError("Stage 13 structural discovery boundary differs")

    expected_runtime = {
        key: str(value) for key, value in dict(config["runtime"]).items()
    }
    runtime = {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": platform.python_version(),
    }
    if runtime != expected_runtime:
        raise RuntimeError(f"Stage 13 runtime differs: {runtime} != {expected_runtime}")

    outputs = {
        key: Path(str(value)) for key, value in dict(config["outputs"]).items()
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Stage 13 discovery outputs exist; pass --overwrite")

    rcsb = dict(config["rcsb"])
    request_kwargs = {
        "timeout_seconds": float(rcsb["request_timeout_seconds"]),
        "maximum_retries": int(rcsb["maximum_retries"]),
        "retry_backoff_seconds": float(rcsb["retry_backoff_seconds"]),
    }
    accession = str(discovery["uniprot_accession"])
    method = str(discovery["experimental_method"])
    payload = search_payload(accession, method)
    search_response = request_json(
        str(rcsb["search_endpoint"]), payload, **request_kwargs
    )
    identifiers = sorted(
        str(value["identifier"]).upper()
        for value in search_response.get("result_set") or []
        if isinstance(value, dict) and value.get("identifier")
    )
    if len(identifiers) != int(search_response.get("total_count", -1)):
        raise ValueError("Stage 13 RCSB search identifiers are incomplete")
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("Stage 13 RCSB search identifiers are invalid")

    entries: list[dict[str, object]] = []
    chunk_size = int(rcsb["metadata_chunk_size"])
    for start in range(0, len(identifiers), chunk_size):
        chunk = identifiers[start : start + chunk_size]
        response = request_json(
            str(rcsb["graphql_endpoint"]),
            {"query": GRAPHQL_QUERY, "variables": {"ids": chunk}},
            **request_kwargs,
        )
        if response.get("errors"):
            raise ValueError(f"Stage 13 RCSB GraphQL errors: {response['errors']}")
        data = response.get("data")
        if not isinstance(data, dict):
            raise ValueError("Stage 13 RCSB GraphQL data are missing")
        entries.extend(
            value for value in data.get("entries") or [] if isinstance(value, dict)
        )
    entries.sort(key=lambda value: str(value.get("rcsb_id", "")))
    if {str(value.get("rcsb_id", "")).upper() for value in entries} != set(
        identifiers
    ):
        raise ValueError("Stage 13 RCSB metadata IDs differ")

    reference_id = str(target["dude_reference_structure"]).upper()
    rows = [
        apply_title_exclusions(
            normalize_entry(entry, accession, eligibility, {reference_id}),
            [str(value) for value in eligibility["excluded_title_patterns"]],
        )
        for entry in entries
    ]
    for row in rows:
        row["is_dude_reference_structure"] = row["pdb_id"] == reference_id
    rows.sort(key=lambda value: str(value["pdb_id"]))
    eligible = [row for row in rows if row["status"] == "metadata_eligible"]
    eligible_new = [
        row for row in eligible if not row["is_dude_reference_structure"]
    ]
    reference_rows = [row for row in rows if row["pdb_id"] == reference_id]
    if len(reference_rows) != 1 or reference_rows[0]["status"] != "metadata_eligible":
        raise ValueError("Stage 13 DUD-E reference is not metadata-eligible")
    minimum = int(
        preregistration["receptor_pool_plan"][
            "metadata_minimum_eligible_count_including_reference"
        ]
    )
    if len(eligible) < minimum:
        raise ValueError("too few Stage 13 metadata-eligible structures")

    search_snapshot = {
        "schema_version": "1.0",
        "retrieved_on": discovery["retrieved_on"],
        "endpoint": rcsb["search_endpoint"],
        "request": payload,
        "total_count": len(identifiers),
        "identifiers": identifiers,
    }
    metadata_snapshot = {
        "schema_version": "1.0",
        "retrieved_on": discovery["retrieved_on"],
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
        "status": "stage13_egfr_metadata_discovery_ok",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "preregistration": {
            "path": preregistration_path.as_posix(),
            "sha256": file_sha256(preregistration_path),
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
            "MAPK14_stage11_rows_read": 0,
            "test_rows_read": 0,
        },
        "outputs": {
            key: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for key, path in outputs.items()
            if key not in {"summary_json", "report_md"}
        },
        "next_gate": (
            "download coordinates, audit sequence and ATP-pocket completeness, "
            "remove covalent or non-ATP-site complexes, align to 2RGP chain A, "
            "and run label-independent max-min structural selection"
        ),
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["summary_json"], summary)
    write_report(outputs["report_md"], summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
