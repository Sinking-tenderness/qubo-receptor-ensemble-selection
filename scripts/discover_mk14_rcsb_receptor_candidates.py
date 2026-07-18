"""Discover label-independent MAPK14 receptor candidates from RCSB PDB."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

try:
    from .prepare_receptor import file_sha256
except ImportError:
    from prepare_receptor import file_sha256


REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "purpose",
    "preregistration",
    "runtime",
    "retrieved_on",
    "rcsb",
    "outputs",
    "interpretation_boundary",
}
REQUIRED_OUTPUT_KEYS = {
    "search_snapshot_json",
    "metadata_snapshot_json",
    "candidate_metadata_csv",
    "eligible_new_candidates_csv",
    "summary_json",
}
GRAPHQL_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    exptl { method }
    rcsb_entry_info { resolution_combined }
    rcsb_accession_info { initial_release_date }
    polymer_entities {
      rcsb_id
      entity_poly {
        rcsb_mutation_count
        rcsb_sample_sequence_length
      }
      rcsb_polymer_entity { pdbx_mutation }
      rcsb_polymer_entity_container_identifiers {
        auth_asym_ids
        reference_sequence_identifiers {
          database_name
          database_accession
          entity_sequence_coverage
          reference_sequence_coverage
        }
      }
    }
    nonpolymer_entities {
      rcsb_id
      pdbx_entity_nonpoly { comp_id name }
      rcsb_nonpolymer_entity { formula_weight }
      rcsb_nonpolymer_entity_container_identifiers {
        auth_asym_ids
        nonpolymer_comp_id
      }
    }
  }
}
""".strip()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path) -> dict[str, object]:
    config = read_json(path)
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"discovery config is missing keys: {', '.join(missing)}")
    preregistration = config["preregistration"]
    runtime = config["runtime"]
    rcsb = config["rcsb"]
    outputs = config["outputs"]
    if not isinstance(preregistration, dict) or set(preregistration) != {
        "path",
        "sha256",
    }:
        raise ValueError("preregistration must contain path and sha256")
    if not isinstance(runtime, dict) or set(runtime) != {
        "conda_environment",
        "python_version",
    }:
        raise ValueError("runtime lock is incomplete")
    if not isinstance(rcsb, dict):
        raise ValueError("rcsb request configuration is missing")
    for key in (
        "search_endpoint",
        "graphql_endpoint",
        "metadata_chunk_size",
        "request_timeout_seconds",
        "maximum_retries",
        "retry_backoff_seconds",
    ):
        if key not in rcsb:
            raise ValueError(f"rcsb configuration is missing: {key}")
    if int(rcsb["metadata_chunk_size"]) < 1:
        raise ValueError("metadata_chunk_size must be positive")
    if int(rcsb["maximum_retries"]) < 1:
        raise ValueError("maximum_retries must be positive")
    if not isinstance(outputs, dict) or set(outputs) != REQUIRED_OUTPUT_KEYS:
        raise ValueError("discovery outputs do not match the required set")
    return config


def check_runtime(config: dict[str, object]) -> dict[str, str]:
    expected = config["runtime"]
    assert isinstance(expected, dict)
    actual = {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": platform.python_version(),
    }
    if actual != {key: str(value) for key, value in expected.items()}:
        raise RuntimeError(f"runtime differs: {actual} != {expected}")
    return actual


def validate_preregistration(preregistration: dict[str, object]) -> None:
    if preregistration.get("target_id") != "MK14":
        raise ValueError("preregistration target must be MK14")
    boundary = preregistration.get("data_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("data boundary is missing")
    if (
        boundary.get("test") != "locked_unreleased"
        or boundary.get("labels_allowed_during_structural_selection") is not False
        or boundary.get("docking_scores_allowed_during_structural_selection")
        is not False
    ):
        raise ValueError("structural discovery boundary changed")
    discovery = preregistration.get("rcsb_discovery")
    if not isinstance(discovery, dict):
        raise ValueError("rcsb discovery rules are missing")
    if (
        discovery.get("uniprot_accession") != "Q16539"
        or discovery.get("experimental_method") != "X-RAY DIFFRACTION"
    ):
        raise ValueError("RCSB target query changed")
    eligibility = discovery.get("metadata_eligibility")
    if not isinstance(eligibility, dict):
        raise ValueError("metadata eligibility rules are missing")
    if int(eligibility.get("required_mutation_count", -1)) != 0:
        raise ValueError("metadata discovery must require wild-type entries")
    future = preregistration.get("future_validation")
    if (
        not isinstance(future, dict)
        or future.get("status") != "not_authorized"
        or future.get("reuse_previous_validation_for_promotion") is not False
    ):
        raise ValueError("future validation boundary changed")


def search_payload(accession: str, method: str) -> dict[str, object]:
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": (
                            "rcsb_polymer_entity_container_identifiers."
                            "reference_sequence_identifiers.database_accession"
                        ),
                        "operator": "exact_match",
                        "value": accession,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": method,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 1000},
            "sort": [
                {
                    "sort_by": "rcsb_accession_info.initial_release_date",
                    "direction": "desc",
                }
            ],
        },
    }


def request_json(
    endpoint: str,
    payload: dict[str, object],
    timeout_seconds: float,
    maximum_retries: int,
    retry_backoff_seconds: float,
) -> dict[str, object]:
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=encoded,
        headers={"Content-Type": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(maximum_retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                value = json.load(response)
            if not isinstance(value, dict):
                raise ValueError("RCSB response root is not an object")
            return value
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 < maximum_retries:
                time.sleep(retry_backoff_seconds * (2**attempt))
    assert last_error is not None
    raise RuntimeError(f"RCSB request failed after retries: {last_error}")


def matching_reference(
    entity: dict[str, object], accession: str
) -> dict[str, object] | None:
    identifiers = entity.get("rcsb_polymer_entity_container_identifiers")
    if not isinstance(identifiers, dict):
        return None
    references = identifiers.get("reference_sequence_identifiers") or []
    matches = [
        reference
        for reference in references
        if isinstance(reference, dict)
        and reference.get("database_name") == "UniProt"
        and reference.get("database_accession") == accession
    ]
    if not matches:
        return None
    return max(
        matches,
        key=lambda value: (
            float(value.get("reference_sequence_coverage") or 0.0),
            float(value.get("entity_sequence_coverage") or 0.0),
        ),
    )


def normalize_entry(
    entry: dict[str, object],
    accession: str,
    eligibility: dict[str, object],
    existing_structure_ids: set[str],
) -> dict[str, object]:
    pdb_id = str(entry.get("rcsb_id", "")).upper()
    polymer_entities = entry.get("polymer_entities") or []
    target_entities: list[tuple[dict[str, object], dict[str, object]]] = []
    for entity in polymer_entities:
        if not isinstance(entity, dict):
            continue
        reference = matching_reference(entity, accession)
        if reference is not None:
            target_entities.append((entity, reference))
    target_entities.sort(
        key=lambda item: (
            -float(item[1].get("reference_sequence_coverage") or 0.0),
            -float(item[1].get("entity_sequence_coverage") or 0.0),
            str(item[0].get("rcsb_id", "")),
        )
    )
    target_entity = target_entities[0][0] if target_entities else {}
    reference = target_entities[0][1] if target_entities else {}
    entity_poly = target_entity.get("entity_poly") or {}
    polymer = target_entity.get("rcsb_polymer_entity") or {}
    identifiers = target_entity.get("rcsb_polymer_entity_container_identifiers") or {}
    target_chains = sorted(str(value) for value in identifiers.get("auth_asym_ids") or [])

    resolution_values = (entry.get("rcsb_entry_info") or {}).get(
        "resolution_combined"
    ) or []
    resolution = min(float(value) for value in resolution_values) if resolution_values else math.nan
    mutation_count = int(entity_poly.get("rcsb_mutation_count") or 0)
    sample_length = int(entity_poly.get("rcsb_sample_sequence_length") or 0)
    reference_coverage = float(reference.get("reference_sequence_coverage") or 0.0)
    entity_coverage = float(reference.get("entity_sequence_coverage") or 0.0)

    excluded = {
        str(value).upper() for value in eligibility["excluded_nonpolymer_ids"]
    }
    minimum_weight = float(eligibility["minimum_nonpolymer_formula_weight_kda"])
    maximum_weight = float(eligibility["maximum_nonpolymer_formula_weight_kda"])
    qualifying_ligands: list[dict[str, object]] = []
    for ligand in entry.get("nonpolymer_entities") or []:
        if not isinstance(ligand, dict):
            continue
        component = ligand.get("pdbx_entity_nonpoly") or {}
        ligand_ids = ligand.get("rcsb_nonpolymer_entity_container_identifiers") or {}
        details = ligand.get("rcsb_nonpolymer_entity") or {}
        comp_id = str(
            component.get("comp_id") or ligand_ids.get("nonpolymer_comp_id") or ""
        ).upper()
        weight_value = details.get("formula_weight")
        weight = float(weight_value) if weight_value is not None else math.nan
        auth_chains = sorted(str(value) for value in ligand_ids.get("auth_asym_ids") or [])
        same_chains = sorted(set(auth_chains) & set(target_chains))
        if (
            comp_id
            and comp_id not in excluded
            and math.isfinite(weight)
            and minimum_weight <= weight <= maximum_weight
            and same_chains
        ):
            qualifying_ligands.append(
                {
                    "comp_id": comp_id,
                    "name": str(component.get("name") or ""),
                    "formula_weight_kda": weight,
                    "target_chains": same_chains,
                }
            )
    qualifying_ligands.sort(key=lambda value: (value["comp_id"], value["name"]))
    ligand_chains = sorted(
        {
            chain
            for ligand in qualifying_ligands
            for chain in ligand["target_chains"]
        }
    )
    selected_chain = ligand_chains[0] if ligand_chains else (target_chains[0] if target_chains else "")

    reasons: list[str] = []
    if not target_entities:
        reasons.append("missing_target_uniprot_entity")
    if not math.isfinite(resolution) or resolution > float(
        eligibility["maximum_resolution_angstrom"]
    ):
        reasons.append("resolution_above_limit")
    if mutation_count != int(eligibility["required_mutation_count"]):
        reasons.append("mutation_count_differs")
    if reference_coverage < float(eligibility["minimum_reference_sequence_coverage"]):
        reasons.append("reference_sequence_coverage_below_limit")
    if not (
        int(eligibility["minimum_sample_sequence_length"])
        <= sample_length
        <= int(eligibility["maximum_sample_sequence_length"])
    ):
        reasons.append("sample_sequence_length_outside_limits")
    if eligibility["require_same_chain_drug_like_nonpolymer"] and not qualifying_ligands:
        reasons.append("missing_same_chain_drug_like_nonpolymer")

    title = entry.get("struct") or {}
    accession_info = entry.get("rcsb_accession_info") or {}
    return {
        "pdb_id": pdb_id,
        "status": "metadata_eligible" if not reasons else "metadata_excluded",
        "exclusion_reasons": ";".join(reasons),
        "is_existing_receptor_structure": pdb_id in existing_structure_ids,
        "title": str(title.get("title") or ""),
        "initial_release_date": str(accession_info.get("initial_release_date") or ""),
        "resolution_angstrom": resolution,
        "target_polymer_entity_id": str(target_entity.get("rcsb_id") or ""),
        "target_auth_chains": ";".join(target_chains),
        "selected_auth_chain": selected_chain,
        "mutation_count": mutation_count,
        "pdbx_mutation_note": str(polymer.get("pdbx_mutation") or ""),
        "sample_sequence_length": sample_length,
        "reference_sequence_coverage": reference_coverage,
        "entity_sequence_coverage": entity_coverage,
        "qualifying_ligand_count": len(qualifying_ligands),
        "qualifying_ligand_ids": ";".join(
            str(value["comp_id"]) for value in qualifying_ligands
        ),
        "qualifying_ligand_weights_kda": ";".join(
            f"{float(value['formula_weight_kda']):.3f}"
            for value in qualifying_ligands
        ),
        "qualifying_ligand_names": ";".join(
            str(value["name"]) for value in qualifying_ligands
        ),
    }


def run_discovery(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = load_config(config_path)
    runtime = check_runtime(config)
    preregistration_record = config["preregistration"]
    assert isinstance(preregistration_record, dict)
    preregistration_path = Path(str(preregistration_record["path"]))
    if not preregistration_path.is_file():
        raise FileNotFoundError(preregistration_path)
    if file_sha256(preregistration_path) != str(
        preregistration_record["sha256"]
    ).upper():
        raise ValueError("preregistration SHA-256 differs")
    preregistration = read_json(preregistration_path)
    validate_preregistration(preregistration)
    upstream = preregistration["upstream_failed_gate"]
    assert isinstance(upstream, dict)
    upstream_path = Path(str(upstream["summary_path"]))
    if file_sha256(upstream_path) != str(upstream["summary_sha256"]).upper():
        raise ValueError("upstream method-gate summary SHA-256 differs")
    upstream_summary = read_json(upstream_path)
    if upstream_summary.get("status") != upstream["required_status"]:
        raise ValueError("upstream method gate does not have the required failure status")

    outputs = config["outputs"]
    assert isinstance(outputs, dict)
    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    existing_outputs = [path for path in output_paths.values() if path.exists()]
    if existing_outputs and not overwrite:
        raise FileExistsError("discovery outputs exist; review before overwrite")
    if overwrite:
        for path in existing_outputs:
            path.unlink()

    discovery = preregistration["rcsb_discovery"]
    assert isinstance(discovery, dict)
    eligibility = discovery["metadata_eligibility"]
    assert isinstance(eligibility, dict)
    rcsb = config["rcsb"]
    assert isinstance(rcsb, dict)
    accession = str(discovery["uniprot_accession"])
    method = str(discovery["experimental_method"])
    payload = search_payload(accession, method)
    request_kwargs = {
        "timeout_seconds": float(rcsb["request_timeout_seconds"]),
        "maximum_retries": int(rcsb["maximum_retries"]),
        "retry_backoff_seconds": float(rcsb["retry_backoff_seconds"]),
    }
    search_response = request_json(
        str(rcsb["search_endpoint"]), payload, **request_kwargs
    )
    identifiers = sorted(
        str(value["identifier"]).upper()
        for value in search_response.get("result_set") or []
        if isinstance(value, dict) and value.get("identifier")
    )
    if not identifiers or len(identifiers) != int(search_response.get("total_count", -1)):
        raise ValueError("RCSB search result identifiers are incomplete")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("RCSB search returned duplicate entry IDs")

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
            raise ValueError(f"RCSB GraphQL returned errors: {response['errors']}")
        data = response.get("data")
        if not isinstance(data, dict):
            raise ValueError("RCSB GraphQL data object is missing")
        chunk_entries = data.get("entries") or []
        entries.extend(value for value in chunk_entries if isinstance(value, dict))
    entries.sort(key=lambda value: str(value.get("rcsb_id", "")))
    if {str(value.get("rcsb_id", "")).upper() for value in entries} != set(
        identifiers
    ):
        raise ValueError("RCSB metadata entry IDs differ from search IDs")

    existing_structures = {
        value.split("_")[1]
        for value in preregistration["pool_expansion"]["existing_receptor_ids"]
    }
    rows = [
        normalize_entry(value, accession, eligibility, existing_structures)
        for value in entries
    ]
    rows.sort(key=lambda value: str(value["pdb_id"]))
    eligible = [value for value in rows if value["status"] == "metadata_eligible"]
    eligible_new = [
        value for value in eligible if not value["is_existing_receptor_structure"]
    ]
    minimum_new = int(
        preregistration["pool_expansion"]["minimum_eligible_new_candidate_count"]
    )
    if len(eligible_new) < minimum_new:
        raise ValueError("too few new metadata-eligible candidates")

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
    write_json(output_paths["search_snapshot_json"], search_snapshot)
    write_json(output_paths["metadata_snapshot_json"], metadata_snapshot)
    write_csv(output_paths["candidate_metadata_csv"], rows)
    write_csv(output_paths["eligible_new_candidates_csv"], eligible_new)

    reason_counts = Counter(
        reason
        for row in rows
        for reason in str(row["exclusion_reasons"]).split(";")
        if reason
    )
    resolutions = [float(value["resolution_angstrom"]) for value in eligible]
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "metadata_discovery_ok",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "preregistration": {
            "path": preregistration_path.as_posix(),
            "sha256": file_sha256(preregistration_path),
        },
        "runtime": runtime,
        "retrieved_on": config["retrieved_on"],
        "query": {
            "uniprot_accession": accession,
            "experimental_method": method,
            "search_result_count": len(identifiers),
        },
        "counts": {
            "metadata_entry_count": len(entries),
            "metadata_eligible_count": len(eligible),
            "existing_eligible_count": sum(
                bool(value["is_existing_receptor_structure"]) for value in eligible
            ),
            "new_metadata_eligible_count": len(eligible_new),
            "metadata_excluded_count": len(rows) - len(eligible),
        },
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "eligible_resolution_angstrom": {
            "minimum": min(resolutions),
            "median": statistics.median(resolutions),
            "maximum": max(resolutions),
        },
        "eligible_new_pdb_ids": [str(value["pdb_id"]) for value in eligible_new],
        "data_boundary": {
            "ligand_labels_read": 0,
            "docking_scores_read": 0,
            "previous_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "outputs": {
            key: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for key, path in output_paths.items()
            if key != "summary_json"
        },
        "next_gate": (
            "coordinate download, chain audit, proper-rotation alignment, "
            "pocket completeness, pocket-proximal ligand check, and structural "
            "max-min selection"
        ),
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(output_paths["summary_json"], summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_discovery(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
