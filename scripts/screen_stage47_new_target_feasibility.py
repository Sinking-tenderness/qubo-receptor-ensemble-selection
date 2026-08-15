"""Screen frozen DUD-E targets for a sufficiently large unseen receptor pool."""

from __future__ import annotations

import argparse
import json
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


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    if file_sha256(Path(__file__)) != config["implementation"]["sha256"]:
        raise ValueError("Stage47 implementation identity differs")
    closure_path = root / config["src_closure"]["path"]
    if file_sha256(closure_path) != config["src_closure"]["sha256"]:
        raise ValueError("Stage47 SRC closure identity differs")
    closure = read_json(closure_path)
    if closure.get("status") != "stage46b_src_human_metadata_pool_closed":
        raise ValueError("SRC branch was not formally closed")
    outputs = {key: root / value for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage47 outputs exist; pass --overwrite")

    rcsb = config["rcsb"]
    kwargs = {
        "timeout_seconds": float(rcsb["request_timeout_seconds"]),
        "maximum_retries": int(rcsb["maximum_retries"]),
        "retry_backoff_seconds": float(rcsb["retry_backoff_seconds"]),
    }
    rules = config["metadata_eligibility"]
    screen_rows: list[dict[str, Any]] = []
    snapshots: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for target in config["candidate_targets"]:
        target_id = target["target_id"]
        accession = target["uniprot_accession"]
        payload = search_payload(accession, config["experimental_method"])
        search = request_json(rcsb["search_endpoint"], payload, **kwargs)
        identifiers = sorted(
            row["identifier"].upper()
            for row in search.get("result_set", [])
            if isinstance(row, dict) and row.get("identifier")
        )
        entries: list[dict[str, Any]] = []
        for start in range(0, len(identifiers), int(rcsb["metadata_chunk_size"])):
            response = request_json(
                rcsb["graphql_endpoint"],
                {"query": GRAPHQL_QUERY, "variables": {"ids": identifiers[start : start + int(rcsb["metadata_chunk_size"])]}},
                **kwargs,
            )
            if response.get("errors"):
                raise ValueError(f"Stage47 RCSB GraphQL errors for {target_id}: {response['errors']}")
            entries.extend(response["data"]["entries"] or [])
        rows = [
            apply_title_exclusions(
                normalize_entry(entry, accession, rules, set()),
                rules["excluded_title_patterns"],
            )
            for entry in entries
        ]
        eligible = [row for row in rows if row["status"] == "metadata_eligible"]
        reasons = Counter(
            reason
            for row in rows
            for reason in row["exclusion_reasons"].split(";")
            if reason
        )
        eligible_for_selection = (
            len(eligible) >= int(config["selection_gate"]["minimum_metadata_eligible_count"])
            and int(target["dude_clustered_active_count"]) >= int(config["selection_gate"]["minimum_dude_clustered_active_count"])
        )
        screen_rows.append(
            {
                **target,
                "xray_search_count": len(identifiers),
                "metadata_eligible_count": len(eligible),
                "metadata_excluded_count": len(rows) - len(eligible),
                "eligible_for_selection": eligible_for_selection,
                "eligible_pdb_ids": ";".join(sorted(row["pdb_id"] for row in eligible)),
                "exclusion_reason_counts_json": json.dumps(dict(sorted(reasons.items())), sort_keys=True),
            }
        )
        for row in rows:
            all_rows.append({"target_id": target_id, **row})
        snapshots[target_id] = {
            "accession": accession,
            "search_request": payload,
            "identifiers": identifiers,
            "entries": entries,
        }

    eligible_targets = [row for row in screen_rows if row["eligible_for_selection"]]
    selected = max(
        eligible_targets,
        key=lambda row: (
            int(row["metadata_eligible_count"]),
            int(row["dude_clustered_active_count"]),
            row["target_id"],
        ),
        default=None,
    )
    write_csv(outputs["screen_csv"], screen_rows)
    write_csv(outputs["candidate_metadata_csv"], all_rows)
    write_json(outputs["snapshot_json"], {"schema_version": "1.0", "retrieved_on": config["retrieved_on"], "targets": snapshots})
    result = {
        "schema_version": "1.0",
        "status": "stage47_new_target_feasibility_screen_complete",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "candidate_count": len(screen_rows),
        "eligible_target_count": len(eligible_targets),
        "selected_target": selected,
        "decision": {
            "new_target_source_intake_authorized": selected is not None,
            "production_docking_authorized": False,
            "fresh_validation_authorized": False,
            "quantum_hardware_authorized": False,
        },
        "data_boundary": {
            "individual_ligand_rows_read": 0,
            "docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
        },
        "outputs": {
            key: {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)}
            for key, path in outputs.items()
            if key != "result_json"
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


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
