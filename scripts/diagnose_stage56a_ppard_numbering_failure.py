"""Diagnose the Stage 56 PPARD author-numbering coordinate-gate failure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import gemmi
from Bio.Align import PairwiseAligner


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verified(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage 56a input identity differs: {path}")
    return path


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def polymer_residues(
    structure: gemmi.Structure, chain_name: str
) -> list[tuple[gemmi.Residue, str]]:
    chains = [chain for chain in structure[0] if chain.name == chain_name]
    if len(chains) != 1:
        raise ValueError(f"author chain is absent or ambiguous: {chain_name}")
    output: list[tuple[gemmi.Residue, str]] = []
    for residue in chains[0]:
        if residue.entity_type != gemmi.EntityType.Polymer:
            continue
        code = gemmi.find_tabulated_residue(residue.name).one_letter_code
        if code and code != " ":
            output.append((residue, code))
    return output


def sequence_mapping(
    reference: list[tuple[gemmi.Residue, str]],
    candidate: list[tuple[gemmi.Residue, str]],
) -> tuple[dict[tuple[int, str], tuple[int, str]], dict[str, Any]]:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(
        "".join(value[1] for value in reference),
        "".join(value[1] for value in candidate),
    )[0]
    mapping: dict[tuple[int, str], tuple[int, str]] = {}
    identities = 0
    offsets: list[int] = []
    for (ref_start, ref_end), (candidate_start, candidate_end) in zip(
        alignment.aligned[0], alignment.aligned[1], strict=True
    ):
        for ref_index, candidate_index in zip(
            range(ref_start, ref_end), range(candidate_start, candidate_end), strict=True
        ):
            ref_residue, ref_code = reference[ref_index]
            candidate_residue, candidate_code = candidate[candidate_index]
            candidate_key = (
                int(candidate_residue.seqid.num),
                str(candidate_residue.seqid.icode).strip(),
            )
            reference_key = (
                int(ref_residue.seqid.num),
                str(ref_residue.seqid.icode).strip(),
            )
            mapping[candidate_key] = reference_key
            identities += int(ref_code == candidate_code)
            offsets.append(candidate_key[0] - reference_key[0])
    offset_counts = Counter(offsets)
    dominant_offset, dominant_count = offset_counts.most_common(1)[0]
    metrics = {
        "reference_residue_count": len(reference),
        "candidate_residue_count": len(candidate),
        "sequence_mapped_residue_count": len(mapping),
        "sequence_identity_fraction": identities / len(mapping),
        "dominant_author_number_offset": dominant_offset,
        "dominant_author_number_offset_fraction": dominant_count / len(mapping),
    }
    return mapping, metrics


def run(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    failed = read_json(inputs["failed_summary"])
    if failed["status"] != "stage56_ppard_coordinate_pool_insufficient_stop":
        raise ValueError("Stage56 source result is not the frozen coordinate failure")
    failed_rows = read_csv(inputs["failed_coordinate_audit"])
    failed_by_pdb = {row["pdb_id"]: row for row in failed_rows}
    metadata = [
        row
        for row in read_csv(inputs["candidate_metadata_csv"])
        if row["target_id"] == "PPARD" and row["status"] == "metadata_eligible"
    ]
    reference_structure = gemmi.read_structure(str(inputs["reference_mmcif"]))
    reference = polymer_residues(reference_structure, "A")
    rows: list[dict[str, Any]] = []
    for candidate in sorted(metadata, key=lambda row: row["pdb_id"]):
        pdb_id = candidate["pdb_id"]
        path = root / str(config["raw_mmcif_path_template"]).format(pdb_id=pdb_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        structure = gemmi.read_structure(str(path))
        _mapping, metrics = sequence_mapping(
            reference,
            polymer_residues(structure, candidate["selected_auth_chain"]),
        )
        original = failed_by_pdb[pdb_id]
        rows.append(
            {
                "pdb_id": pdb_id,
                "chain": candidate["selected_auth_chain"],
                "original_status": original["status"],
                "original_matched_ca_count": int(original["matched_ca_count"]),
                "original_exclusion_reasons": original["exclusion_reasons"],
                **metrics,
                "sequence_mapping_pass": (
                    int(metrics["sequence_mapped_residue_count"])
                    >= int(config["adjudication_gate"]["minimum_sequence_mapped_residue_count"])
                    and float(metrics["sequence_identity_fraction"])
                    >= float(config["adjudication_gate"]["minimum_sequence_identity_fraction"])
                ),
            }
        )
    original_failures = [row for row in rows if row["original_status"] != "coordinate_eligible"]
    systematic = all(
        int(row["original_matched_ca_count"])
        <= int(config["adjudication_gate"]["maximum_original_matched_ca_for_numbering_failure"])
        and int(row["dominant_author_number_offset"])
        == int(config["adjudication_gate"]["expected_failure_mode_offset"])
        and bool(row["sequence_mapping_pass"])
        for row in original_failures
    )
    passing_mapping_count = sum(bool(row["sequence_mapping_pass"]) for row in rows)
    amendment_authorized = (
        systematic
        and passing_mapping_count
        >= int(config["adjudication_gate"]["minimum_sequence_mappable_candidate_count"])
    )
    output_csv = root / config["outputs"]["diagnostic_csv"]
    write_csv(output_csv, rows)
    result = {
        "schema_version": "1.0",
        "status": "stage56a_ppard_author_numbering_failure_adjudicated",
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, implementation),
        "counts": {
            "candidate_count": len(rows),
            "original_coordinate_eligible_count": sum(
                row["original_status"] == "coordinate_eligible" for row in rows
            ),
            "original_coordinate_failure_count": len(original_failures),
            "sequence_mapping_pass_count": passing_mapping_count,
            "systematic_minus36_failure_count": sum(
                int(row["dominant_author_number_offset"]) == -36
                for row in original_failures
            ),
        },
        "diagnosis": {
            "systematic_author_numbering_failure_confirmed": systematic,
            "minimum_sequence_identity_fraction": min(
                float(row["sequence_identity_fraction"]) for row in rows
            ),
            "median_sequence_identity_fraction": statistics.median(
                float(row["sequence_identity_fraction"]) for row in rows
            ),
            "minimum_sequence_mapped_residue_count": min(
                int(row["sequence_mapped_residue_count"]) for row in rows
            ),
            "biological_coordinate_gate_failure_established": False,
        },
        "decision": {
            "sequence_correspondence_amendment_authorized": amendment_authorized,
            "threshold_lowering_authorized": False,
            "docking_authorized": False,
            "fresh_validation_authorized": False,
            "quantum_hardware_authorized": False,
        },
        "amendment_contract": config["amendment_contract"],
        "data_boundary": {
            "docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
        },
        "outputs": {"diagnostic_csv": descriptor(root, output_csv)},
        "interpretation_boundary": config["interpretation_boundary"],
    }
    output_json = root / config["outputs"]["result_json"]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    run(config_path, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
