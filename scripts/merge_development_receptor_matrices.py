"""Merge MD and non-MD development score matrices into one receptor pool."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

try:
    from .align_receptor_structure import file_sha256
except ImportError:
    from align_receptor_structure import file_sha256


METADATA = ("target_id", "ligand_id", "label")
INPUT_KEYS = (
    "development_ligand_manifest",
    "expanded_receptor_manifest",
    "md_primary_matrix",
    "md_sensitivity_matrix",
    "md_warning_table",
    "non_md_primary_matrix",
    "non_md_sensitivity_matrix",
    "non_md_warning_table",
)
OUTPUT_KEYS = (
    "primary_matrix_csv",
    "sensitivity_matrix_csv",
    "warning_table_csv",
    "summary_json",
)
SHA256_PATTERN = re.compile(r"^[0-9A-Fa-f]{64}$")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or not rows:
        raise ValueError(f"CSV is empty: {path}")
    return fields, rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    required = {
        "schema_version",
        "experiment_id",
        "purpose",
        "inputs",
        "input_sha256",
        "expected",
        "outputs",
        "interpretation_boundary",
    }
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("matrix merge configuration is incomplete")
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    outputs = config["outputs"]
    if not isinstance(inputs, dict) or not set(INPUT_KEYS).issubset(inputs):
        raise ValueError("matrix merge inputs are incomplete")
    if not isinstance(hashes, dict) or not set(INPUT_KEYS).issubset(hashes):
        raise ValueError("matrix merge hashes are incomplete")
    if any(not SHA256_PATTERN.fullmatch(str(hashes[key])) for key in INPUT_KEYS):
        raise ValueError("matrix merge input SHA-256 is invalid")
    if not isinstance(outputs, dict) or not set(OUTPUT_KEYS).issubset(outputs):
        raise ValueError("matrix merge outputs are incomplete")
    return config


def verify_file(path: Path, expected_hash: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != expected_hash.upper():
        raise ValueError(f"input SHA-256 differs: {path}")


def matrix_lookup(
    path: Path,
    expected_ligands: dict[str, str],
) -> tuple[list[str], dict[str, dict[str, str]]]:
    fields, rows = read_csv(path)
    if not set(METADATA).issubset(fields):
        raise ValueError(f"matrix metadata columns are incomplete: {path}")
    receptors = [field for field in fields if field not in METADATA]
    lookup = {row["ligand_id"]: row for row in rows}
    if len(lookup) != len(rows) or set(lookup) != set(expected_ligands):
        raise ValueError(f"matrix ligand IDs differ: {path}")
    for ligand_id, row in lookup.items():
        if row["label"] != expected_ligands[ligand_id]:
            raise ValueError(f"matrix label differs: {ligand_id}")
        if any(not math.isfinite(float(row[receptor])) for receptor in receptors):
            raise ValueError(f"matrix contains a non-finite score: {ligand_id}")
    return receptors, lookup


def merge_matrix_groups(
    receptor_order: list[str],
    first: dict[str, dict[str, str]],
    second: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    if set(first) != set(second):
        raise ValueError("matrix groups contain different ligand IDs")
    output: list[dict[str, object]] = []
    for ligand_id in sorted(first):
        if first[ligand_id]["label"] != second[ligand_id]["label"]:
            raise ValueError(f"matrix group label differs: {ligand_id}")
        merged: dict[str, object] = {
            "target_id": first[ligand_id]["target_id"],
            "ligand_id": ligand_id,
            "label": first[ligand_id]["label"],
        }
        for receptor in receptor_order:
            source = first if receptor in first[ligand_id] else second
            if receptor not in source[ligand_id]:
                raise ValueError(f"missing receptor column: {receptor}")
            merged[receptor] = float(source[ligand_id][receptor])
        output.append(merged)
    return output


def merge_warning_rows(
    md_rows: list[dict[str, str]],
    non_md_rows: list[dict[str, str]],
    development_ids: set[str],
    md_receptors: set[str],
    non_md_receptors: set[str],
) -> tuple[list[dict[str, object]], int]:
    md_development = [row for row in md_rows if row["ligand_id"] in development_ids]
    skipped = len(md_rows) - len(md_development)
    if any(row["ligand_id"] not in development_ids for row in non_md_rows):
        raise ValueError("non-MD warning table contains a locked ligand")
    output: list[dict[str, object]] = []
    for group, rows, receptors in (
        ("md", md_development, md_receptors),
        ("non_md", non_md_rows, non_md_receptors),
    ):
        for row in rows:
            if row["receptor_id"] not in receptors:
                raise ValueError(f"warning receptor differs from {group} pool")
            output.append({**row, "source_group": group})
    keys = [(row["ligand_id"], row["receptor_id"]) for row in output]
    if len(keys) != len(set(keys)):
        raise ValueError("merged warning table contains duplicate pairs")
    output.sort(key=lambda row: (str(row["ligand_id"]), str(row["receptor_id"])))
    return output, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    expected = config["expected"]
    outputs = config["outputs"]
    assert isinstance(inputs, dict)
    assert isinstance(hashes, dict)
    assert isinstance(expected, dict)
    assert isinstance(outputs, dict)
    input_paths = {key: Path(str(inputs[key])) for key in INPUT_KEYS}
    for key, path in input_paths.items():
        verify_file(path, str(hashes[key]))
    output_paths = {key: Path(str(outputs[key])) for key in OUTPUT_KEYS}
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("matrix merge outputs exist; use --overwrite")
    if args.overwrite:
        for path in existing:
            path.unlink()

    _, ligand_rows = read_csv(input_paths["development_ligand_manifest"])
    ligand_labels = {row["ligand_id"]: row["label"] for row in ligand_rows}
    if len(ligand_labels) != int(expected["development_ligand_count"]):
        raise ValueError("development ligand count differs")
    if any(row.get("benchmark_split") == "test" for row in ligand_rows):
        raise ValueError("development manifest contains a locked ligand")

    _, receptor_rows = read_csv(input_paths["expanded_receptor_manifest"])
    receptor_order = [row["conformer_id"] for row in receptor_rows]
    md_receptors = {
        row["conformer_id"] for row in receptor_rows if row["source_type"] == "md_cluster_medoid"
    }
    non_md_receptors = set(receptor_order) - md_receptors
    if len(receptor_order) != int(expected["expanded_receptor_count"]):
        raise ValueError("expanded receptor count differs")
    if len(md_receptors) != int(expected["md_receptor_count"]) or len(non_md_receptors) != int(
        expected["non_md_receptor_count"]
    ):
        raise ValueError("receptor source counts differ")

    md_primary_receptors, md_primary = matrix_lookup(
        input_paths["md_primary_matrix"], ligand_labels
    )
    md_sensitivity_receptors, md_sensitivity = matrix_lookup(
        input_paths["md_sensitivity_matrix"], ligand_labels
    )
    non_md_primary_receptors, non_md_primary = matrix_lookup(
        input_paths["non_md_primary_matrix"], ligand_labels
    )
    non_md_sensitivity_receptors, non_md_sensitivity = matrix_lookup(
        input_paths["non_md_sensitivity_matrix"], ligand_labels
    )
    if set(md_primary_receptors) != md_receptors or md_primary_receptors != md_sensitivity_receptors:
        raise ValueError("MD matrix receptor columns differ")
    if set(non_md_primary_receptors) != non_md_receptors or non_md_primary_receptors != non_md_sensitivity_receptors:
        raise ValueError("non-MD matrix receptor columns differ")

    primary_rows = merge_matrix_groups(receptor_order, md_primary, non_md_primary)
    sensitivity_rows = merge_matrix_groups(receptor_order, md_sensitivity, non_md_sensitivity)
    _, md_warning_rows = read_csv(input_paths["md_warning_table"])
    warning_fields, non_md_warning_rows = read_csv(input_paths["non_md_warning_table"])
    warning_rows, skipped_warnings = merge_warning_rows(
        md_warning_rows,
        non_md_warning_rows,
        set(ligand_labels),
        md_receptors,
        non_md_receptors,
    )
    if len(warning_rows) != int(expected["combined_development_warning_count"]):
        raise ValueError("combined warning count differs")
    if skipped_warnings != int(expected["md_locked_warning_rows_skipped"]):
        raise ValueError("skipped locked warning count differs")

    matrix_fields = [*METADATA, *receptor_order]
    write_csv(output_paths["primary_matrix_csv"], matrix_fields, primary_rows)
    write_csv(output_paths["sensitivity_matrix_csv"], matrix_fields, sensitivity_rows)
    write_csv(output_paths["warning_table_csv"], [*warning_fields, "source_group"], warning_rows)
    output_hashes = {
        key: file_sha256(output_paths[key])
        for key in ("primary_matrix_csv", "sensitivity_matrix_csv", "warning_table_csv")
    }
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok_with_seed_stability_warning" if warning_rows else "ok",
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "input_sha256": {key: file_sha256(path) for key, path in sorted(input_paths.items())},
        "ligand_count": len(ligand_labels),
        "label_counts": {
            label: sum(value == label for value in ligand_labels.values())
            for label in sorted(set(ligand_labels.values()))
        },
        "receptor_count": len(receptor_order),
        "receptor_ids": receptor_order,
        "receptor_ligand_pair_count": len(ligand_labels) * len(receptor_order),
        "seed_stability_warning_count": len(warning_rows),
        "warning_counts": {
            "md_development": sum(row["source_group"] == "md" for row in warning_rows),
            "non_md_development": sum(row["source_group"] == "non_md" for row in warning_rows),
            "md_locked_rows_skipped": skipped_warnings,
        },
        "locked_test_score_rows": 0,
        "outputs": {key: output_paths[key].as_posix() for key in OUTPUT_KEYS},
        "output_sha256": output_hashes,
        "interpretation_boundary": config["interpretation_boundary"],
    }
    output_paths["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    with output_paths["summary_json"].open("w", encoding="ascii", newline="\n") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
