"""Subset existing score matrices to a hash-pinned development ligand set."""

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


METADATA_COLUMNS = ("target_id", "ligand_id", "label")
CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "purpose",
    "inputs",
    "input_sha256",
    "expected",
    "outputs",
    "interpretation_boundary",
}
INPUT_KEYS = ("development_ligand_manifest", "primary_matrix", "sensitivity_matrix")
OUTPUT_KEYS = (
    "primary_development_matrix_csv",
    "sensitivity_development_matrix_csv",
    "summary_json",
)
EXPECTED_KEYS = {
    "source_ligand_count",
    "development_ligand_count",
    "skipped_locked_ligand_count",
    "receptor_count",
    "development_label_counts",
}
SHA256_PATTERN = re.compile(r"^[0-9A-Fa-f]{64}$")


def portable_path(value: str) -> Path:
    return Path(value.replace("\\", "/"))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or not rows:
        raise ValueError(f"CSV is empty or lacks a header: {path}")
    return fields, rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    if not fields or not rows:
        raise ValueError(f"cannot write empty matrix: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(config, dict):
        raise ValueError("matrix subset configuration must be a JSON object")
    missing = sorted(CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"matrix subset configuration is missing keys: {', '.join(missing)}")
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    expected = config["expected"]
    outputs = config["outputs"]
    if not isinstance(inputs, dict) or not set(INPUT_KEYS).issubset(inputs):
        raise ValueError("inputs is missing one or more matrix subset paths")
    if not isinstance(hashes, dict) or not set(INPUT_KEYS).issubset(hashes):
        raise ValueError("input_sha256 is missing one or more matrix subset hashes")
    for key in INPUT_KEYS:
        if not SHA256_PATTERN.fullmatch(str(hashes[key])):
            raise ValueError(f"input_sha256.{key} must be a hexadecimal SHA-256")
    if not isinstance(expected, dict) or not EXPECTED_KEYS.issubset(expected):
        raise ValueError("expected is missing one or more matrix subset counts")
    numeric = [key for key in EXPECTED_KEYS if key != "development_label_counts"]
    if any(int(expected[key]) <= 0 for key in numeric):
        raise ValueError("expected matrix subset counts must be positive")
    if int(expected["development_ligand_count"]) + int(
        expected["skipped_locked_ligand_count"]
    ) != int(expected["source_ligand_count"]):
        raise ValueError("expected ligand counts are internally inconsistent")
    labels = expected["development_label_counts"]
    if not isinstance(labels, dict) or not labels or any(int(value) <= 0 for value in labels.values()):
        raise ValueError("development_label_counts must be a non-empty positive count object")
    if sum(int(value) for value in labels.values()) != int(expected["development_ligand_count"]):
        raise ValueError("development_label_counts does not sum to development_ligand_count")
    if not isinstance(outputs, dict) or not set(OUTPUT_KEYS).issubset(outputs):
        raise ValueError("outputs is missing one or more matrix subset paths")
    return config


def verify_file(path: Path, expected_hash: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual != expected_hash.upper():
        raise ValueError(f"{label} SHA-256 differs: expected {expected_hash.upper()}, got {actual}")


def development_metadata(path: Path) -> dict[str, str]:
    _, rows = read_csv(path)
    output: dict[str, str] = {}
    for row in rows:
        ligand_id = row["ligand_id"]
        if ligand_id in output:
            raise ValueError(f"duplicate development ligand ID: {ligand_id}")
        output[ligand_id] = row["label"]
    return output


def subset_matrix_rows(
    fields: list[str],
    rows: list[dict[str, str]],
    development_labels: dict[str, str],
    expected_source_count: int,
    expected_skipped_count: int,
) -> tuple[list[str], list[dict[str, object]], int]:
    if not set(METADATA_COLUMNS).issubset(fields):
        raise ValueError("score matrix is missing one or more metadata columns")
    receptor_columns = [field for field in fields if field not in METADATA_COLUMNS]
    if not receptor_columns:
        raise ValueError("score matrix contains no receptor columns")
    if len(rows) != expected_source_count:
        raise ValueError(f"source matrix row count differs: expected {expected_source_count}, got {len(rows)}")
    seen: set[str] = set()
    output: list[dict[str, object]] = []
    skipped = 0
    for row in rows:
        ligand_id = row["ligand_id"]
        if ligand_id in seen:
            raise ValueError(f"duplicate source matrix ligand ID: {ligand_id}")
        seen.add(ligand_id)
        if ligand_id not in development_labels:
            skipped += 1
            continue
        if row["label"] != development_labels[ligand_id]:
            raise ValueError(f"development label differs in score matrix for {ligand_id}")
        parsed: dict[str, object] = {key: row[key] for key in METADATA_COLUMNS}
        for receptor_id in receptor_columns:
            try:
                score = float(row[receptor_id])
            except ValueError as exc:
                raise ValueError(
                    f"development score is not numeric for {ligand_id}, {receptor_id}"
                ) from exc
            if not math.isfinite(score):
                raise ValueError(f"development score is non-finite for {ligand_id}, {receptor_id}")
            parsed[receptor_id] = score
        output.append(parsed)
    if skipped != expected_skipped_count:
        raise ValueError(f"skipped row count differs: expected {expected_skipped_count}, got {skipped}")
    if {str(row["ligand_id"]) for row in output} != set(development_labels):
        raise ValueError("source matrix does not contain the exact development ligand set")
    output.sort(key=lambda row: str(row["ligand_id"]))
    return receptor_columns, output, skipped


def label_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["label"])
        counts[label] = counts.get(label, 0) + 1
    return counts


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
    input_paths = {key: portable_path(str(inputs[key])) for key in INPUT_KEYS}
    for key, path in input_paths.items():
        verify_file(path, str(hashes[key]), key)
    output_paths = {key: portable_path(str(outputs[key])) for key in OUTPUT_KEYS}
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("matrix subset outputs exist; use --overwrite after review")
    if args.overwrite:
        for path in existing:
            path.unlink()

    development_labels = development_metadata(input_paths["development_ligand_manifest"])
    if len(development_labels) != int(expected["development_ligand_count"]):
        raise ValueError("development manifest ligand count differs from configuration")
    matrix_outputs: dict[str, tuple[list[str], list[dict[str, object]], int]] = {}
    for matrix_key in ("primary_matrix", "sensitivity_matrix"):
        fields, rows = read_csv(input_paths[matrix_key])
        matrix_outputs[matrix_key] = subset_matrix_rows(
            fields,
            rows,
            development_labels,
            int(expected["source_ligand_count"]),
            int(expected["skipped_locked_ligand_count"]),
        )
    primary_receptors, primary_rows, primary_skipped = matrix_outputs["primary_matrix"]
    sensitivity_receptors, sensitivity_rows, sensitivity_skipped = matrix_outputs[
        "sensitivity_matrix"
    ]
    if primary_receptors != sensitivity_receptors:
        raise ValueError("primary and sensitivity receptor columns differ")
    if len(primary_receptors) != int(expected["receptor_count"]):
        raise ValueError("matrix receptor count differs from configuration")
    if [(row["ligand_id"], row["label"]) for row in primary_rows] != [
        (row["ligand_id"], row["label"]) for row in sensitivity_rows
    ]:
        raise ValueError("primary and sensitivity development ligand metadata differs")
    normalized_expected_labels = {
        str(key): int(value) for key, value in dict(expected["development_label_counts"]).items()
    }
    if label_counts(primary_rows) != normalized_expected_labels:
        raise ValueError("development matrix label counts differ from configuration")

    fields = [*METADATA_COLUMNS, *primary_receptors]
    write_csv(output_paths["primary_development_matrix_csv"], fields, primary_rows)
    write_csv(output_paths["sensitivity_development_matrix_csv"], fields, sensitivity_rows)
    output_hashes = {
        key: file_sha256(output_paths[key])
        for key in ("primary_development_matrix_csv", "sensitivity_development_matrix_csv")
    }
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "operation": "ID-only development row subset with locked score cells skipped before numeric parsing",
        "config_path": args.config.as_posix(),
        "config_sha256": file_sha256(args.config),
        "input_sha256": {key: file_sha256(path) for key, path in sorted(input_paths.items())},
        "counts": {
            "source_ligands": int(expected["source_ligand_count"]),
            "development_ligands": len(primary_rows),
            "skipped_locked_ligands": primary_skipped,
            "receptors": len(primary_receptors),
            "development_labels": label_counts(primary_rows),
        },
        "integrity": {
            "primary_and_sensitivity_skipped_counts_match": primary_skipped == sensitivity_skipped,
            "locked_receptor_score_cells_parsed": 0,
            "ranking_metrics_calculated": 0,
            "receptor_selection_performed": False,
        },
        "receptor_ids": primary_receptors,
        "outputs": {key: output_paths[key].as_posix() for key in OUTPUT_KEYS},
        "output_sha256": output_hashes,
        "interpretation_boundary": config["interpretation_boundary"],
    }
    output_paths["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["summary_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
