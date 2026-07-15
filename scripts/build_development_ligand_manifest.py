"""Create a hash-pinned development-only ligand PDBQT manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

try:
    from .align_receptor_structure import file_sha256
except ImportError:
    from align_receptor_structure import file_sha256


REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "purpose",
    "inputs",
    "input_sha256",
    "development_splits",
    "locked_split",
    "expected",
    "outputs",
    "interpretation_boundary",
}
INPUT_KEYS = (
    "full_pdbqt_manifest",
    "scaffold_split_manifest",
    "scaffold_split_summary",
)
OUTPUT_KEYS = ("development_pdbqt_manifest_csv", "summary_json")
EXPECTED_KEYS = {
    "total_ligand_count",
    "development_ligand_count",
    "locked_ligand_count",
    "total_label_counts",
    "development_label_counts",
    "locked_label_counts",
}
SHA256_PATTERN = re.compile(r"^[0-9A-Fa-f]{64}$")


def portable_path(value: str) -> Path:
    return Path(value.replace("\\", "/"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized_counts(value: object, label: str) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty JSON object")
    counts = {str(key): int(count) for key, count in value.items()}
    if any(count <= 0 for count in counts.values()):
        raise ValueError(f"{label} values must be positive")
    return counts


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(config, dict):
        raise ValueError("development ligand configuration must be a JSON object")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"development ligand configuration is missing keys: {', '.join(missing)}")
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    outputs = config["outputs"]
    expected = config["expected"]
    if not isinstance(inputs, dict) or not set(INPUT_KEYS).issubset(inputs):
        raise ValueError("inputs is missing one or more ligand manifest paths")
    if not isinstance(hashes, dict) or not set(INPUT_KEYS).issubset(hashes):
        raise ValueError("input_sha256 is missing one or more ligand manifest hashes")
    for key in INPUT_KEYS:
        if not SHA256_PATTERN.fullmatch(str(hashes[key])):
            raise ValueError(f"input_sha256.{key} must be a hexadecimal SHA-256")
    if not isinstance(outputs, dict) or not set(OUTPUT_KEYS).issubset(outputs):
        raise ValueError("outputs is missing one or more development manifest paths")
    development_splits = config["development_splits"]
    if (
        not isinstance(development_splits, list)
        or not development_splits
        or any(not isinstance(value, str) or not value for value in development_splits)
        or len(set(development_splits)) != len(development_splits)
    ):
        raise ValueError("development_splits must contain unique non-empty strings")
    if not isinstance(config["locked_split"], str) or not config["locked_split"]:
        raise ValueError("locked_split must be a non-empty string")
    if config["locked_split"] in development_splits:
        raise ValueError("locked_split cannot occur in development_splits")
    if not isinstance(expected, dict) or not EXPECTED_KEYS.issubset(expected):
        raise ValueError("expected is missing one or more ligand counts")
    total = int(expected["total_ligand_count"])
    development = int(expected["development_ligand_count"])
    locked = int(expected["locked_ligand_count"])
    if min(total, development, locked) <= 0 or development + locked != total:
        raise ValueError("expected ligand counts must be positive and internally consistent")
    total_labels = normalized_counts(expected["total_label_counts"], "total_label_counts")
    development_labels = normalized_counts(
        expected["development_label_counts"], "development_label_counts"
    )
    locked_labels = normalized_counts(expected["locked_label_counts"], "locked_label_counts")
    if sum(total_labels.values()) != total:
        raise ValueError("total_label_counts does not sum to total_ligand_count")
    if sum(development_labels.values()) != development:
        raise ValueError("development_label_counts does not sum to development_ligand_count")
    if sum(locked_labels.values()) != locked:
        raise ValueError("locked_label_counts does not sum to locked_ligand_count")
    return config


def verify_file(path: Path, expected_hash: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual != expected_hash.upper():
        raise ValueError(f"{label} SHA-256 differs: expected {expected_hash.upper()}, got {actual}")


def count_labels(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["label"])
        counts[label] = counts.get(label, 0) + 1
    return counts


def identifier_set_sha256(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def build_development_rows(
    pdbqt_rows: list[dict[str, str]],
    split_rows: list[dict[str, str]],
    development_splits: set[str],
    locked_split: str,
) -> tuple[list[dict[str, object]], list[str]]:
    pdbqt_by_id = {row["ligand_id"]: row for row in pdbqt_rows}
    split_by_id = {row["ligand_id"]: row for row in split_rows}
    if len(pdbqt_by_id) != len(pdbqt_rows) or len(split_by_id) != len(split_rows):
        raise ValueError("PDBQT or split manifest contains duplicate ligand IDs")
    if set(pdbqt_by_id) != set(split_by_id):
        only_pdbqt = sorted(set(pdbqt_by_id) - set(split_by_id))
        only_split = sorted(set(split_by_id) - set(pdbqt_by_id))
        raise ValueError(
            f"PDBQT and split manifest ligand IDs differ: only_pdbqt={only_pdbqt}, "
            f"only_split={only_split}"
        )
    allowed_splits = {*development_splits, locked_split}
    observed_splits = {row["split"] for row in split_rows}
    if observed_splits != allowed_splits:
        raise ValueError(f"unexpected split values: expected {allowed_splits}, got {observed_splits}")

    development_rows: list[dict[str, object]] = []
    locked_ids: list[str] = []
    for ligand_id in sorted(pdbqt_by_id):
        pdbqt = pdbqt_by_id[ligand_id]
        split = split_by_id[ligand_id]
        if pdbqt.get("label") != split.get("label"):
            raise ValueError(f"label differs between manifests for {ligand_id}")
        if pdbqt.get("canonical_smiles") != split.get("canonical_smiles"):
            raise ValueError(f"canonical SMILES differs between manifests for {ligand_id}")
        split_name = split["split"]
        if split_name == locked_split:
            locked_ids.append(ligand_id)
            continue
        if split_name not in development_splits:
            raise ValueError(f"ligand {ligand_id} has unrecognized split {split_name!r}")
        if pdbqt.get("pdbqt_status") != "ok":
            raise ValueError(f"ligand PDBQT preparation did not pass for {ligand_id}")
        pdbqt_path = portable_path(pdbqt["pdbqt_path"])
        if not pdbqt_path.is_file():
            raise FileNotFoundError(pdbqt_path)
        actual_hash = file_sha256(pdbqt_path)
        recorded_hash = pdbqt.get("pdbqt_sha256", "").strip().upper()
        if recorded_hash and recorded_hash != actual_hash:
            raise ValueError(f"recorded PDBQT SHA-256 differs for {ligand_id}")
        development_rows.append({
            **pdbqt,
            "pdbqt_path": pdbqt_path.as_posix(),
            "pdbqt_sha256": actual_hash,
            "scaffold_smiles": split["scaffold_smiles"],
            "benchmark_split": split_name,
        })
    return development_rows, locked_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = config["inputs"]
    input_hashes = config["input_sha256"]
    outputs = config["outputs"]
    expected = config["expected"]
    assert isinstance(inputs, dict)
    assert isinstance(input_hashes, dict)
    assert isinstance(outputs, dict)
    assert isinstance(expected, dict)

    input_paths = {key: portable_path(str(inputs[key])) for key in INPUT_KEYS}
    for key, path in input_paths.items():
        verify_file(path, str(input_hashes[key]), key)
    output_paths = {key: portable_path(str(outputs[key])) for key in OUTPUT_KEYS}
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("development ligand outputs exist; use --overwrite after review")
    if args.overwrite:
        for path in existing:
            path.unlink()

    pdbqt_rows = read_csv(input_paths["full_pdbqt_manifest"])
    split_rows = read_csv(input_paths["scaffold_split_manifest"])
    split_summary = json.loads(input_paths["scaffold_split_summary"].read_text(encoding="ascii"))
    if split_summary.get("scaffold_disjoint") is not True:
        raise ValueError("scaffold split summary does not report scaffold_disjoint=true")
    if int(split_summary.get("input_rows", -1)) != int(expected["total_ligand_count"]):
        raise ValueError("scaffold split summary input count differs from configuration")
    if len(pdbqt_rows) != int(expected["total_ligand_count"]):
        raise ValueError("full PDBQT manifest count differs from configuration")
    if count_labels([dict(row) for row in pdbqt_rows]) != normalized_counts(
        expected["total_label_counts"], "total_label_counts"
    ):
        raise ValueError("full PDBQT manifest label counts differ from configuration")

    development_rows, locked_ids = build_development_rows(
        pdbqt_rows,
        split_rows,
        set(str(value) for value in config["development_splits"]),
        str(config["locked_split"]),
    )
    if len(development_rows) != int(expected["development_ligand_count"]):
        raise ValueError("development ligand count differs from configuration")
    if len(locked_ids) != int(expected["locked_ligand_count"]):
        raise ValueError("locked ligand count differs from configuration")
    if count_labels(development_rows) != normalized_counts(
        expected["development_label_counts"], "development_label_counts"
    ):
        raise ValueError("development label counts differ from configuration")
    locked_rows = [row for row in split_rows if row["ligand_id"] in set(locked_ids)]
    if count_labels([dict(row) for row in locked_rows]) != normalized_counts(
        expected["locked_label_counts"], "locked_label_counts"
    ):
        raise ValueError("locked label counts differ from configuration")
    development_ids = [str(row["ligand_id"]) for row in development_rows]
    if set(development_ids).intersection(locked_ids):
        raise RuntimeError("locked test ligands leaked into development manifest")

    write_csv(output_paths["development_pdbqt_manifest_csv"], development_rows)
    manifest_hash = file_sha256(output_paths["development_pdbqt_manifest_csv"])
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "operation": "materialize existing train and validation PDBQT rows while excluding locked test",
        "config_path": args.config.as_posix(),
        "config_sha256": file_sha256(args.config),
        "input_sha256": {key: file_sha256(path) for key, path in sorted(input_paths.items())},
        "counts": {
            "total": len(pdbqt_rows),
            "development": len(development_rows),
            "locked": len(locked_ids),
            "development_labels": count_labels(development_rows),
            "locked_labels": count_labels([dict(row) for row in locked_rows]),
        },
        "splits": {
            "development": list(config["development_splits"]),
            "locked": config["locked_split"],
            "scaffold_disjoint": True,
        },
        "integrity": {
            "development_ligand_ids_sha256": identifier_set_sha256(development_ids),
            "locked_ligand_ids_sha256": identifier_set_sha256(locked_ids),
            "per_ligand_pdbqt_sha256_recorded": True,
            "locked_test_ligands_written_to_docking_manifest": 0,
            "docking_score_files_read": 0,
        },
        "outputs": {
            "development_pdbqt_manifest_csv": output_paths[
                "development_pdbqt_manifest_csv"
            ].as_posix(),
            "summary_json": output_paths["summary_json"].as_posix(),
        },
        "output_sha256": {"development_pdbqt_manifest_csv": manifest_hash},
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
