"""Correct the Stage 19c aggregate target label without changing docking data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_receptor import file_sha256


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("corrected rows have inconsistent columns")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def rooted(root: Path, value: str) -> Path:
    path = (root / value.replace("\\", "/")).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path leaves repository root: {value}") from error
    return path


def verified(root: Path, descriptor: dict[str, object]) -> Path:
    path = rooted(root, str(descriptor["path"]))
    expected = str(descriptor["sha256"]).upper()
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"input identity differs: {path}")
    return path


def non_target_digest(rows: list[dict[str, object]]) -> str:
    payload = [
        {key: value for key, value in row.items() if key != "target_id"}
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()


def correct_target_rows(
    rows: list[dict[str, str]], erroneous_target: str, corrected_target: str
) -> list[dict[str, object]]:
    observed = Counter(row.get("target_id", "") for row in rows)
    if observed != Counter({erroneous_target: len(rows)}):
        raise ValueError(f"source target labels differ: {dict(observed)}")
    return [{**row, "target_id": corrected_target} for row in rows]


def descriptor(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def write_report(path: Path, result: dict[str, object]) -> None:
    lines = [
        "# Stage 19c PPARG target-id amendment 01",
        "",
        "The historical Uni-Dock helper wrote `target_id=MK14` into a PPARG score table.",
        "This amendment changes only that metadata field and does not rerun docking.",
        "",
        f"- Corrected rows: {result['corrected_row_count']}",
        f"- Non-target payload SHA-256: `{result['non_target_payload_sha256']}`",
        "- Primary and sensitivity matrices are reused byte-for-byte.",
        "- Validation and test rows read: 0.",
        "",
        "The amended table remains post-hoc exploratory Train-668 evidence only.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, dict(config["implementation"]))
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("amendment implementation path differs")

    inputs = {
        key: verified(root, dict(value))
        for key, value in dict(config["inputs"]).items()
    }
    summary = read_json(inputs["source_summary"])
    audit = read_json(inputs["source_audit"])
    if summary.get("status") != "stage19c_pparg_train668_unidock_matrix_ok":
        raise ValueError("source Stage 19c summary did not pass")
    if (
        audit.get("status")
        != "independent_stage19c_pparg_train668_unidock_matrix_audit_ok"
    ):
        raise ValueError("source Stage 19c audit did not pass")
    if any(int(value) != 0 for value in dict(audit["data_boundary"]).values()):
        raise ValueError("source Stage 19c audit crossed a data boundary")

    rows = read_csv(inputs["source_scores"])
    ligands = read_csv(inputs["ligand_manifest"])
    receptors = read_csv(inputs["receptor_manifest"])
    expected = dict(config["expected"])
    ligand_ids = {row["ligand_id"] for row in ligands}
    receptor_ids = {row["conformer_id"] for row in receptors}
    keys = {
        (row["seed_id"], row["receptor_id"], row["ligand_id"])
        for row in rows
    }
    if (
        len(rows) != int(expected["pair_count"])
        or len(keys) != len(rows)
        or len(ligand_ids) != int(expected["ligand_count"])
        or len(receptor_ids) != int(expected["receptor_count"])
        or {row["seed_id"] for row in rows}
        != set(str(value) for value in expected["seed_ids"])
        or {row["ligand_id"] for row in rows} != ligand_ids
        or {row["receptor_id"] for row in rows} != receptor_ids
    ):
        raise ValueError("source Stage 19c score grid differs")
    if {row["target_id"] for row in ligands} != {config["corrected_target_id"]}:
        raise ValueError("ligand manifest target identity differs")
    if not all(
        ligand_id.startswith(f"{config['corrected_target_id']}_")
        for ligand_id in ligand_ids
    ) or not all(
        receptor_id.startswith(f"{config['corrected_target_id']}_")
        for receptor_id in receptor_ids
    ):
        raise ValueError("PPARG identifier prefixes differ")

    corrected = correct_target_rows(
        rows,
        str(config["erroneous_target_id"]),
        str(config["corrected_target_id"]),
    )
    source_digest = non_target_digest(rows)
    if non_target_digest(corrected) != source_digest:
        raise ValueError("non-target payload changed during amendment")

    outputs = dict(config["outputs"])
    scores_path = rooted(root, str(outputs["corrected_scores_csv"]))
    result_path = rooted(root, str(outputs["result_json"]))
    report_path = rooted(root, str(outputs["report_md"]))
    if not overwrite and any(path.exists() for path in (scores_path, result_path, report_path)):
        raise FileExistsError("amendment outputs exist; pass --overwrite")
    write_csv(scores_path, corrected)
    reloaded = read_csv(scores_path)
    if (
        Counter(row["target_id"] for row in reloaded)
        != Counter({str(config["corrected_target_id"]): len(rows)})
        or non_target_digest(reloaded) != source_digest
    ):
        raise ValueError("written amendment differs")

    result = {
        "schema_version": "1.0",
        "amendment_id": config["amendment_id"],
        "status": "stage19c_pparg_target_id_amendment01_ok",
        "experiment_class": "posthoc_exploratory_train_only_metadata_amendment",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_target_id": config["erroneous_target_id"],
        "corrected_target_id": config["corrected_target_id"],
        "corrected_row_count": len(rows),
        "non_target_payload_sha256": source_digest,
        "numerical_scores_exact": True,
        "pose_paths_and_hashes_exact": True,
        "primary_matrix_reused_byte_exact": True,
        "sensitivity_matrix_reused_byte_exact": True,
        "docking_jobs_started": 0,
        "data_boundary": {"validation_rows_read": 0, "test_rows_read": 0},
        "inputs": {
            key: descriptor(root, path) for key, path in inputs.items()
        },
        "outputs": {
            "corrected_scores_csv": descriptor(root, scores_path),
            "primary_matrix_csv": descriptor(root, inputs["primary_matrix"]),
            "sensitivity_matrix_csv": descriptor(
                root, inputs["sensitivity_matrix"]
            ),
        },
        "next_gate": "run the frozen PPARG Train-668 QUBO and comparator analysis",
        "decision_boundary": config["decision_boundary"],
    }
    write_json(result_path, result)
    write_report(report_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


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
