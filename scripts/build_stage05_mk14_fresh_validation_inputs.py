"""Materialize the preregistered fresh MAPK14 validation inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .prepare_receptor import file_sha256
except ImportError:
    from prepare_receptor import file_sha256


LABELS = {"active", "decoy"}
MATRIX_IDS = ("primary", "sensitivity", "seed0", "seed1", "seed2")


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def read_only_split(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if not rows:
        raise ValueError(f"CSV contains no {split} rows: {path}")
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


def checked_path(record: dict[str, object]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(record["sha256"]).upper():
        raise ValueError(f"input SHA-256 differs: {path}")
    return path


def rows_by_id(
    rows: list[dict[str, str]], label: str
) -> dict[str, dict[str, str]]:
    output = {row["ligand_id"]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"{label} contains duplicate ligand IDs")
    return output


def group_order_key(seed: int, group_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}|{group_id}".encode("utf-8")).hexdigest()
    return digest, group_id


def select_fresh_panel(
    validation_rows: list[dict[str, str]],
    consumed_rows: list[dict[str, str]],
    selection: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    validation_by_id = rows_by_id(validation_rows, "validation split")
    consumed_by_id = rows_by_id(consumed_rows, "consumed validation panel")
    if set(consumed_by_id) - set(validation_by_id):
        raise ValueError("a consumed validation ligand is absent from the split")
    if any(row["split"] != "validation" for row in validation_rows):
        raise ValueError("a non-validation row entered validation selection")
    if any(row["split"] != "validation" for row in consumed_rows):
        raise ValueError("the consumed panel contains a non-validation row")
    if any(row["label"] not in LABELS for row in validation_rows):
        raise ValueError("validation contains an unsupported label")
    for ligand_id, consumed in consumed_by_id.items():
        source = validation_by_id[ligand_id]
        if (
            source["label"] != consumed["label"]
            or source["split_group_id"] != consumed["split_group_id"]
        ):
            raise ValueError(f"consumed metadata differs: {ligand_id}")

    consumed_groups = {row["split_group_id"] for row in consumed_rows}
    eligible_rows = [
        row.copy()
        for row in validation_rows
        if row["split_group_id"] not in consumed_groups
    ]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible_rows:
        groups[row["split_group_id"]].append(row)

    active_groups = {
        group_id
        for group_id, rows in groups.items()
        if any(row["label"] == "active" for row in rows)
    }
    selected_groups = set(active_groups)
    selected_rows = [
        row for group_id in selected_groups for row in groups[group_id]
    ]
    active_count = sum(row["label"] == "active" for row in selected_rows)
    minimum_active = int(selection["minimum_selected_active_count"])
    if active_count < minimum_active:
        raise ValueError(
            f"only {active_count} fresh active rows remain; minimum is {minimum_active}"
        )

    target_decoys = active_count * int(selection["target_decoy_to_active_ratio"])
    decoy_count = sum(row["label"] == "decoy" for row in selected_rows)
    decoy_only_groups = [
        group_id
        for group_id, rows in groups.items()
        if group_id not in selected_groups
        and all(row["label"] == "decoy" for row in rows)
    ]
    decoy_only_groups.sort(
        key=lambda group_id: group_order_key(
            int(selection["selection_seed"]), group_id
        )
    )
    for group_id in decoy_only_groups:
        if decoy_count >= target_decoys:
            break
        selected_groups.add(group_id)
        selected_rows.extend(groups[group_id])
        decoy_count += len(groups[group_id])
    if decoy_count < target_decoys:
        raise ValueError("not enough eligible validation decoys")
    if len(selected_rows) > int(selection["maximum_selected_ligand_count"]):
        raise ValueError("fresh panel exceeds the preregistered maximum size")

    role = str(selection["selection_role"])
    panel: list[dict[str, object]] = [
        {**row, "selection_role": role} for row in selected_rows
    ]
    panel.sort(key=lambda row: (str(row["label"]), str(row["ligand_id"])))
    panel_groups = {str(row["split_group_id"]) for row in panel}
    if panel_groups & consumed_groups:
        raise ValueError("fresh and consumed validation groups overlap")
    if len({str(row["ligand_id"]) for row in panel}) != len(panel):
        raise ValueError("fresh panel contains duplicate ligand IDs")
    return panel, {
        "locked_validation_row_count": len(validation_rows),
        "locked_validation_label_counts": dict(
            sorted(Counter(row["label"] for row in validation_rows).items())
        ),
        "consumed_row_count": len(consumed_rows),
        "consumed_group_count": len(consumed_groups),
        "eligible_row_count_after_group_exclusion": len(eligible_rows),
        "eligible_group_count_after_group_exclusion": len(groups),
        "selected_row_count": len(panel),
        "selected_label_counts": dict(
            sorted(Counter(str(row["label"]) for row in panel).items())
        ),
        "selected_group_count": len(panel_groups),
        "active_group_count": len(active_groups),
        "target_decoy_count": target_decoys,
        "selected_decoy_overshoot": decoy_count - target_decoys,
        "consumed_group_overlap_count": 0,
        "test_rows_selected": 0,
    }


def score_bounds(
    rows: list[dict[str, str]], receptor_ids: list[str]
) -> dict[str, dict[str, float]]:
    bounds: dict[str, dict[str, float]] = {}
    for receptor_id in receptor_ids:
        values = [float(row[receptor_id]) for row in rows]
        bounds[receptor_id] = {
            "minimum": min(values),
            "maximum": max(values),
        }
    return bounds


def freeze_normalization_bounds(
    primary_rows: list[dict[str, str]],
    sensitivity_rows: list[dict[str, str]],
    long_rows: list[dict[str, str]],
    receptor_ids: list[str],
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, object]]:
    primary_by_id = rows_by_id(primary_rows, "train primary matrix")
    sensitivity_by_id = rows_by_id(
        sensitivity_rows, "train sensitivity matrix"
    )
    if set(primary_by_id) != set(sensitivity_by_id):
        raise ValueError("train primary and sensitivity ligand IDs differ")
    if len(primary_by_id) != 696:
        raise ValueError("Train-696 matrix must contain 696 ligands")
    if Counter(row["label"] for row in primary_rows) != Counter(
        {"active": 348, "decoy": 348}
    ):
        raise ValueError("Train-696 labels differ")

    long_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    for row in long_rows:
        key = (row["ligand_id"], row["receptor_id"])
        if key in long_by_pair:
            raise ValueError(f"duplicate Train-696 pair: {key}")
        long_by_pair[key] = row
    expected_pairs = len(primary_by_id) * 8
    if len(long_by_pair) != expected_pairs:
        raise ValueError("Train-696 long matrix pair count differs")

    bounds = {
        "primary": score_bounds(primary_rows, receptor_ids),
        "sensitivity": score_bounds(sensitivity_rows, receptor_ids),
    }
    seed_columns = {
        "seed0": "seed0_representative_score",
        "seed1": "seed1_representative_score",
        "seed2": "seed2_representative_score",
    }
    for seed, column in seed_columns.items():
        bounds[seed] = {}
        for receptor_id in receptor_ids:
            values = [
                float(long_by_pair[(ligand_id, receptor_id)][column])
                for ligand_id in primary_by_id
            ]
            bounds[seed][receptor_id] = {
                "minimum": min(values),
                "maximum": max(values),
            }
    return bounds, {
        "train_ligand_count": len(primary_by_id),
        "train_label_counts": {"active": 348, "decoy": 348},
        "train_full_receptor_pair_count": len(long_by_pair),
        "frozen_receptor_count": len(receptor_ids),
        "matrix_ids": list(MATRIX_IDS),
    }


def fixed_receptor_rows(
    source_rows: list[dict[str, str]], receptor_ids: list[str]
) -> list[dict[str, object]]:
    by_id = {row["conformer_id"]: row for row in source_rows}
    if len(by_id) != len(source_rows):
        raise ValueError("receptor manifest contains duplicate conformer IDs")
    if set(receptor_ids) - set(by_id):
        raise ValueError("a frozen validation receptor is absent")
    output: list[dict[str, object]] = []
    for receptor_id in receptor_ids:
        row = by_id[receptor_id]
        if row["status"] != "ok":
            raise ValueError(f"receptor did not pass preparation: {receptor_id}")
        path = Path(row["receptor_pdbqt"].replace("\\", "/"))
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = file_sha256(path)
        if digest != row["receptor_pdbqt_sha256"].upper():
            raise ValueError(f"receptor hash differs: {receptor_id}")
        output.append(
            {
                **row,
                "receptor_pdbqt": path.as_posix(),
                "receptor_pdbqt_sha256": digest,
                "validation_role": "frozen_comparator_union",
            }
        )
    return output


def run(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = read_json(config_path)
    evidence_path = checked_path(config["decision_evidence"])
    evidence = read_json(evidence_path)
    if evidence.get("status") != config["decision_evidence"]["required_status"]:
        raise ValueError("the positive Train-696 decision evidence changed")
    paths = {
        key: checked_path(record)
        for key, record in dict(config["inputs"]).items()
    }
    split_summary = read_json(paths["grouped_scaffold_split_summary"])
    source_audit = config["source_audit"]
    validation_counts = split_summary["counts"]["validation"]
    if (
        int(validation_counts["active"])
        != int(source_audit["locked_validation_active_rows"])
        or int(validation_counts["decoy"])
        != int(source_audit["locked_validation_decoy_rows"])
    ):
        raise ValueError("locked validation source counts differ")

    validation_rows = read_only_split(
        paths["grouped_scaffold_split"], "validation"
    )
    consumed_rows = read_csv(paths["consumed_validation_panel"])
    panel, selection_audit = select_fresh_panel(
        validation_rows,
        consumed_rows,
        dict(config["fresh_panel_selection"]),
    )
    receptor_ids = [
        str(value) for value in config["frozen_methods"]["fixed_receptor_union"]
    ]
    receptors = fixed_receptor_rows(
        read_csv(paths["receptor_manifest"]), receptor_ids
    )
    bounds, train_audit = freeze_normalization_bounds(
        read_csv(paths["train_primary_matrix"]),
        read_csv(paths["train_sensitivity_matrix"]),
        read_csv(paths["train_aggregated_seed_scores"]),
        receptor_ids,
    )

    outputs = config["outputs"]
    panel_path = Path(str(outputs["fresh_panel_csv"]))
    summary_path = Path(str(outputs["fresh_panel_summary_json"]))
    model_path = Path(str(outputs["frozen_model_artifact_json"]))
    receptor_path = Path(str(outputs["fixed_receptor_manifest_csv"]))
    materialized = (panel_path, summary_path, model_path, receptor_path)
    if not overwrite and any(path.exists() for path in materialized):
        raise FileExistsError("fresh-validation outputs exist; use --overwrite")
    write_csv(panel_path, panel)
    write_csv(receptor_path, receptors)

    config_record = {
        "path": config_path.as_posix(),
        "sha256": file_sha256(config_path),
    }
    input_records = {
        key: {"path": path.as_posix(), "sha256": file_sha256(path)}
        for key, path in paths.items()
    }
    frozen_model = {
        "schema_version": "1.0",
        "authorization_id": config["authorization_id"],
        "status": "fresh_validation_model_frozen_scores_unavailable",
        "config": config_record,
        "decision_evidence": {
            "path": evidence_path.as_posix(),
            "sha256": file_sha256(evidence_path),
            "status": evidence["status"],
        },
        "frozen_methods": config["frozen_methods"],
        "normalization_bounds": bounds,
        "normalization_formula": "(validation_score-train_minimum)/(train_maximum-train_minimum); no clipping",
        "train_audit": train_audit,
        "evaluation": config["evaluation"],
        "acceptance": config["acceptance"],
        "validation_scores_read": 0,
        "test_rows_selected": 0,
        "test_scores_read": 0,
        "interpretation_boundary": config["interpretation_boundary"],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(
        json.dumps(frozen_model, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    summary = {
        "schema_version": "1.0",
        "authorization_id": config["authorization_id"],
        "status": "fresh_validation_panel_ok_preparation_pending",
        "config": config_record,
        "inputs": input_records,
        "selection_audit": selection_audit,
        "train_normalization_audit": train_audit,
        "fixed_receptor_ids": receptor_ids,
        "outputs": {
            "fresh_panel_csv": {
                "path": panel_path.as_posix(),
                "sha256": file_sha256(panel_path),
            },
            "fixed_receptor_manifest_csv": {
                "path": receptor_path.as_posix(),
                "sha256": file_sha256(receptor_path),
            },
            "frozen_model_artifact_json": {
                "path": model_path.as_posix(),
                "sha256": file_sha256(model_path),
            },
        },
        "validation_scores_read": 0,
        "test_rows_selected": 0,
        "test_scores_read": 0,
        "next_gate": config["next_gate"],
        "interpretation_note": config["interpretation_boundary"],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
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
