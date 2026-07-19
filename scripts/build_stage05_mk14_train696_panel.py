"""Build the preregistered 348-active/348-decoy MAPK14 train panel."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

try:
    from .prepare_receptor import file_sha256
except ImportError:
    from prepare_receptor import file_sha256


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


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty panel")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checked_path(record: dict[str, object]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(record["sha256"]).upper():
        raise ValueError(f"input SHA-256 differs: {path}")
    return path


def select_panel(
    split_rows: list[dict[str, str]],
    existing_rows: list[dict[str, str]],
    active_count: int,
    decoy_count: int,
    seed: int,
    selection_role: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    split_by_id = {row["ligand_id"]: row for row in split_rows}
    if len(split_by_id) != len(split_rows):
        raise ValueError("grouped split contains duplicate ligand IDs")
    existing_by_id = {row["ligand_id"]: row for row in existing_rows}
    if len(existing_by_id) != len(existing_rows):
        raise ValueError("existing train panel contains duplicate ligand IDs")
    if set(existing_by_id) - set(split_by_id):
        raise ValueError("an existing train ligand is absent from the grouped split")
    for ligand_id, existing in existing_by_id.items():
        source = split_by_id[ligand_id]
        if (
            source["split"] != "train"
            or existing["split"] != "train"
            or source["label"] != existing["label"]
            or source["split_group_id"] != existing["split_group_id"]
        ):
            raise ValueError(f"existing panel row differs: {ligand_id}")

    train_actives = sorted(
        (
            row.copy()
            for row in split_rows
            if row["split"] == "train" and row["label"] == "active"
        ),
        key=lambda row: row["ligand_id"],
    )
    if len(train_actives) != active_count:
        raise ValueError("the frozen train active count differs")
    if len({row["split_group_id"] for row in train_actives}) != active_count:
        raise ValueError("train actives are not split-group unique")

    existing_decoys = sorted(
        (
            split_by_id[row["ligand_id"]].copy()
            for row in existing_rows
            if row["label"] == "decoy"
        ),
        key=lambda row: row["ligand_id"],
    )
    used_groups = {row["split_group_id"] for row in train_actives}
    if used_groups & {row["split_group_id"] for row in existing_decoys}:
        raise ValueError("an active and existing decoy share a split group")
    used_groups.update(row["split_group_id"] for row in existing_decoys)
    existing_ids = set(existing_by_id)
    candidates = sorted(
        (
            row.copy()
            for row in split_rows
            if row["split"] == "train"
            and row["label"] == "decoy"
            and row["ligand_id"] not in existing_ids
        ),
        key=lambda row: row["ligand_id"],
    )
    random.Random(seed).shuffle(candidates)
    additional_decoys: list[dict[str, str]] = []
    for row in candidates:
        if row["split_group_id"] in used_groups:
            continue
        additional_decoys.append(row)
        used_groups.add(row["split_group_id"])
        if len(existing_decoys) + len(additional_decoys) == decoy_count:
            break
    if len(existing_decoys) + len(additional_decoys) != decoy_count:
        raise ValueError("not enough group-diverse train decoys")

    panel = [*train_actives, *existing_decoys, *additional_decoys]
    for row in panel:
        row["selection_role"] = selection_role
    panel.sort(key=lambda row: (row["label"], row["ligand_id"]))
    selected_ids = {row["ligand_id"] for row in panel}
    if not existing_ids.issubset(selected_ids):
        raise ValueError("the existing train panel is not a strict subset")
    if len(selected_ids) != active_count + decoy_count:
        raise ValueError("expanded panel contains duplicate ligand IDs")
    if len({row["split_group_id"] for row in panel}) != len(panel):
        raise ValueError("expanded panel contains duplicate split groups")
    new_rows = [row.copy() for row in panel if row["ligand_id"] not in existing_ids]
    return panel, new_rows


def run(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = read_json(config_path)
    evidence = checked_path(config["decision_evidence"])
    evidence_value = read_json(evidence)
    if evidence_value.get("status") != config["decision_evidence"]["required_status"]:
        raise ValueError("the fixed-candidate failure evidence changed")
    inputs = config["inputs"]
    selection = config["panel_selection"]
    source_audit = config["source_audit"]
    outputs = config["outputs"]
    assert isinstance(inputs, dict)
    assert isinstance(selection, dict)
    assert isinstance(source_audit, dict)
    assert isinstance(outputs, dict)
    paths = {key: checked_path(record) for key, record in inputs.items()}
    split_rows = read_csv(paths["grouped_scaffold_split"])
    existing_rows = read_csv(paths["existing_train_panel"])
    train_counts = Counter(
        row["label"] for row in split_rows if row["split"] == "train"
    )
    train_groups = {
        label: len(
            {
                row["split_group_id"]
                for row in split_rows
                if row["split"] == "train" and row["label"] == label
            }
        )
        for label in ("active", "decoy")
    }
    if (
        train_counts["active"] != int(source_audit["train_active_rows"])
        or train_counts["decoy"] != int(source_audit["train_decoy_rows"])
        or train_groups["active"]
        != int(source_audit["train_active_unique_split_groups"])
        or train_groups["decoy"]
        != int(source_audit["train_decoy_unique_split_groups"])
    ):
        raise ValueError("frozen grouped-split source counts differ")
    if {row["split"] for row in existing_rows} != {"train"}:
        raise ValueError("the existing panel contains a non-train row")
    panel, new_rows = select_panel(
        split_rows,
        existing_rows,
        int(selection["target_active_count"]),
        int(selection["target_decoy_count"]),
        int(selection["additional_decoy_seed"]),
        str(selection["selection_role"]),
    )
    if len(panel) != int(selection["target_total_count"]):
        raise ValueError("expanded panel count differs")
    if len(new_rows) != int(selection["new_total_count"]):
        raise ValueError("new ligand count differs")
    if Counter(row["label"] for row in new_rows) != Counter(
        {
            "active": int(selection["new_active_count"]),
            "decoy": int(selection["new_decoy_count"]),
        }
    ):
        raise ValueError("new ligand label counts differ")
    if any(row["split"] != "train" for row in panel):
        raise ValueError("a validation or test ligand entered the panel")

    panel_path = Path(str(outputs["expanded_panel_csv"]))
    new_path = Path(str(outputs["new_ligands_csv"]))
    summary_path = Path(str(outputs["panel_summary_json"]))
    if not overwrite and any(path.exists() for path in (panel_path, new_path, summary_path)):
        raise FileExistsError("expanded-panel outputs exist; use --overwrite")
    write_csv(panel_path, panel)
    write_csv(new_path, new_rows)
    existing_ids = {row["ligand_id"] for row in existing_rows}
    summary = {
        "schema_version": "1.0",
        "authorization_id": config["authorization_id"],
        "status": "expanded_train696_panel_ok_preparation_pending",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_counts": {
            "train_active": train_counts["active"],
            "train_decoy": train_counts["decoy"],
            "train_active_groups": train_groups["active"],
            "train_decoy_groups": train_groups["decoy"],
        },
        "panel": {
            "row_count": len(panel),
            "label_counts": dict(sorted(Counter(row["label"] for row in panel).items())),
            "unique_split_group_count": len({row["split_group_id"] for row in panel}),
            "selection_role_counts": dict(
                sorted(Counter(row["selection_role"] for row in panel).items())
            ),
            "existing_panel_rows_retained": sum(
                row["ligand_id"] in existing_ids for row in panel
            ),
            "new_row_count": len(new_rows),
            "new_label_counts": dict(
                sorted(Counter(row["label"] for row in new_rows).items())
            ),
            "validation_rows": 0,
            "test_rows": 0,
        },
        "outputs": {
            "expanded_panel_csv": {
                "path": panel_path.as_posix(),
                "sha256": file_sha256(panel_path),
            },
            "new_ligands_csv": {
                "path": new_path.as_posix(),
                "sha256": file_sha256(new_path),
            },
        },
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
