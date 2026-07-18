import csv

import pytest

from scripts.build_stage05_mk14_expanded_train_remote_bundle import (
    validate_train_only_manifest,
)


def write_manifest(path, role, split):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ligand_id", "selection_role", "split"],
        )
        writer.writeheader()
        for index in range(160):
            writer.writerow(
                {
                    "ligand_id": f"L{index}",
                    "selection_role": role,
                    "split": split,
                }
            )


def test_validate_train_only_manifest_rejects_validation(tmp_path):
    path = tmp_path / "ligands.csv"
    write_manifest(path, "development_validation", "validation")

    with pytest.raises(ValueError, match="not train-only"):
        validate_train_only_manifest(path)


def test_validate_train_only_manifest_accepts_exact_train_rows(tmp_path):
    path = tmp_path / "ligands.csv"
    write_manifest(path, "development_train", "train")

    assert len(validate_train_only_manifest(path)) == 160
