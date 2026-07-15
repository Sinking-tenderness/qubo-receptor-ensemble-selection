import json
from pathlib import Path

import pytest

from scripts.build_development_ligand_manifest import (
    build_development_rows,
    load_config,
)


CONFIG_PATH = Path("configs/stage04_cdk2_development_ligand_manifest.json")


def ligand_row(tmp_path: Path, ligand_id: str, label: str) -> dict[str, str]:
    path = tmp_path / f"{ligand_id}.pdbqt"
    path.write_text("ATOM\n", encoding="ascii")
    return {
        "ligand_id": ligand_id,
        "label": label,
        "canonical_smiles": f"SMILES-{ligand_id}",
        "pdbqt_status": "ok",
        "pdbqt_path": path.as_posix(),
    }


def split_row(ligand_id: str, label: str, split: str) -> dict[str, str]:
    return {
        "ligand_id": ligand_id,
        "label": label,
        "canonical_smiles": f"SMILES-{ligand_id}",
        "scaffold_smiles": f"SCAFFOLD-{ligand_id}",
        "split": split,
    }


def test_build_development_rows_excludes_locked_split(tmp_path: Path):
    pdbqt_rows = [
        ligand_row(tmp_path, "A", "active"),
        ligand_row(tmp_path, "B", "decoy"),
        ligand_row(tmp_path, "C", "active"),
    ]
    split_rows = [
        split_row("A", "active", "train"),
        split_row("B", "decoy", "validation"),
        split_row("C", "active", "test"),
    ]

    development, locked = build_development_rows(
        pdbqt_rows, split_rows, {"train", "validation"}, "test"
    )

    assert [row["ligand_id"] for row in development] == ["A", "B"]
    assert all(row["pdbqt_sha256"] for row in development)
    assert [row["benchmark_split"] for row in development] == ["train", "validation"]
    assert locked == ["C"]


def test_build_development_rows_rejects_label_mismatch(tmp_path: Path):
    pdbqt_rows = [
        ligand_row(tmp_path, "A", "active"),
        ligand_row(tmp_path, "B", "decoy"),
    ]
    split_rows = [
        split_row("A", "decoy", "train"),
        split_row("B", "decoy", "test"),
    ]

    with pytest.raises(ValueError, match="label differs"):
        build_development_rows(pdbqt_rows, split_rows, {"train"}, "test")


def test_load_config_rejects_locked_split_in_development(tmp_path: Path):
    config = json.loads(CONFIG_PATH.read_text(encoding="ascii"))
    config["development_splits"].append("test")
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="ascii")

    with pytest.raises(ValueError, match="locked_split"):
        load_config(path)
