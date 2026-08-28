from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from scripts.seed_ppara_pool30_score_reuse import seed_verified_score_tables


SEEDS = (11, 12, 13)
LIGAND_IDS = ("L001", "L002")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_ligand_manifest(run: Path) -> None:
    rows = []
    for ligand_id in LIGAND_IDS:
        pdbqt = run / "ligands" / f"{ligand_id}.pdbqt"
        _write_text(pdbqt, f"PDBQT {ligand_id}\n")
        rows.append(
            {
                "ligand_id": ligand_id,
                "label": "active",
                "selection_role": "benchmark",
                "pdbqt_path": str(pdbqt),
            }
        )
    with (run / "prepared_ligands.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_receptor_audit(run: Path, receptor_ids: tuple[str, ...]) -> None:
    selected = []
    for receptor_id in receptor_ids:
        pdbqt = run / "receptors" / "prepared" / f"{receptor_id}.pdbqt"
        _write_text(pdbqt, f"RECEPTOR {receptor_id}\n")
        selected.append(
            {
                "conformer_id": receptor_id,
                "rcsb_id": receptor_id,
                "receptor_pdbqt": str(pdbqt),
                "status": "ok",
            }
        )
    (run / "receptor_preparation_audit.json").write_text(
        json.dumps({"selected": selected, "selected_count": len(selected)}),
        encoding="utf-8",
    )


def _write_score_table(path: Path, receptor_id: str, seed: int, ligand_ids: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ligand_id", "receptor_id", "seed", "status", "score"],
        )
        writer.writeheader()
        for ligand_id in ligand_ids:
            writer.writerow(
                {
                    "ligand_id": ligand_id,
                    "receptor_id": receptor_id,
                    "seed": seed,
                    "status": "ok",
                    "score": "-7.0",
                }
            )


def _make_matching_runs(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    for run in (source, destination):
        _write_ligand_manifest(run)
        (run / "docking_box.json").write_text(
            json.dumps({"center_x": 1.0, "center_y": 2.0, "size_x": 22.0}),
            encoding="utf-8",
        )
        (run / "config.snapshot.json").write_text(
            json.dumps(
                {
                    "target_id": "PPARA",
                    "docking": {
                        "seeds": list(SEEDS),
                        "box": {
                            "method": "ligand_bounds",
                            "artifact_path": str(run / "docking_box.json"),
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
    _write_receptor_audit(source, ("R01", "R02"))
    _write_receptor_audit(destination, ("R01", "R02", "R03"))
    for seed in SEEDS:
        for receptor_id in ("R01", "R02"):
            _write_score_table(
                source / "score_tables" / f"seed_{seed}__{receptor_id}.csv",
                receptor_id,
                seed,
                LIGAND_IDS,
            )
    return source, destination


def test_seed_reuse_hardlinks_complete_verified_tables(tmp_path: Path) -> None:
    source, destination = _make_matching_runs(tmp_path)

    audit = seed_verified_score_tables(source, destination)

    destination_table = destination / "score_tables" / "seed_11__R01.csv"
    assert audit["shared_receptor_ids"] == ["R01", "R02"]
    assert audit["linked_table_count"] == 6
    assert os.path.samefile(source / "score_tables" / "seed_11__R01.csv", destination_table)


def test_seed_reuse_rejects_changed_ligand_pdbqt(tmp_path: Path) -> None:
    source, destination = _make_matching_runs(tmp_path)
    _write_text(destination / "ligands" / "L001.pdbqt", "different ligand bytes\n")

    with pytest.raises(ValueError, match="ligand PDBQT hash"):
        seed_verified_score_tables(source, destination)


def test_seed_reuse_rejects_incomplete_score_table(tmp_path: Path) -> None:
    source, destination = _make_matching_runs(tmp_path)
    _write_score_table(
        source / "score_tables" / "seed_11__R01.csv",
        "R01",
        11,
        ("L001",),
    )

    with pytest.raises(ValueError, match="complete"):
        seed_verified_score_tables(source, destination)
