import csv
import json

from scripts.build_stage05_mk14_expanded_matrix_diagnostic_bundle import (
    selected_manifest_paths,
)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_selected_manifest_paths_includes_only_expected_train_inputs(tmp_path) -> None:
    receptor_manifest = tmp_path / "receptors.csv"
    ligand_manifest = tmp_path / "ligands.csv"
    write_csv(
        receptor_manifest,
        ["conformer_id", "receptor_pdbqt"],
        [
            {"conformer_id": "R1", "receptor_pdbqt": "rec/r1.pdbqt"},
            {"conformer_id": "R2", "receptor_pdbqt": "rec/r2.pdbqt"},
        ],
    )
    write_csv(
        ligand_manifest,
        ["ligand_id", "selection_role", "pdbqt_path"],
        [
            {
                "ligand_id": "L1",
                "selection_role": "development_train",
                "pdbqt_path": "lig/l1.pdbqt",
            },
            {
                "ligand_id": "L2",
                "selection_role": "development_train",
                "pdbqt_path": "lig/l2.pdbqt",
            },
        ],
    )
    config = tmp_path / "diagnostic.json"
    config.write_text(
        json.dumps(
            {
                "inputs": {
                    "receptor_manifest": receptor_manifest.name,
                    "ligand_manifest": ligand_manifest.name,
                },
                "expected_cases": [{"ligand_id": "L2", "receptor_id": "R1"}],
            }
        ),
        encoding="ascii",
    )

    assert selected_manifest_paths(tmp_path, config.name) == [
        "lig/l2.pdbqt",
        "rec/r1.pdbqt",
    ]
