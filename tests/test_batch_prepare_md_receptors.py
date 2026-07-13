from pathlib import Path

from scripts.batch_prepare_md_receptors import (
    receptor_output_paths,
    successful_manifest_row,
)


def test_receptor_output_paths_are_isolated_by_conformer(tmp_path: Path):
    paths = receptor_output_paths(tmp_path, "C01")
    assert paths["item_directory"] == tmp_path / "C01"
    assert paths["receptor_pdbqt"].name == "C01_receptor.pdbqt"
    assert paths["item_summary"].name == "preparation_summary.json"


def test_successful_manifest_row_extracts_audited_receptor_values():
    alignment = {
        "conformer_id": "C01",
        "cluster_id": "1",
        "temporal_support_role": "revisited_primary_candidate",
        "aligned_heavy_pdb_path": "aligned.pdb",
        "aligned_heavy_pdb_sha256": "ABC",
    }
    summary = {
        "status": "ok",
        "charge_model": "gasteiger",
        "meeko_version": "0.7.1",
        "prody_version": "2.4.1",
        "outputs": {
            "protein_only_pdb": {
                "path": "protein.pdb", "sha256": "P",
                "audit": {"atom_record_count": 10, "hydrogen_count": 0},
            },
            "prepared_pdb": {
                "path": "prepared.pdb", "sha256": "Q",
                "audit": {"atom_record_count": 20, "hydrogen_count": 10},
            },
            "receptor_pdbqt": {
                "path": "receptor.pdbqt", "sha256": "R",
                "audit": {
                    "atom_record_count": 12,
                    "hetatm_record_count": 0,
                    "residue_count": 3,
                    "hydrogen_like_atom_count": 2,
                    "charge_min": -0.5,
                    "charge_max": 0.3,
                    "autodock_atom_types": ["C", "HD", "N"],
                },
            },
        },
    }
    row = successful_manifest_row(
        alignment,
        summary,
        runtime_seconds=1.25,
        expected_residue_count=3,
        expected_atom_types=["C", "HD", "N"],
    )
    assert row["preparation_status"] == "ok"
    assert row["receptor_pdbqt_atom_count"] == 12
    assert row["receptor_autodock_atom_types"] == "C;HD;N"
    assert row["runtime_seconds"] == 1.25
