import pytest

from scripts.experimental.unidock.build_unidock_rigid_train160_manifest import build_rows


def test_build_rows_replaces_only_declared_pdbqt_fields():
    source = [
        {
            "ligand_id": "L1",
            "label": "decoy",
            "smiles": "CC",
            "pdbqt_path": "old.pdbqt",
            "pdbqt_sha256": "OLD",
            "pdbqt_status": "ok",
            "pdbqt_message": "meeko_ok",
            "pdbqt_atom_count": "3",
            "pdbqt_atom_types": "C;CG0;G0",
            "pdbqt_charge_min": "0",
            "pdbqt_charge_max": "0",
            "torsdof": "0",
        }
    ]
    rigid = [
        {
            **source[0],
            "source_manifest_index": "0",
            "source_pdbqt_sha256": "OLD",
            "sdf_sha256": "SDF",
            "pdbqt_path": "new.pdbqt",
            "pdbqt_sha256": "NEW",
            "pdbqt_message": "meeko_rigid_macrocycles_ok",
            "pdbqt_atom_count": "2",
            "pdbqt_atom_types": "C",
        }
    ]

    rows = build_rows(source, rigid)

    assert rows[0]["smiles"] == "CC"
    assert rows[0]["pdbqt_path"] == "new.pdbqt"
    assert rows[0]["source_pdbqt_path"] == "old.pdbqt"
    assert rows[0]["seed_offset"] == 0
    assert rows[0]["preparation_variant"] == "meeko_rigid_macrocycles"


def test_build_rows_rejects_source_index_drift():
    source = [{"ligand_id": "L1", "pdbqt_path": "x", "pdbqt_sha256": "X"}]
    rigid = [
        {
            "ligand_id": "L1",
            "source_manifest_index": "2",
            "source_pdbqt_sha256": "X",
        }
    ]

    with pytest.raises(ValueError, match="rigid source index differs"):
        build_rows(source, rigid)
