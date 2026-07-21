from pathlib import Path

from scripts.audit_stage05_mk14_fresh_validation_preparation import (
    audit_prepared_rows,
)
from scripts.prepare_receptor import file_sha256


def test_audit_prepared_rows_accepts_warning_sdf_and_complete_pdbqt(
    tmp_path: Path,
):
    pdbqt = tmp_path / "ligand.pdbqt"
    pdbqt.write_text("ATOM      1  C   LIG A   1       0.0 0.0 0.0\n", encoding="ascii")
    common = {
        "ligand_id": "L1",
        "label": "active",
        "split": "validation",
        "split_group_id": "g1",
        "selection_role": "fresh_validation_preregistered",
        "formal_charge": "0",
    }
    panel = [common]
    three_d = [{**common, "prep_status": "warning"}]
    prepared = [
        {
            **common,
            "pdbqt_status": "ok",
            "pdbqt_path": str(pdbqt),
            "pdbqt_sha256": file_sha256(pdbqt),
            "pdbqt_atom_count": "1",
        }
    ]

    audit = audit_prepared_rows(panel, three_d, prepared)

    assert audit["ligand_count"] == 1
    assert audit["three_d_status_counts"] == {"warning": 1}
    assert audit["pdbqt_status_counts"] == {"ok": 1}
    assert audit["test_rows"] == 0
