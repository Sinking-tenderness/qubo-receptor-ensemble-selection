from pathlib import Path

from scripts.run_md_receptor_docking_gate import (
    add_parent_comparison,
    annotate_search_warnings,
    select_fixed_ligands,
)


def test_select_fixed_ligands_preserves_requested_order_and_hash(tmp_path: Path):
    first = tmp_path / "A.pdbqt"
    second = tmp_path / "D.pdbqt"
    first.write_text("A", encoding="ascii")
    second.write_text("D", encoding="ascii")
    from scripts.prepare_receptor import file_sha256

    manifest = [
        {"ligand_id": "D", "label": "decoy", "pdbqt_status": "ok", "pdbqt_path": str(second)},
        {"ligand_id": "A", "label": "active", "pdbqt_status": "ok", "pdbqt_path": str(first)},
    ]
    requested = [
        {"ligand_id": "A", "label": "active", "pdbqt_sha256": file_sha256(first)},
        {"ligand_id": "D", "label": "decoy", "pdbqt_sha256": file_sha256(second)},
    ]
    selected = select_fixed_ligands(manifest, requested)
    assert [row["ligand_id"] for row in selected] == ["A", "D"]
    assert all(row["pdbqt_sha256"] for row in selected)


def test_add_parent_comparison_uses_score_difference():
    rows = [{
        "ligand_id": "A",
        "receptor_id": "R1",
        "representative_score": -9.0,
        "status": "ok",
    }]
    compared = add_parent_comparison(rows, {"A": -8.5})
    assert compared[0]["parent_af2_score"] == -8.5
    assert compared[0]["delta_from_parent_af2"] == -0.5


def test_search_warning_flags_nonnegative_large_delta():
    rows = [{
        "ligand_id": "D",
        "receptor_id": "R",
        "representative_score": 26.95,
        "parent_af2_score": -7.547,
        "delta_from_parent_af2": 34.497,
        "status": "ok",
    }]
    annotated = annotate_search_warnings(rows, True, 5.0)
    assert annotated[0]["search_quality_warning"] is True
    assert annotated[0]["search_quality_warning_reasons"] == (
        "nonnegative_vina_score;large_unfavorable_delta_from_parent"
    )
