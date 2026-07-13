import pytest

from scripts.run_md_receptor_ligand_benchmark import (
    annotate_search_warnings,
    portable_manifest_path,
    score_table_is_complete,
)


def test_score_table_is_complete_requires_all_successful_ligands(tmp_path):
    table = tmp_path / "scores.csv"
    table.write_text(
        "ligand_id,status\nA,ok\nB,ok\n",
        encoding="utf-8",
    )
    assert score_table_is_complete(table, {"A", "B"}) is True
    assert score_table_is_complete(table, {"A", "B", "C"}) is False


def test_score_table_is_complete_rejects_failed_rows(tmp_path):
    table = tmp_path / "scores.csv"
    table.write_text(
        "ligand_id,status\nA,ok\nB,failed\n",
        encoding="utf-8",
    )
    assert score_table_is_complete(table, {"A", "B"}) is False


def test_portable_manifest_path_normalizes_windows_separators():
    path = portable_manifest_path(r"results\runs\receptor.pdbqt")
    assert path.as_posix() == "results/runs/receptor.pdbqt"


def test_annotate_search_warnings_uses_each_ligand_median():
    rows = [
        {"ligand_id": "A", "receptor_id": "R1", "status": "ok", "representative_score": -8.0},
        {"ligand_id": "A", "receptor_id": "R2", "status": "ok", "representative_score": -7.8},
        {"ligand_id": "A", "receptor_id": "R3", "status": "ok", "representative_score": 2.0},
    ]
    annotated = annotate_search_warnings(rows, True, 5.0)
    assert annotated[0]["ligand_median_score"] == pytest.approx(-7.8)
    assert annotated[0]["search_quality_warning"] is False
    assert annotated[2]["search_quality_warning"] is True
    assert annotated[2]["search_quality_warning_reasons"] == (
        "nonnegative_vina_score;large_unfavorable_delta_from_ligand_median"
    )
