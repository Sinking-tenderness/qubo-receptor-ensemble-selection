from pathlib import Path

from scripts.audit_external_target_intake import audit_ism, molblock_sha256


def test_molblock_hash_ignores_dynamic_sdf_properties(tmp_path: Path) -> None:
    first = tmp_path / "first.sdf"
    second = tmp_path / "second.sdf"
    molblock = "\n  test\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n"
    first.write_text(molblock + "> <job_id>\n1\n\n$$$$\n", encoding="utf-8")
    second.write_text(molblock + "> <job_id>\n2\n\n$$$$\n", encoding="utf-8")
    assert molblock_sha256(first) == molblock_sha256(second)


def test_audit_ism_parses_and_counts_unique_molecules(tmp_path: Path) -> None:
    path = tmp_path / "ligands.ism"
    path.write_text("CC mol1\nC[NH3+] mol2\n", encoding="utf-8")
    result = audit_ism(path, 2)
    assert result["row_count"] == 2
    assert result["rdkit_parsed_count"] == 2
    assert result["charged_molecule_count"] == 1
    assert result["rdkit_failure_count"] == 0


def test_audit_ism_preserves_duplicate_source_ids(tmp_path: Path) -> None:
    path = tmp_path / "stereoisomers.ism"
    path.write_text("F[C@H](Cl)Br source1\nF[C@@H](Cl)Br source1\n", encoding="utf-8")
    result = audit_ism(
        path,
        2,
        expected_unique_id_count=1,
        expected_duplicate_id_count=1,
        expected_max_id_multiplicity=2,
    )
    assert result["unique_source_molecule_id_count"] == 1
    assert result["duplicate_source_molecule_id_count"] == 1
    assert result["maximum_source_id_multiplicity"] == 2
