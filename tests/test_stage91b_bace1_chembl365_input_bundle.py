import csv
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage91b_frozen_manifest_is_development_only():
    path = (
        ROOT
        / "data/processed/stage91b_bace1_chembl365_development_ligand_manifest.csv"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 365
    assert {row["role"] for row in rows} == {"development"}
    assert all(row["docking_authorized"].lower() == "true" for row in rows)


def test_stage91b_freeze_records_a_zero_leakage_boundary():
    result = json.loads(
        (ROOT / "data/stage91b_bace1_development_manifest_freeze.json").read_text(
            encoding="ascii"
        )
    )
    assert result["status"] == "stage91b_bace1_development_manifest_frozen"
    assert result["output"]["row_count"] == 365
    assert result["roles_present"] == ["development"]
    assert result["data_boundary"]["confirmation_rows_exported"] == 0
    assert result["data_boundary"]["locked_test_rows_exported"] == 0


def test_stage91b_external_bundle_excludes_all_role_manifest(tmp_path):
    from scripts.build_stage91b_bace1_chembl365_input_bundle import run

    output = tmp_path / "stage91b.tar.gz"
    run(ROOT, output)
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
        assert (
            "data/processed/stage91_bace1_chembl_assay_role_ligand_manifest.csv"
            not in names
        )
        assert (
            "data/processed/stage91b_bace1_chembl365_development_ligand_manifest.csv"
            in names
        )
        manifest = archive.extractfile(
            "data/processed/stage91b_bace1_chembl365_development_ligand_manifest.csv"
        )
        assert manifest is not None
        text = manifest.read().decode("utf-8")
        assert ",confirmation_a," not in text
        assert ",confirmation_b," not in text
        assert ",locked_test," not in text
