import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage91c_freezes_complete_development_grid():
    config = json.loads(
        (
            ROOT
            / "configs/stage91c_bace1_chembl365_unidock113_production.json"
        ).read_text(encoding="ascii")
    )
    assert config["expected"]["receptor_count"] == 34
    assert config["expected"]["ligand_count"] == 365
    assert config["expected"]["seed_count"] == 3
    assert config["expected"]["batch_count"] == 102
    assert config["expected"]["pair_count"] == 37230
    assert config["expected"]["potency_label_counts"] == {
        "high": 248,
        "low": 52,
        "gray": 65,
    }
    assert config["expected"]["confirmation_rows"] == 0
    assert config["expected"]["locked_test_rows"] == 0


def test_stage91c_authorizes_only_development_docking():
    prereg = json.loads(
        (
            ROOT
            / "configs/stage91c_bace1_group_robust_development_docking_preregistration.json"
        ).read_text(encoding="ascii")
    )
    authorization = prereg["authorization"]
    assert authorization["development_docking_authorized"] is True
    assert authorization["confirmation_or_test_docking_authorized"] is False
    assert authorization["objective_coefficient_tuning_after_docking_authorized"] is False
    assert authorization["quantum_hardware_authorized"] is False


def test_stage91c_bundle_contains_only_frozen_development_inputs(tmp_path):
    from scripts.build_stage91c_bace1_chembl365_production_bundle import main
    import sys

    output = tmp_path / "stage91c.tar.gz"
    summary = tmp_path / "summary.json"
    old_argv = sys.argv
    try:
        sys.argv = [
            "builder",
            "--root",
            str(ROOT),
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ]
        assert main() == 0
    finally:
        sys.argv = old_argv
    result = json.loads(summary.read_text(encoding="ascii"))
    assert result["status"] == "ok"
    assert result["gpu_pair_count"] == 37230
    assert result["confirmation_rows"] == 0
    assert result["locked_test_rows"] == 0
    with tarfile.open(output, "r:gz") as archive:
        names = [name.lower() for name in archive.getnames()]
    assert not any("confirmation_a" in name for name in names)
    assert not any("confirmation_b" in name for name in names)
    assert not any("locked_test" in name for name in names)
    assert not any("fresh_validation" in name for name in names)
