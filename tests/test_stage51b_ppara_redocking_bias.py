import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_stage51b_preserves_failed_confirmation_and_allows_only_exploration():
    result = json.loads(
        (ROOT / "data/stage51b_ppara_redocking_bias_diagnostic_result.json").read_text()
    )
    decision = result["decision"]
    assert result["status"] == "stage51b_ppara_redocking_bias_diagnostic_ok"
    assert result["stage51_frozen_outcome"]["technical_gate_pass"] is False
    assert result["stage51_frozen_outcome"]["confirmatory_status_changed"] is False
    assert decision["confirmatory_development_panel_docking_authorized"] is False
    assert decision["exploratory_twenty_receptor_branch_candidate"] is True
    assert all(decision["exploratory_gate_conditions"].values())


def test_stage51b_detects_joint_structure_and_chemistry_association():
    result = json.loads(
        (ROOT / "data/stage51b_ppara_redocking_bias_diagnostic_result.json").read_text()
    )
    association = result["nearest_neighbor_association"]
    assert association["receptor_structure"]["balanced_accuracy"] >= 0.85
    assert association["cognate_ligand_chemistry"]["balanced_accuracy"] >= 0.85
    assert association["receptor_structure"]["permutation_p"] < 0.001
    assert association["cognate_ligand_chemistry"]["permutation_p"] < 0.001
    assert association["dominant_driver"] == "mixed_or_unresolved"


def test_stage51b_audit_verifies_reproducible_outputs():
    audit = json.loads(
        (ROOT / "data/stage51b_ppara_redocking_bias_diagnostic_audit.json").read_text()
    )
    assert audit["status"] == "stage51b_ppara_redocking_bias_diagnostic_audit_ok"
    assert audit["prepared_receptor_count"] == 60
    assert audit["redocking_result_count"] == 180
    assert audit["structural_pair_count"] == 1770
    assert audit["reference_sdf_count"] == 60
    assert audit["new_docking_jobs"] == 0
    for descriptor in audit["outputs"].values():
        path = ROOT / descriptor["path"]
        assert path.is_file()
        assert file_sha256(path) == descriptor["sha256"]
