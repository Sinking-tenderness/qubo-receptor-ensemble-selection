import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage49_ligand_panels_are_frozen_and_disjoint():
    result = json.loads(
        (ROOT / "data/stage49_ppara_ligand_panel_allocation_summary.json").read_text()
    )
    assert result["status"] == "stage49_ppara_ligand_panels_frozen"
    assert result["decision"]["ligand_allocation_gate_passed"] is True
    assert all(result["disjointness"].values())
    assert result["selected_counts"] == {
        "test_active": 93,
        "test_decoy": 1860,
        "train_active": 187,
        "train_decoy": 187,
        "validation_active": 93,
        "validation_decoy": 1860,
    }


def test_stage49b_independent_structural_audit(tmp_path):
    module = load_script(
        "stage49b_audit", "scripts/audit_stage49b_ppara_structural_pool.py"
    )
    result = module.audit(ROOT, tmp_path / "audit.json")
    assert result["status"] == "stage49b_ppara_structural_pool_independent_audit_ok"
    assert result["coordinate_counts"] == {"audited": 75, "eligible": 66, "excluded": 9}
    assert result["selected_receptor_count"] == 64
    assert result["deterministic_maxmin_order_reproduced"] is True


def test_stage49b_protected_boundaries_remain_closed():
    result = json.loads(
        (ROOT / "data/stage49b_ppara_structural_selection_summary.json").read_text()
    )
    assert all(value == 0 for value in result["data_boundary"].values())
    assert result["decision"]["cognate_redocking_input_preparation_authorized"] is True
    assert result["decision"]["production_docking_authorized"] is False
    assert result["decision"]["fresh_validation_release_authorized"] is False
    assert result["decision"]["quantum_hardware_authorized"] is False


def test_stage50_input_audit_freezes_64_cognate_cases():
    module = load_script(
        "stage50_input",
        "scripts/prepare_stage50_ppara_large_pool_redocking_inputs.py",
    )
    config, _, rows, audit = module.validate_inputs(
        ROOT / "configs/stage50_ppara_large_pool_redocking_input_preparation.json",
        ROOT,
    )
    assert len(rows) == 64
    assert audit["status"] == "audit_only_ok"
    assert audit["receptor_count"] == audit["cognate_ligand_count"] == 64
    assert all(value == 0 for value in audit["data_boundary"].values())
    assert config["expected"]["minimum_prepared_receptor_count"] == 24
    assert config["preparation_protocol"]["add_missing_residues"] is False
    assert config["technical_amendment01"]["same_frozen_receptors_retried"] is True
    assert config["technical_amendment01"]["structural_or_docking_threshold_changed"] is False
