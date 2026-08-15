from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage36_exact_screen_completed_without_protected_data() -> None:
    result = json.loads((ROOT / "data/stage36_pparg_consensus_objective_landscape_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "stage36_pparg_consensus_objective_landscape_complete"
    assert result["decision"]["exact_landscape_screen_complete"] is True
    assert result["decision"]["quantum_hardware_authorized"] is False
    assert all(value == 0 for value in result["data_boundary"].values())


def test_stage36_does_not_select_degenerate_consensus_objective() -> None:
    result = json.loads((ROOT / "data/stage36_pparg_consensus_objective_landscape_result.json").read_text(encoding="utf-8"))
    assert result["decision"]["candidate_objective_found"] is False
    assert result["decision"]["selected_objective_id"] is None
    assert result["decision"]["stage37_sparse_qubo_encoding_authorized"] is False


def test_stage36_independent_audit_passed() -> None:
    audit = json.loads((ROOT / "data/stage36_pparg_consensus_objective_landscape_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "stage36_pparg_consensus_objective_landscape_audit_ok"
    assert audit["checks"]["cross_start_threshold_degeneracy_confirmed"] is True
    assert all(audit["checks"].values())
