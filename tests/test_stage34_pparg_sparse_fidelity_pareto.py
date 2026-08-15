from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage34_screen_completed_without_protected_data() -> None:
    result = json.loads((ROOT / "data/stage34_pparg_sparse_fidelity_pareto_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "stage34_pparg_sparse_fidelity_pareto_complete"
    assert result["screen_statistics"]["cell_count"] == 21
    assert all(value == 0 for value in result["data_boundary"].values())
    assert result["decision"]["quantum_advantage_claim_authorized"] is False


def test_centered_residual_improves_fidelity_but_does_not_override_gate() -> None:
    result = json.loads((ROOT / "data/stage34_pparg_sparse_fidelity_pareto_result.json").read_text(encoding="utf-8"))
    centered = result["encoding_records"]["centered_residual_q01"]
    local = result["encoding_records"]["local_redundancy_q02"]
    assert float(centered["maximum_dense_quality_loss"]) < float(local["maximum_dense_quality_loss"])
    assert centered["quality_gate_passed"] is True
    assert centered["sparsity_gate_passed"] is True
    assert centered["stability_gate_passed"] is False
    assert result["decision"]["small_quantum_annealing_application_pilot_authorized"] is False


def test_stage34_independent_audit_passed() -> None:
    audit = json.loads((ROOT / "data/stage34_pparg_sparse_fidelity_pareto_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "stage34_pparg_sparse_fidelity_pareto_audit_ok"
    assert all(audit["checks"].values())
