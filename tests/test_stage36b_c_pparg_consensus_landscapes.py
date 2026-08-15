from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def test_stage36b_creates_shared_states_and_rugged_landscapes() -> None:
    result = load("stage36b_pparg_start_centered_consensus_landscape_result.json")
    assert result["state_statistics"]["8"]["cross_start_shared_frame_fraction"] >= 0.9
    maximum_local = max(
        int(row["strict_local_optimum_count"])
        for objective in result["objective_records"].values()
        for row in objective["cohorts"].values()
    )
    assert maximum_local >= 50
    assert result["decision"]["stage37_sparse_qubo_encoding_authorized"] is False


def test_stage36c_primary_is_heldout_and_not_replaced_by_control() -> None:
    result = load("stage36c_pparg_consensus_blend_replication_result.json")
    assert result["decision"]["primary_objective_id"] == "consensus_support_blend_v1"
    assert result["decision"]["primary_passing_cohort_count"] == 0
    assert result["decision"]["primary_difficulty_replication_passed"] is False
    assert result["decision"]["stage37_sparse_qubo_encoding_authorized"] is False


def test_stage36b_and_stage36c_audits_pass() -> None:
    audit_b = load("stage36b_pparg_start_centered_consensus_landscape_audit.json")
    audit_c = load("stage36c_pparg_consensus_blend_replication_audit.json")
    assert audit_b["status"] == "stage36b_pparg_start_centered_consensus_landscape_audit_ok"
    assert audit_c["status"] == "stage36c_pparg_consensus_blend_replication_audit_ok"
    assert all(audit_b["checks"].values())
    assert all(audit_c["checks"].values())
