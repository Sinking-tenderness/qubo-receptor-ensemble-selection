import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_stage25_config_is_frozen_and_structure_only() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/stage25_bace1_prospective_structure_replication.json").read_text(encoding="ascii")
    )
    assert config["evidence_timing"]["bace1_stage25_objective_outcomes_known_before_freeze"] is False
    assert config["evidence_timing"]["new_docking_jobs"] is False
    assert config["objective"]["neighborhood_fraction"] == 0.10
    assert config["objective"]["k"] == 8
    assert config["objective"]["diversity_weight"] == 0.15


def test_stage25_result_and_audit() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "data/stage25_bace1_prospective_structure_replication_result.json").read_text(encoding="ascii")
    )
    audit = json.loads(
        (root / "data/stage25_bace1_prospective_structure_replication_audit.json").read_text(encoding="ascii")
    )
    assert result["status"] == "stage25_bace1_prospective_structure_replication_complete"
    assert result["target_record"]["candidate_count"] == 49
    assert result["target_record"]["within_tolerance_batch_fraction"] == 1.0
    assert result["decision"]["prospective_structure_replication_gate_passed"] is False
    assert audit["status"] == "stage25_bace1_prospective_structure_replication_audit_ok"
    assert audit["coverage"]["read_rows_recomputed"] == 256
    assert audit["coverage"]["batch_rows_recomputed"] == 4
