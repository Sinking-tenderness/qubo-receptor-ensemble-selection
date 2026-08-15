from __future__ import annotations

import json
from pathlib import Path

from scripts import diagnose_stage41d_bace1_redocking_qualified_coverage as stage41d


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage41d_bace1_redocking_qualified_structural_coverage.json"


def test_stage41d_quantile() -> None:
    assert stage41d.quantile([0.0, 10.0], 0.25) == 2.5
    assert stage41d.quantile([3.0], 0.95) == 3.0


def test_stage41d_result_is_reproducible() -> None:
    result = stage41d.run(CONFIG, ROOT)

    assert result["status"] == "stage41d_conditional_go_new_posthoc_development_route"
    assert result["metrics"]["passing_receptor_count"] == 34
    assert result["metrics"]["failing_receptor_count"] == 15
    assert result["metrics"]["pairwise_q95_retention"] > 0.97
    assert result["metrics"]["diameter_retention"] > 0.84
    assert all(result["criterion_checks"].values())
    assert result["prospective_stage42_pair_count"] == 27132
    assert result["state_count_by_k"]["6"] == 1344904
    assert result["evidence_timing"]["structural_coverage_metrics_known_when_criteria_frozen"] is True


def test_stage41d_distance_matrix_is_complete() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = stage41d.read_csv(ROOT / config["inputs"]["structural_distances"]["path"])
    gate = stage41d.read_csv(ROOT / config["inputs"]["stage41c_gate_results"]["path"])
    distances = stage41d.build_distances(rows, {row["conformer_id"] for row in gate})

    assert len(distances) == 1176
