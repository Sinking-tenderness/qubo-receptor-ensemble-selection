import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_stage101_exposes_the_failed_marginal_signal():
    result = json.loads((ROOT / "data/stage101_marginal_transfer_gate_result.json").read_text(encoding="utf-8"))
    k2 = result["correlations"]["k1_to_k2"]
    assert k2["spearman_r"] < -0.4
    assert k2["spearman_p"] < 0.05
    assert k2["sign_accuracy"] < 0.5
    assert k2["false_positive_count"] >= 8


def test_stage101_loto_never_trains_on_the_held_target():
    rows = read_csv(ROOT / "results/runs/stage101_marginal_transfer_gate/marginal_edges.csv")
    targets = {row["target_id"] for row in rows}
    assert len(rows) == 50
    for row in rows:
        training = set(row["loto_training_targets"].split("|"))
        assert row["target_id"] not in training
        assert training == targets - {row["target_id"]}


def test_stage101_keeps_the_scientific_boundary():
    result = json.loads((ROOT / "data/stage101_marginal_transfer_gate_result.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "data/stage101_marginal_transfer_gate_audit.json").read_text(encoding="utf-8"))
    assert result["oracle_ceiling_mean_target_gain"] > 0.04
    assert result["decision"]["hardware_authorized"] is False
    assert result["decision"]["same_matrix_threshold_tuning_allowed"] is False
    assert result["data_boundary"]["new_docking_jobs"] == 0
    assert audit["status"] == "stage101_independent_audit_ok"
