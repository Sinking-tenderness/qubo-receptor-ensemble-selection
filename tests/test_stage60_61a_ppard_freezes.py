import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text())


def read_csv(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_stage60_freezes_exact_pre_ppard_rank_pair_qubo():
    result = read_json("data/stage60_ppard_transferred_qubo_freeze_result.json")
    audit = read_json("data/stage60_ppard_transferred_qubo_freeze_audit.json")
    stage42f = read_json("data/stage42f_bace1_rank_sensitive_pair_qubo_result.json")
    stage53 = read_json("data/stage53_ppara_large_pool_qubo_transfer_result.json")
    assert result["status"] == "stage60_ppard_transferred_qubo_and_k_rule_frozen"
    assert audit["status"] == "stage60_ppard_transferred_qubo_independent_audit_ok"
    assert result["transferred_objective"] == stage42f["objective"]
    assert result["transferred_objective"] == stage53["objectives"]["rank_pair_qubo"]
    assert audit["objective_coefficient_change_count"] == 0
    assert result["nested_cv"]["k_selection_rule"] == "one_standard_error_smallest_k"


def test_stage60_nested_folds_are_balanced_and_group_isolated():
    outer = read_csv(
        "data/processed/stage60_ppard_full_development_outer_fold_assignments.csv"
    )
    inner = read_csv(
        "data/processed/stage60_ppard_full_development_inner_fold_assignments.csv"
    )
    assert len(outer) == 240
    assert Counter((row["outer_fold"], row["label"]) for row in outer) == Counter(
        {(str(fold), label): 30 for fold in range(4) for label in ("active", "decoy")}
    )
    outer_by_id = {row["ligand_id"]: row["outer_fold"] for row in outer}
    for column in ("split_group_id", "scaffold_smiles"):
        groups = defaultdict(set)
        for row in outer:
            groups[row[column]].add(row["outer_fold"])
        assert max(map(len, groups.values())) == 1
    assert len(inner) == 720
    for outer_fold in range(4):
        rows = [row for row in inner if row["outer_fold"] == str(outer_fold)]
        assert len(rows) == 180
        assert all(outer_by_id[row["ligand_id"]] != str(outer_fold) for row in rows)
        assert Counter((row["inner_fold"], row["label"]) for row in rows) == Counter(
            {(str(fold), label): 30 for fold in range(3) for label in ("active", "decoy")}
        )


def test_stage60_preserves_claim_boundaries():
    result = read_json("data/stage60_ppard_transferred_qubo_freeze_result.json")
    decision = result["decision"]
    assert decision["remaining_development_ligand_preparation_authorized"] is True
    assert decision["remaining_development_docking_authorized"] is True
    assert decision["fresh_validation_authorized"] is False
    assert decision["locked_test_authorized"] is False
    assert decision["quantum_hardware_authorized"] is False
    assert decision["qubo_superiority_claim_authorized"] is False
    assert result["data_boundary"]["ppard_pilot_score_rows_read"] == 0


def test_stage61a_remaining144_exactly_complements_pilot96():
    train = read_csv("data/processed/stage56_ppard_train240_ligand_manifest.csv")
    pilot = read_csv("data/processed/stage56_ppard_pilot96_ligand_manifest.csv")
    remaining = read_csv(
        "data/processed/stage61a_ppard_remaining144_ligand_manifest.csv"
    )
    freeze = read_json("data/stage61a_ppard_remaining144_manifest_freeze.json")
    assert freeze["status"] == "stage61a_ppard_remaining144_manifest_frozen"
    assert len(train) == 240 and len(pilot) == 96 and len(remaining) == 144
    assert remaining == [row for row in train if row["pilot_selected"] == "False"]
    assert Counter(row["label"] for row in remaining) == {
        "active": 72,
        "decoy": 72,
    }
    assert not ({row["ligand_id"] for row in pilot} & {row["ligand_id"] for row in remaining})
    assert {row["pilot_selected"] for row in remaining} == {"False"}


def test_stage61a_config_prepares_no_protected_or_pilot_rows():
    config = read_json(
        "configs/stage61a_ppard_remaining144_unidock_input_preparation.json"
    )
    assert config["expected"]["ligand_count"] == 144
    assert config["expected"]["future_pair_count"] == 12528
    assert config["source"]["required_row_values"]["pilot_selected"] == "False"
    paths = [record["path"].lower() for record in config["inputs"].values()]
    assert all("fresh_validation" not in path for path in paths)
    assert all("locked_test" not in path for path in paths)
    assert all("protected/" not in path for path in paths)
