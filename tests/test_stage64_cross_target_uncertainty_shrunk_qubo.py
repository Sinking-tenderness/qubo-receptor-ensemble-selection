import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_stage42f_bace1_rank_sensitive_pair_qubo import classical_by_size
from scripts.run_stage64_cross_target_uncertainty_shrunk_qubo import (
    classical_by_size_cached,
)


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="ascii"))


def read_csv(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_stage64_implementation_and_inputs_are_hash_locked():
    config = read_json("configs/stage64_cross_target_uncertainty_shrunk_qubo.json")
    descriptors = list(config["implementation"].values())
    descriptors.extend(config["inputs"].values())
    for target in config["targets"].values():
        descriptors.extend(target["inputs"].values())
    for value in descriptors:
        assert sha256(ROOT / value["path"]) == value["sha256"]


def test_stage64_cached_search_is_identical_to_reference_search():
    rng = np.random.default_rng(20260806)
    singleton = rng.normal(size=9)
    complement = rng.normal(size=(9, 9))
    complement = (complement + complement.T) / 2.0
    np.fill_diagonal(complement, 0.0)
    expected, expected_records = classical_by_size(
        9, 6, singleton, complement, 64
    )
    observed, observed_records = classical_by_size_cached(
        9, 6, singleton, complement, 64
    )
    assert observed == expected
    assert observed_records == expected_records


def test_stage64_reconstructs_baseline_and_completes_full_grid():
    result = read_json("data/stage64_cross_target_uncertainty_shrunk_qubo_result.json")
    assert result["status"] == "stage64_cross_target_uncertainty_shrunk_qubo_complete"
    assert result["candidate_count"] == 10
    assert result["fixed_k_metric_count"] == 960
    assert result["baseline_reproduction_cell_count"] == 96
    assert result["analysis_payload_sha256"] == (
        "AC8666988EBCB33B7741A8FCA41E969DBA60023362BB13B75D1A99AB36F984E2"
    )
    assert sum(
        value["score_row_count"] for value in result["target_input_audits"].values()
    ) == 116532


def test_stage64_shrinkage_improves_v1_but_not_pair_off():
    result = read_json("data/stage64_cross_target_uncertainty_shrunk_qubo_result.json")
    selected = result["selected_candidate"]
    assert selected["candidate_id"] == "pair_scale_0p25"
    assert selected["mean_target_gain_over_baseline_v1"] == pytest.approx(
        0.053638055741386026
    )
    assert selected["worst_target_gain_over_baseline_v1"] == pytest.approx(
        0.028265735366831145
    )
    assert selected["positive_target_count"] == 4
    assert selected["mean_target_gain_over_pair_off"] == pytest.approx(
        -0.008570297129599244
    )
    assert selected["worst_target_gain_over_pair_off"] == pytest.approx(
        -0.036586736841193464
    )
    assert selected["nonnegative_target_count_over_pair_off"] == 1
    global_rows = {
        row["candidate_id"]: row
        for row in read_csv(
            "results/runs/stage64_cross_target_uncertainty_shrunk_qubo/global_summary.csv"
        )
    }
    assert float(global_rows["pair_off"]["mean_target_gain_over_baseline_v1"]) == pytest.approx(
        0.062208352870985265
    )
    assert int(global_rows["pair_off"]["nonnegative_target_count"]) == 4


def test_stage64_freeze_gate_and_claim_boundaries_are_conservative():
    result = read_json("data/stage64_cross_target_uncertainty_shrunk_qubo_result.json")
    gate = result["freeze_gate"]
    assert gate["objective_v2_frozen"] is False
    assert gate["loto_mean_gain_over_baseline_v1"] == pytest.approx(
        0.053638055741386026
    )
    assert gate["loto_mean_gain_over_pair_off"] == pytest.approx(
        -0.00857029712959928
    )
    assert gate["loto_positive_target_count"] == 4
    assert gate["loto_positive_target_count_over_pair_off"] == 1
    assert result["decision"]["stage65_nested_k_authorized"] is False
    assert result["decision"]["same_target_retuning_authorized"] is False
    assert result["decision"]["fresh_validation_authorized"] is False
    assert result["decision"]["quantum_hardware_authorized"] is False
    for key in (
        "fresh_validation_rows_read",
        "locked_test_rows_read",
        "new_docking_jobs",
        "quantum_hardware_jobs",
    ):
        assert result["data_boundary"][key] == 0


def test_stage64_independent_audit_passed():
    audit = read_json("data/stage64_cross_target_uncertainty_shrunk_qubo_audit.json")
    assert audit["status"] == (
        "stage64_cross_target_uncertainty_shrunk_qubo_independent_audit_ok"
    )
    assert audit["baseline_reproduction_cells_independently_verified"] == 96
    assert audit["candidate_summaries_independently_recomputed"] is True
    assert audit["loto_selection_independently_recomputed"] is True
    assert audit["freeze_gate_independently_recomputed"] is True
    assert audit["objective_v2_frozen"] is False
    assert audit["source_result"]["sha256"] == sha256(
        ROOT / "data/stage64_cross_target_uncertainty_shrunk_qubo_result.json"
    )
