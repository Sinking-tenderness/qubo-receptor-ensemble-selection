import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_stage67_bedroc_rankbin_qubo import RankUtilityObjective


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="ascii"))


def read_csv(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_stage67_implementation_and_sources_are_hash_locked():
    config = read_json("configs/stage67_bedroc_rankbin_qubo.json")
    for section in ("implementation", "inputs"):
        for value in config[section].values():
            assert sha256(ROOT / value["path"]) == value["sha256"]


def test_stage67_rankbin_identity_matches_cumulative_binary_levels():
    utilities = np.asarray([0.01, 0.13, 0.51, 0.88, 1.0])
    for bin_count in (4, 8, 16, 32):
        quantized = np.floor(bin_count * utilities + 1e-12) / bin_count
        cumulative = sum(
            (utilities >= level / bin_count).astype(float)
            for level in range(1, bin_count + 1)
        ) / bin_count
        assert np.array_equal(quantized, cumulative)
        assert np.max(utilities - quantized) < 1.0 / bin_count + 1e-12


def test_stage67_objective_quantization_is_deterministic():
    ranks = np.asarray(
        [
            [[0.01, 0.10], [0.20, 0.02], [0.05, 0.30], [0.40, 0.04]],
            [[0.02, 0.11], [0.21, 0.03], [0.06, 0.31], [0.41, 0.05]],
            [[0.03, 0.12], [0.22, 0.04], [0.07, 0.32], [0.42, 0.06]],
        ],
        dtype=float,
    )
    labels = np.asarray([1, 1, 0, 0], dtype=int)
    mask = np.ones(4, dtype=bool)
    continuous = RankUtilityObjective(ranks, labels, mask, 20.0, None)
    b32 = RankUtilityObjective(ranks, labels, mask, 20.0, 32)
    first = b32.score((0, 1))
    second = b32.score((1, 0))
    assert first == second
    assert abs(first[0] - continuous.score((0, 1))[0]) < 2.0 / 32


def test_stage67_dimensions_and_pair_off_reproduction():
    result = read_json("data/stage67_bedroc_rankbin_qubo_result.json")
    assert result["status"] == "stage67_bedroc_rankbin_qubo_complete"
    assert result["objective_count"] == 5
    assert result["fixed_k_metric_count"] == 1056
    assert result["pair_off_reproduction_cell_count"] == 96
    assert result["analysis_payload_sha256"] == (
        "9B3F1D132C1DB395F841CD86065C18609F871AE08F7119845096A20C697E5D69"
    )


def test_stage67_continuous_objective_ceiling_fails_all_targets():
    result = read_json("data/stage67_bedroc_rankbin_qubo_result.json")
    continuous = result["continuous_reference"]
    assert continuous["mean_target_gain_over_pair_off"] == pytest.approx(
        -0.05992462215505179
    )
    assert continuous["worst_target_gain_over_pair_off"] == pytest.approx(
        -0.1404400299244411
    )
    assert continuous["nonnegative_target_count_over_pair_off"] == 0
    assert result["route_gate"]["continuous_objective_supported"] is False


def test_stage67_b32_is_faithful_but_cannot_rescue_performance():
    result = read_json("data/stage67_bedroc_rankbin_qubo_result.json")
    rankbin = result["rankbin_reference"]
    assert rankbin["mean_subset_jaccard_vs_continuous"] == pytest.approx(
        0.8967261904761905
    )
    assert rankbin["mean_absolute_train_quantization_error"] == pytest.approx(
        0.006530314635281113
    )
    assert rankbin["mean_target_gain_over_pair_off"] == pytest.approx(
        -0.05487163479988352
    )
    assert rankbin["nonnegative_target_count_over_pair_off"] == 0
    assert result["route_gate"]["rankbin_qubo_freeze_authorized"] is False
    assert result["decision"]["quantum_hardware_authorized"] is False


def test_stage67_target_losses_and_model_scale_are_frozen():
    rows = {
        row["target_id"]: row
        for row in read_csv(
            "results/runs/stage67_bedroc_rankbin_qubo/target_summary.csv"
        )
        if row["objective_id"] == "rankbin_b32"
    }
    assert float(rows["BACE1"]["mean_gain_over_pair_off"]) == pytest.approx(
        -0.013311715239980204
    )
    assert float(rows["PPARG"]["mean_gain_over_pair_off"]) == pytest.approx(
        -0.12395698696592901
    )
    assert float(rows["PPARA"]["mean_gain_over_pair_off"]) == pytest.approx(
        -0.04352503576229351
    )
    assert float(rows["PPARD"]["mean_gain_over_pair_off"]) == pytest.approx(
        -0.03869280123133138
    )
    result = read_json("data/stage67_bedroc_rankbin_qubo_result.json")
    model = result["qubo_model_audit"]
    assert model["total_variable_count_at_b32_k3"] == 67389
    assert model["maximum_target_variable_count_at_b32_k3"] == 21903
    assert model["maximum_factorized_energy_residual"] < 1e-8


def test_stage67_independent_audit_passed():
    audit = read_json("data/stage67_bedroc_rankbin_qubo_audit.json")
    assert audit["status"] == "stage67_bedroc_rankbin_qubo_independent_audit_ok"
    assert audit["pair_off_reproduction_cells_independently_verified"] == 96
    assert audit["same_objective_search_cells_independently_verified"] == 480
    assert audit["factorized_qubo_models_independently_checked"] == 4
    assert audit["continuous_objective_supported"] is False
    assert audit["rankbin_qubo_freeze_authorized"] is False
    assert audit["source_result"]["sha256"] == sha256(
        ROOT / "data/stage67_bedroc_rankbin_qubo_result.json"
    )
