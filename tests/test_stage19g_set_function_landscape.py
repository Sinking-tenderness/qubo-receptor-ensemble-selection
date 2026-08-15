import json
from pathlib import Path

import numpy as np

from scripts.diagnose_stage19g_set_function_landscape import (
    best_mask,
    build_mask_catalog,
    greedy_path,
    pairwise_closure,
    submodularity_diagnostics,
)
from scripts.prepare_receptor import file_sha256


def landscape_values(catalog: dict[str, object], function) -> dict[str, np.ndarray]:
    masks = catalog["nonempty_masks"]
    return {
        "robust_composite": np.asarray(
            [function(int(mask)) for mask in masks], dtype=float
        )
    }


def test_mask_catalog_contains_all_subsets_through_six() -> None:
    catalog = build_mask_catalog(16, 6)
    assert len(catalog["nonempty_masks"]) == 14892
    assert len(catalog["masks_by_size"][3]) == 560
    assert len(catalog["masks_by_size"][6]) == 8008
    for mask in catalog["nonempty_masks"]:
        column = int(catalog["column_by_mask"][int(mask)])
        assert column > 0


def test_additive_landscape_is_greedy_and_pairwise_closed() -> None:
    catalog = build_mask_catalog(4, 3)
    weights = [0.7, 0.5, 0.3, 0.1]
    values = landscape_values(
        catalog,
        lambda mask: sum(
            weight for index, weight in enumerate(weights) if mask & (1 << index)
        ),
    )
    receptors = ["R1", "R2", "R3", "R4"]
    path = greedy_path(values, catalog, receptors)
    for size in range(1, 4):
        exact = best_mask(
            values, catalog, receptors, catalog["masks_by_size"][size]
        )
        assert path[size] == exact
    closure = pairwise_closure(values, catalog, receptors, [0.01, 0.05])
    assert abs(closure["rmse"]) < 1e-12
    assert abs(closure["r2"] - 1.0) < 1e-12
    assert closure["predicted_best_regret"] == 0.0
    submodular = submodularity_diagnostics(values, catalog, 4, 1e-9)
    assert submodular["submodularity_violation_count"] == 0
    assert submodular["negative_marginal_count"] == 0


def test_supermodular_pair_can_trap_greedy() -> None:
    catalog = build_mask_catalog(4, 3)
    weights = [0.9, 0.6, 0.6, 0.0]

    def objective(mask: int) -> float:
        value = sum(
            weight for index, weight in enumerate(weights) if mask & (1 << index)
        )
        if mask & (1 << 1) and mask & (1 << 2):
            value += 1.0
        return value

    values = landscape_values(catalog, objective)
    receptors = ["R1", "R2", "R3", "R4"]
    path = greedy_path(values, catalog, receptors)
    exact = best_mask(values, catalog, receptors, catalog["masks_by_size"][2])
    assert path[2] != exact
    submodular = submodularity_diagnostics(values, catalog, 4, 1e-9)
    assert submodular["submodularity_violation_count"] > 0


def test_stage19g_config_freezes_data_boundary_and_route_thresholds() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/stage19g_cross_target_set_function_landscape.json"
    config = json.loads(config_path.read_text(encoding="ascii"))

    assert file_sha256(root / config["implementation"]["path"]) == (
        config["implementation"]["sha256"]
    )
    assert config["diagnostic"]["expected_nonempty_subset_count"] == 14892
    assert config["route_gate"]["minimum_gap_folds_per_target_size"] == 2
    assert config["route_gate"]["bace1_method_amendment_authorized_by_this_stage"] is False
    assert config["evidence_timing"]["bace1_benchmark_docking_started"] is False


def test_stage19g_result_rejects_cross_target_efficacy_route() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root / "data/stage19g_cross_target_set_function_landscape_result.json"
        ).read_text(encoding="ascii")
    )

    assert result["status"] == (
        "stage19g_cross_target_set_function_landscape_complete"
    )
    assert result["decision"]["cross_target_route"] == (
        "no_cross_target_efficacy_qubo_route_authorized"
    )
    assert result["decision"]["bace1_method_amendment_authorized"] is False
    assert all(
        diagnosis["current_k3_gap_fold_count"] == 0
        for diagnosis in result["target_diagnosis"].values()
    )
    assert result["target_diagnosis"]["MK14"]["stable_higher_order_gate"] is True
    assert result["target_diagnosis"]["PPARG"]["stable_higher_order_gate"] is True
    assert result["data_boundary"]["new_docking_jobs"] == 0


def test_stage19g_audit_identifies_frozen_result() -> None:
    root = Path(__file__).resolve().parents[1]
    audit = json.loads(
        (
            root / "data/stage19g_cross_target_set_function_landscape_audit.json"
        ).read_text(encoding="ascii")
    )

    assert audit["status"] == (
        "stage19g_cross_target_set_function_landscape_audit_ok"
    )
    assert audit["checks"]["all_set_utilities_recomputed_independently"] is True
    assert audit["checks"]["failed_cross_target_route_preserved"] is True
    assert audit["coverage"]["triple_rows_recomputed"] == 5600
    result_path = root / audit["result"]["path"]
    assert file_sha256(result_path) == audit["result"]["sha256"]
