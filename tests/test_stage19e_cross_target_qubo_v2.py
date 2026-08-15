import json
from pathlib import Path

import numpy as np

from scripts.diagnose_stage19e_cross_target_qubo_v2 import (
    alpha_summary_key,
    vectorized_bedroc,
)
from scripts.build_stage19e_cross_target_qubo_v2_bundle import bundle_paths
from scripts.evaluate_virtual_screening import bedroc
from scripts.prepare_receptor import file_sha256


def test_vectorized_bedroc_matches_scalar_reference_with_ties() -> None:
    scores = np.asarray(
        [
            [-2.0, -1.0, -2.0],
            [-2.0, -3.0, -1.0],
            [-1.0, -2.0, -3.0],
            [-4.0, -4.0, -4.0],
            [-3.0, -2.0, -2.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=float,
    )
    labels = np.asarray([1, 0, 1, 0, 1, 0], dtype=int)
    actual = vectorized_bedroc(scores, labels, 20.0)

    expected = []
    for column in range(scores.shape[1]):
        ranked = sorted(
            (
                {
                    "ligand_id": f"L{index:02d}",
                    "binary_label": int(labels[index]),
                    "score": float(scores[index, column]),
                }
                for index in range(len(labels))
            ),
            key=lambda row: (row["score"], row["ligand_id"]),
        )
        expected.append(bedroc(ranked, 20.0))

    assert np.allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_alpha_selection_key_prefers_validation_then_stability() -> None:
    better_mean = {
        "mean_validation_robust_composite": 0.8,
        "worst_validation_robust_composite": 0.5,
        "mean_validation_rank_spearman": 0.4,
        "alpha": 10.0,
    }
    better_worst = {
        "mean_validation_robust_composite": 0.7,
        "worst_validation_robust_composite": 0.6,
        "mean_validation_rank_spearman": 0.9,
        "alpha": 0.1,
    }
    assert min((better_mean, better_worst), key=alpha_summary_key) is better_mean


def test_stage19e_config_is_frozen_before_bace1_docking() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root / "configs/stage19e_cross_target_qubo_v2_nested_diagnostic.json"
    )
    config = json.loads(config_path.read_text(encoding="ascii"))

    assert file_sha256(root / config["implementation"]["path"]) == (
        config["implementation"]["sha256"]
    )
    assert config["evidence_timing"]["bace1_benchmark_docking_started"] is False
    assert config["development_support_gate"]["all_checks_required"] is True
    assert config["development_support_gate"][
        "minimum_positive_folds_of_eight"
    ] == 5
    for target in config["targets"].values():
        assert not any(
            marker in descriptor["path"].lower()
            for descriptor in target["inputs"].values()
            for marker in ("fresh_validation", "locked_test")
        )


def test_stage19e_result_and_independent_audit_reject_v2() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root
            / "data/stage19e_cross_target_qubo_v2_nested_diagnostic_result.json"
        ).read_text(encoding="ascii")
    )
    audit = json.loads(
        (
            root
            / "data/stage19e_cross_target_qubo_v2_nested_diagnostic_audit.json"
        ).read_text(encoding="ascii")
    )

    assert result["status"] == (
        "stage19e_quadratic_v2_not_supported_do_not_amend_bace1"
    )
    assert result["gate"]["passed"] is False
    assert result["gate"]["bace1_v2_amendment_authorized"] is False
    assert result["data_boundary"]["bace1_docking_rows_read"] == 0
    assert audit["status"] == (
        "stage19e_cross_target_qubo_v2_nested_diagnostic_audit_ok"
    )
    assert audit["checks"]["all_outer_method_metrics_scalar_recomputed"] is True
    assert audit["checks"]["failed_gate_reproduced"] is True


def test_stage19e_bundle_contains_no_protected_panel() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = bundle_paths(root)

    assert "data/stage19e_cross_target_qubo_v2_nested_diagnostic_result.json" in paths
    assert "data/stage19e_cross_target_qubo_v2_nested_diagnostic_audit.json" in paths
    assert "data/stage19e_cross_target_qubo_v2_algorithm_record.json" in paths
    assert not any("fresh_validation" in path.lower() for path in paths)
    assert not any("locked_test" in path.lower() for path in paths)
