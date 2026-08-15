import json
from pathlib import Path

from scripts.audit_stage19d_pparg_train_comparators import (
    DEFAULT_CONFIG,
    audit,
)
from scripts.build_stage19d_pparg_train_comparator_bundle import bundle_paths


def test_stage19d_outputs_pass_independent_audit() -> None:
    root = Path(__file__).resolve().parents[1]
    result = audit(DEFAULT_CONFIG, root)

    assert result["status"] == (
        "stage19d_pparg_train668_frozen_comparator_audit_ok"
    )
    assert result["coverage"] == {
        "ligand_count": 668,
        "split_group_count": 451,
        "scaffold_count": 452,
        "outer_fold_count": 4,
        "robust_method_count": 16,
        "primary_only_method_count": 2,
        "oof_prediction_count": 54776,
        "oof_metric_count": 82,
        "fold_selection_count": 72,
        "full_selection_count": 18,
        "model_artifact_count": 6,
    }
    assert result["qubo_diagnostics"] == {
        "full_train_exact_equals_qubo_greedy": True,
        "full_train_exact_differs_from_direct_greedy": True,
        "folds_where_exact_differs_from_qubo_greedy": 3,
        "fold_count": 4,
        "noncardinality_quadratic_term_count": 120,
    }


def test_stage19d_result_preserves_exploratory_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root
            / "data/stage19d_pparg_train668_frozen_comparator_result.json"
        ).read_text(encoding="ascii")
    )

    assert result["experiment_class"] == "posthoc_exploratory_train_only"
    assert result["stage18e_confirmatory_gate"] == "closed_failed_14_of_24"
    assert result["data_boundary"] == {
        "validation_rows_read": 0,
        "test_rows_read": 0,
    }
    assert result["primary_oof_comparison"]["qubo_exact_better"] is False
    assert result["strongest_literature_method"] == "edock_rf_all16"


def test_stage19d_core_bundle_is_train_only_and_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = bundle_paths(root)

    assert "data/stage19d_pparg_train668_frozen_comparator_audit.json" in paths
    assert "data/stage19d_pparg_train668_frozen_comparator_result.json" in paths
    assert (
        "results/runs/stage19d_pparg_train668_frozen_comparators/"
        "oof_predictions.csv"
    ) in paths
    assert sum(path.endswith((".joblib", ".json")) and "/models/" in path for path in paths) == 6
    assert not any("fresh_validation" in path.lower() for path in paths)
    assert not any("locked_test" in path.lower() for path in paths)
