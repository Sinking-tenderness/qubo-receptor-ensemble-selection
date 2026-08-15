import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_stage32a_pparg_md_functional_landscape import (
    best_mask,
    build_catalog,
    fold_assignments,
    normalize_from_train,
    strong_greedy,
)


def test_fold_assignment_is_balanced_and_deterministic() -> None:
    config = {"folds": {"assignment_seed": "unit-test", "fold_count": 4}}
    ligands = [
        {
            "ligand_id": f"{label}_{index:02d}",
            "label": label,
            "split_group_id": f"{label}_group_{index:02d}",
        }
        for label in ("active", "decoy")
        for index in range(80)
    ]
    first, rows = fold_assignments(ligands, config)
    second, _ = fold_assignments(ligands, config)
    assert np.array_equal(first, second)
    assert len(rows) == 160
    for fold in range(4):
        selected = [ligands[index]["label"] for index in np.flatnonzero(first == fold)]
        assert selected.count("active") == 20
        assert selected.count("decoy") == 20


def test_train_cdf_is_frozen_for_holdout_rows_and_midrank_ties() -> None:
    matrix = np.asarray([[1.0], [1.0], [3.0], [0.0], [4.0]])
    train = np.asarray([True, True, True, False, False])
    normalized = normalize_from_train(matrix, train)[:, 0]
    assert np.allclose(normalized, [0.375, 0.375, 0.75, 0.125, 0.875])


def test_catalog_covers_all_subsets_through_size_six() -> None:
    catalog = build_catalog(16, list(range(1, 7)))
    assert len(catalog["nonempty_masks"]) == 14892
    assert all(int(mask).bit_count() in range(1, 7) for mask in catalog["nonempty_masks"])
    assert all(catalog["column"][int(mask) & (int(mask) - 1)] >= 0 for mask in catalog["nonempty_masks"])


def test_strong_greedy_can_be_distinguished_from_exact_oracle() -> None:
    receptor_ids = ["A", "B", "C", "D", "E", "F"]
    catalog = build_catalog(6, [1, 2, 3])
    utility = np.zeros(len(catalog["nonempty_masks"]), dtype=float)
    by_subset = {tuple(combination): 0.1 for size in (1, 2, 3) for combination in itertools.combinations(receptor_ids, size)}
    by_subset.update({("A", "D"): 0.9, ("B", "E"): 0.9, ("C", "F"): 0.9})
    by_subset.update({("A", "D", "E"): 0.9, ("B", "E", "F"): 0.9, ("C", "D", "F"): 0.9})
    by_subset[("A", "B", "C")] = 1.0
    for mask in catalog["nonempty_masks"]:
        subset = tuple(receptor_ids[index] for index in range(6) if int(mask) & (1 << index))
        utility[int(catalog["column"][int(mask)]) - 1] = by_subset[subset]
    values = {"robust": utility}
    exact = best_mask(values, catalog, receptor_ids, catalog["masks_by_size"][3], 1e-12)
    greedy = strong_greedy(3, values, catalog, receptor_ids, 1e-12)
    assert exact != greedy
    assert exact == (1 << 0) | (1 << 1) | (1 << 2)


def test_stage32a_preregistration_and_result_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/stage32a_pparg_md_functional_landscape_analysis.json").read_text(encoding="ascii"))
    assert config["evidence_timing"]["stage32_enrichment_and_subset_outcomes_known_before_analysis_freeze"] is False
    assert config["landscape"]["subset_sizes"] == [1, 2, 3, 4, 5, 6]
    assert config["landscape"]["bedroc_alpha"] == 20.0
    assert config["stage33_gate"]["minimum_qualifying_subset_sizes"] == 2
    result_path = root / config["outputs"]["result_json"]
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="ascii"))
        assert result["coverage"]["fold_comparison_count"] == 24
        assert result["coverage"]["nonempty_subset_count_per_fold"] == 14892
        assert result["data_boundary"]["fresh_validation_rows_read"] == 0
        assert result["data_boundary"]["quantum_hardware_jobs"] == 0
    audit_path = root / config["outputs"]["audit_json"]
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="ascii"))
        assert audit["status"] == "stage32a_pparg_md_functional_landscape_audit_ok"
        assert audit["checks"]["all_24_exact_and_strong_greedy_selections_recomputed"] is True
