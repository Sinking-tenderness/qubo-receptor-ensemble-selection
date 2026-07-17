from pathlib import Path

import pytest

from scripts.run_development_scaffold_cv_gate import (
    add_consensus_constraints,
    candidate_configs,
    consensus_core_from_inner_subsets,
    load_config,
    make_scaffold_folds,
    method_configs,
    stable_receptors_from_inner_subsets,
)


CONFIG_PATH = Path(
    "configs/stage03_cdk2_af2_md_100a100d_development_scaffold_cv_gate.json"
)
EXPANDED_CONFIG_PATH = Path(
    "configs/stage04_cdk2_expanded16_development_scaffold_cv_gate.json"
)
PAIR_CONFIG_PATH = Path(
    "configs/stage04_cdk2_expanded16_pair_bedroc_development_scaffold_cv_gate.json"
)
STABILITY_CONFIG_PATH = Path(
    "configs/stage04_cdk2_expanded16_stability_development_scaffold_cv_gate.json"
)
CONSENSUS_CONFIG_PATH = Path(
    "configs/stage04_cdk2_expanded16_consensus_development_scaffold_cv_gate.json"
)
CORE_PLUS_ONE_CONFIG_PATH = Path(
    "configs/stage04_cdk2_expanded16_core_plus_one_development_scaffold_cv_gate.json"
)
CORE_PLUS_TWO_CONFIG_PATH = Path(
    "configs/stage04_cdk2_expanded16_core_plus_two_development_scaffold_cv_gate.json"
)
FIXED2_MEAN_CONFIG_PATH = Path(
    "configs/stage04_cdk2_expanded16_fixed2_mean_development_scaffold_cv_gate.json"
)


def test_fixed_development_cv_config_preregisters_test_lock_and_pass_rule():
    config = load_config(CONFIG_PATH)
    assert config["cross_validation"]["development_splits"] == [
        "train",
        "validation",
    ]
    assert config["cross_validation"]["evaluate_locked_test"] is False
    assert config["model"]["subset_sizes"] == [1, 2, 3]
    assert config["acceptance"]["minimum_primary_bedroc_delta"] == 0.02
    assert (
        config["acceptance"]["minimum_primary_bedroc_bootstrap_ci95_low"]
        == 0.0
    )


def test_scaffold_fold_assignment_never_splits_a_scaffold_group():
    rows = [
        {
            "ligand_id": "A1",
            "label": "active",
            "scaffold_smiles": "SA",
        },
        {
            "ligand_id": "A2",
            "label": "active",
            "scaffold_smiles": "SA",
        },
        {
            "ligand_id": "A3",
            "label": "active",
            "scaffold_smiles": "SB",
        },
        {
            "ligand_id": "A4",
            "label": "active",
            "scaffold_smiles": "SC",
        },
        {
            "ligand_id": "D1",
            "label": "decoy",
            "scaffold_smiles": "SD",
        },
        {
            "ligand_id": "D2",
            "label": "decoy",
            "scaffold_smiles": "SE",
        },
        {
            "ligand_id": "D3",
            "label": "decoy",
            "scaffold_smiles": "SF",
        },
        {
            "ligand_id": "D4",
            "label": "decoy",
            "scaffold_smiles": "SG",
        },
        {
            "ligand_id": "D5",
            "label": "decoy",
            "scaffold_smiles": "SH",
        },
        {
            "ligand_id": "D6",
            "label": "decoy",
            "scaffold_smiles": "SI",
        },
    ]
    assignment = make_scaffold_folds(rows, 3, 42)
    assert assignment["A1"] == assignment["A2"]
    for fold in range(3):
        labels = {
            row["label"]
            for row in rows
            if assignment[row["ligand_id"]] == fold
        }
        assert labels == {"active", "decoy"}


def test_preregistered_qubo_grid_has_expected_family_sizes():
    model = load_config(CONFIG_PATH)["model"]
    coverage = candidate_configs(model, "coverage_qubo")
    discriminative = candidate_configs(model, "discriminative_qubo")
    assert len(coverage) == 111
    assert len(discriminative) == 333
    assert all(
        config["weights"]["decoy_exposure"] == 0.0 for config in coverage
    )
    assert all(
        config["weights"]["decoy_exposure"] > 0.0
        for config in discriminative
    )


def test_pair_bedroc_qubo_grid_adds_pair_ensemble_utility_weight():
    model = dict(load_config(CONFIG_PATH)["model"])
    model["families"] = [
        "coverage_qubo",
        "discriminative_qubo",
        "pair_bedroc_qubo",
    ]
    model["weight_grids"] = {
        **model["weight_grids"],
        "ensemble_pair_utility": [0.5, 1.0, 2.0],
    }
    candidates = candidate_configs(model, "pair_bedroc_qubo")

    assert len(candidates) == 327
    assert all(
        config["weights"]["ensemble_pair_utility"] > 0.0
        for config in candidates
        if config["target_size"] > 1
    )


def test_expanded_config_preserves_gate_and_excludes_locked_scores():
    config = load_config(EXPANDED_CONFIG_PATH)
    assert len(config["receptor_ids"]) == 16
    assert config["cross_validation"]["matrices_exclude_locked_split"] is True
    assert config["cross_validation"]["evaluate_locked_test"] is False
    assert config["model"]["subset_sizes"] == [1, 2, 3]
    assert config["acceptance"]["minimum_primary_bedroc_delta"] == 0.02
    assert method_configs(config["model"], 16)["all_receptors"] == [
        {
            "family": "all_receptors",
            "target_size": 16,
            "aggregation": "min_score",
        },
        {
            "family": "all_receptors",
            "target_size": 16,
            "aggregation": "mean_score",
        },
    ]


def test_pair_bedroc_config_adds_pair_family_without_unlocking_test():
    config = load_config(PAIR_CONFIG_PATH)

    assert config["model"]["families"] == [
        "coverage_qubo",
        "discriminative_qubo",
        "pair_bedroc_qubo",
    ]
    assert config["cross_validation"]["evaluate_locked_test"] is False
    assert config["cross_validation"]["matrices_exclude_locked_split"] is True
    methods = method_configs(config["model"], 16)
    assert "pair_bedroc_qubo" in methods
    assert len(methods["pair_bedroc_qubo"]) == 327


def test_stability_config_uses_inner_contexts_and_keeps_test_locked():
    config = load_config(STABILITY_CONFIG_PATH)

    assert config["model"]["stability_weight"] == 1.0
    assert "stability_qubo" in config["model"]["families"]
    assert config["cross_validation"]["evaluate_locked_test"] is False
    methods = method_configs(config["model"], 16)
    assert len(methods["stability_qubo"]) == 111
    assert all(
        trial["weights"]["stability"] == 1.0
        for trial in methods["stability_qubo"]
    )


def test_consensus_receptors_require_two_of_three_inner_selections():
    receptors = ["R1", "R2", "R3", "R4"]
    required = stable_receptors_from_inner_subsets(
        [["R1", "R2"], ["R1", "R3"], ["R1", "R4"]],
        receptors,
        2.0 / 3.0,
    )
    assert required == ("R1",)


def test_consensus_candidate_grid_uses_only_two_and_three_receptor_budgets():
    model = dict(load_config(EXPANDED_CONFIG_PATH)["model"])
    model["families"] = [
        "coverage_qubo",
        "discriminative_qubo",
        "consensus_qubo",
    ]
    model["consensus_min_inner_frequency"] = 2.0 / 3.0
    model["consensus_subset_sizes"] = [2, 3]
    candidates = candidate_configs(model, "consensus_qubo")
    assert {candidate["target_size"] for candidate in candidates} == {2, 3}
    assert all(
        candidate["weights"]["decoy_exposure"] == 0.0
        for candidate in candidates
    )
    constrained = add_consensus_constraints(candidates[:1], ("R1", "R2"))
    assert constrained[0]["required_receptors"] == ["R1", "R2"]
    assert "required_receptors" not in candidates[0]
    constrained = add_consensus_constraints(candidates, ("R1", "R2", "R3"))
    assert {candidate["target_size"] for candidate in constrained} == {3}


def test_consensus_config_preregisters_inner_frequency_and_keeps_test_locked():
    config = load_config(CONSENSUS_CONFIG_PATH)
    assert config["model"]["consensus_subset_sizes"] == [2, 3]
    assert config["model"]["consensus_min_inner_frequency"] == pytest.approx(
        2.0 / 3.0
    )
    assert "consensus_qubo" in config["model"]["families"]
    assert config["cross_validation"]["evaluate_locked_test"] is False
    assert config["cross_validation"]["matrices_exclude_locked_split"] is True


def test_core_plus_one_selects_two_qualified_receptors_and_one_residual_slot():
    receptors = ["R1", "R2", "R3", "R4"]
    core = consensus_core_from_inner_subsets(
        [["R1", "R2"], ["R1", "R2"], ["R1", "R3"]],
        receptors,
        2.0 / 3.0,
        2,
    )
    assert core == ("R1", "R2")
    model = dict(load_config(CORE_PLUS_ONE_CONFIG_PATH)["model"])
    methods = method_configs(model, 16)
    assert {candidate["target_size"] for candidate in methods["core_plus_one_qubo"]} == {3}
    assert len(methods["core_plus_one_qubo"]) == 54


def test_core_plus_two_uses_one_core_and_two_residual_slots():
    model = dict(load_config(CORE_PLUS_TWO_CONFIG_PATH)["model"])
    methods = method_configs(model, 16)
    assert {candidate["target_size"] for candidate in methods["core_plus_two_qubo"]} == {3}
    assert len(methods["core_plus_two_qubo"]) == 54


def test_fixed2_mean_config_reduces_coverage_model_selection_space():
    config = load_config(FIXED2_MEAN_CONFIG_PATH)
    methods = method_configs(config["model"], 16)
    assert config["model"]["subset_sizes"] == [2]
    assert config["model"]["aggregation_methods"] == ["mean_score"]
    assert len(methods["coverage_qubo"]) == 27
