from pathlib import Path

from scripts.run_development_scaffold_cv_gate import (
    candidate_configs,
    load_config,
    make_scaffold_folds,
    method_configs,
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
