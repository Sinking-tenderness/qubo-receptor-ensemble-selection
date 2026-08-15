from pathlib import Path

from scripts.prepare_receptor import file_sha256
from scripts.experimental.unidock.build_stage11_mk14_fresh_validation_bundle import (
    CONFIG,
    bundle_paths,
)
from scripts.experimental.unidock.evaluate_stage11_mk14_fresh_validation_confirmation import (
    paired_bootstrap,
    seed_matrices,
)
from scripts.experimental.unidock.prepare_stage11_mk14_fresh_validation_inputs import (
    validate_source_inputs,
)
from scripts.experimental.unidock.run_stage11_mk14_fresh_validation_confirmation import (
    validate_config,
)
from scripts.experimental.unidock.run_unidock_gpu_equivalence import read_json


def test_stage11_source_inputs_and_candidate_provenance_are_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    config = read_json(root / CONFIG)
    receptors, ligands, macrocycles, audit = validate_source_inputs(root, config)

    assert len(receptors) == 6
    assert len(ligands) == 1576
    assert len(macrocycles) == 54
    assert {row["label"] for row in macrocycles} == {"decoy"}
    assert audit["train_score_rows"] == 0
    assert audit["test_rows"] == 0
    assert audit["candidate_subsets"] == {
        "exact_pair_synergy": [
            "MK14_2BAJ_aligned",
            "MK14_2QD9_reference",
            "MK14_3BV2_aligned",
        ],
        "qubo_forward_greedy": [
            "MK14_3BV2_aligned",
            "MK14_3ITZ_aligned",
            "MK14_3KQ7_aligned",
        ],
        "direct_bedroc_greedy": [
            "MK14_2BAJ_aligned",
            "MK14_2QD9_reference",
            "MK14_3KQ7_aligned",
        ],
        "full_train_exact_secondary": [
            "MK14_2BAJ_aligned",
            "MK14_3BV2_aligned",
            "MK14_4AAC_aligned",
        ],
    }


def test_stage11_config_and_implementation_hashes_are_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    config = read_json(root / CONFIG)
    validate_config(config)
    for descriptor in config["implementation"].values():
        path = root / str(descriptor["path"])
        assert file_sha256(path) == str(descriptor["sha256"]).upper()


def test_stage11_seed_matrix_construction_and_paired_bootstrap() -> None:
    ligands = [
        {"ligand_id": "A", "label": "active", "selection_role": "validation"},
        {"ligand_id": "D", "label": "decoy", "selection_role": "validation"},
    ]
    score_rows = []
    for seed_id in ("seed0", "seed1", "seed2"):
        for ligand_id, label in (("A", "active"), ("D", "decoy")):
            for receptor_id in ("R1", "R2"):
                score_rows.append(
                    {
                        "seed_id": seed_id,
                        "ligand_id": ligand_id,
                        "label": label,
                        "receptor_id": receptor_id,
                        "gpu_score": "-8.0" if ligand_id == "A" else "-4.0",
                    }
                )
    matrices = seed_matrices(score_rows, ligands, ["R1", "R2"])
    assert matrices["seed2"][0]["R1"] == "-8.0"
    assert matrices["seed0"][1]["R2"] == "-4.0"

    candidate = {}
    comparator = {}
    groups = {}
    for index in range(20):
        ligand_id = f"L{index:02d}"
        label = "active" if index < 10 else "decoy"
        candidate[ligand_id] = {
            "label": label,
            "score": -10.0 if label == "active" else 0.0,
        }
        comparator[ligand_id] = {
            "label": label,
            "score": 0.0 if label == "active" else -10.0,
        }
        groups[ligand_id] = ligand_id
    bootstrap = paired_bootstrap(
        {"candidate": candidate, "control": comparator},
        groups,
        "candidate",
        ["control"],
        100,
        123,
    )
    assert bootstrap["valid_replicates"] == 100
    assert bootstrap["deltas"]["control"]["lower_95pct"] > 0.0


def test_stage11_bundle_contains_structures_but_no_old_validation_scores() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = bundle_paths(root)

    assert sum(path.endswith(".pdbqt") for path in paths) == 1582
    assert sum(path.endswith(".sdf") for path in paths) == 54
    assert not any("locked_test" in path.lower() for path in paths)
    assert not any("fresh_validation_e32_" in path.lower() for path in paths)
    assert not any("fresh_validation_result.json" in path.lower() for path in paths)
