from pathlib import Path

from scripts.experimental.unidock.audit_stage09_mk14_train696_production import (
    compare_matrix,
)
from scripts.experimental.unidock.build_stage09_mk14_train696_production_bundle import (
    CONFIG,
    FIXED_PATHS,
    bundle_paths,
)
from scripts.experimental.unidock.run_stage09_mk14_train696_production import (
    matrix_rows,
    read_json,
    selected_records,
    validate_inputs,
)


def test_stage09_actual_inputs_are_complete_train_only_and_macrocycle_safe() -> None:
    root = Path(__file__).resolve().parents[1]
    config = read_json(root / CONFIG)
    receptors, ligands, audit = validate_inputs(root, config)
    assert len(receptors) == 16
    assert len(ligands) == 696
    assert audit["label_counts"] == {"active": 348, "decoy": 348}
    assert audit["preparation_variant_counts"] == {
        "meeko_rigid_macrocycles": 15,
        "original_meeko_flexible": 681,
    }
    assert audit["macrocycle_closure_pseudoatom_ligand_count"] == 0
    assert audit["expected_batch_count"] == 48
    assert audit["expected_pair_count"] == 33408
    assert audit["validation_rows"] == 0
    assert audit["test_rows"] == 0


def test_stage09_matrix_aggregations_and_independent_comparison() -> None:
    ligands = [
        {
            "ligand_id": "L1",
            "label": "active",
            "selection_role": "development_train_expanded",
        }
    ]
    receptors = ["R1", "R2"]
    rows = []
    for seed_id, first, second in (
        ("seed0", -8.0, -5.0),
        ("seed1", -7.0, -6.0),
        ("seed2", -9.0, -4.0),
    ):
        rows.extend(
            [
                {
                    "seed_id": seed_id,
                    "ligand_id": "L1",
                    "receptor_id": "R1",
                    "gpu_score": first,
                },
                {
                    "seed_id": seed_id,
                    "ligand_id": "L1",
                    "receptor_id": "R2",
                    "gpu_score": second,
                },
            ]
        )
    median = matrix_rows(rows, ligands, receptors, "median")
    minimum = matrix_rows(rows, ligands, receptors, "minimum")
    assert median[0]["R1"] == -8.0
    assert median[0]["R2"] == -5.0
    assert minimum[0]["R1"] == -9.0
    assert minimum[0]["R2"] == -6.0
    compare_matrix(
        [{key: str(value) for key, value in median[0].items()}],
        [{key: str(value) for key, value in row.items()} for row in rows],
        ligands,
        receptors,
        "median",
    )


def test_stage09_filters_preserve_frozen_order() -> None:
    records = [{"seed_id": "seed0"}, {"seed_id": "seed1"}, {"seed_id": "seed2"}]
    selected = selected_records(records, "seed_id", ["seed2", "seed0"])
    assert [row["seed_id"] for row in selected] == ["seed0", "seed2"]


def test_stage09_bundle_is_self_contained_and_boundary_clean() -> None:
    paths = bundle_paths(Path.cwd())
    assert CONFIG in paths
    assert any(
        path.endswith("run_stage09_mk14_train696_production_remote.sh")
        for path in FIXED_PATHS
    )
    assert sum(path.endswith("_receptor.pdbqt") for path in paths) == 16
    assert sum(path.endswith(".pdbqt") for path in paths) == 712
    assert not any("fresh_validation" in path for path in paths)
    assert not any("locked_test" in path for path in paths)
