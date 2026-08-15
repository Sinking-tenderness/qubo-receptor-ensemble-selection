from pathlib import Path

import pytest

from scripts.prepare_receptor import file_sha256
from scripts.screen_stage10_mk14_expanded16_qubo_greedy import (
    build_matrices,
    fixed_cardinality_exact,
    fixed_cardinality_greedy,
    read_csv,
    read_json,
    rooted,
)


def test_fixed_cardinality_exact_detects_a_forward_greedy_trap() -> None:
    coefficients = {
        "constant": 0.0,
        "linear": {"A": -10.0, "B": -9.0, "C": -8.0},
        "quadratic": {
            "A__B": 100.0,
            "A__C": 100.0,
            "B__C": -100.0,
        },
    }

    exact, exact_energy = fixed_cardinality_exact(
        coefficients, ["A", "B", "C"], 2
    )
    greedy, greedy_energy, path = fixed_cardinality_greedy(
        coefficients, ["A", "B", "C"], 2
    )

    assert exact == ("B", "C")
    assert greedy == ("A", "B")
    assert exact_energy == pytest.approx(-117.0)
    assert greedy_energy == pytest.approx(81.0)
    assert path[-1]["subset"] == ["A", "B"]


def test_build_matrices_preserves_all_seed_receptor_cells() -> None:
    receptor_ids = ["R1", "R2"]
    manifest_rows = [
        {"ligand_id": "L1", "label": "active"},
        {"ligand_id": "L2", "label": "decoy"},
    ]
    primary_rows = [
        {"ligand_id": "L1", "label": "active", "R1": "-8", "R2": "-5"},
        {"ligand_id": "L2", "label": "decoy", "R1": "-4", "R2": "-6"},
    ]
    sensitivity_rows = [
        {"ligand_id": "L1", "label": "active", "R1": "-9", "R2": "-6"},
        {"ligand_id": "L2", "label": "decoy", "R1": "-5", "R2": "-7"},
    ]
    score_rows = []
    for seed_index, seed_id in enumerate(("seed0", "seed1", "seed2")):
        for ligand_index, ligand_id in enumerate(("L1", "L2")):
            for receptor_index, receptor_id in enumerate(receptor_ids):
                score_rows.append(
                    {
                        "seed_id": seed_id,
                        "ligand_id": ligand_id,
                        "receptor_id": receptor_id,
                        "gpu_score": str(
                            -1.0 - seed_index - ligand_index - receptor_index
                        ),
                    }
                )

    matrices = build_matrices(
        primary_rows,
        sensitivity_rows,
        score_rows,
        manifest_rows,
        receptor_ids,
    )

    assert set(matrices) == {
        "primary",
        "sensitivity",
        "seed0",
        "seed1",
        "seed2",
    }
    assert matrices["primary"]["L1"]["R1"] == -8.0
    assert matrices["sensitivity"]["L2"]["R2"] == -7.0
    assert matrices["seed2"]["L2"]["R2"] == -5.0


def test_stage10_frozen_objectives_match_their_recorded_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/stage10_mk14_expanded16_qubo_greedy_screen.json"
    config = read_json(config_path)

    implementation = dict(config["implementation"])
    assert file_sha256(rooted(root, str(implementation["path"]))) == str(
        implementation["sha256"]
    ).upper()

    for family, raw_spec in dict(config["objective_specs"]).items():
        spec = dict(raw_spec)
        source = read_json(rooted(root, str(dict(spec["source_result"])["path"])))
        assert source["status"] == spec["required_status"]
        selected = dict(source["selected_qubo"])
        frozen = dict(spec["frozen_candidate"])
        assert selected["family"] == family == frozen["family"]
        assert selected["target_size"] == frozen["target_size"]
        assert selected["aggregation"] == frozen["aggregation"]
        assert {
            key: float(value) for key, value in dict(selected["weights"]).items()
        } == {
            key: float(value) for key, value in dict(frozen["weights"]).items()
        }


def test_stage10_inputs_are_train_only_and_boundary_clean() -> None:
    root = Path(__file__).resolve().parents[1]
    config = read_json(
        root / "configs/stage10_mk14_expanded16_qubo_greedy_screen.json"
    )
    inputs = dict(config["inputs"])
    ligand_rows = read_csv(rooted(root, str(dict(inputs["ligand_manifest"])["path"])))
    stage09_summary = read_json(
        rooted(root, str(dict(inputs["stage09_summary"])["path"]))
    )
    stage09_audit = read_json(
        rooted(root, str(dict(inputs["stage09_audit"])["path"]))
    )

    assert len(ligand_rows) == 696
    assert {row["split"] for row in ligand_rows} == {"train"}
    assert {row["selection_role"] for row in ligand_rows} == {
        "development_train_expanded"
    }
    assert stage09_summary["status"] == "stage09_train696_unidock_matrix_ok"
    assert (
        stage09_audit["status"]
        == "independent_stage09_train696_unidock_matrix_audit_ok"
    )
    assert stage09_summary["data_boundary"] == {
        "validation_rows_read": 0,
        "test_rows_read": 0,
    }
    assert all(int(value) == 0 for value in stage09_audit["data_boundary"].values())
