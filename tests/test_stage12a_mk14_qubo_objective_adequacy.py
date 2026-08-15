import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.diagnose_stage12a_mk14_qubo_objective_adequacy import (
    build_explicit_qubo,
    choose_highest,
    design_matrix,
    exact_all_cardinalities,
)
from scripts.prepare_receptor import file_sha256


def test_design_matrix_contains_linear_and_selected_pair_terms() -> None:
    subsets = [("R1", "R2"), ("R1", "R3")]
    matrix, names = design_matrix(subsets, ["R1", "R2", "R3"], True)

    assert names == [
        "x::R1",
        "x::R2",
        "x::R3",
        "xx::R1__R2",
        "xx::R1__R3",
        "xx::R2__R3",
    ]
    assert matrix.tolist() == [
        [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    ]


def test_choose_highest_breaks_ties_by_subset_id() -> None:
    subsets = [("R1", "R3"), ("R1", "R2"), ("R2", "R3")]
    assert choose_highest(np.array([0.5, 0.5, 0.4]), subsets) == 1


def test_explicit_qubo_preserves_surrogate_optimum_and_cardinality() -> None:
    receptor_ids = ["R1", "R2", "R3"]
    feature_names = [
        "x::R1",
        "x::R2",
        "x::R3",
        "xx::R1__R2",
        "xx::R1__R3",
        "xx::R2__R3",
    ]
    model = SimpleNamespace(
        intercept_=0.0,
        coef_=np.array([0.1, 0.1, 0.1, 1.0, 0.0, 0.0]),
    )
    qubo = build_explicit_qubo(model, feature_names, receptor_ids, 2, 20.0)
    subset, _ = exact_all_cardinalities(receptor_ids, qubo)

    assert subset == ("R1", "R2")


def test_stage12a_config_freezes_implementation_and_excludes_stage11_inputs() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root / "configs/stage12a_mk14_qubo_objective_adequacy_posthoc.json"
    )
    config = json.loads(config_path.read_text(encoding="ascii"))
    implementation = config["implementation"]

    assert file_sha256(root / implementation["path"]) == implementation["sha256"]
    assert not any("stage11" in key.lower() for key in config["inputs"])
    assert config["expected"]["stage11_validation_rows"] == 0
