import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import (
    build_qubo,
    maxmin_seeded,
    maxsum_greedy,
    q_energy,
)
from scripts.prepare_receptor import file_sha256


def test_structure_qubo_expansion_matches_reduced_energy() -> None:
    ids = ["A", "B", "C"]
    matrix = np.array(
        [
            [0.0, 0.2, 0.8],
            [0.2, 0.0, 0.5],
            [0.8, 0.5, 0.0],
        ]
    )
    quality = {value: 0.0 for value in ids}
    qubo = build_qubo(ids, matrix, quality, 2, 4.0, 1.0, 0.0)
    subset = ("A", "C")
    expected = q_energy(subset, ids, matrix, quality, 2, 4.0, 1.0, 0.0)
    assignment_energy = float(qubo["constant"])
    assignment_energy += sum(qubo["linear"][value] for value in subset)
    assignment_energy += qubo["quadratic"]["A::C"]
    assert assignment_energy == pytest.approx(expected)


def test_structure_baselines_are_deterministic() -> None:
    ids = ["A", "B", "C", "D"]
    matrix = np.array(
        [
            [0.0, 0.1, 0.9, 0.3],
            [0.1, 0.0, 0.4, 0.8],
            [0.9, 0.4, 0.0, 0.2],
            [0.3, 0.8, 0.2, 0.0],
        ]
    )
    assert maxmin_seeded(ids, matrix, 3, "A") == ("A", "C", "D")
    assert maxmin_seeded(ids, matrix, 3, "A") == maxmin_seeded(ids, matrix, 3, "A")
    assert maxsum_greedy(ids, matrix, 3) == maxsum_greedy(ids, matrix, 3)


def test_stage21_config_is_structure_only_and_no_hardware() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/stage21_structure_aware_qubo.json").read_text(encoding="ascii")
    )
    assert config["evidence_timing"]["new_docking_jobs"] is False
    assert config["evidence_timing"]["quantum_hardware_execution"] is False
    assert config["diagnostic"]["lambda_quality"] == 0.0
    paths = [
        value
        for target in config["targets"].values()
        for value in target["inputs"].values()
    ]
    assert not any("fresh_validation" in value or "locked_test" in value for value in paths)


def test_stage21_result_and_audit_pass() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "data/stage21_structure_aware_qubo_result.json").read_text(encoding="ascii")
    )
    audit = json.loads(
        (root / "data/stage21_structure_aware_qubo_audit.json").read_text(encoding="ascii")
    )
    assert result["status"] == "stage21_structure_aware_qubo_train_only_complete"
    assert result["decision"]["new_docking_authorized"] is False
    assert result["decision"]["stable_difference_observed"] is True
    assert audit["status"] == "stage21_structure_aware_qubo_audit_ok"
    assert audit["coverage"]["selection_rows_recomputed"] == 30
    assert audit["coverage"]["restart_rows_recomputed"] == 240
    assert file_sha256(root / audit["result"]["path"]) == audit["result"]["sha256"]
