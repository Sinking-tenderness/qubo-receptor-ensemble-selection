from copy import deepcopy
from pathlib import Path

import pytest

from scripts.experimental.unidock.run_stage08_mk14_expanded16_redocking import (
    merge_receptor_manifests,
    read_json,
    summarize_receptor_redocking_gate,
    validate_inputs,
    validate_protocol,
)
from scripts.experimental.unidock.build_stage08_mk14_expanded16_redocking_bundle import (
    CONFIG,
    FIXED_PATHS,
    bundle_paths,
)


def test_frozen_protocol_rejects_search_change() -> None:
    root = Path(__file__).resolve().parents[1]
    config = read_json(root / "configs/stage08_mk14_expanded16_unidock_redocking.json")
    protocol = deepcopy(config["unidock"])
    validate_protocol(protocol)
    protocol["max_step"] = 79
    with pytest.raises(ValueError, match="max_step"):
        validate_protocol(protocol)


def test_receptor_gate_requires_two_of_three_and_median_threshold() -> None:
    rows = [
        {
            "conformer_id": "R1",
            "top_ranked_rmsd_angstrom": value,
            "top_ranked_pose_success": value <= 2.0,
        }
        for value in (1.2, 1.8, 4.5)
    ]
    result = summarize_receptor_redocking_gate(rows, ["R1"], 2.0, 2)
    assert result[0]["gate_pass"] is True
    assert result[0]["successful_seed_count"] == 2
    rows[1]["top_ranked_rmsd_angstrom"] = 2.1
    rows[1]["top_ranked_pose_success"] = False
    result = summarize_receptor_redocking_gate(rows, ["R1"], 2.0, 2)
    assert result[0]["gate_pass"] is False


def test_manifest_merge_rejects_duplicate_receptors() -> None:
    row = {
        "conformer_id": "R1",
        "receptor_pdbqt": "r.pdbqt",
        "receptor_pdbqt_sha256": "A",
        "status": "ok",
    }
    with pytest.raises(ValueError, match="duplicate"):
        merge_receptor_manifests([row], [dict(row)])


def test_repository_stage08_redocking_input_audit() -> None:
    root = Path(__file__).resolve().parents[1]
    config = read_json(root / "configs/stage08_mk14_expanded16_unidock_redocking.json")
    _, new_rows, existing_rows, audit = validate_inputs(root, config)
    assert audit["status"] == "audit_only_ok"
    assert len(new_rows) == 8
    assert len(existing_rows) == 8
    assert audit["expected_redocking_pair_count"] == 24
    assert audit["validation_rows"] == 0
    assert audit["test_rows"] == 0


def test_stage08_bundle_is_self_contained_and_has_no_validation_inputs() -> None:
    paths = bundle_paths(Path.cwd())
    assert CONFIG in paths
    assert any(path.endswith("run_stage08_mk14_expanded16_redocking_remote.sh") for path in FIXED_PATHS)
    assert sum(path.endswith("_receptor.pdbqt") for path in paths) == 8
    assert not any("fresh_validation" in path for path in paths)
    assert not any("locked_test" in path for path in paths)
