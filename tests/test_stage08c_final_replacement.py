from pathlib import Path

from scripts.select_stage08c_mk14_final_replacement import read_json
from scripts.experimental.unidock.build_stage08c_mk14_final_replacement_bundle import (
    CONFIG,
    FIXED_PATHS,
    bundle_paths,
)
from scripts.experimental.unidock.run_stage08c_mk14_final_replacement_redocking import (
    validate_inputs,
)


def test_stage08c_excludes_zero_distance_duplicate_and_selects_1oz1() -> None:
    path = Path("data/stage08c_mk14_final_replacement_selection_summary.json")
    if not path.exists():
        return
    result = read_json(path)
    assert result["status"] == "stage08c_final_replacement_selection_ok"
    assert result["propagated_equivalence_exclusions"] == ["MK14_4A9Y_aligned"]
    assert result["replacement_receptor_ids"] == ["MK14_1OZ1_aligned"]
    assert result["final_receptor_count_if_pass"] == 16


def test_stage08c_input_audit_has_15_plus_1_receptors() -> None:
    root = Path(__file__).resolve().parents[1]
    config = read_json(root / CONFIG)
    _, current, selected, _, audit = validate_inputs(root, config)
    assert len(current) == 15
    assert selected["conformer_id"] == "MK14_1OZ1_aligned"
    assert audit["expected_redocking_pair_count"] == 3
    assert audit["final_receptor_count_if_pass"] == 16
    assert audit["validation_rows"] == 0
    assert audit["test_rows"] == 0


def test_stage08c_bundle_is_self_contained_and_boundary_clean() -> None:
    paths = bundle_paths(Path.cwd())
    assert CONFIG in paths
    assert any(
        path.endswith("run_stage08c_mk14_final_replacement_remote.sh")
        for path in FIXED_PATHS
    )
    assert sum(path.endswith("_receptor.pdbqt") for path in paths) == 15
    assert any(path.endswith("1OZ1.pdb") for path in paths)
    assert any(path.endswith("1OZ1_A_to_2QD9_A.pdb") for path in paths)
    assert not any("fresh_validation" in path for path in paths)
    assert not any("locked_test" in path for path in paths)
