from pathlib import Path

from scripts.select_stage08b_mk14_replacements import read_json, run_selection
from scripts.experimental.unidock.build_stage08b_mk14_replacement_bundle import (
    CONFIG,
    FIXED_PATHS,
    bundle_paths,
)
from scripts.experimental.unidock.run_stage08b_mk14_replacement_redocking import (
    validate_inputs,
)


def test_repository_replacement_selection_is_deterministic() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "data/stage08b_mk14_expanded16_replacement_selection_summary.json"
    if output.exists():
        result = read_json(output)
    else:
        result = run_selection(
            root / "configs/stage08b_mk14_expanded16_replacement_preregistration.json"
        )
    assert result["status"] == "stage08b_replacement_selection_ok"
    assert result["replacement_receptor_ids"] == [
        "MK14_3ITZ_aligned",
        "MK14_2BAK_aligned",
    ]
    assert result["final_receptor_count_if_both_pass"] == 16
    assert result["data_boundary"]["previous_validation_rows_read"] == 0
    assert result["data_boundary"]["test_rows_read"] == 0


def test_stage08b_recovery_input_audit_has_14_plus_2_receptors() -> None:
    root = Path(__file__).resolve().parents[1]
    config = read_json(root / CONFIG)
    _, replacements, current, _, audit = validate_inputs(root, config)
    assert len(current) == 14
    assert len(replacements) == 2
    assert audit["expected_redocking_pair_count"] == 6
    assert audit["final_receptor_count_if_both_pass"] == 16
    assert audit["validation_rows"] == 0
    assert audit["test_rows"] == 0


def test_stage08b_bundle_is_self_contained_and_boundary_clean() -> None:
    paths = bundle_paths(Path.cwd())
    assert CONFIG in paths
    assert any(path.endswith("run_stage08b_mk14_replacement_remote.sh") for path in FIXED_PATHS)
    assert sum(path.endswith("_receptor.pdbqt") for path in paths) == 14
    assert not any("fresh_validation" in path for path in paths)
    assert not any("locked_test" in path for path in paths)
