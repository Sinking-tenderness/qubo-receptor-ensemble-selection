import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_stage32_pparg_md_functional_pilot import frozen_frames, frozen_ligands, read_json
from scripts.experimental.unidock.run_stage32_pparg_md_functional_pilot import FROZEN_PROFILE, FROZEN_SEEDS, matrix_rows, validate_config


def test_stage32_config_is_prospective_and_protected() -> None:
    root = Path(__file__).resolve().parents[1]
    config = read_json(root / "configs/stage32_pparg_md_functional_complementarity_pilot.json")
    validate_config(config)
    assert config["evidence_timing"]["stage32_docking_outcomes_known_before_freeze"] is False
    assert config["evidence_timing"]["fresh_validation_rows_permitted"] is False
    assert config["evidence_timing"]["test_rows_permitted"] is False
    assert config["functional_analysis_preregistration"]["bedroc_alpha"] == 20.0
    assert FROZEN_PROFILE == ("enhanced", 1024, 80)
    assert len(FROZEN_SEEDS) == 3


def test_stage32_frame_panel_is_two_per_start_without_labels() -> None:
    root = Path(__file__).resolve().parents[1]
    config = read_json(root / "configs/stage32_pparg_md_functional_complementarity_pilot.json")
    frames = frozen_frames(config, root)
    assert len(frames) == 16
    assert Counter(int(row["start_index"]) for row in frames) == Counter({value: 2 for value in range(8)})
    assert {int(row["temporal_maximin_rank"]) for row in frames} == {0, 1}
    assert len({row["frame_id"] for row in frames}) == 16


def test_stage32_ligand_panel_is_balanced_and_scaffold_unique() -> None:
    root = Path(__file__).resolve().parents[1]
    config = read_json(root / "configs/stage32_pparg_md_functional_complementarity_pilot.json")
    ligands = frozen_ligands(config, root)
    assert Counter(row["label"] for row in ligands) == Counter({"active": 80, "decoy": 80})
    for label in ("active", "decoy"):
        groups = [row["split_group_id"] for row in ligands if row["label"] == label]
        assert len(groups) == len(set(groups)) == 80
    assert {row["split"] for row in ligands} == {"train"}
    assert {row["selection_role"] for row in ligands} == {"development_train"}


def test_matrix_aggregation_preserves_ligand_metadata() -> None:
    ligands = [{"ligand_id": "L1", "label": "active", "selection_role": "development_train", "split_group_id": "G1"}]
    rows = [
        {"ligand_id": "L1", "receptor_id": "R1", "gpu_score": value}
        for value in (-8.0, -7.0, -9.0)
    ]
    median = matrix_rows(rows, ligands, ["R1"], "median")
    minimum = matrix_rows(rows, ligands, ["R1"], "minimum")
    assert median[0]["R1"] == -8.0
    assert minimum[0]["R1"] == -9.0
    assert median[0]["split_group_id"] == "G1"


def test_selection_result_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "data/stage32_pparg_md_functional_pilot_input_preparation_result.json"
    if not path.is_file():
        return
    result = json.loads(path.read_text(encoding="ascii"))
    assert result["counts"]["receptors"] == 16
    assert result["counts"]["ligands"] == 160
    assert result["counts"]["expected_pairs"] == 7680
    assert result["data_boundary"]["fresh_validation_rows_read"] == 0
    assert result["data_boundary"]["test_rows_read"] == 0
