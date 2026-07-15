import json
from pathlib import Path

import pytest

from scripts.subset_score_matrices_to_development import (
    load_config,
    subset_matrix_rows,
)


CONFIG_PATH = Path("configs/stage04_cdk2_md_development_matrix_subset.json")


def test_subset_matrix_skips_locked_scores_before_numeric_parsing():
    fields = ["target_id", "ligand_id", "label", "R1", "R2"]
    rows = [
        {
            "target_id": "CDK2",
            "ligand_id": "DEV",
            "label": "active",
            "R1": "-8.0",
            "R2": "-7.0",
        },
        {
            "target_id": "CDK2",
            "ligand_id": "LOCKED",
            "label": "decoy",
            "R1": "MUST_NOT_BE_PARSED",
            "R2": "MUST_NOT_BE_PARSED",
        },
    ]

    receptors, output, skipped = subset_matrix_rows(
        fields, rows, {"DEV": "active"}, 2, 1
    )

    assert receptors == ["R1", "R2"]
    assert output == [{
        "target_id": "CDK2",
        "ligand_id": "DEV",
        "label": "active",
        "R1": -8.0,
        "R2": -7.0,
    }]
    assert skipped == 1


def test_subset_matrix_rejects_non_numeric_development_score():
    fields = ["target_id", "ligand_id", "label", "R1"]
    rows = [{"target_id": "CDK2", "ligand_id": "DEV", "label": "active", "R1": "bad"}]

    with pytest.raises(ValueError, match="not numeric"):
        subset_matrix_rows(fields, rows, {"DEV": "active"}, 1, 0)


def test_load_config_rejects_inconsistent_expected_counts(tmp_path: Path):
    config = json.loads(CONFIG_PATH.read_text(encoding="ascii"))
    config["expected"]["skipped_locked_ligand_count"] = 39
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="ascii")

    with pytest.raises(ValueError, match="internally inconsistent"):
        load_config(path)
