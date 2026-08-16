import pytest

from scripts.experimental.unidock.run_unidock_batch_targeted import (
    apply_target_id,
    infer_target_id,
)


def test_infer_target_id_requires_one_target() -> None:
    assert infer_target_id(
        [
            {"ligand_id": "P1", "target_id": "PPARG"},
            {"ligand_id": "P2", "target_id": "PPARG"},
        ]
    ) == "PPARG"

    with pytest.raises(ValueError, match="one non-empty target_id"):
        infer_target_id(
            [
                {"ligand_id": "P1", "target_id": "PPARG"},
                {"ligand_id": "E1", "target_id": "EGFR"},
            ]
        )


def test_apply_target_id_changes_metadata_only() -> None:
    source = [
        {
            "target_id": "MK14",
            "ligand_id": "PPARG_active_L1",
            "receptor_id": "PPARG_R1",
            "gpu_score": -9.25,
        }
    ]
    corrected = apply_target_id(source, "PPARG")

    assert source[0]["target_id"] == "MK14"
    assert corrected[0] == {
        **source[0],
        "target_id": "PPARG",
    }
