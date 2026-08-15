import pytest

from scripts.amend_stage19c_pparg_target_id import (
    correct_target_rows,
    non_target_digest,
)


def test_target_id_amendment_preserves_non_target_payload() -> None:
    source = [
        {
            "target_id": "MK14",
            "seed_id": "seed0",
            "receptor_id": "PPARG_R1",
            "ligand_id": "PPARG_L1",
            "gpu_score": "-8.25",
        }
    ]
    corrected = correct_target_rows(source, "MK14", "PPARG")

    assert corrected[0]["target_id"] == "PPARG"
    assert non_target_digest(corrected) == non_target_digest(source)


def test_target_id_amendment_rejects_nonuniform_source_labels() -> None:
    with pytest.raises(ValueError, match="source target labels differ"):
        correct_target_rows(
            [{"target_id": "MK14"}, {"target_id": "PPARG"}],
            "MK14",
            "PPARG",
        )
