import pytest

from scripts.experimental.unidock.merge_unidock_rigid_cpu_reference import merge_rows


def row(receptor: str, ligand: str, label: str, score: str):
    return {
        "target_id": "MK14",
        "receptor_id": receptor,
        "ligand_id": ligand,
        "label": label,
        "representative_score": score,
        "representative_method": "pose_rank_1",
        "status": "ok",
    }


def test_merge_rows_replaces_exact_rigid_pairs():
    original = [
        row("R1", "A", "active", "-8.0"),
        row("R1", "D", "decoy", "-7.0"),
        row("R2", "A", "active", "-8.5"),
        row("R2", "D", "decoy", "-7.5"),
    ]
    rigid = [
        row("R1", "D", "decoy", "-9.0"),
        row("R2", "D", "decoy", "-9.5"),
    ]

    merged = merge_rows(original, rigid, {"R1", "R2"}, {"D"})

    by_key = {(item["receptor_id"], item["ligand_id"]): item for item in merged}
    assert by_key[("R1", "A")]["representative_score"] == "-8.0"
    assert by_key[("R1", "D")]["representative_score"] == "-9.0"
    assert by_key[("R1", "D")]["source_score_protocol"].endswith(
        "rigid_macrocycle_replacement"
    )


def test_merge_rows_requires_complete_replacement_grid():
    original = [row("R1", "D", "decoy", "-7.0")]

    with pytest.raises(ValueError, match="replacement keys differ"):
        merge_rows(original, [], {"R1"}, {"D"})
