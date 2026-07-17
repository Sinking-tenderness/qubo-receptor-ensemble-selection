import pytest

from scripts.audit_stage05_development_matrix import (
    audit_aggregated_rows,
    validate_matrix,
)


SEEDS = ["seed0", "seed1", "seed2"]
RECEPTORS = ["R1", "R2"]


def aggregated_rows():
    rows = []
    for ligand_id, label, role, base in (
        ("A", "active", "development_train", -9.0),
        ("D", "decoy", "development_validation", -7.0),
    ):
        for receptor_index, receptor_id in enumerate(RECEPTORS):
            values = [
                base + receptor_index * 0.2,
                base + receptor_index * 0.2 - 0.1,
                base + receptor_index * 0.2 + 0.1,
            ]
            rows.append(
                {
                    "ligand_id": ligand_id,
                    "label": label,
                    "selection_role": role,
                    "receptor_id": receptor_id,
                    "status": "ok",
                    "seed0_representative_score": str(values[0]),
                    "seed1_representative_score": str(values[1]),
                    "seed2_representative_score": str(values[2]),
                    "minimum_representative_score": str(min(values)),
                    "median_representative_score": str(sorted(values)[1]),
                    "maximum_representative_score": str(max(values)),
                    "seed_score_range": str(max(values) - min(values)),
                }
            )
    return rows


def run_audit(rows):
    return audit_aggregated_rows(
        rows,
        SEEDS,
        expected_ligand_count=2,
        expected_receptor_count=2,
        allowed_roles={"development_train", "development_validation"},
        maximum_nonnegative_pairs=0,
        maximum_seed_range=2.0,
    )


def test_stable_complete_matrix_passes_admission() -> None:
    flags, summary = run_audit(aggregated_rows())

    assert flags == []
    assert summary["admission_passed"] is True
    assert summary["aggregated_pair_count"] == 4


def test_nonnegative_and_seed_range_failures_are_retained() -> None:
    rows = aggregated_rows()
    row = rows[0]
    values = [-9.0, 0.5, -8.8]
    row.update(
        {
            "seed0_representative_score": str(values[0]),
            "seed1_representative_score": str(values[1]),
            "seed2_representative_score": str(values[2]),
            "minimum_representative_score": str(min(values)),
            "median_representative_score": str(sorted(values)[1]),
            "maximum_representative_score": str(max(values)),
            "seed_score_range": str(max(values) - min(values)),
        }
    )

    flags, summary = run_audit(rows)

    assert summary["admission_passed"] is False
    assert summary["nonnegative_seed_score_pair_count"] == 1
    assert summary["excessive_seed_range_pair_count"] == 1
    assert flags[0]["flag_reasons"] == (
        "nonnegative_seed_score;seed_score_range_exceeded"
    )


def test_stored_aggregation_must_be_reproducible() -> None:
    rows = aggregated_rows()
    rows[0]["median_representative_score"] = "-99"

    with pytest.raises(ValueError, match="aggregate value differs"):
        run_audit(rows)


def test_primary_matrix_must_match_aggregated_long_rows() -> None:
    rows = aggregated_rows()
    matrix = [
        {
            "ligand_id": ligand_id,
            **{
                receptor_id: next(
                    row["median_representative_score"]
                    for row in rows
                    if row["ligand_id"] == ligand_id
                    and row["receptor_id"] == receptor_id
                )
                for receptor_id in RECEPTORS
            },
        }
        for ligand_id in ("A", "D")
    ]
    validate_matrix(matrix, rows, RECEPTORS, "median_representative_score")
    matrix[0]["R1"] = "-1"

    with pytest.raises(ValueError, match="matrix score differs"):
        validate_matrix(matrix, rows, RECEPTORS, "median_representative_score")
