from scripts.audit_stage05_expanded_e32_matrix import audit_consensus_rows


def row(ligand_id, scores):
    minimum = min(scores)
    median = sorted(scores)[1]
    maximum = max(scores)
    return {
        "ligand_id": ligand_id,
        "receptor_id": "R1",
        "selection_role": "development_train",
        "status": "ok",
        "seed0_representative_score": str(scores[0]),
        "seed1_representative_score": str(scores[1]),
        "seed2_representative_score": str(scores[2]),
        "minimum_representative_score": str(minimum),
        "median_representative_score": str(median),
        "maximum_representative_score": str(maximum),
        "seed_score_range": str(maximum - minimum),
    }


def test_consensus_gate_tolerates_one_unfavorable_outlier() -> None:
    flags, audit = audit_consensus_rows(
        [row("L1", [-13.18, -13.15, -11.8])],
        ["seed0", "seed1", "seed2"],
        expected_ligands=1,
        expected_receptors=1,
        consensus_delta=0.5,
        minimum_consensus_replicates=2,
        maximum_median_minus_minimum=0.5,
        maximum_nonnegative_median_pairs=0,
    )

    assert flags == []
    assert audit["admission_passed"] is True
    assert audit["minimum_consensus_replicate_count"] == 2


def test_consensus_gate_rejects_when_only_one_seed_finds_best_basin() -> None:
    flags, audit = audit_consensus_rows(
        [row("L1", [-12.94, -10.78, -10.76])],
        ["seed0", "seed1", "seed2"],
        expected_ligands=1,
        expected_receptors=1,
        consensus_delta=0.5,
        minimum_consensus_replicates=2,
        maximum_median_minus_minimum=0.5,
        maximum_nonnegative_median_pairs=0,
    )

    assert len(flags) == 1
    assert audit["admission_passed"] is False
    assert "insufficient_consensus_replicates" in flags[0]["flag_reasons"]
