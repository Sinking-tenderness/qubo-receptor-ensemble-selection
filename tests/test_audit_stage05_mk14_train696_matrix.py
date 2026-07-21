from scripts.audit_stage05_mk14_train696_matrix import audit_rows


def test_audit_retains_seed_disagreement_as_uncertainty() -> None:
    ligands = {
        "A": {"ligand_id": "A", "label": "active"},
        "D": {"ligand_id": "D", "label": "decoy"},
    }
    rows = []
    for ligand_id, values in (("A", (-9.0, -8.9, -8.8)), ("D", (-8.0, -1.0, -7.9))):
        ordered = sorted(values)
        rows.append(
            {
                "ligand_id": ligand_id,
                "label": ligands[ligand_id]["label"],
                "selection_role": "development_train_expanded",
                "receptor_id": "R1",
                "status": "ok",
                "seed_count": "3",
                "seed0_representative_score": str(values[0]),
                "seed1_representative_score": str(values[1]),
                "seed2_representative_score": str(values[2]),
                "minimum_representative_score": str(ordered[0]),
                "median_representative_score": str(ordered[1]),
                "maximum_representative_score": str(ordered[2]),
                "seed_score_range": str(ordered[2] - ordered[0]),
            }
        )

    flags, audit = audit_rows(
        rows,
        ligands,
        ["R1"],
        ["seed0", "seed1", "seed2"],
        "development_train_expanded",
        consensus_delta=0.5,
        minimum_consensus=2,
        range_threshold=2.0,
        extreme_range_threshold=5.0,
    )

    assert audit["nonnegative_median_score_pair_count"] == 0
    assert audit["raw_range_diagnostic_pair_count"] == 1
    assert audit["extreme_raw_range_diagnostic_pair_count"] == 1
    assert audit["minimum_score_consensus_diagnostic_pair_count"] == 0
    assert flags[0]["ligand_id"] == "D"
    assert "extreme_raw_seed_range_diagnostic" in flags[0]["flag_reasons"]
