from scripts.run_vina_search_ladder import summarize_case


def make_rows(e8: float, e16: float, e32: float):
    return [
        {
            "case_id": "case",
            "receptor_id": "receptor",
            "ligand_id": "ligand",
            "label": "active",
            "protocol_id": protocol,
            "seed": 7,
            "status": "ok",
            "docking_score": score,
        }
        for protocol, score in (("e8", e8), ("e16", e16), ("e32", e32))
    ]


def test_penultimate_protocol_passes_when_e16_e32_are_close() -> None:
    summary = summarize_case(make_rows(-8.0, -8.5, -8.7), ["e8", "e16", "e32"], [7], 0.5)

    assert summary["penultimate_protocol_stable"] is True
    assert summary["maximum_absolute_highest_pair_delta"] == 0.2


def test_penultimate_protocol_fails_when_e16_e32_are_far_apart() -> None:
    summary = summarize_case(make_rows(-8.0, -8.2, -9.0), ["e8", "e16", "e32"], [7], 0.5)

    assert summary["penultimate_protocol_stable"] is False
    assert summary["stability_failure_reasons"] == "highest_pair_delta_exceeded"


def test_highest_protocol_seed_range_is_gated_for_multiple_seeds() -> None:
    rows = []
    for seed, e16, e32 in ((7, -8.5, -8.6), (8, -8.4, -9.3)):
        rows.extend(
            {
                "case_id": "case",
                "receptor_id": "receptor",
                "ligand_id": "ligand",
                "label": "active",
                "protocol_id": protocol,
                "seed": seed,
                "status": "ok",
                "docking_score": score,
            }
            for protocol, score in (("e16", e16), ("e32", e32))
        )

    summary = summarize_case(rows, ["e16", "e32"], [7, 8], 1.0, 0.5)

    assert summary["e32_seed_range"] == 0.7
    assert summary["penultimate_protocol_stable"] is False
    assert summary["stability_failure_reasons"] == "highest_protocol_seed_range_exceeded"
