import pytest

from scripts.run_vina_search_robustness import (
    summarize_case,
    validate_search_configs,
)


def make_rows(e4_scores, e8_scores):
    rows = []
    seeds = [11, 22, 33]
    for protocol, scores in (("e4", e4_scores), ("e8", e8_scores)):
        for seed, score in zip(seeds, scores, strict=True):
            rows.append({
                "case_id": "C05_D0026",
                "receptor_id": "C05",
                "ligand_id": "D0026",
                "label": "decoy",
                "protocol": protocol,
                "seed": seed,
                "status": "ok",
                "docking_score": score,
            })
    return rows


def test_validate_search_configs_allows_only_exhaustiveness_to_differ():
    e4 = {
        "center_x": "0.52", "center_y": "27.06", "center_z": "8.97",
        "size_x": "18", "size_y": "18", "size_z": "16",
        "exhaustiveness": "4", "num_modes": "1", "cpu": "4",
    }
    e8 = {**e4, "exhaustiveness": "8"}
    validate_search_configs(e4, e8, 4)
    with pytest.raises(ValueError, match="differ outside exhaustiveness"):
        validate_search_configs(e4, {**e8, "size_x": "20"}, 4)


def test_summarize_case_passes_stable_paired_scores():
    summary = summarize_case(
        make_rows([-7.8, -7.9, -7.85], [-7.82, -7.88, -7.86]),
        [11, 22, 33],
        True,
        1.0,
        0.5,
    )
    assert summary["acceptance_pass"] is True
    assert summary["e4_seed_range"] == pytest.approx(0.1)
    assert summary["maximum_absolute_paired_delta"] == pytest.approx(0.02)


def test_summarize_case_preserves_multiple_instability_reasons():
    summary = summarize_case(
        make_rows([26.95, -7.8, -7.9], [-7.8, -7.82, -7.88]),
        [11, 22, 33],
        True,
        1.0,
        0.5,
    )
    assert summary["acceptance_pass"] is False
    assert summary["nonnegative_score_count"] == 1
    assert summary["acceptance_failure_reasons"] == (
        "nonnegative_score;e4_seed_range_exceeded;paired_e4_e8_delta_exceeded"
    )
