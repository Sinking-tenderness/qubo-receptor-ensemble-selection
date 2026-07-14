import hashlib

import pytest

from scripts.run_vina_warning_diagnostics import (
    summarize_case,
    validate_protocol_configs,
)


def write_search_config(path, exhaustiveness):
    path.write_text(
        "\n".join(
            [
                "center_x = 0.52",
                "center_y = 27.06",
                "center_z = 8.97",
                "size_x = 18",
                "size_y = 18",
                "size_z = 16",
                f"exhaustiveness = {exhaustiveness}",
                "num_modes = 1",
                "cpu = 4",
                "",
            ]
        ),
        encoding="ascii",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_validate_protocol_configs_allows_only_exhaustiveness_to_differ(tmp_path):
    e32 = tmp_path / "e32.txt"
    e64 = tmp_path / "e64.txt"
    protocols = [
        {
            "protocol_id": "e32",
            "config_path": str(e32),
            "config_sha256": write_search_config(e32, 32),
            "expected_exhaustiveness": 32,
        },
        {
            "protocol_id": "e64",
            "config_path": str(e64),
            "config_sha256": write_search_config(e64, 64),
            "expected_exhaustiveness": 64,
        },
    ]
    resolved = validate_protocol_configs(protocols, workers=8, max_total_cpu=32)
    assert [row["protocol_id"] for row in resolved] == ["e32", "e64"]

    e64.write_text(e64.read_text(encoding="ascii").replace("size_x = 18", "size_x = 20"))
    protocols[1]["config_sha256"] = hashlib.sha256(e64.read_bytes()).hexdigest().upper()
    with pytest.raises(ValueError, match="differ outside exhaustiveness"):
        validate_protocol_configs(protocols, workers=8, max_total_cpu=32)


def diagnostic_rows(low_scores, high_scores, source_score):
    seeds = [20260946, 20360946, 20460946]
    rows = []
    for protocol_id, exhaustiveness, scores in (
        ("e32", 32, low_scores),
        ("e64", 64, high_scores),
    ):
        for seed, score in zip(seeds, scores, strict=True):
            rows.append(
                {
                    "case_id": "C03_D0016",
                    "receptor_id": "C03",
                    "ligand_id": "D0016",
                    "label": "decoy",
                    "source_score": source_score,
                    "source_seed": seeds[0],
                    "protocol_id": protocol_id,
                    "exhaustiveness": exhaustiveness,
                    "seed": seed,
                    "status": "ok",
                    "docking_score": score,
                }
            )
    return rows, seeds


def thresholds():
    return {
        "flag_nonnegative_high_protocol_scores": True,
        "source_reproduction_tolerance_kcal_per_mol": 0.001,
        "maximum_high_protocol_seed_range_kcal_per_mol": 1.0,
        "maximum_paired_protocol_delta_kcal_per_mol": 1.0,
    }


def test_summarize_case_confirms_search_instability_when_positive_source_is_rescued():
    rows, seeds = diagnostic_rows(
        [70.78, -3.1, -3.2],
        [-3.25, -3.2, -3.3],
        70.78,
    )
    summary = summarize_case(
        rows,
        [{"protocol_id": "e32"}, {"protocol_id": "e64"}],
        seeds,
        thresholds(),
    )
    assert summary["source_reproduction_delta"] == pytest.approx(0.0)
    assert summary["source_positive_rescued"] is True
    assert summary["persistent_nonnegative"] is False
    assert summary["diagnostic_classification"] == "search_instability_confirmed"
    assert summary["acceptance_pass"] is False


def test_summarize_case_preserves_persistent_unfavorable_outcome():
    rows, seeds = diagnostic_rows([5.0, 4.0, 3.0], [2.0, 2.2, 2.1], 5.0)
    summary = summarize_case(
        rows,
        [{"protocol_id": "e32"}, {"protocol_id": "e64"}],
        seeds,
        thresholds(),
    )
    assert summary["persistent_nonnegative"] is True
    assert summary["diagnostic_classification"] == "persistent_unfavorable"
    assert "high_protocol_nonnegative_score" in summary["acceptance_failure_reasons"]
