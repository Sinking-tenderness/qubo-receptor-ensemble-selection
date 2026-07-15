from pathlib import Path

import pytest

from scripts.prepare_receptor import file_sha256
from scripts.run_vina_aggregate_warning_diagnostics import (
    load_config,
    resolve_cases,
    select_warning_rows,
    summarize_case,
)


CONFIG_PATH = Path(
    "configs/stage03_cdk2_af2_md_100a100d_e64_high_risk_diagnostics_linux.json"
)


def test_fixed_config_declares_eight_pairs_and_three_seed_replicates():
    config = load_config(CONFIG_PATH)
    assert config["selection"]["expected_case_count"] == 8
    assert len(config["expected_cases"]) == 8
    assert [row["replicate_id"] for row in config["seed_replicates"]] == [
        "seed0",
        "seed1",
        "seed2",
    ]


def test_select_warning_rows_uses_configured_reason_union():
    rows = [
        {
            "receptor_id": "R1",
            "ligand_id": "L1",
            "seed_stability_warning_reasons": "seed_range_exceeded",
        },
        {
            "receptor_id": "R1",
            "ligand_id": "L2",
            "seed_stability_warning_reasons": (
                "seed_range_exceeded;minimum_median_delta_exceeded"
            ),
        },
        {
            "receptor_id": "R2",
            "ligand_id": "L3",
            "seed_stability_warning_reasons": "insufficient_favorable_replicates",
        },
    ]
    selected = select_warning_rows(
        rows,
        ["minimum_median_delta_exceeded", "insufficient_favorable_replicates"],
    )
    assert [(row["receptor_id"], row["ligand_id"]) for row in selected] == [
        ("R1", "L2"),
        ("R2", "L3"),
    ]


def test_resolve_cases_matches_pairs_hashes_and_derives_original_ligand_seeds(
    tmp_path,
):
    receptor_path = tmp_path / "receptor.pdbqt"
    first_ligand_path = tmp_path / "first.pdbqt"
    selected_ligand_path = tmp_path / "selected.pdbqt"
    receptor_path.write_text("RECEPTOR\n", encoding="ascii")
    first_ligand_path.write_text("FIRST\n", encoding="ascii")
    selected_ligand_path.write_text("SELECTED\n", encoding="ascii")

    receptor_hash = file_sha256(receptor_path)
    selected_ligand_hash = file_sha256(selected_ligand_path)
    selected_rows = [
        {
            "receptor_id": "R1",
            "ligand_id": "L2",
            "label": "decoy",
            "score_seed0": "-3.0",
            "score_seed1": "7.0",
            "score_seed2": "-2.0",
            "base_seed_seed0": "100",
            "base_seed_seed1": "200",
            "base_seed_seed2": "300",
            "minimum_score": "-3.0",
            "median_score": "-2.0",
            "seed_range": "10.0",
            "favorable_replicate_count": "2",
            "seed_stability_warning_reasons": (
                "seed_range_exceeded;minimum_median_delta_exceeded"
            ),
        }
    ]
    expected_cases = [
        {
            "case_id": "R1_L2",
            "receptor_id": "R1",
            "receptor_pdbqt_sha256": receptor_hash,
            "ligand_id": "L2",
            "ligand_pdbqt_sha256": selected_ligand_hash,
            "label": "decoy",
        }
    ]
    receptor_rows = [
        {
            "conformer_id": "R1",
            "preparation_status": "ok",
            "receptor_pdbqt_path": receptor_path.as_posix(),
            "receptor_pdbqt_sha256": receptor_hash,
        }
    ]
    ligand_rows = [
        {
            "ligand_id": "L1",
            "label": "active",
            "pdbqt_status": "ok",
            "pdbqt_path": first_ligand_path.as_posix(),
        },
        {
            "ligand_id": "L2",
            "label": "decoy",
            "pdbqt_status": "ok",
            "pdbqt_path": selected_ligand_path.as_posix(),
        },
    ]
    seeds = [
        {"replicate_id": "seed0", "base_seed": 100},
        {"replicate_id": "seed1", "base_seed": 200},
        {"replicate_id": "seed2", "base_seed": 300},
    ]

    resolved = resolve_cases(
        selected_rows,
        expected_cases,
        receptor_rows,
        ligand_rows,
        seeds,
    )
    assert resolved[0]["actual_seeds"] == {
        "seed0": 101,
        "seed1": 201,
        "seed2": 301,
    }
    assert resolved[0]["source_seed"] == 101

    wrong_expected = [{**expected_cases[0], "ligand_id": "L1"}]
    with pytest.raises(ValueError, match="differ from expected cases"):
        resolve_cases(
            selected_rows,
            wrong_expected,
            receptor_rows,
            ligand_rows,
            seeds,
        )


def diagnostic_case(source_median):
    return {
        "case_id": "C03_D0016",
        "receptor_id": "C03",
        "ligand_id": "D0016",
        "label": "decoy",
        "source_e32_minimum_score": -3.635,
        "source_e32_median_score": source_median,
        "source_e32_seed_range": 88.905,
        "source_e32_favorable_replicate_count": 1,
        "source_e32_scores": {
            "seed0": 7.417,
            "seed1": -3.635,
            "seed2": 85.27,
        },
        "actual_seeds": {"seed0": 116, "seed1": 216, "seed2": 316},
    }


def thresholds():
    return {
        "minimum_favorable_replicates": 2,
        "maximum_seed_range_kcal_per_mol": 1.0,
        "maximum_minimum_median_delta_kcal_per_mol": 1.0,
        "flag_nonnegative_median_score": True,
    }


@pytest.mark.parametrize(
    ("source_median", "scores", "expected_classification", "acceptance_pass"),
    [
        (7.417, [-3.8, -3.7, -3.6], "critical_pair_rescued_by_e64", True),
        (7.417, [7.0, -3.0, 85.0], "persistent_primary_failure", False),
        (
            -7.337,
            [-8.5, -7.3, -7.2],
            "favorable_but_seed_variable_at_e64",
            False,
        ),
        (-7.337, [-7.4, -7.3, -7.2], "stable_at_e64", True),
    ],
)
def test_summarize_case_classifies_e64_same_seed_outcomes(
    source_median,
    scores,
    expected_classification,
    acceptance_pass,
):
    rows = [
        {"seed": seed, "status": "ok", "docking_score": score}
        for seed, score in zip([116, 216, 316], scores, strict=True)
    ]
    summary = summarize_case(diagnostic_case(source_median), rows, thresholds())
    assert summary["diagnostic_classification"] == expected_classification
    assert summary["acceptance_pass"] is acceptance_pass
    assert summary["source_e32_score_seed0"] == pytest.approx(7.417)
    assert summary["e64_score_seed2"] == pytest.approx(scores[2])
