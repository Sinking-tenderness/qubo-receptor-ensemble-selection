import json
from pathlib import Path

from scripts.experimental.unidock.build_stage07b_unidock_confirmation_bundle import (
    CONFIG,
    FIXED_PATHS,
)
from scripts.experimental.unidock.evaluate_stage07b_unidock_enhanced_confirmation import (
    profile_summary_rows,
    select_profile,
)
from scripts.experimental.unidock.run_stage07b_unidock_enhanced_confirmation import (
    CANDIDATE_PROFILES,
    PROFILE_ORDER,
    audit_batch_poses,
    merged_protocol,
    pdbqt_atom_signature,
    validate_config,
    validate_inputs,
)
from scripts.experimental.unidock.run_unidock_gpu_equivalence import (
    file_sha256,
    read_json,
)


CONFIG_PATH = Path(
    "configs/stage07b_mk14_unidock113_train160_enhanced_confirmation.json"
)


def atom_line(serial: int, atom_type: str) -> str:
    return (
        f"ATOM  {serial:5d}  C{serial:<2d} LIG A   1      "
        f"0.000   0.000   0.000  0.00  0.00     0.000 {atom_type}\n"
    )


def test_stage07b_config_uses_preregistered_factorial_ladder():
    config = read_json(CONFIG_PATH)

    validate_config(config)

    observed = {
        profile_id: (
            merged_protocol(config, profile_id)["exhaustiveness"],
            merged_protocol(config, profile_id)["max_step"],
        )
        for profile_id in PROFILE_ORDER
    }
    assert observed == {
        "detail_recheck": (512, 40),
        "step_extended": (512, 80),
        "depth_extended": (1024, 40),
        "enhanced": (1024, 80),
    }
    assert CANDIDATE_PROFILES == (
        "step_extended",
        "depth_extended",
        "enhanced",
    )


def test_stage07b_actual_inputs_are_train_only_and_macrocycle_safe():
    config = read_json(CONFIG_PATH)

    receptors, ligands, audit = validate_inputs(Path.cwd(), config)

    assert len(receptors) == 4
    assert len(ligands) == 160
    assert audit["profile_count"] == 4
    assert audit["expected_pair_count"] == 7680
    assert audit["macrocycle_closure_pseudoatom_ligand_count"] == 0
    assert audit["preparation_variant_counts"] == {
        "meeko_rigid_macrocycles": 4,
        "original_meeko_flexible": 156,
    }
    assert audit["validation_rows"] == audit["test_rows"] == 0


def test_pose_integrity_audit_detects_count_and_type_mismatches(tmp_path):
    input_ok = tmp_path / "input_ok.pdbqt"
    input_bad = tmp_path / "input_bad.pdbqt"
    output_ok = tmp_path / "output_ok.pdbqt"
    output_bad = tmp_path / "output_bad.pdbqt"
    input_ok.write_text(atom_line(1, "C") + atom_line(2, "OA"), encoding="ascii")
    input_bad.write_text(atom_line(1, "C") + atom_line(2, "N"), encoding="ascii")
    output_ok.write_text(
        "REMARK VINA RESULT: -8.0 0.0 0.0\n"
        + atom_line(1, "C")
        + atom_line(2, "OA"),
        encoding="ascii",
    )
    output_bad.write_text(
        "REMARK VINA RESULT: -7.0 0.0 0.0\n" + atom_line(1, "C"),
        encoding="ascii",
    )
    ligands = [
        {"ligand_id": "ok", "pdbqt_path": input_ok.name},
        {"ligand_id": "bad", "pdbqt_path": input_bad.name},
    ]
    rows = [
        {
            "ligand_id": "ok",
            "pose_count": 1,
            "output_pose_path": output_ok.name,
        },
        {
            "ligand_id": "bad",
            "pose_count": 1,
            "output_pose_path": output_bad.name,
        },
    ]

    assert pdbqt_atom_signature(input_ok) == (2, {"C": 1, "OA": 1})
    audited, summary = audit_batch_poses(tmp_path, ligands, rows)

    assert audited[0]["pose_integrity_status"] == "ok"
    assert audited[1]["pose_integrity_status"] == "failed"
    assert summary["failure_count"] == 1
    assert summary["mismatches"][0]["ligand_id"] == "bad"


def test_stage07b_selects_fastest_candidate_passing_every_gate():
    config = read_json(CONFIG_PATH)
    comparisons = []
    stability = []
    batches = []
    for profile_id in PROFILE_ORDER:
        comparisons.append(
            {
                "profile_id": profile_id,
                "spearman_vs_reference": 0.98,
                "top5pct_overlap_vs_reference": 0.875,
                "bedroc_delta_vs_reference": 0.01,
            }
        )
        stability.append(
            {
                "profile_id": profile_id,
                "spearman": 0.96,
                "top5pct_overlap": 0.875,
                "absolute_bedroc_delta": 0.01,
            }
        )
        elapsed = {
            "detail_recheck": 9.0,
            "step_extended": 10.0,
            "depth_extended": 12.0,
            "enhanced": 20.0,
        }[profile_id]
        for index in range(12):
            batches.append(
                {
                    "profile_id": profile_id,
                    "elapsed_seconds": str(elapsed + index / 100),
                    "engine_warning_count": (
                        "2" if profile_id == "depth_extended" and index == 0 else "0"
                    ),
                    "pose_integrity_failure_count": "0",
                }
            )

    summaries = profile_summary_rows(
        config, comparisons, stability, batches
    )

    assert not summaries[0]["selection_eligible"]
    assert summaries[1]["selection_eligible"]
    assert not summaries[2]["selection_eligible"]
    assert summaries[3]["selection_eligible"]
    assert select_profile(summaries) == "step_extended"


def test_stage07b_config_hashes_match_implementations():
    config = json.loads(CONFIG_PATH.read_text(encoding="ascii"))

    for descriptor in config["implementation"].values():
        assert file_sha256(Path(descriptor["path"])) == descriptor["sha256"]


def test_stage07b_bundle_contains_execution_and_diagnostics_tools():
    assert CONFIG in FIXED_PATHS
    assert "scripts/experimental/unidock/run_stage07_unidock_sensitivity.py" in FIXED_PATHS
    assert any("run_stage07b_unidock" in path for path in FIXED_PATHS)
    assert any("evaluate_stage07b_unidock" in path for path in FIXED_PATHS)
    assert any("build_stage07b_pose_diagnostics.py" in path for path in FIXED_PATHS)
    assert any("enhanced_confirmation_execution.md" in path for path in FIXED_PATHS)
