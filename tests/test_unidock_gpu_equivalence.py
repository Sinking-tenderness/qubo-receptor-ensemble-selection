from pathlib import Path

import pytest

from scripts.experimental.unidock.audit_unidock_gpu_equivalence import (
    average_ranks,
    best_receptor_agreement,
    group_metrics,
    pearson,
    quantile,
    spearman,
    top_overlap,
)
from scripts.experimental.unidock.run_unidock_gpu_equivalence import (
    macrocycle_closure_atom_types,
    map_pose_outputs,
    parse_vina_pose,
    unidock_input_compatibility,
    unidock_command,
    unidock_log_warnings,
    validate_gpu_score,
)


def ligand(ligand_id: str) -> dict[str, str]:
    return {
        "ligand_id": ligand_id,
        "pdbqt_path": f"ligands/{ligand_id}.pdbqt",
    }


def test_parse_vina_pose_reads_first_mode_and_counts_modes(tmp_path: Path):
    pose = tmp_path / "L1.pdbqt"
    pose.write_text(
        "MODEL 1\n"
        "REMARK VINA RESULT: -9.125 0.000 0.000\n"
        "ENDMDL\n"
        "MODEL 2\n"
        "REMARK VINA RESULT: -8.500 1.000 2.000\n"
        "ENDMDL\n",
        encoding="ascii",
    )

    score, count = parse_vina_pose(pose)

    assert score == -9.125
    assert count == 2


def test_map_pose_outputs_accepts_preserved_and_out_suffix_names(tmp_path: Path):
    (tmp_path / "L1.pdbqt").write_text(
        "REMARK VINA RESULT: -8.0 0 0\n", encoding="ascii"
    )
    (tmp_path / "L2_out.pdbqt").write_text(
        "REMARK VINA RESULT: -7.0 0 0\n", encoding="ascii"
    )

    mapped = map_pose_outputs(tmp_path, [ligand("L1"), ligand("L2")])

    assert mapped["L1"].name == "L1.pdbqt"
    assert mapped["L2"].name == "L2_out.pdbqt"


def test_macrocycle_closure_atom_types_detect_meeko_pseudoatoms(tmp_path: Path):
    ligand_path = tmp_path / "macrocycle.pdbqt"
    ligand_path.write_text(
        "ATOM      1  C   UNL     1       0.0 0.0 0.0  1.00 0.00 0.0 C\n"
        "ATOM      2  C   UNL     1       0.0 0.0 0.0  1.00 0.00 0.0 CG0\n"
        "ATOM      3  G   UNL     1       0.0 0.0 0.0  1.00 0.00 0.0 G0\n",
        encoding="ascii",
    )

    assert macrocycle_closure_atom_types(ligand_path) == ["CG0", "G0"]


def test_input_compatibility_blocks_pseudoatoms_by_default():
    config = {
        "data_boundary": {"allowed_split": "train"},
        "unidock": {},
    }
    audit = {
        "macrocycle_closure_pseudoatom_ligand_count": 4,
        "validation_rows": 0,
        "test_rows": 0,
    }

    decision = unidock_input_compatibility(config, audit)

    assert decision["status"] == "blocked"
    assert decision["compatible"] is False


def test_input_compatibility_allows_explicit_train_diagnostic_only():
    config = {
        "data_boundary": {"allowed_split": "train"},
        "unidock": {
            "macrocycle_closure_pseudoatom_policy": "allow_train_diagnostic"
        },
    }
    audit = {
        "macrocycle_closure_pseudoatom_ligand_count": 1,
        "validation_rows": 0,
        "test_rows": 0,
    }

    assert unidock_input_compatibility(config, audit)["compatible"] is True

    audit["validation_rows"] = 1
    assert unidock_input_compatibility(config, audit)["compatible"] is False


def test_gpu_score_guard_rejects_nonphysical_magnitude():
    validate_gpu_score(-14.5, "L1")

    with pytest.raises(ValueError, match="nonphysical Uni-Dock score"):
        validate_gpu_score(825120.125, "L2")


def test_unidock_log_warnings_counts_output_coordinate_mismatch(tmp_path: Path):
    path = tmp_path / "unidock.log"
    path.write_text(
        "WARNING: in add_to_output_container, adding the 1th ligand\n"
        "t.coords.size()=26, out[0].coords.size()=27\n",
        encoding="ascii",
    )

    warnings = unidock_log_warnings(path)

    assert warnings["output_container_warning_count"] == 1
    assert warnings["coordinate_size_mismatch_count"] == 1
    assert warnings["total_count"] == 2


def test_correlation_helpers_handle_ties_and_monotonic_scores():
    assert average_ranks([1.0, 1.0, 3.0]) == [1.5, 1.5, 3.0]
    assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert spearman([1.0, 1.0, 3.0], [2.0, 2.0, 5.0]) == pytest.approx(
        1.0
    )
    assert quantile([0.0, 10.0], 0.95) == 9.5


def test_top_overlap_uses_lower_scores_as_better():
    ligand_ids = [f"L{index}" for index in range(20)]
    cpu = [float(index) for index in range(20)]
    gpu = [float(index) for index in range(20)]
    gpu[0], gpu[1] = gpu[1], gpu[0]

    overlap, count = top_overlap(ligand_ids, cpu, gpu, 0.10)

    assert count == 2
    assert overlap == 1.0


def test_unidock_command_freezes_detail_parameters(tmp_path: Path):
    protocol = {
        "scoring": "vina",
        "exhaustiveness": 512,
        "max_step": 40,
        "refine_step": 5,
        "num_modes": 1,
        "energy_range": 3,
        "verbosity": 1,
        "box": {
            "center_x": -0.49,
            "center_y": 3.26,
            "center_z": 21.83,
            "size_x": 22,
            "size_y": 24,
            "size_z": 32,
        },
    }

    command = unidock_command(
        "unidock",
        tmp_path / "receptor.pdbqt",
        tmp_path / "ligands.index",
        tmp_path / "poses",
        protocol,
        20260801,
    )

    assert command[command.index("--exhaustiveness") + 1] == "512"
    assert command[command.index("--max_step") + 1] == "40"
    assert command[command.index("--seed") + 1] == "20260801"
    assert command[command.index("--num_modes") + 1] == "1"


def test_group_metrics_use_explicit_cpu_and_gpu_score_columns():
    rows = []
    for index in range(20):
        rows.append(
            {
                "seed_id": "seed0",
                "receptor_id": "R1",
                "ligand_id": f"L{index:02d}",
                "cpu_vina_e32_score": float(index),
                "gpu_unidock_detail_score": float(index),
                "score_delta_gpu_minus_cpu": 0.0,
            }
        )

    metrics = group_metrics(rows)
    agreement = best_receptor_agreement(rows)

    assert metrics[0]["spearman"] == pytest.approx(1.0)
    assert metrics[0]["top5pct_overlap"] == 1.0
    assert agreement["agreement_fraction"] == 1.0
