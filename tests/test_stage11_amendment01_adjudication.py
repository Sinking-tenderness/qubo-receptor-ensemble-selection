from pathlib import Path

from scripts.experimental.unidock.adjudicate_stage11_mk14_fresh_validation_amendment01 import (
    bedroc_from_labels,
    build_matrices,
    parse_pose,
    ranked_metrics,
)


def test_parse_pose_reads_score_shape_and_atom_types(tmp_path: Path) -> None:
    pose = tmp_path / "pose.pdbqt"
    pose.write_text(
        "MODEL 1\n"
        "REMARK VINA RESULT:   -8.125      0.000      0.000\n"
        "ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 C\n"
        "ATOM      2  O   UNL     1       1.000   0.000   0.000  1.00  0.00    -0.100 OA\n"
        "ENDMDL\n",
        encoding="ascii",
    )

    assert parse_pose(pose) == {
        "score": -8.125,
        "atom_count": 2,
        "atom_types": ["C", "OA"],
    }


def test_positive_outlier_is_ignored_by_median_and_minimum_aggregation() -> None:
    manifest = [
        {
            "ligand_id": "A",
            "label": "active",
            "selection_role": "validation",
        }
    ]
    values = {
        ("seed0", "A", "R"): -8.0,
        ("seed1", "A", "R"): -7.0,
        ("seed2", "A", "R"): 222.0,
    }

    primary, minimum, _ = build_matrices(values, manifest, ["R"], "raw")
    clipped_primary, clipped_minimum, _ = build_matrices(
        values, manifest, ["R"], "clip_100"
    )

    assert primary[0]["R"] == clipped_primary[0]["R"] == -7.0
    assert minimum[0]["R"] == clipped_minimum[0]["R"] == -8.0


def test_independent_metrics_rank_lower_scores_first() -> None:
    rows = [
        {"ligand_id": "A1", "label": "active", "R": -10.0},
        {"ligand_id": "A2", "label": "active", "R": -9.0},
        {"ligand_id": "D1", "label": "decoy", "R": -2.0},
        {"ligand_id": "D2", "label": "decoy", "R": -1.0},
    ]
    metrics = ranked_metrics(rows, ("R",))

    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc_average_precision"] == 1.0
    assert metrics["bedroc_alpha_20"] == 1.0
    assert bedroc_from_labels([1, 1, 0, 0]) == 1.0
