import csv
from pathlib import Path

import pytest

from scripts.experimental.unidock.diagnose_unidock_gpu_equivalence_failure import (
    comparison_metrics,
    manifest_pseudoatom_summary,
)


def comparison_row(
    ligand_id: str,
    label: str,
    cpu: float,
    gpu: float,
) -> dict[str, object]:
    return {
        "seed_id": "seed0",
        "receptor_id": "R1",
        "ligand_id": ligand_id,
        "label": label,
        "cpu_vina_e32_score": cpu,
        "gpu_unidock_detail_score": gpu,
        "score_delta_gpu_minus_cpu": gpu - cpu,
        "absolute_score_delta": abs(gpu - cpu),
    }


def test_comparison_metrics_preserve_score_direction_and_labels():
    rows = [
        comparison_row(
            f"L{index:02d}",
            "active" if index < 10 else "decoy",
            float(index),
            float(index) + 0.1,
        )
        for index in range(20)
    ]

    metrics = comparison_metrics(rows)

    assert metrics["pair_count"] == 20
    assert metrics["ligand_count"] == 20
    assert metrics["overall_spearman"] == pytest.approx(1.0)
    assert metrics["median_group_top5pct_overlap"] == 1.0
    assert metrics["active_decoy_mean_delta_gap_kcal_per_mol"] == pytest.approx(
        0.0
    )


def test_manifest_pseudoatom_summary_uses_numbered_closure_types(
    tmp_path: Path,
):
    path = tmp_path / "ligands.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ligand_id", "label", "pdbqt_atom_types"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "ligand_id": "L1",
                    "label": "active",
                    "pdbqt_atom_types": "A;C;N;OA",
                },
                {
                    "ligand_id": "L2",
                    "label": "decoy",
                    "pdbqt_atom_types": "A;C;CG0;G0;OA",
                },
            ]
        )

    summary = manifest_pseudoatom_summary(path, "fixture")

    assert summary["ligand_count"] == 2
    assert summary["macrocycle_closure_pseudoatom_ligand_count"] == 1
    assert summary["affected_label_counts"] == {"decoy": 1}
