import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("rdkit")

from qubo_receptor_ensemble.experiment import build_problem_stage, evaluate_stage, solve_stage
from qubo_receptor_ensemble.full_workflow import FULL_WORKFLOW_STAGES, FullExperimentConfig


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_full_workflow_persists_adaptive_k_before_building_problem(tmp_path: Path) -> None:
    matrix_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for number in range(1, 5):
        for label, prefix, r1, r2 in (
            ("active", "A", -float(7 - number), -10.0 - number),
            ("decoy", "D", -10.0 + number, -1.0),
        ):
            ligand_id = prefix + str(number)
            matrix_rows.append(
                {
                    "ligand_id": ligand_id,
                    "label": label,
                    "R1": r1,
                    "R2": r2,
                }
            )
            manifest_rows.append(
                {
                    "ligand_id": ligand_id,
                    "label": label,
                    "selection_role": "development_train",
                    "pdbqt_path": "inputs/" + ligand_id + ".pdbqt",
                    "scaffold_smiles": prefix + str(number),
                    "outer_fold": number % 2,
                }
            )

    matrix_path = _write_csv(tmp_path / "matrix.csv", matrix_rows)
    ligand_manifest = _write_csv(
        tmp_path / "ligands.csv",
        [{**row, "scaffold_smiles": ""} for row in manifest_rows],
    )
    source_ligand_manifest = _write_csv(
        tmp_path / "source_ligands.csv",
        [
            {"ligand_id": row["ligand_id"], "scaffold_smiles": row["scaffold_smiles"]}
            for row in manifest_rows
        ],
    )
    receptor_manifest = _write_csv(
        tmp_path / "receptors.csv",
        [
            {"conformer_id": "R1", "receptor_pdbqt": "R1.pdbqt"},
            {"conformer_id": "R2", "receptor_pdbqt": "R2.pdbqt"},
        ],
    )
    run_directory = tmp_path / "run"
    paths = {
        "run_directory": run_directory,
        "prepared_ligand_manifest": ligand_manifest,
        "source_ligand_manifest": source_ligand_manifest,
        "selected_receptor_manifest": receptor_manifest,
        "primary_matrix": matrix_path,
        "problem": run_directory / "problem.json",
        "selection": run_directory / "selection.json",
        "evaluation": run_directory / "evaluation.json",
    }
    config = FullExperimentConfig(
        path=tmp_path / "config.json",
        data_root=tmp_path,
        data={
            "experiment_id": "adaptive-test",
            "target_id": "TEST",
            "problem": {
                "type": "receptor_subset",
                "strategy": "qubo",
                "target_size": 1,
                "weights": {"redundancy": 0.0, "count": 0.0, "size": 1.0},
                "k_policy": {
                    "mode": "adaptive",
                    "selector": "mechanistic_bootstrap_lcb",
                    "candidates": [1, 2],
                    "scaffold_field": "scaffold_smiles",
                    "inner_fold_count": 2,
                    "bootstrap_iterations": 50,
                    "lower_quantile": 0.025,
                    "rescue_fractions": [0.01, 0.05],
                    "random_seed": 13,
                },
            },
            "solve": {"backend": "exact"},
            "evaluate": {"metrics": ["bedroc_alpha_20"]},
        },
        paths=paths,
        stages=FULL_WORKFLOW_STAGES,
        start_stage="build_problem",
        end_stage="persist",
    )

    built = build_problem_stage(config, matrix_path)

    assert built["adaptive_cardinality"]["selected_k"] in {1, 2}
    decision_path = run_directory / "adaptive_cardinality.json"
    assert decision_path.is_file()
    payload = json.loads(paths["problem"].read_text(encoding="ascii"))
    assert payload["problem_config"]["target_size"] == payload["adaptive_cardinality"]["selected_k"]

    solved = solve_stage(config, paths["problem"])
    selection = json.loads(paths["selection"].read_text(encoding="ascii"))
    assert solved["selection_path"] == paths["selection"]
    assert selection["adaptive_cardinality"] == payload["adaptive_cardinality"]

    evaluated = evaluate_stage(config, paths["selection"])
    evaluation = json.loads(paths["evaluation"].read_text(encoding="ascii"))
    assert evaluated["evaluation_path"] == paths["evaluation"]
    assert evaluation["adaptive_cardinality"] == payload["adaptive_cardinality"]
