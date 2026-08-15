"""Independently audit the Stage60 PPARD QUBO and nested-CV freeze."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def descriptor_path(root: Path, descriptor: dict[str, Any]) -> Path:
    path = root / str(descriptor["path"])
    if not path.is_file() or sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"Stage60 descriptor differs: {path}")
    return path


def assert_group_isolation(
    rows: list[dict[str, str]], fold_column: str, context: str
) -> None:
    for group_column in ("split_group_id", "scaffold_smiles"):
        assignments: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            assignments[row[group_column]].add(row[fold_column])
        if any(len(values) != 1 for values in assignments.values()):
            raise ValueError(f"{context} {group_column} crosses folds")


def run(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(root / "configs/stage60_ppard_transferred_qubo_freeze.json")
    result = read_json(root / "data/stage60_ppard_transferred_qubo_freeze_result.json")
    if result["status"] != "stage60_ppard_transferred_qubo_and_k_rule_frozen":
        raise ValueError("Stage60 result did not complete")
    for descriptor in result["outputs"].values():
        descriptor_path(root, descriptor)

    stage42f = read_json(
        descriptor_path(root, config["inputs"]["stage42f_result"])
    )
    stage53 = read_json(descriptor_path(root, config["inputs"]["stage53_result"]))
    frozen = config["transferred_objective"]
    if frozen != stage42f["objective"] or frozen != stage53["objectives"]["rank_pair_qubo"]:
        raise ValueError("Stage60 objective is not an exact pre-PPARD transfer")

    candidates = read_csv(
        descriptor_path(root, result["outputs"]["candidate_audit_csv"])
    )
    selected = [row for row in candidates if row["selected"] == "True"]
    if len(selected) != 1 or selected[0]["candidate_id"] != frozen["objective_id"]:
        raise ValueError("Stage60 selected-candidate identity differs")
    if any(row["prior_support"] != "unsupported" for row in candidates):
        raise ValueError("Stage60 concealed prior objective failures")

    outer = read_csv(
        descriptor_path(root, result["outputs"]["outer_fold_assignments_csv"])
    )
    inner = read_csv(
        descriptor_path(root, result["outputs"]["inner_fold_assignments_csv"])
    )
    if len(outer) != 240 or len({row["ligand_id"] for row in outer}) != 240:
        raise ValueError("Stage60 outer-fold dimensions differ")
    outer_counts = Counter((row["outer_fold"], row["label"]) for row in outer)
    if outer_counts != Counter(
        {(str(fold), label): 30 for fold in range(4) for label in ("active", "decoy")}
    ):
        raise ValueError("Stage60 outer folds are not exactly balanced")
    assert_group_isolation(outer, "outer_fold", "outer")
    if len(inner) != 720:
        raise ValueError("Stage60 inner-fold dimensions differ")
    outer_by_id = {row["ligand_id"]: row["outer_fold"] for row in outer}
    for outer_fold in range(4):
        rows = [row for row in inner if row["outer_fold"] == str(outer_fold)]
        if len(rows) != 180 or len({row["ligand_id"] for row in rows}) != 180:
            raise ValueError("Stage60 inner outer-train dimensions differ")
        if any(outer_by_id[row["ligand_id"]] == str(outer_fold) for row in rows):
            raise ValueError("Stage60 outer holdout leaked into an inner fold")
        counts = Counter((row["inner_fold"], row["label"]) for row in rows)
        if counts != Counter(
            {(str(fold), label): 30 for fold in range(3) for label in ("active", "decoy")}
        ):
            raise ValueError("Stage60 inner folds are not exactly balanced")
        assert_group_isolation(rows, "inner_fold", f"outer{outer_fold} inner")

    model = read_json(
        descriptor_path(root, result["outputs"]["model_record_json"])
    )
    if model["objective"] != frozen or model["coefficient_changes_after_stage42f"] != 0:
        raise ValueError("Stage60 model record differs")
    if model["nested_cv"]["k_selection_rule"] != "one_standard_error_smallest_k":
        raise ValueError("Stage60 k-selection rule differs")
    decision = result["decision"]
    if not all(
        decision[key]
        for key in (
            "transferred_qubo_objective_frozen",
            "nested_k_stopping_rule_frozen",
            "remaining_development_manifest_freeze_authorized",
            "remaining_development_ligand_preparation_authorized",
            "remaining_development_docking_authorized",
        )
    ):
        raise ValueError("Stage60 next development step was not authorized")
    if any(
        decision[key]
        for key in (
            "fresh_validation_authorized",
            "locked_test_authorized",
            "quantum_hardware_authorized",
            "qubo_superiority_claim_authorized",
            "solver_novelty_claim_authorized",
        )
    ):
        raise ValueError("Stage60 overstepped its claim boundary")

    audit = {
        "schema_version": "1.0",
        "audit_id": "stage60-ppard-transferred-qubo-independent-audit-v1",
        "status": "stage60_ppard_transferred_qubo_independent_audit_ok",
        "objective_exact_pre_ppard_transfer": True,
        "objective_coefficient_change_count": 0,
        "prior_candidate_failures_disclosed": True,
        "selected_candidate_count": 1,
        "outer_assignment_rows": len(outer),
        "inner_assignment_rows": len(inner),
        "outer_fold_label_count": 30,
        "inner_fold_label_count": 30,
        "outer_holdout_inner_leak_count": 0,
        "split_group_cross_fold_count": 0,
        "scaffold_cross_fold_count": 0,
        "protected_score_rows_read": 0,
        "fresh_validation_authorized": False,
        "quantum_hardware_authorized": False,
        "next_step": "freeze and prepare only the remaining 144 development-train ligands",
    }
    output = output if output.is_absolute() else root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage60_ppard_transferred_qubo_freeze_audit.json"),
    )
    args = parser.parse_args()
    run(args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
