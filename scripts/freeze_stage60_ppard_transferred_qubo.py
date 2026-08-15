"""Freeze the transferred PPARD QUBO and outcome-blind nested-CV folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verified(root: Path, descriptor: dict[str, Any]) -> Path:
    path = root / str(descriptor["path"])
    if not path.is_file() or sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"frozen input differs: {path}")
    return path


def make_group_folds(
    rows: list[dict[str, str]], fold_count: int, seed: int
) -> dict[str, int]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["split_group_id"]].append(row)
    generator = random.Random(seed)
    items = list(groups.items())
    generator.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    totals = {
        label: sum(row["label"] == label for row in rows)
        for label in ("active", "decoy")
    }
    targets = {label: totals[label] / fold_count for label in totals}
    counts = [{label: 0 for label in totals} for _ in range(fold_count)]
    assignments: dict[str, int] = {}
    for _, group in items:
        group_counts = {
            label: sum(row["label"] == label for row in group)
            for label in totals
        }

        def cost(fold: int) -> tuple[float, float, int]:
            projected = {
                label: counts[fold][label] + group_counts[label]
                for label in totals
            }
            overflow = sum(
                max(0.0, projected[label] - targets[label])
                / max(1.0, targets[label])
                for label in totals
            )
            distance = sum(
                abs(projected[label] - targets[label]) / max(1.0, targets[label])
                for label in totals
            )
            return overflow, distance, fold

        fold = min(range(fold_count), key=cost)
        for row in group:
            assignments[row["ligand_id"]] = fold
        for label in totals:
            counts[fold][label] += group_counts[label]
    if len(assignments) != len(rows):
        raise ValueError("group-fold assignment is incomplete")
    return assignments


def validate_group_isolation(
    rows: list[dict[str, str]], assignments: dict[str, int]
) -> None:
    for column in ("split_group_id", "scaffold_smiles"):
        observed: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            observed[row[column]].add(assignments[row["ligand_id"]])
        if any(len(folds) != 1 for folds in observed.values()):
            raise ValueError(f"{column} crosses frozen folds")


def candidate_audit(
    config: dict[str, Any], inputs: dict[str, Path]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage37 = read_json(inputs["stage37_result"])
    stage40 = read_json(inputs["stage40_result"])
    stage42f = read_json(inputs["stage42f_result"])
    stage53 = read_json(inputs["stage53_result"])
    selected = dict(config["transferred_objective"])
    if selected != stage42f["objective"]:
        raise ValueError("Stage60 objective differs from the frozen Stage42f QUBO")
    if selected != stage53["objectives"]["rank_pair_qubo"]:
        raise ValueError("Stage60 objective differs from the Stage53 transfer")
    if stage37["decision"]["functional_objective_supported"]:
        raise ValueError("Stage37 support status unexpectedly changed")
    if stage40["decision"]["bedroc_aligned_objective_supported"]:
        raise ValueError("Stage40 support status unexpectedly changed")
    if stage42f["decision"]["rank_sensitive_pair_qubo_supported"]:
        raise ValueError("Stage42f support status unexpectedly changed")
    if stage53["decision"]["frozen_qubo_application_transfer_supported"]:
        raise ValueError("Stage53 support status unexpectedly changed")
    rank_bedroc = float(
        stage53["full_data_methods"]["rank_pair_qubo_exact"]["robust_bedroc"]
    )
    coverage_bedroc = float(
        stage53["full_data_methods"]["coverage_qubo_exact"]["robust_bedroc"]
    )
    if rank_bedroc <= coverage_bedroc:
        raise ValueError("pre-PPARD rank-pair evidence no longer exceeds coverage")
    rows = [
        {
            "candidate_id": "seed_robust_functional_complementarity_v1",
            "polynomial_class": "QUBO_with_auxiliary_encoding",
            "prior_targets": "MK14+PPARG+PPARA",
            "prior_support": "unsupported",
            "selected": False,
            "reason": "Earlier coverage objective and weaker PPARA transfer BEDROC",
        },
        {
            "candidate_id": "bedroc_aligned_signed_mobius_v1",
            "polynomial_class": "cubic_HUBO",
            "prior_targets": "MK14+PPARG",
            "prior_support": "unsupported",
            "selected": False,
            "reason": "Not a native second-order QUBO and failed its frozen efficacy gate",
        },
        {
            "candidate_id": selected["objective_id"],
            "polynomial_class": "fixed_k_QUBO",
            "prior_targets": "BACE1+PPARA",
            "prior_support": "unsupported",
            "selected": True,
            "reason": "Most recent transferable second-order objective aligned to BEDROC20 and k=1..6",
        },
    ]
    evidence = {
        "stage42f_best_combination_over_single_bedroc_gain": float(
            stage42f["decision"]["best_combination_over_single_bedroc_gain"]
        ),
        "stage53_rank_pair_full_robust_bedroc": rank_bedroc,
        "stage53_coverage_full_robust_bedroc": coverage_bedroc,
        "stage53_rank_pair_minus_single_full_bedroc": float(
            stage53["decision"]["full_data_rank_pair_over_single_bedroc_gain"]
        ),
        "stage53_solver_novelty_detected": bool(
            stage53["decision"]["solver_novelty_detected"]
        ),
    }
    return rows, evidence


def validate_manifest(
    train_rows: list[dict[str, str]], pilot_rows: list[dict[str, str]]
) -> dict[str, Any]:
    if len(train_rows) != 240 or Counter(row["label"] for row in train_rows) != {
        "active": 120,
        "decoy": 120,
    }:
        raise ValueError("PPARD Train-240 dimensions differ")
    if {row["split"] for row in train_rows} != {"train"} or {
        row["selection_role"] for row in train_rows
    } != {"development_train"}:
        raise ValueError("PPARD Train-240 crossed a split boundary")
    if len({row["ligand_id"] for row in train_rows}) != 240:
        raise ValueError("PPARD Train-240 ligand IDs are not unique")
    pilot_from_train = [row for row in train_rows if row["pilot_selected"] == "True"]
    remaining = [row for row in train_rows if row["pilot_selected"] == "False"]
    if len(pilot_from_train) != 96 or Counter(
        row["label"] for row in pilot_from_train
    ) != {"active": 48, "decoy": 48}:
        raise ValueError("PPARD Pilot-96 identity differs")
    if len(remaining) != 144 or Counter(row["label"] for row in remaining) != {
        "active": 72,
        "decoy": 72,
    }:
        raise ValueError("PPARD remaining development panel differs")
    if [row["ligand_id"] for row in pilot_from_train] != [
        row["ligand_id"] for row in pilot_rows
    ]:
        raise ValueError("PPARD Pilot-96 order differs from Train-240")
    train_by_id = {row["ligand_id"]: row for row in train_rows}
    if any(train_by_id[row["ligand_id"]] != row for row in pilot_rows):
        raise ValueError("PPARD Pilot-96 rows differ from Train-240")
    scaffold_groups: dict[str, set[str]] = defaultdict(set)
    split_scaffolds: dict[str, set[str]] = defaultdict(set)
    for row in train_rows:
        scaffold_groups[row["scaffold_smiles"]].add(row["split_group_id"])
        split_scaffolds[row["split_group_id"]].add(row["scaffold_smiles"])
    if any(len(values) != 1 for values in scaffold_groups.values()) or any(
        len(values) != 1 for values in split_scaffolds.values()
    ):
        raise ValueError("PPARD scaffold and split-group identities are not one-to-one")
    return {
        "development_ligand_count": len(train_rows),
        "development_label_counts": dict(
            sorted(Counter(row["label"] for row in train_rows).items())
        ),
        "pilot_ligand_count": len(pilot_from_train),
        "remaining_ligand_count": len(remaining),
        "remaining_label_counts": dict(
            sorted(Counter(row["label"] for row in remaining).items())
        ),
        "scaffold_group_count": len(scaffold_groups),
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    decision = result["decision"]
    evidence = result["pre_ppard_candidate_evidence"]
    lines = [
        "# Stage60 PPARD transferred QUBO and k-rule freeze",
        "",
        "## Frozen choice",
        "",
        f"- Objective: `{result['transferred_objective']['objective_id']}`.",
        "- Candidate sizes: k=1..6.",
        "- Selection: nested scaffold CV and the one-standard-error smallest-k rule.",
        "- No PPARD docking score was used to fit or change an objective coefficient.",
        "",
        "## Prior evidence boundary",
        "",
        f"- Stage42f combination-over-single signal: {evidence['stage42f_best_combination_over_single_bedroc_gain']:.6f}.",
        f"- Stage53 rank-pair versus coverage full BEDROC: {evidence['stage53_rank_pair_full_robust_bedroc']:.6f} versus {evidence['stage53_coverage_full_robust_bedroc']:.6f}.",
        "- The objective was unsupported on prior targets and has not beaten the strongest classical optimizer.",
        "",
        "## Authorization",
        "",
        f"- Remaining development preparation: {'YES' if decision['remaining_development_ligand_preparation_authorized'] else 'NO'}.",
        f"- Fresh validation: {'YES' if decision['fresh_validation_authorized'] else 'NO'}.",
        f"- Quantum hardware: {'YES' if decision['quantum_hardware_authorized'] else 'NO'}.",
        "",
        result["interpretation_boundary"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, dict(config["implementation"])["freezer"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage60 implementation identity differs")
    inputs = {
        key: verified(root, descriptor)
        for key, descriptor in dict(config["inputs"]).items()
    }
    if any(
        marker in path.relative_to(root).as_posix().lower()
        for path in inputs.values()
        for marker in ("fresh_validation", "locked_test", "protected/")
    ):
        raise ValueError("Stage60 input crossed a protected split boundary")
    outputs = {key: root / value for key, value in dict(config["outputs"]).items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage60 outputs exist; pass --overwrite")

    stage55 = read_json(inputs["stage55_preregistration"])
    stage59 = read_json(inputs["stage59_result"])
    stage59_audit = read_json(inputs["stage59_audit"])
    if (
        stage55["frozen_protocol"]["gate_actions"]["pass"]
        != "freeze the transferred QUBO objective and stopping rule, then authorize docking the remaining development-train ligands"
    ):
        raise ValueError("Stage55 pass action differs")
    if not stage59["decision"]["functional_complementarity_gate_passed"]:
        raise ValueError("Stage59 functional gate did not pass")
    if not stage59["decision"]["transferred_qubo_objective_freeze_authorized"]:
        raise ValueError("Stage59 did not authorize the Stage60 freeze")
    if (
        stage59_audit["status"]
        != "stage59_ppard_functional_complementarity_independent_audit_ok"
    ):
        raise ValueError("Stage59 independent audit did not pass")

    train_rows = read_csv(inputs["train_manifest"])
    pilot_rows = read_csv(inputs["pilot_manifest"])
    manifest_summary = validate_manifest(train_rows, pilot_rows)
    candidates, prior_evidence = candidate_audit(config, inputs)

    cv = dict(config["nested_cv"])
    outer_count = int(cv["outer_fold_count"])
    outer_assignments = make_group_folds(
        train_rows, outer_count, int(cv["outer_fold_seed"])
    )
    validate_group_isolation(train_rows, outer_assignments)
    outer_counts = Counter(
        (outer_assignments[row["ligand_id"]], row["label"])
        for row in train_rows
    )
    required_outer = Counter(
        {(fold, label): 30 for fold in range(outer_count) for label in ("active", "decoy")}
    )
    if outer_counts != required_outer:
        raise ValueError("Stage60 outer folds are not exactly class balanced")
    outer_rows = [
        {
            "ligand_id": row["ligand_id"],
            "label": row["label"],
            "split_group_id": row["split_group_id"],
            "scaffold_smiles": row["scaffold_smiles"],
            "pilot_selected": row["pilot_selected"],
            "outer_fold": outer_assignments[row["ligand_id"]],
        }
        for row in train_rows
    ]

    inner_rows: list[dict[str, Any]] = []
    inner_counts: dict[str, dict[str, int]] = {}
    seeds = [int(value) for value in cv["inner_fold_seeds"]]
    if len(seeds) != outer_count:
        raise ValueError("Stage60 inner-fold seed count differs")
    for outer_fold in range(outer_count):
        outer_train = [
            row
            for row in train_rows
            if outer_assignments[row["ligand_id"]] != outer_fold
        ]
        inner_count = int(cv["inner_fold_count"])
        inner_assignments = make_group_folds(
            outer_train, inner_count, seeds[outer_fold]
        )
        validate_group_isolation(outer_train, inner_assignments)
        counts = Counter(
            (inner_assignments[row["ligand_id"]], row["label"])
            for row in outer_train
        )
        required = Counter(
            {(fold, label): 30 for fold in range(inner_count) for label in ("active", "decoy")}
        )
        if counts != required:
            raise ValueError("Stage60 inner folds are not exactly class balanced")
        for fold in range(inner_count):
            for label in ("active", "decoy"):
                inner_counts[f"outer{outer_fold}_inner{fold}_{label}"] = {
                    "count": counts[(fold, label)]
                }
        inner_rows.extend(
            {
                "outer_fold": outer_fold,
                "ligand_id": row["ligand_id"],
                "label": row["label"],
                "split_group_id": row["split_group_id"],
                "scaffold_smiles": row["scaffold_smiles"],
                "inner_fold": inner_assignments[row["ligand_id"]],
            }
            for row in outer_train
        )

    write_csv(outputs["candidate_audit_csv"], candidates)
    write_csv(outputs["outer_fold_assignments_csv"], outer_rows)
    write_csv(outputs["inner_fold_assignments_csv"], inner_rows)
    model_record = {
        "schema_version": "1.0",
        "model_id": "stage60-ppard-transferred-rank-pair-qubo-v1",
        "status": "stage60_ppard_transferred_qubo_frozen",
        "objective": config["transferred_objective"],
        "nested_cv": config["nested_cv"],
        "comparators": config["comparators"],
        "claim_gates": config["claim_gates"],
        "coefficient_changes_after_stage42f": 0,
        "ppard_pilot_outcomes_used_for_coefficient_selection": False,
        "ppard_pilot_outcomes_used_for_weight_fitting": False,
    }
    write_json(outputs["model_record_json"], model_record)
    result = {
        "schema_version": "1.0",
        "status": "stage60_ppard_transferred_qubo_and_k_rule_frozen",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": sha256(config_path),
        },
        "manifest_summary": manifest_summary,
        "pre_ppard_candidate_evidence": prior_evidence,
        "candidate_count": len(candidates),
        "selected_candidate_count": sum(bool(row["selected"]) for row in candidates),
        "transferred_objective": config["transferred_objective"],
        "nested_cv": config["nested_cv"],
        "fold_summary": {
            "outer_assignment_rows": len(outer_rows),
            "outer_fold_label_count": 30,
            "inner_assignment_rows": len(inner_rows),
            "inner_fold_label_count": 30,
            "split_group_cross_fold_count": 0,
            "scaffold_cross_fold_count": 0,
        },
        "data_boundary": {
            "stage59_gate_decision_read": 1,
            "ppard_pilot_score_rows_read": 0,
            "remaining_development_train_score_rows_read": 0,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "decision": {
            "transferred_qubo_objective_frozen": True,
            "nested_k_stopping_rule_frozen": True,
            "remaining_development_manifest_freeze_authorized": True,
            "remaining_development_ligand_preparation_authorized": True,
            "remaining_development_docking_authorized": True,
            "fresh_validation_authorized": False,
            "locked_test_authorized": False,
            "quantum_hardware_authorized": False,
            "qubo_superiority_claim_authorized": False,
            "solver_novelty_claim_authorized": False,
        },
        "outputs": {
            key: {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for key, path in outputs.items()
            if key not in ("result_json", "report_md")
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_report(outputs["report_md"], result)
    result["outputs"]["report_md"] = {
        "path": outputs["report_md"].relative_to(root).as_posix(),
        "sha256": sha256(outputs["report_md"]),
        "size_bytes": outputs["report_md"].stat().st_size,
    }
    write_json(outputs["result_json"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
