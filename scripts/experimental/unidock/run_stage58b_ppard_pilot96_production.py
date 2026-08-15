"""Run the prospective PPARD Pilot-96 x 29 x 3 Uni-Dock matrix."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from scripts.experimental.unidock import run_stage52b_ppara_train374_production as base


common = base.common
FROZEN_SEEDS = (("seed0", 20260801), ("seed1", 20260802), ("seed2", 20260803))
FROZEN_PROFILE = ("enhanced", 1024, 80, 5, 1, 3)
ORIGINAL_FINALIZE = base.finalize


def validate_config(config: dict[str, object]) -> None:
    required = {
        "schema_version", "experiment_id", "purpose", "implementation",
        "data_boundary", "inputs", "expected", "unidock", "execution",
        "outputs", "decision_boundary",
    }
    if set(config) != required:
        raise ValueError("Stage58b config keys differ")
    boundary = dict(config["data_boundary"])
    if any(
        int(boundary[key]) != 0
        for key in ("fresh_validation_rows_permitted", "locked_test_rows_permitted")
    ):
        raise ValueError("Stage58b crossed a protected data boundary")
    seeds = tuple(
        (str(row["seed_id"]), int(row["base_seed"]))
        for row in dict(config["inputs"])["seeds"]
    )
    if seeds != FROZEN_SEEDS:
        raise ValueError("Stage58b seed ledger differs")
    expected = dict(config["expected"])
    fixed_counts = {
        "receptor_count": 29,
        "ligand_count": 96,
        "seed_count": 3,
        "batch_count": 87,
        "pair_count": 8352,
        "fresh_validation_rows": 0,
        "locked_test_rows": 0,
    }
    for key, value in fixed_counts.items():
        if int(expected[key]) != value:
            raise ValueError(f"Stage58b expected count differs: {key}")
    protocol = dict(config["unidock"])
    profile = (
        str(protocol["profile_id"]),
        int(protocol["exhaustiveness"]),
        int(protocol["max_step"]),
        int(protocol["refine_step"]),
        int(protocol["num_modes"]),
        int(protocol["energy_range"]),
    )
    if profile != FROZEN_PROFILE:
        raise ValueError("Stage58b frozen Uni-Dock profile differs")
    if str(protocol["required_package_version"]) != "1.1.3":
        raise ValueError("Stage58b Uni-Dock version differs")


def validate_inputs(
    root: Path, config: dict[str, object]
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    inputs = dict(config["inputs"])
    expected = dict(config["expected"])
    boundary = dict(config["data_boundary"])
    receptor_path = common.verified_path(root, dict(inputs["receptor_manifest"]))
    receptor_summary = common.read_json(
        common.verified_path(root, dict(inputs["receptor_manifest_summary"]))
    )
    ligand_path = common.verified_path(root, dict(inputs["ligand_manifest"]))
    preregistration = common.read_json(
        common.verified_path(root, dict(inputs["preregistration"]))
    )
    input_audit = common.read_json(
        common.verified_path(root, dict(inputs["input_preparation_audit"]))
    )
    preparation = common.read_json(
        common.verified_path(root, dict(inputs["ligand_preparation_summary"]))
    )
    stage57 = common.read_json(
        common.verified_path(root, dict(inputs["stage57_summary"]))
    )
    profile = common.read_json(
        common.verified_path(root, dict(inputs["profile_freeze_result"]))
    )

    if preregistration.get("experiment_id") != (
        "stage55-ppard-small-pilot-preregistration-20260805-v1"
    ):
        raise ValueError("the prospective Stage55 PPARD pilot was not frozen")
    frozen = dict(preregistration["frozen_protocol"])
    pilot = dict(frozen["pilot_panel"])
    docking = dict(frozen["docking"])
    structure = dict(frozen["structural_pool"])
    if (
        int(pilot["active_count"]) != 48
        or int(pilot["decoy_count"]) != 48
        or int(pilot["outer_fold_count"]) != 4
        or pilot["selection_role"] != "development_train_pilot"
    ):
        raise ValueError("Stage55 Pilot-96 protocol differs")
    if (
        int(structure["minimum_cognate_redocking_pass_count"]) != 24
        or structure["retain_all_hard_gate_passing_structures"] is not True
        or structure["max_min_or_outcome_informed_compression_permitted"] is not False
    ):
        raise ValueError("Stage55 PPARD structural-pool rule differs")
    if (
        docking["engine"] != "Uni-Dock 1.1.3"
        or docking["profile"] != "enhanced"
        or int(docking["exhaustiveness"]) != 1024
        or int(docking["max_step"]) != 80
        or [int(value) for value in docking["seeds"]]
        != [seed for _, seed in FROZEN_SEEDS]
    ):
        raise ValueError("Stage55 PPARD docking protocol differs")
    if receptor_summary.get("status") != "stage58b_ppard_passing29_receptor_manifest_ok":
        raise ValueError("the Stage58b receptor freeze did not pass")
    if receptor_summary.get("all_receptors_passed_three_of_three_seeds") is not True:
        raise ValueError("Stage58b receptor freeze contains an unstable redocking pass")
    if input_audit.get("status") != (
        "independent_stage58a_ppard_pilot96_input_audit_ok"
    ):
        raise ValueError("the independent Stage58a input audit did not pass")
    if int(input_audit["receptor_count_frozen_for_next_stage"]) != 29:
        raise ValueError("Stage58a receptor authorization differs")
    if preparation.get("status") != "stage58a_ppard_pilot96_unidock_inputs_ok":
        raise ValueError("the Stage58a ligand preparation did not pass")
    if any(
        int(preparation["data_boundary"][key]) != 0
        for key in (
            "docking_scores_read", "fresh_validation_rows_read", "locked_test_rows_read"
        )
    ):
        raise ValueError("Stage58a preparation exposed outcome or protected data")
    if (
        stage57.get("status") != "stage57_ppard_cognate_redocking_gate_ok"
        or int(stage57["passed_receptor_count"]) != 29
        or not bool(stage57["technical_gate_pass"])
        or int(stage57["pose_integrity_failure_count"]) != 0
        or int(stage57["unresolved_warning_event_count"]) != 0
    ):
        raise ValueError("Stage57 did not authorize PPARD Pilot-96 docking")
    if profile.get("status") != "unidock_profile_frozen_train_only" or profile.get(
        "selected_profile_id"
    ) != "enhanced":
        raise ValueError("the train-only Uni-Dock profile differs")

    receptors = common.read_csv(receptor_path)
    ligands = common.read_csv(ligand_path)
    receptor_ids = [row["conformer_id"] for row in receptors]
    expected_ids = [str(value) for value in expected["receptor_ids"]]
    if receptor_ids != expected_ids or receptor_ids != receptor_summary["receptor_ids"]:
        raise ValueError("Stage58b receptor order differs")
    if len(receptors) != 29 or any(
        row["status"] != "ok"
        or row["stage57_gate_pass"] != "True"
        or row["stage57_seed_count"] != "3"
        or row["stage57_successful_seed_count"] != "3"
        for row in receptors
    ):
        raise ValueError("Stage58b receptor manifest differs")
    if len(ligands) != 96 or len({row["ligand_id"] for row in ligands}) != 96:
        raise ValueError("Stage58b ligand count or IDs differ")
    labels = Counter(row["label"] for row in ligands)
    expected_labels = Counter(
        {key: int(value) for key, value in dict(expected["label_counts"]).items()}
    )
    if labels != expected_labels:
        raise ValueError("Stage58b ligand labels differ")
    if {row["split"] for row in ligands} != {boundary["allowed_split"]}:
        raise ValueError("Stage58b exposed a non-train ligand")
    if {row["selection_role"] for row in ligands} != {
        boundary["allowed_selection_role"]
    } or {row["pilot_role"] for row in ligands} != {boundary["required_pilot_role"]}:
        raise ValueError("Stage58b pilot selection role differs")
    if {row["pilot_selected"] for row in ligands} != {"True"}:
        raise ValueError("Stage58b contains a non-pilot ligand")
    fold_labels = Counter(
        f"fold{row['pilot_outer_fold']}_{row['label']}" for row in ligands
    )
    expected_fold_labels = Counter(
        {f"fold{fold}_{label}": 12 for fold in range(4) for label in ("active", "decoy")}
    )
    if fold_labels != expected_fold_labels:
        raise ValueError("Stage58b outer-fold balance differs")
    if any(row["pdbqt_status"] != "ok" for row in ligands):
        raise ValueError("Stage58b ligand manifest contains a failed PDBQT")
    for rows, path_column, hash_column, id_column in (
        (receptors, "receptor_pdbqt", "receptor_pdbqt_sha256", "conformer_id"),
        (ligands, "pdbqt_path", "pdbqt_sha256", "ligand_id"),
    ):
        for row in rows:
            path = common.rooted_path(root, row[path_column])
            if not path.is_file() or common.file_sha256(path) != row[hash_column].upper():
                raise ValueError(f"Stage58b prepared input differs: {row[id_column]}")
    pseudoatom_ids = [
        row["ligand_id"]
        for row in ligands
        if common.macrocycle_closure_atom_types(common.rooted_path(root, row["pdbqt_path"]))
    ]
    if pseudoatom_ids:
        raise ValueError(f"Stage58b ligands retain closure pseudoatoms: {pseudoatom_ids}")
    variants = Counter(row["preparation_variant"] for row in ligands)
    if variants != Counter(
        {key: int(value) for key, value in preparation["preparation_variant_counts"].items()}
    ):
        raise ValueError("Stage58b preparation variants differ")
    return receptors, ligands, {
        "status": "audit_only_ok",
        "target_id": "PPARD",
        "experiment_class": "prospective outcome-blind development pilot",
        "receptor_count": len(receptors),
        "receptor_ids": receptor_ids,
        "ligand_count": len(ligands),
        "label_counts": dict(sorted(labels.items())),
        "fold_label_counts": dict(sorted(fold_labels.items())),
        "preparation_variant_counts": dict(sorted(variants.items())),
        "macrocycle_closure_pseudoatom_ligand_count": 0,
        "seed_count": len(FROZEN_SEEDS),
        "expected_batch_count": int(expected["batch_count"]),
        "expected_pair_count": int(expected["pair_count"]),
        "fresh_validation_rows": 0,
        "locked_test_rows": 0,
    }


def finalize(*args: object, **kwargs: object) -> dict[str, object]:
    result = ORIGINAL_FINALIZE(*args, **kwargs)
    root = Path(args[0])
    config = dict(args[2])
    outputs = dict(config["outputs"])
    if result["status"] == "stage52b_partial_ok":
        result["status"] = "stage58b_partial_ok"
        common.write_json(common.rooted_path(root, str(outputs["progress_json"])), result)
        return result
    if result["status"] != "stage52b_ppara_train374_unidock_matrix_ok":
        return result
    result.update(
        {
            "status": "stage58b_ppard_pilot96_unidock_matrix_ok",
            "operation": "prospective outcome-blind development-pilot Uni-Dock score generation",
            "experiment_class": "prospective outcome-blind development pilot",
            "next_gate": (
                "run the independent Stage58b matrix audit, then apply the frozen "
                "Stage59 functional-complementarity gate"
            ),
        }
    )
    result.pop("stage51_gate_status", None)
    progress_path = common.rooted_path(root, str(outputs["progress_json"]))
    progress = common.read_json(progress_path)
    progress["status"] = "stage58b_production_complete"
    common.write_json(progress_path, progress)
    common.write_json(common.rooted_path(root, str(outputs["summary_json"])), result)
    return result


def run(
    config_path: Path,
    root: Path,
    unidock: str | None,
    audit_only: bool,
    resume: bool,
    seed_ids: list[str] | None,
    receptor_ids: list[str] | None,
    finalize_only: bool,
) -> dict[str, object]:
    root = root.resolve()
    config = common.read_json(config_path.resolve())
    descriptor = dict(config["implementation"])["production_adapter"]
    adapter_path = common.rooted_path(root, str(descriptor["path"]))
    if adapter_path.resolve() != Path(__file__).resolve():
        raise ValueError("Stage58b production adapter path differs")
    if common.file_sha256(adapter_path) != str(descriptor["sha256"]).upper():
        raise ValueError("Stage58b production adapter hash differs")
    original_validate_config = base.validate_config
    original_validate_inputs = base.validate_inputs
    original_finalize = base.finalize
    try:
        base.validate_config = validate_config
        base.validate_inputs = validate_inputs
        base.finalize = finalize
        return base.run(
            config_path, root, unidock, audit_only, resume, seed_ids,
            receptor_ids, finalize_only,
        )
    finally:
        base.validate_config = original_validate_config
        base.validate_inputs = original_validate_inputs
        base.finalize = original_finalize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--unidock", default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed-id", action="append", default=None)
    parser.add_argument("--receptor-id", action="append", default=None)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only and args.finalize_only:
        parser.error("--audit-only and --finalize-only are mutually exclusive")
    result = run(
        args.config, args.root, args.unidock, args.audit_only, args.resume,
        args.seed_id, args.receptor_id, args.finalize_only,
    )
    return 0 if result["status"] in {
        "audit_only_ok", "stage58b_partial_ok",
        "stage58b_ppard_pilot96_unidock_matrix_ok",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
