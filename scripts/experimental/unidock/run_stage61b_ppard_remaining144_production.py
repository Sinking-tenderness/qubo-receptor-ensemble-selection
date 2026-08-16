"""Run the frozen PPARD Remaining-144 x 29 x 3 Uni-Dock matrix."""

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
        raise ValueError("Stage61b config keys differ")
    boundary = dict(config["data_boundary"])
    if any(
        int(boundary[key]) != 0
        for key in ("fresh_validation_rows_permitted", "locked_test_rows_permitted")
    ):
        raise ValueError("Stage61b crossed a protected data boundary")
    seeds = tuple(
        (str(row["seed_id"]), int(row["base_seed"]))
        for row in dict(config["inputs"])["seeds"]
    )
    if seeds != FROZEN_SEEDS:
        raise ValueError("Stage61b seed ledger differs")
    expected = dict(config["expected"])
    fixed_counts = {
        "receptor_count": 29,
        "ligand_count": 144,
        "seed_count": 3,
        "batch_count": 87,
        "pair_count": 12528,
        "fresh_validation_rows": 0,
        "locked_test_rows": 0,
    }
    for key, value in fixed_counts.items():
        if int(expected[key]) != value:
            raise ValueError(f"Stage61b expected count differs: {key}")
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
        raise ValueError("Stage61b frozen Uni-Dock profile differs")
    if str(protocol["required_package_version"]) != "1.1.3":
        raise ValueError("Stage61b Uni-Dock version differs")


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
    stage60 = common.read_json(
        common.verified_path(root, dict(inputs["stage60_result"]))
    )
    stage60_audit = common.read_json(
        common.verified_path(root, dict(inputs["stage60_audit"]))
    )
    model = common.read_json(
        common.verified_path(root, dict(inputs["stage60_model_record"]))
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
        raise ValueError("the prospective Stage55 PPARD protocol was not frozen")
    frozen = dict(preregistration["frozen_protocol"])
    docking = dict(frozen["docking"])
    if (
        docking["engine"] != "Uni-Dock 1.1.3"
        or docking["profile"] != "enhanced"
        or int(docking["exhaustiveness"]) != 1024
        or int(docking["max_step"]) != 80
        or [int(value) for value in docking["seeds"]]
        != [seed for _, seed in FROZEN_SEEDS]
    ):
        raise ValueError("Stage55 PPARD docking protocol differs")
    if frozen["gate_actions"]["pass"] != (
        "freeze the transferred QUBO objective and stopping rule, then authorize "
        "docking the remaining development-train ligands"
    ):
        raise ValueError("Stage55 remaining-development action differs")
    if stage60.get("status") != "stage60_ppard_transferred_qubo_and_k_rule_frozen":
        raise ValueError("Stage60 QUBO freeze did not complete")
    if not stage60["decision"]["remaining_development_docking_authorized"]:
        raise ValueError("Stage60 did not authorize Remaining-144 docking")
    if stage60["decision"]["fresh_validation_authorized"]:
        raise ValueError("Stage60 unexpectedly opened fresh validation")
    if stage60_audit.get("status") != "stage60_ppard_transferred_qubo_independent_audit_ok":
        raise ValueError("Stage60 independent audit did not pass")
    if (
        model.get("status") != "stage60_ppard_transferred_qubo_frozen"
        or int(model["coefficient_changes_after_stage42f"]) != 0
        or model["ppard_pilot_outcomes_used_for_weight_fitting"] is not False
    ):
        raise ValueError("Stage60 QUBO model record differs")
    if receptor_summary.get("status") != "stage58b_ppard_passing29_receptor_manifest_ok":
        raise ValueError("the Stage58b receptor freeze did not pass")
    if receptor_summary.get("all_receptors_passed_three_of_three_seeds") is not True:
        raise ValueError("the PPARD receptor freeze contains an unstable pass")
    if input_audit.get("status") != (
        "independent_stage61a_ppard_remaining144_input_audit_ok"
    ):
        raise ValueError("the independent Stage61a input audit did not pass")
    if int(input_audit["future_receptor_count"]) != 29:
        raise ValueError("Stage61a receptor authorization differs")
    if preparation.get("status") != "stage61a_ppard_remaining144_unidock_inputs_ok":
        raise ValueError("the Stage61a ligand preparation did not pass")
    if any(
        int(preparation["data_boundary"][key]) != 0
        for key in (
            "docking_scores_read", "fresh_validation_rows_read", "locked_test_rows_read"
        )
    ):
        raise ValueError("Stage61a preparation exposed outcome or protected data")
    if (
        stage57.get("status") != "stage57_ppard_cognate_redocking_gate_ok"
        or int(stage57["passed_receptor_count"]) != 29
        or not bool(stage57["technical_gate_pass"])
        or int(stage57["pose_integrity_failure_count"]) != 0
        or int(stage57["unresolved_warning_event_count"]) != 0
    ):
        raise ValueError("Stage57 did not authorize PPARD production docking")
    if profile.get("status") != "unidock_profile_frozen_train_only" or profile.get(
        "selected_profile_id"
    ) != "enhanced":
        raise ValueError("the train-only Uni-Dock profile differs")

    receptors = common.read_csv(receptor_path)
    ligands = common.read_csv(ligand_path)
    receptor_ids = [row["conformer_id"] for row in receptors]
    expected_ids = [str(value) for value in expected["receptor_ids"]]
    if receptor_ids != expected_ids or receptor_ids != receptor_summary["receptor_ids"]:
        raise ValueError("Stage61b receptor order differs")
    if len(receptors) != 29 or any(
        row["status"] != "ok"
        or row["stage57_gate_pass"] != "True"
        or row["stage57_seed_count"] != "3"
        or row["stage57_successful_seed_count"] != "3"
        for row in receptors
    ):
        raise ValueError("Stage61b receptor manifest differs")
    if len(ligands) != 144 or len({row["ligand_id"] for row in ligands}) != 144:
        raise ValueError("Stage61b ligand count or IDs differ")
    labels = Counter(row["label"] for row in ligands)
    expected_labels = Counter(
        {key: int(value) for key, value in dict(expected["label_counts"]).items()}
    )
    if labels != expected_labels:
        raise ValueError("Stage61b ligand labels differ")
    if {row["split"] for row in ligands} != {boundary["allowed_split"]}:
        raise ValueError("Stage61b exposed a non-train ligand")
    if {row["selection_role"] for row in ligands} != {
        boundary["allowed_selection_role"]
    }:
        raise ValueError("Stage61b selection role differs")
    if {row["pilot_selected"] for row in ligands} != {
        boundary["required_pilot_selected"]
    }:
        raise ValueError("Stage61b contains a Pilot-96 ligand")
    if {row["pilot_outer_fold"] for row in ligands} != {""} or {
        row["pilot_role"] for row in ligands
    } != {""}:
        raise ValueError("Stage61b ligand retains a pilot role")
    if any(row["pdbqt_status"] != "ok" for row in ligands):
        raise ValueError("Stage61b ligand manifest contains a failed PDBQT")
    for rows, path_column, hash_column, id_column in (
        (receptors, "receptor_pdbqt", "receptor_pdbqt_sha256", "conformer_id"),
        (ligands, "pdbqt_path", "pdbqt_sha256", "ligand_id"),
    ):
        for row in rows:
            path = common.rooted_path(root, row[path_column])
            if not path.is_file() or common.file_sha256(path) != row[hash_column].upper():
                raise ValueError(f"Stage61b prepared input differs: {row[id_column]}")
    pseudoatom_ids = [
        row["ligand_id"]
        for row in ligands
        if common.macrocycle_closure_atom_types(common.rooted_path(root, row["pdbqt_path"]))
    ]
    if pseudoatom_ids:
        raise ValueError(f"Stage61b ligands retain closure pseudoatoms: {pseudoatom_ids}")
    variants = Counter(row["preparation_variant"] for row in ligands)
    if variants != Counter(
        {key: int(value) for key, value in preparation["preparation_variant_counts"].items()}
    ):
        raise ValueError("Stage61b preparation variants differ")
    return receptors, ligands, {
        "status": "audit_only_ok",
        "target_id": "PPARD",
        "experiment_class": "prospective remaining development completion",
        "receptor_count": len(receptors),
        "receptor_ids": receptor_ids,
        "ligand_count": len(ligands),
        "label_counts": dict(sorted(labels.items())),
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
        result["status"] = "stage61b_partial_ok"
        common.write_json(common.rooted_path(root, str(outputs["progress_json"])), result)
        return result
    if result["status"] != "stage52b_ppara_train374_unidock_matrix_ok":
        return result
    result.update(
        {
            "status": "stage61b_ppard_remaining144_unidock_matrix_ok",
            "operation": "prospective remaining-development Uni-Dock score generation",
            "experiment_class": "prospective remaining development completion",
            "next_gate": (
                "run the independent Stage61b matrix audit, merge with Stage58b "
                "Pilot-96, then apply the frozen Stage60 nested QUBO protocol"
            ),
        }
    )
    result.pop("stage51_gate_status", None)
    progress_path = common.rooted_path(root, str(outputs["progress_json"]))
    progress = common.read_json(progress_path)
    progress["status"] = "stage61b_production_complete"
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
        raise ValueError("Stage61b production adapter path differs")
    if common.file_sha256(adapter_path) != str(descriptor["sha256"]).upper():
        raise ValueError("Stage61b production adapter hash differs")
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
        "audit_only_ok", "stage61b_partial_ok",
        "stage61b_ppard_remaining144_unidock_matrix_ok",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
