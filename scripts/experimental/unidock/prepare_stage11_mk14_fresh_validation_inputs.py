"""Prepare the frozen six-receptor MAPK14 fresh-validation Uni-Dock inputs."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    from .prepare_stage09_mk14_train696_inputs import (
        PDBQT_FIELDS,
        prepare_rigid_pdbqt,
    )
    from .run_unidock_gpu_equivalence import (
        file_sha256,
        macrocycle_closure_atom_types,
        read_csv,
        read_json,
        relative_path,
        rooted_path,
        verified_path,
        write_csv,
        write_json,
    )
    from scripts.batch_prepare_ligand_pdbqt import find_meeko_script, safe_filename
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.batch_prepare_ligand_pdbqt import find_meeko_script, safe_filename
    from scripts.experimental.unidock.prepare_stage09_mk14_train696_inputs import (
        PDBQT_FIELDS,
        prepare_rigid_pdbqt,
    )
    from scripts.experimental.unidock.run_unidock_gpu_equivalence import (
        file_sha256,
        macrocycle_closure_atom_types,
        read_csv,
        read_json,
        relative_path,
        rooted_path,
        verified_path,
        write_csv,
        write_json,
    )


MACROCYCLE_TYPE = re.compile(r"^(?:CG|G)\d+$")


def verify_implementation(
    root: Path, config: dict[str, object], key: str, expected: Path
) -> None:
    descriptor = dict(config["implementation"])[key]
    path = rooted_path(root, str(descriptor["path"]))
    if path.resolve() != expected.resolve():
        raise ValueError(f"Stage 11 implementation path differs: {key}")
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"Stage 11 implementation hash differs: {key}")


def split_subset(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.split("+") if part)


def selected_trial(
    rows: list[dict[str, str]], selector: dict[str, object]
) -> dict[str, str]:
    selected = [
        row
        for row in rows
        if row["context_id"] == str(selector["context_id"])
        and row["objective_family"] == str(selector["objective_family"])
        and row["coefficient_source"] == str(selector["coefficient_source"])
        and int(row["target_size"]) == int(selector["target_size"])
    ]
    if len(selected) != 1:
        raise ValueError(f"Stage 11 selector did not identify one trial: {selector}")
    return selected[0]


def candidate_subsets(config: dict[str, object]) -> dict[str, tuple[str, ...]]:
    return {
        candidate_id: tuple(str(value) for value in dict(value)["subset"])
        for candidate_id, value in dict(config["candidates"]).items()
    }


def has_manifest_macrocycle(row: dict[str, str]) -> bool:
    return any(
        MACROCYCLE_TYPE.fullmatch(atom_type)
        for atom_type in row["pdbqt_atom_types"].split(";")
    )


def validate_source_inputs(
    root: Path, config: dict[str, object]
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, object],
]:
    root = root.resolve()
    inputs = dict(config["inputs"])
    expected = dict(config["expected"])
    boundary = dict(config["data_boundary"])
    source_receptors = read_csv(
        verified_path(root, dict(inputs["source_receptor_manifest"]))
    )
    ligands = read_csv(verified_path(root, dict(inputs["source_ligand_manifest"])))
    admission = read_json(
        verified_path(root, dict(inputs["receptor_admission_audit"]))
    )
    profile = read_json(verified_path(root, dict(inputs["profile_freeze_result"])))
    stage10 = read_json(verified_path(root, dict(inputs["stage10_result"])))
    trials = read_csv(verified_path(root, dict(inputs["stage10_trials"])))

    if admission.get("status") != "independent_stage08c_final_replacement_audit_ok":
        raise ValueError("Stage 11 receptor admission audit did not pass")
    if profile.get("status") != "unidock_profile_frozen_train_only":
        raise ValueError("Stage 11 Uni-Dock profile source did not pass")
    if profile.get("selected_profile_id") != "enhanced":
        raise ValueError("Stage 11 Uni-Dock profile differs")
    if stage10.get("status") != "stage10_expanded16_qubo_greedy_screen_complete":
        raise ValueError("Stage 11 Stage 10 source did not pass")
    if any(int(value) != 0 for value in dict(stage10["data_boundary"]).values()):
        raise ValueError("Stage 10 crossed a data boundary")

    candidates = candidate_subsets(config)
    selectors = dict(config["candidate_provenance"])
    fold_trial = selected_trial(trials, dict(selectors["outer_fold_3_primary"]))
    full_trial = selected_trial(trials, dict(selectors["full_train_primary"]))
    observed = {
        "exact_pair_synergy": split_subset(fold_trial["exact_subset"]),
        "qubo_forward_greedy": split_subset(fold_trial["qubo_greedy_subset"]),
        "direct_bedroc_greedy": split_subset(
            fold_trial["direct_metric_greedy_subset"]
        ),
        "full_train_exact_secondary": split_subset(full_trial["exact_subset"]),
    }
    if observed != candidates:
        raise ValueError("Stage 11 frozen candidate subsets differ from Stage 10")
    if fold_trial["strict_objective_failure"] != "True":
        raise ValueError("Stage 11 source trial was not a strict greedy failure")

    union_ids = [str(value) for value in expected["receptor_ids"]]
    if set(union_ids) != set().union(*map(set, candidates.values())):
        raise ValueError("Stage 11 receptor union differs from frozen candidates")
    source_by_id = {row["conformer_id"]: row for row in source_receptors}
    if len(source_by_id) != len(source_receptors):
        raise ValueError("Stage 11 source receptor manifest contains duplicates")
    try:
        receptors = [source_by_id[receptor_id] for receptor_id in union_ids]
    except KeyError as error:
        raise ValueError(f"Stage 11 receptor is absent: {error.args[0]}") from error
    if any(row["status"] != "ok" for row in receptors):
        raise ValueError("Stage 11 receptor union contains a failed receptor")
    for receptor in receptors:
        path = rooted_path(root, receptor["receptor_pdbqt"])
        if not path.is_file() or file_sha256(path) != receptor[
            "receptor_pdbqt_sha256"
        ].upper():
            raise ValueError(
                f"Stage 11 receptor identity differs: {receptor['conformer_id']}"
            )

    if len(ligands) != int(expected["ligand_count"]):
        raise ValueError("Stage 11 ligand count differs")
    if len({row["ligand_id"] for row in ligands}) != len(ligands):
        raise ValueError("Stage 11 ligand manifest contains duplicate IDs")
    labels = Counter(row["label"] for row in ligands)
    if labels != Counter(
        {key: int(value) for key, value in dict(expected["label_counts"]).items()}
    ):
        raise ValueError("Stage 11 ligand labels differ")
    if {row["split"] for row in ligands} != {boundary["allowed_split"]}:
        raise ValueError("Stage 11 observed a non-validation ligand")
    if {row["selection_role"] for row in ligands} != {
        boundary["allowed_selection_role"]
    }:
        raise ValueError("Stage 11 ligand selection role differs")
    if any(row["pdbqt_status"] != "ok" for row in ligands):
        raise ValueError("Stage 11 source manifest contains a failed PDBQT")

    macrocycles = [row for row in ligands if has_manifest_macrocycle(row)]
    if len(macrocycles) != int(expected["macrocycle_replacement_count"]):
        raise ValueError("Stage 11 macrocycle replacement count differs")
    if Counter(row["label"] for row in macrocycles) != Counter(
        {"decoy": len(macrocycles)}
    ):
        raise ValueError("Stage 11 macrocycle label inventory differs")
    macrocycle_ids = {row["ligand_id"] for row in macrocycles}
    for row in ligands:
        pdbqt = rooted_path(root, row["pdbqt_path"])
        if not pdbqt.is_file() or file_sha256(pdbqt) != row["pdbqt_sha256"].upper():
            raise ValueError(f"Stage 11 source PDBQT differs: {row['ligand_id']}")
        if row["ligand_id"] in macrocycle_ids:
            if not macrocycle_closure_atom_types(pdbqt):
                raise ValueError(
                    f"Stage 11 registered macrocycle lacks pseudoatoms: {row['ligand_id']}"
                )
            sdf = rooted_path(root, row["sdf_path"])
            if not sdf.is_file():
                raise FileNotFoundError(sdf)

    audit = {
        "status": "source_inputs_ok",
        "receptor_count": len(receptors),
        "receptor_ids": union_ids,
        "ligand_count": len(ligands),
        "label_counts": dict(sorted(labels.items())),
        "macrocycle_replacement_count": len(macrocycles),
        "macrocycle_ligand_ids": sorted(macrocycle_ids),
        "candidate_subsets": {
            key: list(value) for key, value in candidates.items()
        },
        "validation_rows": len(ligands),
        "train_score_rows": 0,
        "test_rows": 0,
    }
    return receptors, ligands, macrocycles, audit


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    verify_implementation(root, config, "preparer", Path(__file__))
    verify_implementation(
        root,
        config,
        "stage09_preparation_helper",
        Path(__file__).with_name("prepare_stage09_mk14_train696_inputs.py"),
    )
    receptors, ligands, macrocycles, source_audit = validate_source_inputs(
        root, config
    )
    meeko_version = importlib.metadata.version("meeko")
    if meeko_version != str(dict(config["preparation"])["meeko_version"]):
        raise ValueError(f"Stage 11 Meeko version differs: {meeko_version}")
    meeko_script = find_meeko_script()

    outputs = dict(config["outputs"])
    receptor_output = rooted_path(root, str(outputs["prepared_receptor_manifest"]))
    ligand_output = rooted_path(root, str(outputs["prepared_ligand_manifest"]))
    summary_output = rooted_path(root, str(outputs["preparation_summary"]))
    rigid_directory = rooted_path(root, str(outputs["rigid_macrocycle_directory"]))
    if not overwrite and any(
        path.exists() for path in (receptor_output, ligand_output, summary_output)
    ):
        raise FileExistsError("Stage 11 prepared inputs exist; pass --overwrite")

    candidates = candidate_subsets(config)
    receptor_rows: list[dict[str, object]] = []
    for receptor in receptors:
        receptor_id = receptor["conformer_id"]
        roles = sorted(
            candidate_id
            for candidate_id, subset in candidates.items()
            if receptor_id in subset
        )
        receptor_rows.append({**receptor, "stage11_candidate_roles": ";".join(roles)})

    macrocycle_ids = {row["ligand_id"] for row in macrocycles}
    rigid_by_id: dict[str, dict[str, object]] = {}
    for source_index, row in enumerate(ligands):
        if row["ligand_id"] not in macrocycle_ids:
            continue
        sdf = rooted_path(root, row["sdf_path"])
        destination = rigid_directory / f"{safe_filename(row['ligand_id'])}.pdbqt"
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        prepared = prepare_rigid_pdbqt(meeko_script, sdf, destination)
        prepared["pdbqt_path"] = relative_path(root, destination)
        rigid_by_id[row["ligand_id"]] = {
            **prepared,
            "source_manifest_index": source_index,
            "source_pdbqt_path": row["pdbqt_path"],
            "source_pdbqt_sha256": row["pdbqt_sha256"],
            "sdf_sha256": file_sha256(sdf),
        }

    ligand_rows: list[dict[str, object]] = []
    for source_index, source in enumerate(ligands):
        ligand_id = source["ligand_id"]
        row: dict[str, object] = {
            **source,
            "source_manifest_index": source_index,
            "seed_offset": source_index,
            "preparation_variant": "original_meeko_flexible",
            "source_pdbqt_path": source["pdbqt_path"],
            "source_pdbqt_sha256": source["pdbqt_sha256"],
        }
        if ligand_id in rigid_by_id:
            rigid = rigid_by_id[ligand_id]
            for field in PDBQT_FIELDS:
                row[field] = rigid[field]
            row["preparation_variant"] = "meeko_rigid_macrocycles"
            row["sdf_sha256"] = rigid["sdf_sha256"]
        ligand_rows.append(row)

    remaining = []
    for row in ligand_rows:
        path = rooted_path(root, str(row["pdbqt_path"]))
        if file_sha256(path) != str(row["pdbqt_sha256"]).upper():
            raise ValueError(f"Stage 11 prepared PDBQT differs: {row['ligand_id']}")
        if macrocycle_closure_atom_types(path):
            remaining.append(str(row["ligand_id"]))
    if remaining:
        raise ValueError(f"Stage 11 prepared inputs retain pseudoatoms: {remaining}")
    variants = Counter(str(row["preparation_variant"]) for row in ligand_rows)
    expected_variants = Counter(
        {
            "original_meeko_flexible": len(ligand_rows) - len(macrocycles),
            "meeko_rigid_macrocycles": len(macrocycles),
        }
    )
    if variants != expected_variants:
        raise ValueError("Stage 11 preparation variants differ")

    write_csv(receptor_output, receptor_rows)
    write_csv(ligand_output, ligand_rows)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage11_fresh_validation_unidock_inputs_ok",
        "operation": "validation-only Uni-Dock compatibility preparation",
        "config": {
            "path": relative_path(root, config_path),
            "sha256": file_sha256(config_path),
        },
        "source_audit": source_audit,
        "preparation_variant_counts": dict(sorted(variants.items())),
        "closure_pseudoatom_ligand_count": 0,
        "order_preserved": [row["ligand_id"] for row in ligand_rows]
        == [row["ligand_id"] for row in ligands],
        "meeko": {
            "version": meeko_version,
            "script": meeko_script.as_posix(),
            "rigid_macrocycles": True,
        },
        "outputs": {
            "receptor_manifest": {
                "path": relative_path(root, receptor_output),
                "sha256": file_sha256(receptor_output),
            },
            "ligand_manifest": {
                "path": relative_path(root, ligand_output),
                "sha256": file_sha256(ligand_output),
            },
        },
        "data_boundary": {
            "validation_rows_read": len(ligand_rows),
            "train_score_rows_read": 0,
            "test_rows_read": 0,
        },
        "interpretation_note": (
            "This step prepares frozen validation structures only. It does not "
            "read a validation score, calculate enrichment, or alter a candidate."
        ),
    }
    write_json(summary_output, result)
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
