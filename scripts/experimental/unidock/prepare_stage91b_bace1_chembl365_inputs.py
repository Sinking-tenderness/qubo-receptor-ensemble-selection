from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
from collections import Counter
from pathlib import Path

from scripts.experimental.unidock import prepare_stage42b_bace1_train266_inputs as common


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = common.read_json(config_path)
    stage91_path = root / str(config["inputs"]["stage91_result"])
    freeze_path = root / str(config["inputs"]["development_manifest_freeze"])
    source_path = root / str(config["inputs"]["development_ligand_manifest"])
    stage91 = common.read_json(stage91_path)
    freeze = common.read_json(freeze_path)
    if stage91.get("status") != "stage91_bace1_group_robust_rescue_preregistered":
        raise ValueError("Stage91 preregistration did not pass")
    if freeze.get("status") != "stage91b_bace1_development_manifest_frozen":
        raise ValueError("Stage91b development manifest freeze did not pass")
    if common.file_sha256(source_path) != str(freeze["output"]["sha256"]).upper():
        raise ValueError("Stage91b development manifest identity differs")

    source_rows = common.read_csv(source_path)
    if any(row["role"] != "development" for row in source_rows):
        raise ValueError("a nondevelopment row entered Stage91b")
    if any(str(row["docking_authorized"]).lower() != "true" for row in source_rows):
        raise ValueError("a development ligand lacks preparation authorization")
    expected = dict(config["expected"])
    if len(source_rows) != int(expected["ligand_count"]):
        raise ValueError("Stage91b development ligand count differs")
    labels = Counter(row["potency_label"] for row in source_rows)
    expected_labels = Counter(
        {key: int(value) for key, value in dict(expected["potency_label_counts"]).items()}
    )
    if labels != expected_labels:
        raise ValueError("Stage91b potency-label counts differ")
    core_series = {row["scaffold_group_id"] for row in source_rows if str(row["core_series"]).lower() == "true"}
    if len(core_series) != int(expected["core_series_count"]):
        raise ValueError("Stage91b core-series count differs")

    protocol = dict(config["preparation"])
    rdkit_version = importlib.metadata.version("rdkit")
    meeko_version = importlib.metadata.version("meeko")
    if rdkit_version != str(protocol["rdkit_version"]):
        raise ValueError(f"RDKit version differs: {rdkit_version}")
    if meeko_version != str(protocol["meeko_version"]):
        raise ValueError(f"Meeko version differs: {meeko_version}")
    meeko_script = common.find_meeko_script()

    outputs = dict(config["outputs"])
    run_directory = root / str(outputs["run_directory"])
    sdf_directory = run_directory / "sdf"
    pdbqt_directory = run_directory / "pdbqt"
    manifest_path = root / str(outputs["manifest_csv"])
    summary_path = root / str(outputs["summary_json"])
    if not overwrite and (manifest_path.exists() or summary_path.exists()):
        raise FileExistsError("Stage91b outputs exist; pass --overwrite")
    sdf_directory.mkdir(parents=True, exist_ok=True)
    pdbqt_directory.mkdir(parents=True, exist_ok=True)

    tasks = []
    for index, row in enumerate(source_rows):
        prepared_source = {
            **row,
            "source_smiles": row["canonical_smiles"],
            "label": row["potency_label"],
            "split": "development",
            "selection_role": "chembl_single_assay_development",
        }
        tasks.append(
            {
                "row": prepared_source,
                "root": str(root),
                "sdf_directory": str(sdf_directory),
                "pdbqt_directory": str(pdbqt_directory),
                "meeko_script": str(meeko_script),
                "index": index,
                "overwrite": overwrite,
                "base_seed": int(protocol["rdkit_embed_base_seed"]),
                "seed_offsets": list(protocol["deterministic_retry_seed_offsets"]),
            }
        )

    prepared_by_index: dict[int, dict[str, object]] = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(protocol["local_worker_count"])
    ) as executor:
        futures = {
            executor.submit(common.prepare_one, task): int(task["index"])
            for task in tasks
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            index = futures[future]
            prepared_by_index[index] = future.result()
            if completed % 25 == 0 or completed == len(tasks):
                print(f"prepared {completed}/{len(tasks)}", flush=True)
    prepared_rows = [prepared_by_index[index] for index in range(len(tasks))]
    if [row["ligand_id"] for row in prepared_rows] != [row["ligand_id"] for row in source_rows]:
        raise ValueError("Stage91b output order differs")
    if any(row["pdbqt_status"] != "ok" for row in prepared_rows):
        raise ValueError("Stage91b contains a failed PDBQT")
    for row in prepared_rows:
        for path_key, hash_key in (
            ("sdf_path", "sdf_sha256"),
            ("pdbqt_path", "pdbqt_sha256"),
        ):
            common.verified(root, {"path": row[path_key], "sha256": row[hash_key]})
        if common.macrocycle_closure_atom_types(root / str(row["pdbqt_path"])):
            raise ValueError(f"closure pseudoatom remains: {row['ligand_id']}")

    common.write_csv(manifest_path, prepared_rows)
    variants = Counter(str(row["preparation_variant"]) for row in prepared_rows)
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage91b_bace1_chembl365_unidock_inputs_ok",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": common.file_sha256(config_path),
        },
        "source_manifest": {
            "path": source_path.relative_to(root).as_posix(),
            "sha256": common.file_sha256(source_path),
        },
        "ligand_count": len(prepared_rows),
        "potency_label_counts": dict(sorted(labels.items())),
        "core_series_count": len(core_series),
        "preparation_variant_counts": dict(sorted(variants.items())),
        "closure_pseudoatom_ligand_count": 0,
        "failed_ligand_count": 0,
        "order_preserved": True,
        "runtime": {
            "rdkit_version": rdkit_version,
            "meeko_version": meeko_version,
        },
        "data_boundary": {
            "development_rows_read": len(prepared_rows),
            "confirmation_rows_read": 0,
            "locked_test_rows_read": 0,
            "docking_scores_read": 0,
        },
        "output": {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": common.file_sha256(manifest_path),
        },
        "next_gate": "independent Stage91b input audit before development docking",
    }
    common.write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage91b_bace1_chembl365_unidock_input_preparation.json"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
