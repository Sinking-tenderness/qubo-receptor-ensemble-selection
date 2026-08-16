"""Merge rigid-macrocycle CPU replacements into five-receptor Train-160 scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .run_unidock_gpu_equivalence import (
        file_sha256,
        read_csv,
        read_json,
        write_csv,
        write_json,
    )
except ImportError:
    from run_unidock_gpu_equivalence import (
        file_sha256,
        read_csv,
        read_json,
        write_csv,
        write_json,
    )


def verified_path(descriptor: dict[str, object]) -> Path:
    path = Path(str(descriptor["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = file_sha256(path)
    if observed != str(descriptor["sha256"]).upper():
        raise ValueError(f"SHA-256 differs for {path}: {observed}")
    return path


def score_key(row: dict[str, str]) -> tuple[str, str]:
    return row["receptor_id"], row["ligand_id"]


def merge_rows(
    original_rows: list[dict[str, str]],
    rigid_rows: list[dict[str, str]],
    receptor_ids: set[str],
    replacement_ids: set[str],
) -> list[dict[str, object]]:
    selected = [row for row in original_rows if row["receptor_id"] in receptor_ids]
    original_by_key = {score_key(row): row for row in selected}
    if len(original_by_key) != len(selected):
        raise ValueError("selected original CPU scores contain duplicate keys")
    rigid_by_key = {score_key(row): row for row in rigid_rows}
    if len(rigid_by_key) != len(rigid_rows):
        raise ValueError("rigid CPU scores contain duplicate keys")
    expected_replacements = {
        (receptor_id, ligand_id)
        for receptor_id in receptor_ids
        for ligand_id in replacement_ids
    }
    if set(rigid_by_key) != expected_replacements:
        raise ValueError("rigid CPU replacement keys differ")
    if not expected_replacements.issubset(original_by_key):
        raise ValueError("rigid CPU replacements are absent from original scores")

    output: list[dict[str, object]] = []
    for key, original in sorted(
        original_by_key.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        replacement = rigid_by_key.get(key)
        source = replacement if replacement is not None else original
        if source["status"] != "ok":
            raise ValueError(f"non-ok CPU source score: {key}")
        if replacement is not None and replacement["label"] != original["label"]:
            raise ValueError(f"CPU replacement label differs: {key}")
        output.append(
            {
                "target_id": source["target_id"],
                "ligand_id": source["ligand_id"],
                "label": source["label"],
                "receptor_id": source["receptor_id"],
                "representative_score": source["representative_score"],
                "representative_method": source["representative_method"],
                "status": "ok",
                "source_score_protocol": (
                    "official_vina_e32_rigid_macrocycle_replacement"
                    if replacement is not None
                    else "official_vina_e32_original_nonmacrocycle"
                ),
            }
        )
    return output


def verify_rigid_raw_seeds(
    rigid_summary: dict[str, object],
    base_seed: int,
    offsets: dict[str, int],
) -> dict[str, int]:
    descriptor = rigid_summary["outputs"]["combined_raw_scores_csv"]
    path = verified_path(descriptor)
    observed: dict[str, set[int]] = {}
    for row in read_csv(path):
        if row["status"] != "ok":
            raise ValueError(f"rigid raw CPU score failed: {row['ligand_id']}")
        observed.setdefault(row["ligand_id"], set()).add(int(row["seed"]))
    expected = {
        ligand_id: base_seed + offset
        for ligand_id, offset in offsets.items()
    }
    normalized = {
        ligand_id: next(iter(values))
        for ligand_id, values in observed.items()
        if len(values) == 1
    }
    if normalized != expected:
        raise ValueError(
            f"rigid CPU seeds differ: expected={expected}, observed={normalized}"
        )
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="ascii"))
    receptor_ids = set(str(value) for value in config["receptor_ids"])
    replacement_ids = set(str(value) for value in config["replacement_ligand_ids"])
    offsets = {
        str(key): int(value) for key, value in config["seed_offsets"].items()
    }
    if set(offsets) != replacement_ids:
        raise ValueError("seed-offset ligand IDs differ from replacement IDs")
    aggregate_rows: list[dict[str, object]] = []

    for seed in config["seeds"]:
        seed_id = str(seed["seed_id"])
        base_seed = int(seed["base_seed"])
        original_summary_path = verified_path(seed["original_summary"])
        original_scores_path = verified_path(seed["original_scores"])
        rigid_summary_path = verified_path(seed["rigid_summary"])
        rigid_scores_path = verified_path(seed["rigid_scores"])
        original_summary = read_json(original_summary_path)
        rigid_summary = read_json(rigid_summary_path)
        if int(original_summary["docking_parameters"]["base_seed"]) != base_seed:
            raise ValueError(f"original base seed differs: {seed_id}")
        if int(rigid_summary["docking_parameters"]["base_seed"]) != base_seed:
            raise ValueError(f"rigid base seed differs: {seed_id}")
        observed_seeds = verify_rigid_raw_seeds(
            rigid_summary, base_seed, offsets
        )
        merged = merge_rows(
            read_csv(original_scores_path),
            read_csv(rigid_scores_path),
            receptor_ids,
            replacement_ids,
        )
        expected_pairs = int(config["expected_pair_count_per_seed"])
        if len(merged) != expected_pairs:
            raise ValueError(f"merged CPU pair count differs: {seed_id}")
        output_scores = Path(str(seed["output_scores_path"]))
        output_summary = Path(str(seed["output_summary_path"]))
        if not args.overwrite and (
            output_scores.exists() or output_summary.exists()
        ):
            raise FileExistsError(f"merged CPU outputs exist: {seed_id}")
        write_csv(output_scores, merged)
        summary = {
            "schema_version": "1.0",
            "experiment_id": f"{config['experiment_id']}-{seed_id}",
            "status": "ok",
            "operation": "five-receptor Train-160 CPU reference with 780 inherited nonmacrocycle scores and 20 rigid-macrocycle replacement scores",
            "config": {
                "path": args.config.as_posix(),
                "sha256": file_sha256(args.config),
            },
            "docking_parameters": {
                "base_seed": base_seed,
                "representative_method": "pose_rank_1",
                "vina_version": "1.2.7",
                "exhaustiveness": 32,
                "cpu_per_replacement_job": 2,
            },
            "receptor_count": len(receptor_ids),
            "ligand_count": int(config["expected_ligand_count"]),
            "observed_receptor_ligand_pairs": len(merged),
            "inherited_nonmacrocycle_pair_count": len(merged)
            - len(replacement_ids) * len(receptor_ids),
            "rigid_macrocycle_replacement_pair_count": len(replacement_ids)
            * len(receptor_ids),
            "measured_wall_runtime_seconds": float(
                rigid_summary["measured_wall_runtime_seconds"]
            ),
            "timing_scope": "incremental 20-pair rigid-macrocycle replacement run only; inherited score timing is recorded separately in the GPU equivalence config",
            "rigid_raw_seeds": observed_seeds,
            "sources": {
                "original_summary": seed["original_summary"],
                "original_scores": seed["original_scores"],
                "rigid_summary": seed["rigid_summary"],
                "rigid_scores": seed["rigid_scores"],
            },
            "outputs": {
                "representative_scores": {
                    "path": output_scores.as_posix(),
                    "sha256": file_sha256(output_scores),
                }
            },
            "interpretation_note": config["interpretation_boundary"],
        }
        write_json(output_summary, summary)
        aggregate_rows.append(
            {
                "seed_id": seed_id,
                "base_seed": base_seed,
                "pair_count": len(merged),
                "output_scores_path": output_scores.as_posix(),
                "output_scores_sha256": file_sha256(output_scores),
                "output_summary_path": output_summary.as_posix(),
                "output_summary_sha256": file_sha256(output_summary),
            }
        )

    aggregate_path = Path(str(config["aggregate_summary_path"]))
    aggregate = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "seed_count": len(aggregate_rows),
        "receptor_count": len(receptor_ids),
        "ligand_count": int(config["expected_ligand_count"]),
        "pair_count_per_seed": int(config["expected_pair_count_per_seed"]),
        "replacement_ligand_count": len(replacement_ids),
        "seed_outputs": aggregate_rows,
        "throughput_reference": config["throughput_reference"],
        "interpretation_note": config["interpretation_boundary"],
    }
    write_json(aggregate_path, aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
