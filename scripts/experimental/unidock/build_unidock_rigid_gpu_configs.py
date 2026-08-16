"""Build frozen consumed-train GPU configs from merged rigid CPU references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .run_unidock_gpu_equivalence import file_sha256, read_json, write_json
except ImportError:
    from run_unidock_gpu_equivalence import file_sha256, read_json, write_json


RECEPTOR_IDS = [
    "MK14_2BAJ_aligned",
    "MK14_2QD9_reference",
    "MK14_3K3J_aligned",
    "MK14_3KQ7_aligned",
    "MK14_3MPT_aligned",
]
BOX = {
    "center_x": -0.49,
    "center_y": 3.26,
    "center_z": 21.83,
    "size_x": 22,
    "size_y": 24,
    "size_z": 32,
}
GATE = {
    "require_complete_pairs": True,
    "maximum_overall_median_absolute_score_delta_kcal_per_mol": 0.5,
    "maximum_overall_p95_absolute_score_delta_kcal_per_mol": 1.0,
    "minimum_group_spearman": 0.95,
    "minimum_median_group_top5pct_overlap": 0.8,
    "minimum_throughput_speedup_vs_recorded_32vcpu": 5.0,
    "maximum_active_decoy_mean_delta_gap_kcal_per_mol": 0.5,
}
PROFILES = (
    {
        "profile_id": "detail_rigid_fix",
        "exhaustiveness": 512,
        "max_step": 40,
        "config_path": "configs/stage05_mk14_unidock_rigid_train160_detail_gpu_equivalence.json",
        "run_name": "stage05_mk14_unidock_rigid_train160_detail_gpu_equivalence",
        "purpose": "Isolate the effect of replacing four unsupported flexible-macrocycle PDBQTs while retaining the original Uni-Dock detail search parameters.",
    },
    {
        "profile_id": "enhanced_rigid_search",
        "exhaustiveness": 1024,
        "max_step": 80,
        "config_path": "configs/stage05_mk14_unidock_rigid_train160_enhanced_gpu_equivalence.json",
        "run_name": "stage05_mk14_unidock_rigid_train160_enhanced_gpu_equivalence",
        "purpose": "Test whether doubled Uni-Dock search threads and steps rescue the remaining consumed-train rank disagreement after rigid-macrocycle compatibility repair.",
    },
)


def verified(path: Path, expected_hash: str | None = None) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = file_sha256(path)
    if expected_hash is not None and observed != expected_hash.upper():
        raise ValueError(f"SHA-256 differs for {path}: {observed}")
    return {"path": path.as_posix(), "sha256": observed}


def build_config(
    profile: dict[str, object],
    receptor_manifest: dict[str, str],
    ligand_manifest: dict[str, str],
    cpu_runs: list[dict[str, object]],
    throughput_reference: dict[str, object],
) -> dict[str, object]:
    run_name = str(profile["run_name"])
    outputs = {
        "run_directory": f"results/runs/{run_name}",
        "gpu_scores_csv": f"results/runs/{run_name}/gpu_scores.csv",
        "gpu_batch_runs_csv": f"results/runs/{run_name}/gpu_batch_runs.csv",
        "gpu_summary_json": f"results/runs/{run_name}/gpu_run_summary.json",
        "pairwise_comparison_csv": f"results/runs/{run_name}/equivalence_pairwise.csv",
        "group_metrics_csv": f"results/runs/{run_name}/equivalence_group_metrics.csv",
        "equivalence_summary_json": f"results/runs/{run_name}/equivalence_summary.json",
    }
    return {
        "schema_version": "1.0",
        "experiment_id": f"{run_name.replace('_', '-')}-v1",
        "purpose": profile["purpose"],
        "data_boundary": {
            "allowed_split": "train",
            "allowed_selection_role": "development_train",
            "ligand_count": 160,
            "label_counts": {"active": 80, "decoy": 80},
            "validation_rows_permitted": 0,
            "test_rows_permitted": 0,
            "enrichment_metrics_permitted": False,
            "interpretation": "This post-failure protocol diagnostic uses consumed Train-160 rows only and cannot validate QUBO, enrichment, biological activity, or generalization.",
        },
        "inputs": {
            "receptor_manifest": receptor_manifest,
            "ligand_manifest": ligand_manifest,
            "cpu_seed_runs": cpu_runs,
        },
        "expected": {
            "receptor_ids": RECEPTOR_IDS,
            "receptor_count": 5,
            "ligand_count": 160,
            "seed_count": 3,
            "pair_count_per_seed": 800,
            "total_gpu_pair_count": 2400,
            "cpu_reference_pair_count_per_seed": 800,
        },
        "throughput_reference": throughput_reference,
        "unidock": {
            "executable": "unidock",
            "required_package_version": "1.1.3",
            "scoring": "vina",
            "profile_id": profile["profile_id"],
            "exhaustiveness": profile["exhaustiveness"],
            "max_step": profile["max_step"],
            "refine_step": 5,
            "num_modes": 1,
            "energy_range": 3,
            "verbosity": 1,
            "cuda_visible_devices": "0",
            "macrocycle_closure_pseudoatom_policy": "reject",
            "maximum_absolute_score_kcal_per_mol": 100.0,
            "box": BOX,
        },
        "equivalence_gate": GATE,
        "outputs": outputs,
        "decision_boundary": "A pass remains consumed-train evidence only. It does not authorize mixing engines, opening fresh validation/test rows, or claiming QUBO benefit; any later full-data GPU protocol requires a separate preregistration.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-reference-summary", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    aggregate = read_json(args.cpu_reference_summary)
    if aggregate.get("status") != "ok" or int(aggregate["seed_count"]) != 3:
        raise ValueError("merged CPU reference summary is incomplete")
    receptor_manifest = verified(
        Path("data/processed/stage05_mk14_unidock_train160_receptor_manifest.csv")
    )
    ligand_manifest = verified(
        Path("data/processed/stage05_mk14_unidock_rigid_train160_pdbqt_manifest.csv")
    )
    cpu_runs: list[dict[str, object]] = []
    for item in aggregate["seed_outputs"]:
        summary = verified(
            Path(str(item["output_summary_path"])),
            str(item["output_summary_sha256"]),
        )
        scores = verified(
            Path(str(item["output_scores_path"])),
            str(item["output_scores_sha256"]),
        )
        cpu_runs.append(
            {
                "seed_id": item["seed_id"],
                "base_seed": item["base_seed"],
                "summary_path": summary["path"],
                "summary_sha256": summary["sha256"],
                "scores_path": scores["path"],
                "scores_sha256": scores["sha256"],
            }
        )
    throughput_reference = dict(aggregate["throughput_reference"])
    outputs = []
    for profile in PROFILES:
        path = Path(str(profile["config_path"]))
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"GPU diagnostic config exists: {path}")
        config = build_config(
            profile,
            receptor_manifest,
            ligand_manifest,
            cpu_runs,
            throughput_reference,
        )
        write_json(path, config)
        outputs.append(
            {
                "profile_id": profile["profile_id"],
                "path": path.as_posix(),
                "sha256": file_sha256(path),
            }
        )
    result = {
        "schema_version": "1.0",
        "status": "ok",
        "cpu_reference_summary": {
            "path": args.cpu_reference_summary.as_posix(),
            "sha256": file_sha256(args.cpu_reference_summary),
        },
        "gpu_configs": outputs,
        "validation_rows": 0,
        "test_rows": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
