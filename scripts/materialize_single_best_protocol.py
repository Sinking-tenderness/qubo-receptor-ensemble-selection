"""Materialize a locked-test single-receptor protocol from development data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids


METRIC_KEYS = (
    "bedroc_alpha_20",
    "pr_auc_average_precision",
    "roc_auc",
    "EF1%",
    "EF5%",
    "EF10%",
)
MATRIX_METADATA_COLUMNS = ("target_id", "ligand_id", "label")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("receptor ranking is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require_hash(path: Path, expected: str) -> str:
    observed = file_sha256(path)
    if observed != expected.upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return observed


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    required = {
        "schema_version",
        "experiment_id",
        "purpose",
        "upstream_selector",
        "stability_evidence",
        "inputs",
        "input_sha256",
        "receptor_ids",
        "expected",
        "selection",
        "outputs",
        "interpretation_boundary",
    }
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("single-best materialization config is incomplete")
    receptors = config["receptor_ids"]
    if (
        not isinstance(receptors, list)
        or not receptors
        or len({str(value) for value in receptors}) != len(receptors)
    ):
        raise ValueError("receptor_ids must be nonempty and unique")
    selection = config["selection"]
    if (
        selection.get("protocol_id") != "single_best_fail_closed_v1"
        or selection.get("method") != "single_best"
        or selection.get("aggregation") != "min_score"
        or selection.get("selection_metric") != "bedroc_alpha_20"
        or selection.get("score_direction") != "lower_is_better"
        or selection.get("development_splits") != ["train", "validation"]
        or selection.get("locked_split") != "test"
        or selection.get("tie_breakers")
        != ["pr_auc_average_precision", "roc_auc", "receptor_id"]
    ):
        raise ValueError("single-best materialization rule is invalid")
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    if set(inputs) != {
        "primary_matrix",
        "receptor_manifest",
        "sensitivity_matrix",
        "split_manifest",
    } or set(hashes) != set(inputs):
        raise ValueError("materialization inputs and hashes are incomplete")
    outputs = config["outputs"]
    if set(outputs) != {"receptor_ranking_csv", "protocol_json"}:
        raise ValueError("materialization outputs are incomplete")
    return config


def validate_upstream_selector(specification: dict[str, object]) -> dict[str, object]:
    path = Path(str(specification["path"]))
    require_hash(path, str(specification["sha256"]))
    summary = json.loads(path.read_text(encoding="ascii"))
    selected = summary.get("selected_protocol", {})
    if (
        summary.get("status")
        != "fallback_selected_no_reliable_qubo_candidate"
        or selected.get("protocol_id") != specification["protocol_id"]
        or selected.get("method") != "single_best"
        or summary.get("test_lock", {}).get("scores_evaluated") is not False
    ):
        raise ValueError("upstream selector did not preserve the fallback test lock")
    return {
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "status": summary["status"],
        "protocol_id": selected["protocol_id"],
    }


def partition_manifest(
    rows: list[dict[str, str]],
    development_splits: set[str],
    locked_split: str,
) -> tuple[dict[str, str], set[str], set[str]]:
    if not rows:
        raise ValueError("split manifest is empty")
    ids = [row["ligand_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("split manifest has duplicate ligand IDs")
    labels = {row["ligand_id"]: row["label"] for row in rows}
    if not set(labels.values()).issubset({"active", "decoy"}):
        raise ValueError("split manifest contains unsupported labels")
    development_ids = {
        row["ligand_id"]
        for row in rows
        if row["split"] in development_splits
    }
    locked_ids = {
        row["ligand_id"] for row in rows if row["split"] == locked_split
    }
    known_splits = development_splits | {locked_split}
    if any(row["split"] not in known_splits for row in rows):
        raise ValueError("split manifest contains an unexpected split")
    if not development_ids or not locked_ids or development_ids & locked_ids:
        raise ValueError("development and locked split partition is invalid")
    return labels, development_ids, locked_ids


def validate_expected_counts(
    labels: dict[str, str],
    development_ids: set[str],
    locked_ids: set[str],
    expected: dict[str, object],
) -> None:
    if len(development_ids) != int(expected["development_ligand_count"]):
        raise ValueError("development ligand count differs")
    if len(locked_ids) != int(expected["locked_test_ligand_count"]):
        raise ValueError("locked-test ligand count differs")
    for key, ligand_ids in (
        ("development_label_counts", development_ids),
        ("locked_test_label_counts", locked_ids),
    ):
        observed = Counter(labels[ligand_id] for ligand_id in ligand_ids)
        configured = {
            str(label): int(count)
            for label, count in expected[key].items()
        }
        if dict(observed) != configured:
            raise ValueError(f"{key} differs")


def read_development_matrix(
    path: Path,
    receptor_ids: list[str],
    labels: dict[str, str],
    development_ids: set[str],
    locked_ids: set[str],
) -> list[dict[str, object]]:
    raw_rows = read_csv(path)
    if not raw_rows:
        raise ValueError("development matrix is empty")
    expected_columns = set(MATRIX_METADATA_COLUMNS) | set(receptor_ids)
    if set(raw_rows[0]) != expected_columns:
        raise ValueError("development matrix columns differ")
    matrix_ids = [row["ligand_id"] for row in raw_rows]
    if len(matrix_ids) != len(set(matrix_ids)):
        raise ValueError("development matrix has duplicate ligand IDs")
    if set(matrix_ids) != development_ids:
        raise ValueError("matrix IDs are not exactly the development split")
    if set(matrix_ids) & locked_ids:
        raise ValueError("locked-test ligand appears in development matrix")
    rows: list[dict[str, object]] = []
    for row in raw_rows:
        ligand_id = row["ligand_id"]
        if row["label"] != labels[ligand_id]:
            raise ValueError(f"label differs for {ligand_id}")
        scores = {receptor_id: float(row[receptor_id]) for receptor_id in receptor_ids}
        if any(not math.isfinite(value) for value in scores.values()):
            raise ValueError(f"non-finite score for {ligand_id}")
        rows.append(
            {
                "target_id": row["target_id"],
                "ligand_id": ligand_id,
                "label": row["label"],
                **scores,
            }
        )
    return rows


def rank_receptors(
    matrix_rows: list[dict[str, object]], receptor_ids: list[str]
) -> list[dict[str, object]]:
    ranking: list[dict[str, object]] = []
    for receptor_id in receptor_ids:
        records = {
            str(row["ligand_id"]): {
                "label": row["label"],
                "score": float(row[receptor_id]),
            }
            for row in matrix_rows
        }
        metrics = ranked_metrics_with_ids(records)
        ranking.append(
            {
                "receptor_id": receptor_id,
                **{key: metrics[key] for key in METRIC_KEYS},
                "ligand_count": metrics["ligand_count"],
                "active_count": metrics["active_count"],
                "top10_active_count": metrics["top10_active_count"],
                "top10_ligand_ids": json.dumps(metrics["top10_ligand_ids"]),
            }
        )
    ranking.sort(
        key=lambda row: (
            -float(row["bedroc_alpha_20"]),
            -float(row["pr_auc_average_precision"]),
            -float(row["roc_auc"]),
            str(row["receptor_id"]),
        )
    )
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
        row["selected"] = index == 1
    return ranking


def resolve_receptor_artifact(
    path: Path, receptor_ids: set[str], selected_receptor: str
) -> dict[str, object]:
    rows = read_csv(path)
    if not rows:
        raise ValueError("receptor manifest is empty")
    ids = [row["conformer_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != receptor_ids:
        raise ValueError("receptor manifest IDs differ")
    if any(row["preparation_status"] != "ok" for row in rows):
        raise ValueError("receptor manifest contains a failed preparation")
    selected = next(
        row for row in rows if row["conformer_id"] == selected_receptor
    )
    pdb_path = Path(selected["pdb_path"])
    pdbqt_path = Path(selected["receptor_pdbqt_path"])
    return {
        "source_type": selected["source_type"],
        "source_identifier": selected["source_identifier"],
        "chain": selected["chain"],
        "selected_altloc": selected["selected_altloc"],
        "pdb": {
            "path": pdb_path.as_posix(),
            "sha256": require_hash(pdb_path, selected["pdb_sha256"]),
        },
        "receptor_pdbqt": {
            "path": pdbqt_path.as_posix(),
            "sha256": require_hash(
                pdbqt_path, selected["receptor_pdbqt_sha256"]
            ),
        },
        "pocket_residue_count": int(selected["pocket_residue_count"]),
        "pdbqt_atom_count": int(selected["pdbqt_atom_count"]),
        "pdbqt_hetatm_count": int(selected["pdbqt_hetatm_count"]),
        "pdbqt_hydrogen_like_atom_count": int(
            selected["pdbqt_hydrogen_like_atom_count"]
        ),
        "pdbqt_charge_min": float(selected["pdbqt_charge_min"]),
        "pdbqt_charge_max": float(selected["pdbqt_charge_max"]),
        "pdbqt_autodock_atom_types": selected[
            "pdbqt_autodock_atom_types"
        ].split(";"),
    }


def collect_stability_evidence(
    specification: dict[str, object],
    receptor_ids: set[str],
    expected_repeat_count: int,
    expected_outer_fold_count: int,
) -> dict[str, object]:
    path = Path(str(specification["path"]))
    require_hash(path, str(specification["sha256"]))
    repeated = json.loads(path.read_text(encoding="ascii"))
    if (
        repeated.get("test_lock", {}).get("scores_evaluated") is not False
        or repeated.get("successful_repeat_count") != expected_repeat_count
        or repeated.get("requested_repeat_count") != expected_repeat_count
        or repeated.get("test_lock", {}).get(
            "all_successful_repeat_audits_passed"
        )
        is not True
    ):
        raise ValueError("stability summary is incomplete or unlocked")
    counts: Counter[str] = Counter()
    fold_keys: set[tuple[int, int]] = set()
    source_runs = repeated.get("source_runs", [])
    if len(source_runs) != expected_repeat_count:
        raise ValueError("stability source-run count differs")
    for source in source_runs:
        seed = int(source["fold_seed"])
        source_summary_path = Path(source["summary"]["path"])
        require_hash(source_summary_path, source["summary"]["sha256"])
        source_summary = json.loads(source_summary_path.read_text(encoding="ascii"))
        if source_summary.get("test_lock", {}).get("scores_evaluated") is not False:
            raise ValueError("stability source evaluated locked test scores")
        output = source_summary["outputs"]["outer_fold_results_csv"]
        outer_path = Path(output["path"])
        require_hash(outer_path, output["sha256"])
        rows = [row for row in read_csv(outer_path) if row["method"] == "single_best"]
        if len(rows) != expected_outer_fold_count:
            raise ValueError("single-best outer-fold count differs")
        for row in rows:
            receptor_id = row["subset"]
            if receptor_id not in receptor_ids or "+" in receptor_id:
                raise ValueError("single-best stability subset is invalid")
            key = (seed, int(row["outer_fold"]))
            if key in fold_keys:
                raise ValueError("duplicate stability outer fold")
            fold_keys.add(key)
            counts[receptor_id] += 1
    total = expected_repeat_count * expected_outer_fold_count
    if len(fold_keys) != total:
        raise ValueError("stability outer-fold coverage differs")
    return {
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "repeat_count": expected_repeat_count,
        "outer_fold_count_per_repeat": expected_outer_fold_count,
        "total_outer_folds": total,
        "single_best_selection_counts": dict(sorted(counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    outputs = {key: Path(str(value)) for key, value in config["outputs"].items()}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("materialized protocol outputs already exist")

    upstream = validate_upstream_selector(config["upstream_selector"])
    input_paths = {
        key: Path(str(value)) for key, value in config["inputs"].items()
    }
    input_records = {
        key: {
            "path": path.as_posix(),
            "sha256": require_hash(path, config["input_sha256"][key]),
        }
        for key, path in input_paths.items()
    }
    selection = config["selection"]
    manifest_rows = read_csv(input_paths["split_manifest"])
    labels, development_ids, locked_ids = partition_manifest(
        manifest_rows,
        set(selection["development_splits"]),
        str(selection["locked_split"]),
    )
    validate_expected_counts(
        labels, development_ids, locked_ids, config["expected"]
    )
    receptor_ids = [str(value) for value in config["receptor_ids"]]
    matrix_rows = read_development_matrix(
        input_paths["primary_matrix"],
        receptor_ids,
        labels,
        development_ids,
        locked_ids,
    )
    ranking = rank_receptors(matrix_rows, receptor_ids)
    write_csv(outputs["receptor_ranking_csv"], ranking)
    selected = ranking[0]
    sensitivity_rows = read_development_matrix(
        input_paths["sensitivity_matrix"],
        receptor_ids,
        labels,
        development_ids,
        locked_ids,
    )
    sensitivity_ranking = rank_receptors(sensitivity_rows, receptor_ids)
    sensitivity_selected = next(
        row
        for row in sensitivity_ranking
        if row["receptor_id"] == selected["receptor_id"]
    )
    selected_artifact = resolve_receptor_artifact(
        input_paths["receptor_manifest"],
        set(receptor_ids),
        str(selected["receptor_id"]),
    )
    stability = collect_stability_evidence(
        config["stability_evidence"],
        set(receptor_ids),
        int(config["expected"]["stability_repeat_count"]),
        int(config["expected"]["stability_outer_fold_count"]),
    )
    selected_count = int(
        stability["single_best_selection_counts"].get(
            selected["receptor_id"], 0
        )
    )
    stability["selected_receptor_outer_fold_count"] = selected_count
    stability["selected_receptor_outer_fold_fraction"] = (
        selected_count / int(stability["total_outer_folds"])
    )

    implementation_path = Path(__file__)
    protocol = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "operation": "development-only single-best protocol materialization",
        "status": "single_best_protocol_materialized_test_locked",
        "config": {
            "path": args.config.as_posix(),
            "sha256": file_sha256(args.config),
        },
        "implementation": {
            "path": f"scripts/{implementation_path.name}",
            "sha256": file_sha256(implementation_path),
        },
        "upstream_selector": upstream,
        "inputs": input_records,
        "selection_rule": selection,
        "selected_receptor": {
            "receptor_id": selected["receptor_id"],
            "development_metrics_primary": {
                key: selected[key] for key in METRIC_KEYS
            },
            "development_rank_primary": selected["rank"],
            "development_metrics_sensitivity": {
                key: sensitivity_selected[key] for key in METRIC_KEYS
            },
            "development_rank_sensitivity": sensitivity_selected["rank"],
            "receptor_count_compared": len(receptor_ids),
            "artifact": selected_artifact,
        },
        "stability_evidence": stability,
        "data_audit": {
            "development_ligand_count": len(development_ids),
            "locked_test_ligand_count": len(locked_ids),
            "primary_matrix_locked_test_overlap_count": len(
                {str(row["ligand_id"]) for row in matrix_rows} & locked_ids
            ),
            "sensitivity_matrix_locked_test_overlap_count": len(
                {str(row["ligand_id"]) for row in sensitivity_rows}
                & locked_ids
            ),
        },
        "test_lock": {
            "split": selection["locked_split"],
            "score_cells_read": 0,
            "scores_evaluated": False,
            "metrics_computed": False,
            "release_authorized": False,
        },
        "outputs": {
            "receptor_ranking_csv": {
                "path": outputs["receptor_ranking_csv"].as_posix(),
                "sha256": file_sha256(outputs["receptor_ranking_csv"]),
            }
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    outputs["protocol_json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["protocol_json"].write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "selected_receptor": selected["receptor_id"],
                "development_bedroc_alpha_20": selected[
                    "bedroc_alpha_20"
                ],
                "sensitivity_bedroc_alpha_20": sensitivity_selected[
                    "bedroc_alpha_20"
                ],
                "outer_fold_selection_fraction": stability[
                    "selected_receptor_outer_fold_fraction"
                ],
                "test_evaluated": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
