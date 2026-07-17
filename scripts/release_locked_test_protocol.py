"""Run one authorized evaluation of a fixed protocol on the locked test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids


BOOTSTRAP_METRICS = (
    "roc_auc",
    "pr_auc_average_precision",
    "bedroc_alpha_20",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_hash(path: Path, expected: str) -> str:
    observed = file_sha256(path)
    if observed != expected.upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return observed


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("test ranking is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    required = {
        "schema_version",
        "experiment_id",
        "purpose",
        "authorization",
        "inputs",
        "input_sha256",
        "fixed_protocol",
        "expected",
        "evaluation",
        "outputs",
        "interpretation_boundary",
    }
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("locked-test release config is incomplete")
    authorization = config["authorization"]
    if (
        authorization.get("approved") is not True
        or authorization.get("approved_by") != "project_owner"
        or authorization.get("scope")
        != "one_time_fixed_protocol_test_evaluation"
        or not authorization.get("authorization_id")
        or not authorization.get("approved_on")
    ):
        raise ValueError("locked-test release authorization is invalid")
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    expected_inputs = {
        "materialized_protocol",
        "primary_score_matrix",
        "sensitivity_score_matrix",
        "source_summary",
        "split_manifest",
    }
    if set(inputs) != expected_inputs or set(hashes) != expected_inputs:
        raise ValueError("locked-test release inputs are incomplete")
    fixed = config["fixed_protocol"]
    if (
        fixed.get("protocol_id") != "single_best_fail_closed_v1"
        or fixed.get("method") != "single_best"
        or fixed.get("receptor_id") != "CDK2_AF2_MD2NS_C06_F077"
        or not fixed.get("receptor_pdbqt_sha256")
    ):
        raise ValueError("fixed test protocol is invalid")
    evaluation = config["evaluation"]
    if (
        evaluation.get("locked_split") != "test"
        or evaluation.get("ranking_order")
        != ["docking_score_ascending", "ligand_id_ascending"]
        or evaluation.get("metrics")
        != [
            "roc_auc",
            "pr_auc_average_precision",
            "bedroc_alpha_20",
            "EF1%",
            "EF5%",
            "EF10%",
        ]
        or int(evaluation.get("bootstrap_iterations", 0)) <= 0
        or int(evaluation.get("bootstrap_seed", 0)) <= 0
        or evaluation.get("acceptance_threshold") is not None
    ):
        raise ValueError("locked-test evaluation rule is invalid")
    outputs = config["outputs"]
    if set(outputs) != {
        "primary_rankings_csv",
        "sensitivity_rankings_csv",
        "release_marker_json",
        "summary_json",
    }:
        raise ValueError("locked-test release outputs are incomplete")
    return config


def validate_materialized_protocol(
    path: Path, expected_hash: str, fixed: dict[str, object]
) -> dict[str, object]:
    require_hash(path, expected_hash)
    protocol = json.loads(path.read_text(encoding="ascii"))
    selected = protocol.get("selected_receptor", {})
    artifact = selected.get("artifact", {}).get("receptor_pdbqt", {})
    if (
        protocol.get("status")
        != "single_best_protocol_materialized_test_locked"
        or protocol.get("selection_rule", {}).get("protocol_id")
        != fixed["protocol_id"]
        or selected.get("receptor_id") != fixed["receptor_id"]
        or artifact.get("sha256") != fixed["receptor_pdbqt_sha256"]
        or protocol.get("test_lock", {}).get("scores_evaluated") is not False
        or protocol.get("test_lock", {}).get("release_authorized") is not False
    ):
        raise ValueError("materialized protocol does not match fixed release")
    artifact_path = Path(artifact["path"])
    require_hash(artifact_path, artifact["sha256"])
    return {
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "protocol_id": fixed["protocol_id"],
        "receptor_id": selected["receptor_id"],
        "receptor_pdbqt": {
            "path": artifact_path.as_posix(),
            "sha256": file_sha256(artifact_path),
        },
    }


def read_split_manifest(
    path: Path, locked_split: str
) -> tuple[dict[str, str], set[str], set[str]]:
    rows = read_csv_dicts(path)
    ids = [row["ligand_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("split manifest is empty or has duplicate IDs")
    labels = {row["ligand_id"]: row["label"] for row in rows}
    if not set(labels.values()).issubset({"active", "decoy"}):
        raise ValueError("split manifest contains unsupported labels")
    locked_ids = {
        row["ligand_id"] for row in rows if row["split"] == locked_split
    }
    development_ids = set(ids) - locked_ids
    if not locked_ids or not development_ids or locked_ids & development_ids:
        raise ValueError("locked-test partition is invalid")
    return labels, development_ids, locked_ids


def validate_expected_counts(
    labels: dict[str, str],
    development_ids: set[str],
    locked_ids: set[str],
    expected: dict[str, object],
) -> None:
    if len(labels) != int(expected["total_ligand_count"]):
        raise ValueError("total ligand count differs")
    if len(development_ids) != int(expected["development_ligand_count"]):
        raise ValueError("development ligand count differs")
    if len(locked_ids) != int(expected["locked_test_ligand_count"]):
        raise ValueError("locked-test ligand count differs")
    observed = {
        label: sum(labels[ligand_id] == label for ligand_id in locked_ids)
        for label in ("active", "decoy")
    }
    configured = {
        key: int(value)
        for key, value in expected["locked_test_label_counts"].items()
    }
    if observed != configured:
        raise ValueError("locked-test label counts differ")


def read_fixed_receptor_scores(
    path: Path,
    receptor_id: str,
    labels: dict[str, str],
    locked_ids: set[str],
) -> dict[str, dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("score matrix is empty") from exc
        required = {"ligand_id", "label", receptor_id}
        if not required.issubset(header):
            raise ValueError("score matrix lacks the fixed receptor")
        ligand_index = header.index("ligand_id")
        label_index = header.index("label")
        score_index = header.index(receptor_id)
        seen: set[str] = set()
        records: dict[str, dict[str, object]] = {}
        for values in reader:
            if len(values) != len(header):
                raise ValueError("score matrix row width differs")
            ligand_id = values[ligand_index]
            if ligand_id in seen:
                raise ValueError("score matrix has duplicate ligand IDs")
            seen.add(ligand_id)
            if ligand_id not in labels or values[label_index] != labels[ligand_id]:
                raise ValueError(f"score-matrix label differs for {ligand_id}")
            if ligand_id not in locked_ids:
                continue
            score = float(values[score_index])
            if not math.isfinite(score):
                raise ValueError(f"non-finite test score for {ligand_id}")
            records[ligand_id] = {"label": labels[ligand_id], "score": score}
    if seen != set(labels):
        raise ValueError("score matrix IDs do not match the split manifest")
    if set(records) != locked_ids:
        raise ValueError("fixed-receptor test scores are incomplete")
    return records


def ranking_rows(
    records: dict[str, dict[str, object]], matrix_role: str
) -> list[dict[str, object]]:
    ranked_ids = sorted(
        records,
        key=lambda ligand_id: (float(records[ligand_id]["score"]), ligand_id),
    )
    return [
        {
            "rank": rank,
            "ligand_id": ligand_id,
            "label": records[ligand_id]["label"],
            "docking_score": records[ligand_id]["score"],
            "ranking_score": -float(records[ligand_id]["score"]),
            "matrix_role": matrix_role,
        }
        for rank, ligand_id in enumerate(ranked_ids, start=1)
    ]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stratified_bootstrap_metrics(
    records: dict[str, dict[str, object]], iterations: int, seed: int
) -> dict[str, dict[str, float | int]]:
    active = [row for row in records.values() if row["label"] == "active"]
    decoy = [row for row in records.values() if row["label"] == "decoy"]
    if not active or not decoy:
        raise ValueError("stratified bootstrap requires both labels")
    rng = random.Random(seed)
    samples = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(iterations):
        selected = [rng.choice(active) for _ in active] + [
            rng.choice(decoy) for _ in decoy
        ]
        synthetic = {
            f"sample_{index:04d}": {
                "label": row["label"],
                "score": row["score"],
            }
            for index, row in enumerate(selected)
        }
        metrics = ranked_metrics_with_ids(synthetic)
        for metric in BOOTSTRAP_METRICS:
            samples[metric].append(float(metrics[metric]))
    return {
        metric: {
            "mean": statistics.fmean(values),
            "ci95_low": percentile(values, 0.025),
            "ci95_high": percentile(values, 0.975),
            "iterations": iterations,
        }
        for metric, values in samples.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--authorize-locked-test-release",
        action="store_true",
        help="consume the one-time project-owner authorization",
    )
    args = parser.parse_args()
    if not args.authorize_locked_test_release:
        raise PermissionError("explicit locked-test release flag is required")
    config = load_config(args.config)
    output_paths = {
        key: Path(str(value)) for key, value in config["outputs"].items()
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "locked-test release has already started; outputs cannot be overwritten"
        )

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
    fixed = config["fixed_protocol"]
    protocol = validate_materialized_protocol(
        input_paths["materialized_protocol"],
        config["input_sha256"]["materialized_protocol"],
        fixed,
    )
    source_summary = json.loads(
        input_paths["source_summary"].read_text(encoding="ascii")
    )
    if (
        source_summary.get("ligand_count") != int(config["expected"]["total_ligand_count"])
        or source_summary.get("receptor_count")
        != int(config["expected"]["source_receptor_count"])
    ):
        raise ValueError("source score summary dimensions differ")

    release_marker = {
        "schema_version": "1.0",
        "status": "locked_test_release_consumed",
        "authorization": config["authorization"],
        "config": {
            "path": args.config.as_posix(),
            "sha256": file_sha256(args.config),
        },
        "materialized_protocol": protocol,
        "overwrite_allowed": False,
    }
    output_paths["release_marker_json"].parent.mkdir(
        parents=True, exist_ok=True
    )
    output_paths["release_marker_json"].write_text(
        json.dumps(release_marker, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    evaluation = config["evaluation"]
    labels, development_ids, locked_ids = read_split_manifest(
        input_paths["split_manifest"], str(evaluation["locked_split"])
    )
    validate_expected_counts(
        labels, development_ids, locked_ids, config["expected"]
    )
    receptor_id = str(fixed["receptor_id"])
    primary_records = read_fixed_receptor_scores(
        input_paths["primary_score_matrix"],
        receptor_id,
        labels,
        locked_ids,
    )
    sensitivity_records = read_fixed_receptor_scores(
        input_paths["sensitivity_score_matrix"],
        receptor_id,
        labels,
        locked_ids,
    )
    write_csv(
        output_paths["primary_rankings_csv"],
        ranking_rows(primary_records, "across_seed_median_primary"),
    )
    write_csv(
        output_paths["sensitivity_rankings_csv"],
        ranking_rows(sensitivity_records, "across_seed_minimum_sensitivity"),
    )
    primary_metrics = ranked_metrics_with_ids(primary_records)
    sensitivity_metrics = ranked_metrics_with_ids(sensitivity_records)
    iterations = int(evaluation["bootstrap_iterations"])
    seed = int(evaluation["bootstrap_seed"])
    primary_bootstrap = stratified_bootstrap_metrics(
        primary_records, iterations, seed
    )
    sensitivity_bootstrap = stratified_bootstrap_metrics(
        sensitivity_records, iterations, seed + 1
    )

    implementation_path = Path(__file__)
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "operation": "authorized one-time fixed-protocol locked-test evaluation",
        "status": "locked_test_evaluated_once",
        "authorization": config["authorization"],
        "config": {
            "path": args.config.as_posix(),
            "sha256": file_sha256(args.config),
        },
        "implementation": {
            "path": f"scripts/{implementation_path.name}",
            "sha256": file_sha256(implementation_path),
        },
        "fixed_protocol": protocol,
        "inputs": input_records,
        "evaluation_rule": evaluation,
        "test_data": {
            "ligand_count": len(locked_ids),
            "active_count": sum(labels[value] == "active" for value in locked_ids),
            "decoy_count": sum(labels[value] == "decoy" for value in locked_ids),
            "receptor_score_columns_evaluated": [receptor_id],
            "receptor_score_column_count": 1,
        },
        "metrics": {
            "primary": primary_metrics,
            "sensitivity": sensitivity_metrics,
        },
        "stratified_bootstrap": {
            "primary": primary_bootstrap,
            "sensitivity": sensitivity_bootstrap,
        },
        "test_release": {
            "consumed": True,
            "repeat_or_overwrite_allowed": False,
            "release_marker": {
                "path": output_paths["release_marker_json"].as_posix(),
                "sha256": file_sha256(output_paths["release_marker_json"]),
            },
        },
        "outputs": {
            key: {
                "path": output_paths[key].as_posix(),
                "sha256": file_sha256(output_paths[key]),
            }
            for key in ("primary_rankings_csv", "sensitivity_rankings_csv")
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    output_paths["summary_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "receptor_id": receptor_id,
                "test_ligand_count": len(locked_ids),
                "primary_metrics": {
                    metric: primary_metrics[metric]
                    for metric in evaluation["metrics"]
                },
                "sensitivity_bedroc_alpha_20": sensitivity_metrics[
                    "bedroc_alpha_20"
                ],
                "release_consumed": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
