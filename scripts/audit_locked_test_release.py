"""Independently audit a completed fixed-protocol locked-test release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids


METRIC_KEYS = (
    "roc_auc",
    "pr_auc_average_precision",
    "bedroc_alpha_20",
    "EF1%",
    "EF5%",
    "EF10%",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    required = {
        "schema_version",
        "experiment_id",
        "purpose",
        "inputs",
        "input_sha256",
        "expected",
        "output_json",
        "interpretation_boundary",
    }
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("locked-test audit config is incomplete")
    expected_inputs = {
        "release_summary",
        "release_marker",
        "primary_rankings",
        "sensitivity_rankings",
        "split_manifest",
    }
    if (
        set(config["inputs"]) != expected_inputs
        or set(config["input_sha256"]) != expected_inputs
    ):
        raise ValueError("locked-test audit inputs are incomplete")
    expected = config["expected"]
    if (
        expected.get("status") != "locked_test_evaluated_once"
        or expected.get("receptor_id") != "CDK2_AF2_MD2NS_C06_F077"
        or int(expected.get("test_ligand_count", 0)) <= 0
        or int(expected.get("bootstrap_iterations", 0)) <= 0
    ):
        raise ValueError("locked-test audit expectations are invalid")
    return config


def locked_manifest_records(
    path: Path, split: str
) -> dict[str, str]:
    rows = read_csv(path)
    records = {
        row["ligand_id"]: row["label"]
        for row in rows
        if row["split"] == split
    }
    if not records or not set(records.values()).issubset({"active", "decoy"}):
        raise ValueError("locked manifest records are invalid")
    return records


def audit_ranking(
    path: Path,
    expected_labels: dict[str, str],
    matrix_role: str,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    rows = read_csv(path)
    if len(rows) != len(expected_labels):
        raise ValueError("test ranking row count differs")
    ids = [row["ligand_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_labels):
        raise ValueError("test ranking IDs differ")
    if [int(row["rank"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("test ranking sequence differs")
    records: dict[str, dict[str, object]] = {}
    previous_key: tuple[float, str] | None = None
    for row in rows:
        ligand_id = row["ligand_id"]
        score = float(row["docking_score"])
        ranking_score = float(row["ranking_score"])
        if (
            row["label"] != expected_labels[ligand_id]
            or row["matrix_role"] != matrix_role
            or not math.isfinite(score)
            or abs(ranking_score + score) > 1e-12
        ):
            raise ValueError(f"test ranking row differs for {ligand_id}")
        key = (score, ligand_id)
        if previous_key is not None and key < previous_key:
            raise ValueError("test ranking is not deterministically sorted")
        previous_key = key
        records[ligand_id] = {"label": row["label"], "score": score}
    return records, ranked_metrics_with_ids(records)


def compare_metrics(
    observed: dict[str, object], expected: dict[str, object]
) -> float:
    differences = [
        abs(float(observed[key]) - float(expected[key])) for key in METRIC_KEYS
    ]
    if observed["top10_ligand_ids"] != expected["top10_ligand_ids"]:
        raise ValueError("top-10 ligand IDs differ")
    if int(observed["top10_active_count"]) != int(
        expected["top10_active_count"]
    ):
        raise ValueError("top-10 active count differs")
    maximum = max(differences, default=0.0)
    if maximum > 1e-12:
        raise ValueError("released metrics do not reproduce")
    return maximum


def validate_bootstrap(
    summary: dict[str, object], expected_iterations: int
) -> None:
    for matrix in ("primary", "sensitivity"):
        for metric in ("roc_auc", "pr_auc_average_precision", "bedroc_alpha_20"):
            row = summary["stratified_bootstrap"][matrix][metric]
            values = [
                float(row["ci95_low"]),
                float(row["mean"]),
                float(row["ci95_high"]),
            ]
            if (
                int(row["iterations"]) != expected_iterations
                or any(not math.isfinite(value) for value in values)
                or values[0] > values[2]
            ):
                raise ValueError("bootstrap record is invalid")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output_path = Path(str(config["output_json"]))
    if output_path.exists() and not args.overwrite:
        raise FileExistsError("locked-test audit output already exists")
    inputs = {key: Path(str(value)) for key, value in config["inputs"].items()}
    input_records = {
        key: {
            "path": path.as_posix(),
            "sha256": require_hash(path, config["input_sha256"][key]),
        }
        for key, path in inputs.items()
    }
    summary = json.loads(inputs["release_summary"].read_text(encoding="ascii"))
    marker = json.loads(inputs["release_marker"].read_text(encoding="ascii"))
    expected = config["expected"]
    if (
        summary.get("status") != expected["status"]
        or summary.get("fixed_protocol", {}).get("receptor_id")
        != expected["receptor_id"]
        or summary.get("test_release", {}).get("consumed") is not True
        or summary.get("test_release", {}).get("repeat_or_overwrite_allowed")
        is not False
        or marker.get("status") != "locked_test_release_consumed"
        or marker.get("overwrite_allowed") is not False
        or summary.get("authorization", {}).get("authorization_id")
        != marker.get("authorization", {}).get("authorization_id")
    ):
        raise ValueError("release summary or marker state is invalid")
    if summary["outputs"]["primary_rankings_csv"]["sha256"] != file_sha256(
        inputs["primary_rankings"]
    ) or summary["outputs"]["sensitivity_rankings_csv"]["sha256"] != file_sha256(
        inputs["sensitivity_rankings"]
    ):
        raise ValueError("release summary output hashes differ")
    if summary["test_release"]["release_marker"]["sha256"] != file_sha256(
        inputs["release_marker"]
    ):
        raise ValueError("release marker hash differs")
    locked_labels = locked_manifest_records(
        inputs["split_manifest"], str(expected["locked_split"])
    )
    if len(locked_labels) != int(expected["test_ligand_count"]):
        raise ValueError("locked-test ligand count differs")
    primary_records, primary_metrics = audit_ranking(
        inputs["primary_rankings"],
        locked_labels,
        "across_seed_median_primary",
    )
    sensitivity_records, sensitivity_metrics = audit_ranking(
        inputs["sensitivity_rankings"],
        locked_labels,
        "across_seed_minimum_sensitivity",
    )
    if set(primary_records) != set(sensitivity_records):
        raise ValueError("primary and sensitivity test IDs differ")
    max_difference = max(
        compare_metrics(primary_metrics, summary["metrics"]["primary"]),
        compare_metrics(
            sensitivity_metrics, summary["metrics"]["sensitivity"]
        ),
    )
    validate_bootstrap(summary, int(expected["bootstrap_iterations"]))
    implementation_path = Path(__file__)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "operation": "independent locked-test release audit from ranking CSVs",
        "status": "ok",
        "config": {
            "path": args.config.as_posix(),
            "sha256": file_sha256(args.config),
        },
        "implementation": {
            "path": f"scripts/{implementation_path.name}",
            "sha256": file_sha256(implementation_path),
        },
        "inputs": input_records,
        "receptor_id": expected["receptor_id"],
        "test_ligand_count": len(locked_labels),
        "metrics_reproduced": True,
        "maximum_metric_absolute_difference": max_difference,
        "primary_top10_active_count": primary_metrics["top10_active_count"],
        "primary_top10_ligand_ids": primary_metrics["top10_ligand_ids"],
        "bootstrap_records_valid": True,
        "release_consumed": True,
        "rerun_allowed": False,
        "interpretation_boundary": config["interpretation_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "test_ligand_count": result["test_ligand_count"],
                "metrics_reproduced": result["metrics_reproduced"],
                "maximum_metric_absolute_difference": max_difference,
                "release_consumed": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
