"""Adjudicate the completed Stage86 Dirac-3 physical rescue result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def archive_members(path: Path) -> tuple[dict[str, bytes], int]:
    with tarfile.open(path, "r:gz") as archive:
        members = {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }
    manifest_names = [name for name in members if name.endswith("/bundle_manifest.sha256")]
    if len(manifest_names) != 1:
        raise ValueError("Stage86 result archive must contain one bundle manifest")
    manifest_name = manifest_names[0]
    prefix = PurePosixPath(manifest_name).parent
    verified_count = 0
    for line in members[manifest_name].decode("ascii").splitlines():
        expected, relative = line.split(maxsplit=1)
        member_name = str(prefix / relative.lstrip("*"))
        if member_name not in members:
            raise ValueError(f"Stage86 result member is missing: {relative}")
        if sha256_bytes(members[member_name]) != expected.upper():
            raise ValueError(f"Stage86 result member hash differs: {relative}")
        verified_count += 1
    return members, verified_count


def unique_member(members: dict[str, bytes], suffix: str) -> bytes:
    matches = [value for name, value in members.items() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one Stage86 archive member ending in {suffix}")
    return matches[0]


def weighted(rows: list[dict[str, str]], predicate: Any) -> int:
    return sum(int(row["num_occurrences"]) for row in rows if predicate(row))


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    archive_descriptor = config["inputs"]["result_archive"]
    archive_path = root / archive_descriptor["path"]
    if sha256(archive_path) != archive_descriptor["sha256"]:
        raise ValueError("Stage86 result archive identity differs")
    members, verified_member_count = archive_members(archive_path)

    rescue = json.loads(unique_member(members, "/rescue.json").decode("ascii"))
    response = json.loads(
        unique_member(
            members, "/ppara_of0_k10_nonnegative_gauge_exact.response.json"
        ).decode("ascii")
    )
    rows = list(
        csv.DictReader(
            io.StringIO(unique_member(members, "/rescue_samples.csv").decode("ascii"))
        )
    )

    mapping_descriptor = config["inputs"]["mapping"]
    mapping_path = root / mapping_descriptor["path"]
    if sha256(mapping_path) != mapping_descriptor["sha256"]:
        raise ValueError("Stage86 mapping identity differs")
    mapping = read_json(mapping_path)
    target_k = int(mapping["k"])
    threshold = int(mapping["quality_threshold"])

    total_samples = sum(int(row["num_occurrences"]) for row in rows)
    cardinality_ok = weighted(rows, lambda row: int(row["selected_count"]) == target_k)
    quality_ok = weighted(rows, lambda row: int(row["deficit"]) <= threshold)
    receptor_constraints_ok = weighted(
        rows,
        lambda row: int(row["selected_count"]) == target_k
        and int(row["deficit"]) <= threshold,
    )
    residuals_zero = weighted(
        rows,
        lambda row: all(int(value) == 0 for value in row["residuals"].split("+")),
    )
    count_distribution = Counter()
    for row in rows:
        count_distribution[int(row["selected_count"])] += int(row["num_occurrences"])

    import sys

    sys.path.insert(0, str(root))
    from scripts.experimental.quantum import (  # type: ignore
        run_stage86_nonnegative_gauge_dirac_rescue as runner,
    )

    stage86_config = read_json(root / config["inputs"]["stage86_config"])
    prepared = read_json(root / config["inputs"]["stage86_prepared"])
    ctx = runner.context(root, stage86_config, prepared)
    repairable_rows = [
        row
        for row in rows
        if int(row["selected_count"]) == target_k
        and int(row["deficit"]) <= threshold
    ]
    repaired: dict[str, Any] | None = None
    if repairable_rows:
        candidate = min(repairable_rows, key=lambda row: float(row["original_objective"]))
        selected_ids = set(candidate["selected_subset"].split("+"))
        subset = tuple(
            index
            for index, receptor_id in enumerate(mapping["receptor_ids"])
            if receptor_id in selected_ids
        )
        sample, residuals = runner.s86.assignment_for_subset(
            ctx["encoding"], ctx["cell"]["model"], subset
        )
        vector = [sample[index + 1] for index in range(len(mapping["variable_order"]))]
        decoded = runner.decode(ctx, vector)
        if residuals != [0, 0, 0, 0] or not decoded["feasible"]:
            raise ValueError("Stage86 canonical auxiliary repair did not become feasible")

        feasible_objectives = []
        for trial in itertools.combinations(range(len(mapping["receptor_ids"])), target_k):
            deficit = sum(int(mapping["integer_deficits"][index]) for index in trial)
            if deficit <= threshold:
                feasible_objectives.append(
                    runner.s75.variable_energy(
                        ctx["cell"]["model"], trial, float(mapping["reward_value"])
                    )
                )
        feasible_objectives.sort()
        objective = float(candidate["original_objective"])
        rank = 1 + sum(value < objective - 1e-12 for value in feasible_objectives)
        exact_objective = float(
            mapping["quantized_exact"]["selected_original_objective"]
        )
        repaired = {
            "raw_solution_index": int(candidate["solution_index"]),
            "raw_residuals": [int(value) for value in candidate["residuals"].split("+")],
            "raw_residual_l1": sum(
                abs(int(value)) for value in candidate["residuals"].split("+")
            ),
            "canonical_repair_feasible": True,
            "selected_subset": candidate["selected_subset"],
            "original_objective": objective,
            "exact_objective": exact_objective,
            "objective_gap": objective - exact_objective,
            "feasible_subset_count": len(feasible_objectives),
            "rank_among_feasible_subsets": rank,
            "percentile_from_best": 100.0 * (rank - 1) / (len(feasible_objectives) - 1),
            "better_than_feasible_median": objective
            < feasible_objectives[len(feasible_objectives) // 2],
            "repaired_normalized_energy": float(decoded["energy"]),
            "exact_normalized_energy": float(
                mapping["quantized_exact"]["normalized_energy"]
            ),
        }

    expected = config["expected_execution"]
    device_usage = float(response["job_info"]["job_result"]["device_usage_s"])
    checks = {
        "archive_manifest_verified": verified_member_count == 11,
        "device_job_completed": response.get("status") == "COMPLETED",
        "device_job_count_matches": int(rescue["qci_device_jobs"])
        == int(expected["device_jobs"]),
        "sample_count_matches": total_samples == int(expected["samples"]),
        "device_usage_matches": device_usage
        == float(expected["device_usage_seconds"]),
        "allocation_before_matches": int(rescue["allocation_before"]["seconds"])
        == int(expected["allocation_before_seconds"]),
        "allocation_after_matches": int(rescue["allocation_after"]["seconds"])
        == int(expected["allocation_after_seconds"]),
        "no_raw_feasible_sample": weighted(rows, lambda row: row["feasible"] == "True")
        == int(expected["feasible_samples"]),
        "no_exact_optimum_sample": weighted(
            rows, lambda row: row["exact_optimum"] == "True"
        )
        == int(expected["exact_optimum_samples"]),
        "no_below_certificate_sample": weighted(
            rows, lambda row: row["below_certificate"] == "True"
        )
        == int(expected["below_certificate_samples"]),
        "repaired_candidate_not_competitive": repaired is not None
        and not bool(repaired["better_than_feasible_median"]),
    }
    if not all(checks.values()):
        raise ValueError(f"Stage86a adjudication checks failed: {checks}")

    result = {
        "schema_version": "1.0",
        "status": "stage86_dirac3_physical_rescue_failed_stop_global_penalty_route",
        "archive": {
            "path": archive_descriptor["path"],
            "sha256": archive_descriptor["sha256"],
            "verified_manifest_member_count": verified_member_count,
        },
        "execution": {
            "job_id": response["job_info"]["job_id"],
            "device_jobs": int(rescue["qci_device_jobs"]),
            "sample_count": total_samples,
            "device_usage_seconds": device_usage,
            "remaining_free_seconds": int(rescue["allocation_after"]["seconds"]),
        },
        "constraint_fidelity": {
            "target_k": target_k,
            "quality_threshold": threshold,
            "selected_count_distribution": {
                str(key): value for key, value in sorted(count_distribution.items())
            },
            "cardinality_ok_count": cardinality_ok,
            "quality_ok_count": quality_ok,
            "receptor_constraints_ok_count": receptor_constraints_ok,
            "auxiliary_residuals_zero_count": residuals_zero,
            "fully_feasible_count": int(rescue["feasible_sample_count"]),
            "exact_optimum_count": int(rescue["exact_optimum_sample_count"]),
        },
        "canonical_auxiliary_repair_diagnostic": repaired,
        "checks": checks,
        "primary_endpoint_passed": False,
        "additional_qci_dirac3_global_penalty_jobs_authorized": 0,
        "full_production_authorized": False,
        "quantum_advantage_claim_authorized": False,
        "next_action": config["decision"]["failure_action"],
        "future_hardware_boundary": config["decision"]["future_hardware_boundary"],
        "interpretation": (
            "The globally exact algebraic encoding passed locally, but the physical "
            "Dirac-3 sampler returned no fully feasible sample. The only receptor-level "
            "feasible sample becomes valid after deterministic auxiliary repair but ranks "
            "below the median of the certified feasible search space. This is a physical "
            "sampling-fidelity failure for this encoding and interface, not a biological "
            "failure of the receptor-selection objective and not evidence for or against "
            "quantum advantage in general."
        ),
    }

    result_path = root / config["outputs"]["result_json"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    report_path = root / config["outputs"]["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    assert repaired is not None
    report_path.write_text(
        "\n".join(
            [
                "# Stage86a QCI Dirac-3 rescue failure adjudication",
                "",
                "The one frozen Dirac-3 job completed normally, returned 25 samples, "
                "and used 22 device seconds. No raw sample was fully feasible and none "
                "recovered the certified exact optimum.",
                "",
                "| Diagnostic | Passing samples |",
                "|---|---:|",
                f"| Correct cardinality (k={target_k}) | {cardinality_ok} / {total_samples} |",
                f"| Quality threshold | {quality_ok} / {total_samples} |",
                f"| Both receptor constraints | {receptor_constraints_ok} / {total_samples} |",
                f"| Zero auxiliary residuals | {residuals_zero} / {total_samples} |",
                f"| Fully feasible raw encoding | 0 / {total_samples} |",
                f"| Exact optimum | 0 / {total_samples} |",
                "",
                "One sample satisfied the scientific receptor constraints and was one "
                "auxiliary unit from a valid encoding. Deterministic auxiliary repair "
                f"makes it feasible, but its original objective is {repaired['original_objective']:.6f} "
                f"versus the exact optimum {repaired['exact_objective']:.6f}. It ranks "
                f"{repaired['rank_among_feasible_subsets']:,} of "
                f"{repaired['feasible_subset_count']:,} feasible subsets and is worse "
                "than the feasible median.",
                "",
                "The primary endpoint failed. No additional QCI Dirac-3 global-penalty "
                "job is authorized, and the remaining 73 free seconds should not be spent "
                "on repetitions or post-hoc schedule tuning.",
                "",
                "This result rejects the current physical encoding/interface combination. "
                "It does not reject the receptor-selection objective or establish a general "
                "claim about quantum hardware.",
                "",
            ]
        ),
        encoding="ascii",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/stage86a_qci_dirac3_rescue_failure_adjudication.json",
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = run((root / args.config).resolve(), root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
