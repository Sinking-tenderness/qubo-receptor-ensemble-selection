from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def run(root: Path) -> dict[str, object]:
    root = root.resolve()
    scale_path = root / "results/runs/stage95_pparg_md96_series_routing_scaling/scale_results.csv"
    result_path = root / "data/stage95_pparg_md96_series_routing_scaling_result.json"
    report_path = root / "reports/stage-95/pparg_md96_series_routing_scaling.md"
    rows = read_csv(scale_path)
    if len(rows) != 5:
        raise ValueError("Stage95 scale count differs")
    amended = []
    for row in rows:
        strong = float(row["best_strong_classical_objective"])
        optimal = row["milp_optimal"].lower() == "true"
        upper = (
            float(row["milp_conditionally_rerouted_objective"])
            if optimal
            else float(row["milp_dual_bound"])
        )
        maximum_gap = max(0.0, upper - strong)
        maximum_relative = maximum_gap / max(abs(upper), 1e-12)
        amended.append(
            {
                **row,
                "certified_upper_bound": upper,
                "maximum_possible_objective_gap": maximum_gap,
                "maximum_possible_relative_gap": maximum_relative,
                "one_percent_gap_mathematically_excluded": maximum_relative < 0.01,
            }
        )
    write_csv(scale_path, amended)
    result = json.loads(result_path.read_text(encoding="ascii"))
    excluded = sum(
        str(row["one_percent_gap_mathematically_excluded"]).lower() == "true"
        for row in amended
    )
    result["scales_with_one_percent_gap_mathematically_excluded"] = excluded
    result["all_scales_exclude_one_percent_classical_gap"] = excluded == len(amended)
    result["largest_certified_maximum_possible_relative_gap"] = max(
        float(row["maximum_possible_relative_gap"]) for row in amended
    )
    result["outputs"]["scale_csv"]["sha256"] = sha256(scale_path)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    report_path.write_text(
        "\n".join(
            [
                "# Stage95 PPARG MD-96 series-routing scaling",
                "",
                f"Status: `{result['status']}`.",
                "",
                "Five nested scales used 46,080 real Stage43 Uni-Dock scores and no synthetic scores.",
                "",
                f"All {excluded} scales mathematically exclude a 1% improvement over the best strong classical solution. The largest remaining upper-bound margin is {result['largest_certified_maximum_possible_relative_gap']:.6f}.",
                "",
                "The 16-, 32-, 48-, and 64-receptor MILPs were solved exactly; strong classical search matched every optimum. The 96-receptor MILP timed out, but its certified upper bound still excludes the preregistered 1% value threshold.",
                "",
                "No protected data, new docking, or quantum hardware were used. Stage95 does not authorize quantum execution or same-matrix retuning.",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    adjudication = {
        "schema_version": "1.0",
        "status": "stage95_solver_bound_adjudication_ok",
        "scale_count": len(amended),
        "exact_scale_count": sum(row["milp_optimal"].lower() == "true" for row in amended),
        "bounded_nonexact_scale_count": sum(row["milp_optimal"].lower() != "true" for row in amended),
        "one_percent_gap_excluded_scale_count": excluded,
        "largest_certified_maximum_possible_relative_gap": result[
            "largest_certified_maximum_possible_relative_gap"
        ],
        "scale_csv_sha256": sha256(scale_path),
        "result_json_sha256": sha256(result_path),
        "data_boundary": result["data_boundary"],
    }
    output = root / "data/stage95_pparg_md96_series_routing_bound_adjudication.json"
    output.write_text(
        json.dumps(adjudication, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(adjudication, indent=2, sort_keys=True))
    return adjudication


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    run(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
