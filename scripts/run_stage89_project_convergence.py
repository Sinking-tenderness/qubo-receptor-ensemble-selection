import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="ascii"))


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def metric(document, *keys):
    value = document
    for key in keys:
        value = value[key]
    return value


def run(root: Path, config_path: Path):
    config = read_json(root / config_path)
    inputs = config["inputs"]
    documents = {
        name: read_json(root / path)
        for name, path in inputs.items()
        if name != "qci_physical_poc_report"
    }
    physical_report = (root / inputs["qci_physical_poc_report"]).read_text(
        encoding="ascii"
    )

    mk14 = documents["mk14_fresh_validation"]
    qubo_bedroc = metric(
        mk14, "method_metrics", "pair_synergy_qubo", "primary", "bedroc_alpha_20"
    )
    greedy_bedroc = metric(
        mk14, "method_metrics", "nested_greedy_final", "primary", "bedroc_alpha_20"
    )
    single_bedroc = metric(
        mk14, "method_metrics", "single_best", "primary", "bedroc_alpha_20"
    )
    exhaustive_bedroc = metric(
        mk14, "method_metrics", "nested_exhaustive_final", "primary", "bedroc_alpha_20"
    )

    stage74 = documents["larger_k_classical_scaling"]
    stage75 = documents["variable_k_cqm"]
    stage80 = documents["local_move_hardness"]
    stage86a = documents["qci_global_penalty_adjudication"]
    stage87 = documents["quantum_value_instance_gate"]
    stage88 = documents["chemotype_portfolio_gate"]

    assert mk14["all_checks_passed"] is True
    assert abs(qubo_bedroc - greedy_bedroc) < 1e-12
    assert qubo_bedroc > single_bedroc
    assert qubo_bedroc > exhaustive_bedroc
    assert stage74["exact_validation"]["strong_classical_exact_cell_success_rate"] == 1.0
    assert stage75["solver_performance"]["joint_classical_exact_frontier_match_rate"] == 1.0
    assert "500/500" in physical_report
    assert "0/300" in physical_report
    assert stage80["summary"]["local_trap_candidate_count"] == 0
    assert stage86a["constraint_fidelity"]["fully_feasible_count"] == 0
    assert stage87["strict_instance_gate_passed"] is False
    assert stage88["chemotype_balanced_cqm_design_authorized"] is False

    rows = [
        {
            "claim_id": "C1",
            "claim": "A sparse receptor ensemble can improve early recognition in the independently validated MK14 setting.",
            "status": "supported_with_scope",
            "evidence": (
                f"Fresh-validation BEDROC: ensemble={qubo_bedroc:.6f}, "
                f"single={single_bedroc:.6f}, nested exhaustive={exhaustive_bedroc:.6f}."
            ),
            "boundary": "Within-target MK14 evidence; not universal across proteins.",
        },
        {
            "claim_id": "C2",
            "claim": "The present QUBO receptor selector outperforms greedy selection.",
            "status": "not_supported",
            "evidence": (
                f"MK14 QUBO and frozen greedy selected the same subset and both had "
                f"BEDROC={qubo_bedroc:.6f}."
            ),
            "boundary": "Do not claim QUBO-over-greedy or cross-target superiority.",
        },
        {
            "claim_id": "C3",
            "claim": "The constrained receptor-selection problem can be represented and independently checked as a QUBO/CQM.",
            "status": "supported",
            "evidence": (
                f"Stage75 built {stage75['encoding_summary']['cqm_model_count']} CQM models; "
                f"all certified frontier assignments were feasible and the maximum energy residual was "
                f"{stage75['encoding_summary']['maximum_frontier_energy_encoding_residual']:.3e}."
            ),
            "boundary": "An exact encoding does not imply hardware fidelity or quantum advantage.",
        },
        {
            "claim_id": "C4",
            "claim": "Protein-derived local QUBO controls can be executed faithfully on physical quantum-optimization hardware.",
            "status": "supported_as_poc",
            "evidence": "Stage79 Dirac-3 confirmation recovered 500/500 certified optima, improved 200/200 positive controls, and produced 0/300 false improvements.",
            "boundary": "The 38-40 variable controls are easy for exact and local classical solvers.",
        },
        {
            "claim_id": "C5",
            "claim": "The current local-move hardware task exposes a reproducible classical local-search trap.",
            "status": "refuted_on_current_benchmark",
            "evidence": (
                f"Stage80 screened {stage80['summary']['subproblem_count']} subproblems and found "
                f"{stage80['summary']['local_trap_candidate_count']} multi-move trap candidates."
            ),
            "boundary": "Stage79 remains a hardware PoC, not the core advantage experiment.",
        },
        {
            "claim_id": "C6",
            "claim": "The global unconstrained penalty encoding is sampled faithfully by Dirac-3.",
            "status": "refuted_for_current_interface",
            "evidence": (
                f"Stage86 returned {stage86a['constraint_fidelity']['fully_feasible_count']} fully feasible "
                f"samples out of {stage86a['execution']['sample_count']}; canonical repair ranked "
                f"{stage86a['canonical_auxiliary_repair_diagnostic']['rank_among_feasible_subsets']} of "
                f"{stage86a['canonical_auxiliary_repair_diagnostic']['feasible_subset_count']}."
            ),
            "boundary": "This is an interface/encoding failure, not a biological refutation.",
        },
        {
            "claim_id": "C7",
            "claim": "The study demonstrates quantum advantage or quantum speedup.",
            "status": "not_established",
            "evidence": (
                f"Strong classical solvers matched all {stage87['historical_hardness_summary']['stage74_certified_exact_cell_count']} "
                f"Stage74 certified cells and all {stage87['historical_hardness_summary']['stage75_exact_frontier_cell_count']} "
                "Stage75 exact-frontier cells."
            ),
            "boundary": "No advantage, scaling, or time-to-solution claim is authorized.",
        },
        {
            "claim_id": "C8",
            "claim": "The workflow discovers a new active drug molecule.",
            "status": "not_tested",
            "evidence": "No prospective wet-lab activity assay was performed.",
            "boundary": "Describe this as receptor-ensemble selection for virtual screening, not drug discovery.",
        },
    ]

    output_csv = root / config["outputs"]["claim_evidence_csv"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "schema_version": "1.0",
        "status": "stage89_project_converged_claims_frozen",
        "project_positioning": "feasibility_and_boundary_study",
        "frozen_thesis": (
            "Constraint-aware quantum optimization can faithfully represent receptor-ensemble "
            "selection and can execute small protein-derived controls on physical hardware, but "
            "current instances do not establish a classical-hard regime or quantum advantage."
        ),
        "supported_claim_count": sum(
            row["status"] in {"supported", "supported_with_scope", "supported_as_poc"}
            for row in rows
        ),
        "claim_count": len(rows),
        "critical_results": {
            "mk14_primary_bedroc": {
                "qubo": qubo_bedroc,
                "greedy": greedy_bedroc,
                "single_receptor": single_bedroc,
                "nested_exhaustive": exhaustive_bedroc,
            },
            "stage79_physical_confirmation_optimum_hits": "500/500",
            "stage80_multi_move_local_traps": stage80["summary"]["local_trap_candidate_count"],
            "stage86_fully_feasible_physical_samples": stage86a["constraint_fidelity"]["fully_feasible_count"],
            "stage87_quantum_worthy_instance_gate_passed": stage87["strict_instance_gate_passed"],
            "stage88_chemotype_gate_passed": stage88["chemotype_balanced_cqm_design_authorized"],
        },
        "authorization": config["freeze_policy"],
        "paper_title_candidate": "Constraint-Aware Quantum Optimization for Protein Receptor-Ensemble Selection: Physical-Hardware Feasibility and Practical Limits",
        "two_week_delivery_plan": [
            {
                "dates": "2026-08-11 to 2026-08-12",
                "deliverable": "Freeze the claim-evidence ledger and manuscript outline.",
            },
            {
                "dates": "2026-08-13 to 2026-08-15",
                "deliverable": "Prepare the biological result, hardware control, and limitation figures/tables.",
            },
            {
                "dates": "2026-08-16 to 2026-08-20",
                "deliverable": "Draft abstract, introduction, methods, results, and limitations from frozen evidence.",
            },
            {
                "dates": "2026-08-21 to 2026-08-22",
                "deliverable": "Run claim, numeric, and reproducibility audits; assemble the review package.",
            },
        ],
        "future_research_gate": (
            "A new experimental branch requires a preregistered, independently validated instance "
            "where the exact solution differs from and improves on strong classical search, and where "
            "the search space cannot be exhaustively solved in subsecond time."
        ),
        "data_boundary": {
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "inputs": {
            name: {"path": path, "sha256": sha256(root / path)}
            for name, path in inputs.items()
        },
        "outputs": {
            "claim_evidence_csv": {
                "path": str(output_csv.relative_to(root)).replace("\\", "/"),
                "sha256": sha256(output_csv),
            }
        },
    }

    result_path = root / config["outputs"]["result_json"]
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )

    report = f"""# Stage89 project convergence and claim freeze

## Decision

The project is now frozen as a **feasibility-and-boundary study**, not a quantum-advantage study. No new objective-function search, target docking, or hardware spending is authorized in the current phase. Manuscript preparation and reproducibility packaging are authorized.

## Defensible thesis

{result['frozen_thesis']}

## What succeeded

1. **Biological signal:** On fresh MK14 validation, the three-receptor ensemble reached BEDROC `{qubo_bedroc:.6f}`, compared with `{single_bedroc:.6f}` for the single receptor and `{exhaustive_bedroc:.6f}` for the nested exhaustive comparator.
2. **Model fidelity:** Stage75 produced `{stage75['encoding_summary']['cqm_model_count']}` explicit constrained models with feasible certified frontier assignments and negligible encoding residual.
3. **Physical-hardware PoC:** Stage79 recovered `500/500` certified optima, all `200/200` positive-control improvements, and `0/300` false improvements.
4. **Honest boundary map:** Stage80 found no multi-move local trap; Stage86 produced no fully feasible global-penalty sample; Stage87 and Stage88 both blocked further hardware work.

## What must not be claimed

- QUBO superiority over greedy: the MK14 QUBO and frozen greedy solutions were identical.
- Cross-target superiority: repeated multi-target objective searches did not establish it.
- Quantum advantage or speedup: strong classical methods matched all certified exact references.
- New-drug discovery: no prospective wet-lab activity assay was performed.

## Paper spine

1. Receptor ensembles can improve early virtual-screening recognition, but selecting them is constrained and target dependent.
2. The selection problem can be represented as auditable QUBO/CQM models with exact certificates.
3. A physical optimizer can faithfully solve small protein-derived positive and negative controls.
4. The same study identifies why stronger claims currently fail: local tasks are classically easy, while the meaningful global penalty encoding loses physical feasibility.
5. The contribution is a reproducible feasibility and limit map, with a preregistered gate for any future advantage experiment.

## Two-week delivery plan

| Dates | Deliverable |
|---|---|
| Aug 11-12 | Freeze claims, evidence ledger, and manuscript outline. |
| Aug 13-15 | Produce the biological, hardware-control, and limitation figures/tables. |
| Aug 16-20 | Draft the manuscript from frozen evidence. |
| Aug 21-22 | Audit claims and numbers; assemble the reproducibility package. |

## Reopening rule

New docking or hardware work requires a new preregistration and an independently validated instance where the exact solution differs from and improves on strong classical search, while the search space is no longer exhaustively trivial. Until that gate exists, additional objective tuning or hardware spending would add activity rather than evidence.
"""
    report_path = root / config["outputs"]["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="ascii")

    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage89_project_convergence.json"),
    )
    args = parser.parse_args()
    run(args.root.resolve(), args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
