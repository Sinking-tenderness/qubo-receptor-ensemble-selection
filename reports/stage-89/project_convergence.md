# Stage89 project convergence and claim freeze

## Decision

The project is now frozen as a **feasibility-and-boundary study**, not a quantum-advantage study. No new objective-function search, target docking, or hardware spending is authorized in the current phase. Manuscript preparation and reproducibility packaging are authorized.

## Defensible thesis

Constraint-aware quantum optimization can faithfully represent receptor-ensemble selection and can execute small protein-derived controls on physical hardware, but current instances do not establish a classical-hard regime or quantum advantage.

## What succeeded

1. **Biological signal:** On fresh MK14 validation, the three-receptor ensemble reached BEDROC `0.550861`, compared with `0.376023` for the single receptor and `0.525635` for the nested exhaustive comparator.
2. **Model fidelity:** Stage75 produced `80` explicit constrained models with feasible certified frontier assignments and negligible encoding residual.
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
