# Stage19--Stage102A group meeting summary

## Opening statement

This period focused on one question: can a QUBO reliably decide both which receptor conformations to use and how many to use? We found one reproducible two-receptor benefit on FA10, but no transferable rule across targets. The discussion today is whether to continue with a target-conditional formulation instead of claiming a universal QUBO or quantum advantage.

## Slide 1: What we learned

**Title:** QUBO can identify target-specific complementarity, but it is not yet a universal selector

- Success: the workflow now spans structure pools, MD conformations, Uni-Dock matrices, nested validation, exact QUBO solutions, and physical hardware controls.
- Limitation: a receptor combination that helps one protein can hurt another, and inner-fold marginal gains do not yet transfer reliably.
- Discussion: should the central claim become "first detect whether complementarity exists, then optimize the smallest useful ensemble"?

## Slide 2: Experiments from Stage19 to Stage101

| Phase | What was tested | Main result |
|---|---|---|
| Stage19--20 | Repaired coverage QUBOs, noise stability, and adaptive $k=1\ldots6$ | More receptors were not automatically better; the conservative global recommendation was $k=1$. |
| Stage21--32 | Structure-aware QUBOs and PPARG MD conformations | Structure diversity can be encoded, but structural novelty does not guarantee screening complementarity. |
| Stage33--71 | Sparse, rank-aware, robust, higher-order, and cross-target objectives on MK14, PPARG, BACE1, PPARA, and PPARD | Some targets improved, others degraded; no objective consistently beat the classical baselines. |
| Stage72--89 | Constraint-native formulations and QCI/D-Wave hardware readiness | A simple local hardware control succeeded, but the meaningful global constrained problem was not solved reliably; no quantum-advantage claim is supported. |
| Stage90--101 | Chemical-series rescue, objective repair, adaptive stopping, and marginal transfer | A useful adaptive-$k$ oracle exists, but the inner-validation signal predicts the next receptor poorly across targets. |

**Stage101 diagnosis:** the outer-oracle adaptive-$k$ ceiling was about `+0.0412 BEDROC`, so useful choices exist. The bottleneck is learning when to add the next receptor, not the absence of any useful ensemble.

## Slide 3: Stage102A new results

**Title:** FA10 benefits from two conformations; EGFR should stop at one

Technical execution:

- EGFR: 12 receptors, 600 ligands, 3 seeds, 21,600 scores.
- FA10: 13 receptors, 600 ligands, 3 seeds, 23,400 scores.
- Total: 75/75 batches and 45,000/45,000 scores completed; no missing batch, non-finite median score, or pose-integrity failure.

| Target and method | BEDROC $\alpha=20$ | Gain over selected single | Interpretation |
|---|---:|---:|---|
| EGFR single | 0.3716 | 0 | Best conservative choice |
| EGFR fixed $k=2$ | 0.3435 | -0.0281 | Adding a receptor hurts |
| EGFR outer oracle | 0.3778 | +0.0062 | Even the unattainable ceiling is small |
| FA10 single | 0.7266 | 0 | Strong baseline |
| FA10 fixed $k=2$ | **0.7590** | **+0.0324** | Clear two-receptor complementarity |
| FA10 fixed $k=3$ | 0.7264 | -0.0002 | The third receptor removes the gain |
| FA10 marginal-LCB stopping | 0.7552 | +0.0285 | Nearly recovers the oracle choice |

The FA10 pair `FA10_2PHB_aligned + FA10_5K0H_aligned` was selected in all five outer folds. When this frozen median-selected pair was evaluated on each docking seed, its gains were `+0.0308`, `+0.0272`, and `+0.0267`, so the pair benefit is not a single-seed accident.

Bad result: EGFR has almost no exploitable ensemble ceiling, and the combined provisional one-standard-error gate still fails. The formal Stage102 mechanistic-bootstrap/Ridge gate has not been executed, so PARP1 and quantum hardware remain locked.

## Slide 4: Decision and next week

**Title:** Stop forcing every protein into a multi-receptor QUBO

| Time | Action | Decision criterion |
|---|---|---|
| Day 1--2 | Freeze the Stage102A audit and implement the preregistered mechanistic-bootstrap and held-target Ridge rules | No target-specific retuning |
| Day 3 | Run leave-one-target-out evaluation on all seven development targets | Mean gain $\geq0.02$, worst gain $\geq-0.02$, and at least four targets gain $\geq0.02$ |
| Day 4 | Diagnose whether structural or functional features separate FA10-like from EGFR-like targets | The rule must identify both a positive and a stop case |
| Day 5 | Go/No-Go | Only a full Phase-A gate pass releases untouched PARP1 validation; otherwise stop this adaptive-QUBO claim |

**One question for discussion:** Do we agree to make "detect complementarity before QUBO optimization" the main scientific hypothesis, rather than requiring a multi-receptor QUBO to improve every target?

## Bounded conclusion

The project has not demonstrated quantum advantage or a universal QUBO. It has produced a reproducible negative map and one stable positive mechanism: FA10 needs exactly two complementary conformations, whereas EGFR should remain at one. This makes the next test narrower and more falsifiable than the earlier objective-search loop.
