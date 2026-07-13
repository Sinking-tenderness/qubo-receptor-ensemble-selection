# Stage 3 CDK2 AF2 MD Vina Search Robustness Record

## Scope

This pilot tested whether AutoDock Vina exhaustiveness 4 was sufficiently
stable for expansion beyond the four-ligand MD receptor execution gate. Four
receptor-ligand cases were each docked with three fixed random seeds at paired
exhaustiveness settings of 4 and 8, giving 24 runs.

The experiment evaluated stochastic search stability. It did not evaluate
virtual-screening enrichment or receptor subset quality.

## Case Selection

| Case | Receptor role | Ligand label | Reason |
|---|---|---|---|
| C00-A0029 | low-temporal-support exploratory medoid | active | stress-test an early trajectory state |
| C03-D0004 | revisited primary medoid | decoy | cover a stable gate case and a decoy |
| C05-D0026 | revisited primary medoid | decoy | revisit the catastrophic exhaustiveness-1 failure |
| C07-A0071 | revisited primary medoid | active | cover a late trajectory state and second active |

Each of the four gate ligands appeared exactly once. The selected cases span
four receptors, two actives, two decoys, and both exploratory and primary MD
medoid roles.

## Fixed Protocol

- AutoDock Vina: 1.2.7
- Seeds: `20260821`, `20260822`, `20260823`
- Paired search settings: exhaustiveness 4 and 8
- Output modes: 1
- CPU per Vina process: 4
- Execution: serial, maximum total CPU 4
- Box center: (0.52, 27.06, 8.97) A
- Box size: (18, 18, 16) A
- Experiment config SHA-256:
  `0535FC19036B00B7A1C502EAD0378BAFB318226E054B32444CC0B43685AA5147`

Receptor, ligand, box, seed, CPU allocation, Vina version, and output settings
were paired. Exhaustiveness was the only protocol difference within each
paired comparison.

## Prespecified Acceptance Rules

A case passed only if all six runs completed and all of these conditions held:

- no nonnegative Vina score;
- exhaustiveness-4 score range across seeds no greater than 1.0 kcal/mol; and
- maximum absolute paired exhaustiveness-4 versus exhaustiveness-8 difference
  no greater than 0.5 kcal/mol.

These are search-quality controls, not biological activity thresholds.

## Results

- Expected runs: 24
- Successful Vina runs: 24
- Technical failures: 0
- Cases passing all acceptance rules: 2 of 4
- Final status: `completed_with_search_instability`

| Case | e4 mean | e4 seed range | e8 mean | e8 seed range | Max paired delta | Pass |
|---|---:|---:|---:|---:|---:|:---:|
| C00-A0029 | -3.578 | 17.349 | -9.396 | 0.017 | 17.332 | No |
| C03-D0004 | -7.987 | 0.041 | -7.987 | 0.041 | 0.000 | Yes |
| C05-D0026 | -7.820 | 0.035 | -7.820 | 0.035 | 0.000 | Yes |
| C07-A0071 | -8.048 | 0.923 | -8.358 | 0.009 | 0.917 | No |

Scores and differences are in kcal/mol.

## Failure Analysis

### C00-A0029

The three exhaustiveness-4 scores were `+7.942`, `-9.268`, and `-9.407`.
The paired exhaustiveness-8 scores were `-9.390`, `-9.390`, and `-9.407`.
The positive score occurred for only one random seed and was rescued by
exhaustiveness 8 using that same seed. It is therefore a stochastic search
failure rather than a fixed property of the C00 receptor or A0029 ligand.

### C07-A0071

For seed `20260821`, exhaustiveness 4 returned `-7.440`, while exhaustiveness 8
returned `-8.357`, a 0.917 kcal/mol difference. This was not a catastrophic
positive score, but it exceeded the paired acceptance threshold and shows that
exhaustiveness 4 can miss a more favorable pose even when its output looks
superficially plausible.

### Stable Cases

C03-D0004 and C05-D0026 produced identical paired exhaustiveness-4 and
exhaustiveness-8 scores for every seed. The earlier C05-D0026
exhaustiveness-1 failure was not repeated at either higher setting.

## Runtime

| Protocol | Runs | Total Vina time (s) | Mean per run (s) | Range (s) |
|---|---:|---:|---:|---:|
| Exhaustiveness 4 | 12 | 359.081 | 29.923 | 12.162-52.376 |
| Exhaustiveness 8 | 12 | 705.497 | 58.791 | 21.891-91.168 |

Exhaustiveness 8 required approximately 1.96 times the measured Vina runtime
of exhaustiveness 4 in this serial four-CPU pilot. The full 24-run wall time was
1,065.440 seconds.

## Output Integrity

- Final summary SHA-256:
  `73C2C08A5FB62A6FD45F00F7505CCF20A635C4BD4566862E76356F59AB3ACF1D`
- Raw 24-run table SHA-256:
  `4A9A34BA9D1B1CFC9F8CB530529FD86D25AE4198872BA422E06D1E6D582C60EF`
- Four-case summary SHA-256:
  `ECCCC041C8362781147F191A4A3FD761B566E897791A104A4CFADF5592D9DF70`

Generated poses, logs, and run tables remain ignored local artifacts under
`results/runs/`. The protocol, implementation, tests, and summary report are
tracked in Git.

## Decision

Exhaustiveness 4 is rejected as the primary expansion protocol. It failed the
prespecified search-stability rules in two of four cases, including one
catastrophic positive score. Using this setting for a receptor score matrix
could create false ligand ranks and false conformer complementarity that would
then contaminate enrichment metrics and the QUBO objective.

Exhaustiveness 8 is the current candidate protocol. It showed narrow seed
ranges of 0.009-0.041 kcal/mol across all four pilot cases, but this limited
pilot does not establish robustness over all eight receptors or the full
ligand set.

The next gate should run the original two-active/two-decoy panel across all
eight MD receptors at exhaustiveness 8. A paired CPU-throughput check should
also determine whether four parallel one-CPU jobs or serial four-CPU jobs give
better wall-clock performance without changing scores. Only after that gate
passes should the benchmark expand in stages.
