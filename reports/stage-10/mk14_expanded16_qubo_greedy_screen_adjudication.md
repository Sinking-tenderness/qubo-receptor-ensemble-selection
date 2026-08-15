# Stage 10 MAPK14 Expanded16 QUBO-Greedy Adjudication

## Decision

Stage 10 passes the narrow mechanistic gate: the expanded 16-receptor problem
contains real forward-greedy traps for frozen quadratic objectives. It does not
yet pass a method-superiority gate. The positive held-out signal is concentrated
in a small number of Train-696 outer folds, and the fit on all 696 training rows
does not retain a QUBO-versus-greedy difference at the frozen target size of
three receptors.

## Evidence Boundary

- Input: the independently audited Stage 09 Uni-Dock 1.1.3 Train-696 matrix.
- Dimensions: 16 receptors, 696 training ligands, and three paired seeds.
- Search: target sizes 2 through 8, four coefficient sources, and four frozen
  Train-696 outer folds plus the full training fit.
- Objectives: `coverage_qubo`, `pair_utility_qubo`, and
  `pair_synergy_qubo`, with previously selected weights transferred without
  retuning.
- No fresh-validation or locked-test row was read.

## Stage 09 Input Gate

- All 48 Uni-Dock batches completed.
- All 33,408 receptor-ligand-seed poses were present and independently audited.
- Pose-integrity failures: 0.
- Known warnings: 16; unresolved warnings: 0.
- GPU batch runtime: 7,101.15 seconds (1.97 hours), or 4.7046 pairs/second.

## Main Screen

| Objective | Trials | Greedy missed QUBO optimum | Failure rate | Maximum regret |
|---|---:|---:|---:|---:|
| Coverage | 140 | 36 | 25.71% | 0.184734 |
| Pair synergy | 140 | 27 | 19.29% | 0.274998 |
| Pair utility | 140 | 1 | 0.71% | 0.006186 |
| Total | 420 | 64 | 15.24% | 0.274998 |

Among the 60 strict failures from outer-fold fits, 25 exact-QUBO solutions had
higher primary, mean-seed, and worst-seed held-out BEDROC than both the QUBO
forward-greedy solution and the direct robust-BEDROC greedy solution. These 25
cases were not evenly distributed: fold 0 contributed 10, fold 1 contributed 2,
fold 2 contributed 1, and fold 3 contributed 12.

## Frozen Size-Three Result

The strongest case was the primary-coefficient `pair_synergy_qubo` fit that
excluded outer fold 3:

| Method | Receptor subset | Fold-3 BEDROC |
|---|---|---:|
| Exact QUBO | 2BAJ + 2QD9 + 3BV2 | 0.958182 |
| Direct BEDROC greedy | 2BAJ + 2QD9 + 3KQ7 | 0.888996 |
| QUBO forward greedy | 3BV2 + 3ITZ + 3KQ7 | 0.809817 |

The exact-QUBO delta was +0.069186 versus direct greedy and +0.148365 versus
QUBO forward greedy. The same exact subset was also the exhaustive direct-metric
optimum for this training fold. Therefore this case supports global combination
optimization over forward greedy, but it does not isolate an advantage caused
only by the quadratic surrogate.

The size-three signal was fold-specific. All four robust pair-synergy failures
(primary plus three seed coefficient sources) occurred in outer fold 3. No
pair-utility failure occurred at size three. Coverage had one size-three failure,
but its exact solution did not improve held-out BEDROC over its own greedy path.

## Negative Evidence

On the full Train-696 primary fit at size three:

- Coverage and pair-synergy exact QUBO selected `2BAJ + 3BV2 + 4AAC`, exactly
  matching their QUBO greedy solutions. BEDROC was 0.946625.
- Direct robust-BEDROC greedy selected `2BAJ + 2QD9 + 3BV2` with BEDROC
  0.955655.
- Pair-utility exact QUBO also matched its own greedy solution and reached
  BEDROC 0.926348.

The four full-training greedy failures occurred only for seed-specific coverage
objectives at other target sizes. In every one, the exact objective optimum had
lower primary BEDROC than the corresponding QUBO greedy and direct greedy
solutions. Objective optimality is therefore not interchangeable with screening
quality.

## Interpretation

The expanded receptor pool solved the original structural question: greedy can
miss the global optimum of the frozen QUBO. It did not yet solve the scientific
claim: the globally optimal QUBO solution is not consistently better on unseen
ligands. Nothing in this stage demonstrates runtime advantage, quantum advantage,
or superiority over exhaustive classical search. With 16 choose 3 equal to only
560 subsets, exhaustive classical verification remains trivial and mandatory.

## Recommended Independent Gate

Freeze the exploratory fold-3 candidate and its controls before accessing the
existing fresh-validation labels or scores:

- Candidate A, exact pair-synergy QUBO: `2BAJ + 2QD9 + 3BV2`.
- Control B, QUBO forward greedy: `3BV2 + 3ITZ + 3KQ7`.
- Control C, direct BEDROC greedy: `2BAJ + 2QD9 + 3KQ7`.
- Secondary fit-all QUBO candidate D: `2BAJ + 3BV2 + 4AAC`.

Their union contains six receptors: 2BAJ, 2QD9, 3BV2, 3ITZ, 3KQ7, and 4AAC.
The fresh-validation workload is therefore 1,576 ligands x 6 receptors x 3
seeds = 28,368 docking pairs. At the measured Stage 09 throughput, the expected
GPU batch time is about 6,030 seconds (1.68 hours), before packaging and audit
overhead. Exactly 54 validation decoys require preregistered rigid-macrocycle
PDBQT substitution before Uni-Dock execution.

The confirmatory endpoint should be paired BEDROC with alpha 20. Candidate A
must exceed both controls B and C for the primary median score and for mean- and
worst-seed BEDROC, with paired bootstrap uncertainty reported. No receptor,
weight, aggregation, or search-depth choice may be changed after validation
scores are opened. A success would support a QUBO-formulated global-search
application; it would still not establish quantum computational advantage.
