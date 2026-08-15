# Stage80 Local Move-QUBO Hardness Screen

## Result

The canonical historical-development screen covered `100`
fixed-k protein-derived local move-QUBOs. Only
`3` subproblems contained a strict single-move
improvement. No subproblem contained an improving nonconflicting pair, and the
frozen warm-start tabu screen found no multi-move improvement or local-trap
candidate.

The eligible move count ranged from `7`
to `510`. Up to
`474` binary move variables fit the documented
Dirac-3 level limit, but the added variables did not create a harder scientific
decision.

## Decision

- Additional QCI local scaling run authorized: `false`.
- Mechanical move-cap scaling rejected: `true`.
- Global variable-k reformulation review authorized: `true`.

The Stage79 physical result remains valid, but its local repair task is too easy
for a scaling or advantage claim. The next defensible route is to revisit the
full variable-k QUBO for Dirac-3, where all-to-all connectivity removes minor
embedding and float32 precision may permit a tighter penalty construction.

## Boundary

This screen does not prove that no higher-order trap exists. It reports that no
two-move trap or tabu-detected multi-move improvement was found under the frozen
canonical protocol. It used no new docking data and submitted no cloud or
physical-hardware job.
