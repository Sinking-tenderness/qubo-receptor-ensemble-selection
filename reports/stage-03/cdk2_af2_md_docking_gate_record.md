# Stage 3 CDK2 AF2 MD Docking Gate Record

## Scope

This gate tested whether all eight prepared MD medoid receptors could use one
fixed AutoDock Vina workflow and produce a complete receptor-by-ligand score
matrix. Two actives and two decoys were inherited from the AF2 parent smoke
test, giving 32 receptor-ligand pairs.

This was an execution and parsing gate. It was not an enrichment benchmark and
was not designed to compare receptor quality.

## Fixed Protocol

- AutoDock Vina: 1.2.7
- Receptors: 8 aligned and identically prepared MD medoids
- Ligands: `CDK2_A0029`, `CDK2_A0071`, `CDK2_D0004`, `CDK2_D0026`
- Box center: (0.52, 27.06, 8.97) A
- Box size: (18, 18, 16) A
- Exhaustiveness: 1
- Output modes: 1
- CPU per Vina process: 1
- Parallel workers: 4
- Maximum total docking CPU: 4
- Representative score: pose rank 1
- Gate config SHA-256:
  `987F62F6CBDE3F5FAFA966D39A723FD4742B752766BB5B952CAE7F7A38475537`

The receptor preparation, ligand files, Vina executable, box, ligand-specific
seeds, runner, and score parser were held fixed. Receptor coordinates were the
intended experimental variable.

## Execution Result

- Requested receptors: 8
- Requested ligands: 4
- Expected receptor-ligand pairs: 32
- Successful pairs: 32
- Failed pairs: 0
- Technical status: `ok_with_search_warning`
- Search-quality warning pairs: 1

All receptors produced Vina output, parsed scores, long-format records, and a
complete score matrix.

## Score Matrix

All values are Vina scores in kcal/mol. Lower values are more favorable under
the Vina scoring convention.

| Ligand | Label | C00 | C01 | C02 | C03 | C04 | C05 | C06 | C07 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0029 | active | -9.321 | -9.986 | -9.367 | -10.010 | -7.546 | -7.769 | -7.844 | -7.387 |
| A0071 | active | -7.684 | -7.860 | -7.471 | -7.847 | -7.505 | -7.972 | -7.151 | -8.363 |
| D0004 | decoy | -7.893 | -7.662 | -7.197 | -8.186 | -6.989 | -7.781 | -7.948 | -8.130 |
| D0026 | decoy | -7.407 | -8.229 | -7.732 | -8.153 | -5.848 | **+26.950** | -7.160 | -7.544 |

The C05-D0026 value is retained in the raw output and matrix for auditability,
but it is flagged as a failed low-exhaustiveness search.

## Search-Failure Audit

The automated warning policy flags a pair when either condition is met:

- the representative Vina score is nonnegative; or
- the score is more than 5.0 kcal/mol less favorable than the corresponding
  parent AF2 smoke-test score.

Only `C05 x D0026` was flagged:

- Exhaustiveness 1 score: +26.950 kcal/mol
- Parent AF2 score: -7.547 kcal/mol
- Delta from parent: +34.497 kcal/mol
- Warning reasons:
  `nonnegative_vina_score;large_unfavorable_delta_from_parent`

The same receptor, ligand, box, and random seed (`20260805`) were rerun at
higher search intensities:

| Exhaustiveness | Vina score (kcal/mol) |
|---:|---:|
| 1 | +26.950 |
| 4 | -7.847 |
| 8 | -7.847 |
| 16 | -7.805 |

The agreement among exhaustiveness 4, 8, and 16 shows that the positive score
was an exhaustiveness-1 search failure. It does not indicate a corrupted
receptor, misplaced box, or intrinsically unfavorable C05-D0026 interaction.

## Output Integrity

- Gate summary SHA-256:
  `1DFBA5E0F699987F21B7158EF8EFF069A124E1C2B91FB0B7159BEF6A560625DE`
- Score matrix SHA-256:
  `9E571F877B0BC02245D125F156AC2260B6907287C8B04E4A861E63405A1E92E3`
- Representative long table SHA-256:
  `6690629BBFDA7ADC44B1726D03E26B8685C1DC1D9514174570CAA7AFE132C2A1`

Generated poses, logs, receptor files, and run tables remain ignored local
artifacts under `results/runs/`. Git records the protocol, warning logic,
tests, and this summary report.

## Decision

The eight-receptor workflow passes the technical gate: the shared box, Vina
execution, output parsing, failure preservation, and score-matrix construction
all work across the complete MD medoid set.

The exhaustiveness-1 matrix is excluded from enrichment calculations, receptor
selection, and QUBO objectives because one of only 32 pairs suffered a severe
search failure. A score this unstable could distort ligand ranks and create
false receptor complementarity.

Exhaustiveness 4 is the current candidate for the larger benchmark because it
rescued the failed pair and matched exhaustiveness 8 for that case. Before
expansion, it must be checked across multiple receptors, ligands, and random
seeds. Agreement for one rescued pair is not sufficient to establish global
search robustness.
