# Stage 3 CDK2 AF2 MD E32 Warning Diagnostics Record

## Scope

The uniform exhaustiveness-32 matrix completed all 480 receptor-ligand jobs but
flagged five nonnegative, strongly unfavorable scores. This diagnostic reran
those five fixed pairs with three paired random seeds at exhaustiveness 32 and
64. Receptor, ligand, box, Vina version, CPU allocation, and preparation files
were held fixed.

The diagnostic was designed to distinguish stochastic search instability from
persistent receptor-ligand incompatibility. Its results are not used to replace
individual cells in either the exhaustiveness-8 or exhaustiveness-32 matrix.

## Protocol

- Source matrix: eight MD receptors by 30 actives and 30 decoys
- Flagged receptor-ligand pairs: 5
- Search settings: exhaustiveness 32 and 64
- Seeds per pair and setting: 3
- Total Vina runs: 30
- Parallel layout: 8 workers x 4 CPUs
- Box: center `(0.52, 27.06, 8.97)`, size `(18, 18, 16)` Angstrom
- Vina: AutoDock Vina 1.2.7 Linux x86-64
- Measured wall time: 502.632 seconds

The original source seed was retained for each pair. Two additional seeds used
fixed offsets of 100000 and 200000.

## Technical Outcome

- Expected runs: 30
- Successful runs: 30
- Failed runs: 0
- Source-score reproduction delta: exactly 0.0 kcal/mol for all five pairs
- Diagnostic classification: `search_instability_confirmed` for all five pairs
- Overall status: `completed_with_search_instability`

Exact source-score reproduction rules out accidental changes to the receptor,
ligand, seed, box, executable, or score parser as the explanation for the five
warnings.

## Pair Results

| Pair | Label | Source e32 | E32 range across seeds | E64 range across seeds | Maximum paired e32/e64 delta | Result |
|---|---|---:|---:|---:|---:|---|
| C00-A0009 | active | +1.551 | -9.837 to +1.551 | -9.837 to +1.551 | 0.000 | seed-sensitive search failure |
| C01-A0012 | active | +4.790 | -9.475 to +4.790 | -9.475 to +4.790 | 0.000 | seed-sensitive search failure |
| C07-A0012 | active | +16.530 | -9.493 to +16.530 | -9.493 to +16.530 | 0.000 | seed-sensitive search failure |
| C07-A0013 | active | +163.500 | -8.758 to +163.500 | -8.758 to +163.500 | 0.006 | seed-sensitive search failure |
| C03-D0016 | decoy | +70.780 | -4.547 to +70.780 | -4.915 to -4.150 | 74.930 | seed- and intensity-sensitive failure |

Each pair produced at least one favorable negative score without changing the
box or molecular inputs. The five source values therefore do not demonstrate
that these ligands are physically unable to fit the corresponding receptor
conformers.

For the first four active pairs, exhaustiveness 32 and 64 produced nearly
identical scores for every paired seed. Increasing exhaustiveness alone did not
rescue the unfavorable source seed. For C03-D0016, exhaustiveness 64 removed the
positive source result and produced a stable negative range of 0.765 kcal/mol.
The benchmark therefore contains both seed-sensitive and search-intensity-
sensitive cases.

## Integrity

- Source e32 warning table SHA-256:
  `B8E7295A6ABF8979F6F309E7649E222E358CF9FF0E6F091AD65698010DD36041`
- Diagnostic raw runs SHA-256:
  `FF727380F4DE114E8DF335C99A1F6FA202BEC75FF92C2B8C62D3FCC3ED3091DB`
- Diagnostic case summary SHA-256:
  `F41F2CD6E5B08F72FEB44D02E8727C841D9C149C9F1D94EB0EA359FE1E6AEFD4`
- Diagnostic summary SHA-256:
  `C7D32F45D3DF6408170008F717DD80707CEE4E4E5D9054DDF85D32677B0C59F4`

Generated poses, logs, and diagnostic tables remain ignored run artifacts.

## Decision

A single-seed exhaustiveness-64 matrix is rejected because four of five source
failures remained unchanged at exhaustiveness 64 for the same seed. Expanding
the box is not justified at this stage because alternate seeds found favorable
poses with the current box.

The next candidate protocol is three independent exhaustiveness-32 seeds for
every one of the 480 receptor-ligand pairs. The existing uniform e32 matrix is
retained as seed replicate zero. Two additional complete matrices use base-seed
offsets of 100000 and 200000. Scores will be aggregated uniformly across all
pairs; no warning-only cell replacement is permitted. Minimum-across-seed and
median-across-seed matrices will both be retained, together with seed-range and
agreement warnings, before enrichment or QUBO analysis is resumed.
