# Stage 13g EGFR Cognate-Redocking Failure Adjudication

## Decision

Stage 13f completed all 48 planned Uni-Dock jobs with zero unresolved warnings
and zero pose-integrity failures, but only 12 of 16 receptors passed the frozen
three-seed RMSD gate. The EGFR confirmatory technical gate is therefore closed.

| Receptor | Median RMSD (A) | Range (A) | Seeds passing | Citation diagnostic |
| --- | ---: | ---: | ---: | --- |
| EGFR_9H47_aligned | 10.195 | 8.625-10.320 | 0/3 | covalent/irreversible wording |
| EGFR_8SC7_aligned | 12.998 | 12.977-13.085 | 0/3 | none |
| EGFR_5EM8_aligned | 7.495 | 7.494-7.517 | 0/3 | none |
| EGFR_5UGB_aligned | 3.398 | 3.299-3.527 | 0/3 | covalent/irreversible wording |

## Interpretation

The failures are reproducible across seeds and are not runtime failures. Citation
wording is a mechanism hypothesis only; no explicit protein-ligand covalent bond
was present in the admitted coordinates. The result does not identify a unique cause.

The single frozen reserve could raise the cohort to at most 13 passing receptors,
below the required 16. Lowering the RMSD cutoff, adding post-result receptors, or
screening a 12-receptor subset cannot rescue the confirmatory EGFR endpoint.

## Next Step

Do not run EGFR Train-696. Preserve this valid negative technical result and proceed
to FA10, the next target frozen in the multi-target master preregistration. Any later
EGFR 12-receptor experiment must be labeled post-hoc exploratory.
