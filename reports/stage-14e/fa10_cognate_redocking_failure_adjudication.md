# Stage 14e FA10 Cognate-Redocking Failure Adjudication

## Decision

Stage 14d completed all 48 planned Uni-Dock jobs with zero unresolved warnings
and zero pose-integrity failures, but only 13 of 16 receptors passed the frozen
three-seed RMSD gate. The FA10 confirmatory technical gate is therefore closed.

| Receptor | Median RMSD (A) | Range (A) | Seeds passing | Failure class |
| --- | ---: | ---: | ---: | --- |
| FA10_1F0S_aligned | 2.090 | 2.060-2.188 | 0/3 | three-seed stable near-threshold pose mismatch |
| FA10_1LPG_aligned | 9.136 | 9.076-9.141 | 0/3 | three-seed stable alternative pose |
| FA10_2J2U_aligned | 7.340 | 7.332-7.359 | 0/3 | three-seed stable alternative pose |

## Interpretation

The failures are reproducible across seeds and are not runtime or pose-integrity
failures. FA10_1F0S is consistently close to, but above, the preregistered 2.0 A
cutoff; this does not authorize rounding or threshold relaxation. FA10_1LPG and
FA10_2J2U reproducibly favor distant alternative poses. The run does not identify
a unique molecular cause for any failure.

The single frozen reserve could raise the cohort to at most 14 passing receptors,
below the required 16. Lowering the cutoff, adding post-result receptors, or
screening a 13-receptor subset cannot rescue the confirmatory FA10 endpoint.

## Next Step

Do not run FA10 Train-696. Preserve this valid negative technical result and proceed
to HIVPR, the final target frozen in the multi-target master preregistration. Any
later FA10 13-receptor experiment must be labeled post-hoc exploratory.
