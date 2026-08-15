# Stage86a QCI Dirac-3 rescue failure adjudication

The one frozen Dirac-3 job completed normally, returned 25 samples, and used 22 device seconds. No raw sample was fully feasible and none recovered the certified exact optimum.

| Diagnostic | Passing samples |
|---|---:|
| Correct cardinality (k=10) | 6 / 25 |
| Quality threshold | 13 / 25 |
| Both receptor constraints | 1 / 25 |
| Zero auxiliary residuals | 3 / 25 |
| Fully feasible raw encoding | 0 / 25 |
| Exact optimum | 0 / 25 |

One sample satisfied the scientific receptor constraints and was one auxiliary unit from a valid encoding. Deterministic auxiliary repair makes it feasible, but its original objective is -0.874360 versus the exact optimum -2.838705. It ranks 13,540 of 18,552 feasible subsets and is worse than the feasible median.

The primary endpoint failed. No additional QCI Dirac-3 global-penalty job is authorized, and the remaining 73 free seconds should not be spent on repetitions or post-hoc schedule tuning.

This result rejects the current physical encoding/interface combination. It does not reject the receptor-selection objective or establish a general claim about quantum hardware.
