# Stage90 ChEMBL analog-series intake

Status: `stage90_no_full_rescue_intake_candidate`.

Accepted activities: `25512` across `2079` single-assay endpoints.
Analog-series candidates: `15`; full rescue candidates: `0`.

| Target | Assay | Type | Molecules | Repeat series | Series molecules | High | Low | Receptors | Full gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| BACE1 | CHEMBL3706316 | IC50 | 365 | 6 | 258 | 248 | 52 | 34 | False |
| BACE1 | CHEMBL5730402 | Ki | 358 | 11 | 164 | 223 | 33 | 34 | False |
| BACE1 | CHEMBL3888176 | IC50 | 221 | 6 | 149 | 168 | 26 | 34 | False |
| BACE1 | CHEMBL5733314 | IC50 | 246 | 6 | 138 | 229 | 1 | 34 | False |
| BACE1 | CHEMBL3706381 | IC50 | 172 | 4 | 138 | 106 | 12 | 34 | False |
| BACE1 | CHEMBL3888178 | IC50 | 217 | 5 | 134 | 157 | 17 | 34 | False |
| BACE1 | CHEMBL3887679 | IC50 | 286 | 7 | 131 | 225 | 11 | 34 | False |
| BACE1 | CHEMBL3705666 | IC50 | 185 | 7 | 95 | 30 | 56 | 34 | False |
| BACE1 | CHEMBL5736881 | IC50 | 163 | 5 | 91 | 115 | 8 | 34 | False |
| BACE1 | CHEMBL5736880 | IC50 | 162 | 5 | 90 | 142 | 4 | 34 | False |

## Decision

No source passed all preregistered chemistry, potency, and receptor-pool gates. Do not start docking or quantum work.

This intake used public ChEMBL records only. It read no protected validation/test rows and launched no docking or hardware job.
