# Stage90a BACE1 rescue source freeze

Status: `stage90a_bace1_rescue_source_freeze_passed`.

Stage90 remains failed; its 50-receptor threshold was not changed. The rescue route instead uses the independent, pre-existing Stage41d certificate for 34 redocking-qualified receptors and 1,676,115 k=1..6 states.

| Role | Assay | Document | Molecules | Repeat series | Series molecules | High | Low | Locked |
|---|---|---|---:|---:|---:|---:|---:|---|
| development | CHEMBL3706316 | CHEMBL3639225 | 365 | 6 | 258 | 248 | 52 | False |
| confirmation_a | CHEMBL3888176 | CHEMBL3886610 | 221 | 6 | 149 | 168 | 26 | True |
| confirmation_b | CHEMBL3705666 | CHEMBL3638878 | 185 | 7 | 95 | 30 | 56 | True |
| locked_test | CHEMBL3888330 | CHEMBL3886197 | 115 | 4 | 58 | 59 | 20 | True |

All six assay pairs have zero molecule, Bemis-Murcko scaffold, and document overlap.

## Decision

Stage91 may freeze the development ligands and preregister the group-balanced large-pool comparison. Confirmation and locked-test docking remain prohibited. No quantum hardware route is reopened.
