# Stage85 Mixed-radix Dirac Calibration Preparation

Three exact-oracle fixed-k calibration instances were translated into
native mixed-integer degree-two Dirac-3 polynomials.

- `ppara_of0_k10_high_k_exact`: 27 variables, 127 levels, 331 terms.
- `bace1_of0_k4_medium_pool_exact`: 41 variables, 171 levels, 784 terms.
- `pparg_of0_k3_large_pool_exact`: 103 variables, 364 levels, 5098 terms.

No QCI query or device job was performed. Run allocation-only preflight
first; the three physical jobs remain separately authorization-gated.
This package tests physical solver fidelity, not efficacy or quantum advantage.
