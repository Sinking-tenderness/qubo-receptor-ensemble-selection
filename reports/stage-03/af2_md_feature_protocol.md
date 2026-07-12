# AF2/MD Pocket Feature Protocol

## Scientific boundary

The Stage 2 negative result closes the current score-only QUBO formulation.
Stage 3 does not retune it on the same benchmark. It creates a separate,
train-only structural feature path for future receptor subset selection.

## Conformer sources

1. Download the canonical AlphaFold DB model from the official API and retain
   its API response, model version, source URL, SHA-256, and pLDDT audit.
2. Align AF2 and future MD frames to the 1AQ1 chain-A coordinate frame using
   sequence-matched C-alpha Kabsch alignment.
3. Treat the AF2 model as an apo-like predicted structure. pLDDT is confidence,
   not proof that its ATP/STU pocket is biologically active.
4. Do not label a simulated frame as useful before docking and independent
   scaffold-aware evaluation.

## Pocket feature schema

The initial geometry schema is defined relative to the 1AQ1 STU reference
ligand after alignment:

- reference pocket residue identity;
- residue presence or absence;
- side-chain centroid coordinates;
- minimum residue-to-reference-ligand distance;
- pocket C-alpha RMSD to the reference pocket.

These geometry values are not interaction energies. Future ligand-specific
features must be calculated from docked poses and include active coverage,
decoy exposure, contact types, and pose-quality flags.

## Validation gate

Any new QUBO must be built on training data only and compared against
single-best, all-conformer, random, clustering, and greedy baselines under
scaffold-group outer CV. QAOA is considered only after the classical objective
has a stable positive result with a confidence interval excluding zero.
