# CDK2 AF2 CUDA Equilibration Record

## Scope

This record summarizes the completed CUDA equilibration for the AF2-derived
apo-like CDK2 system. It documents a candidate starting state for a 2 ns
technical production pilot. It does not claim converged CDK2 conformational
sampling.

## Protocol and completion

- Experiment: `stage03-cdk2-af2-apo-cuda-equilibration-v1`.
- OpenMM: `8.5.2.dev-36a30cb`.
- Platform: RTX 4090 CUDA, mixed precision.
- System: 46,206 atoms; 298 protein C-alpha atoms.
- Minimization: bounded at 500 iterations.
- NVT: 100 ps, 50,000 steps.
- NPT: 500 ps, 250,000 steps.
- Timestep: 2 fs; metric/checkpoint interval: 10 ps.
- Completed without resume in 122.953 seconds.
- All 300,000 dynamics steps completed without a CUDA error or non-finite
  recorded metric.

## Late-window assessment

Statistics below use the twelve reported NPT points from 390 to 500 ps.

| Metric | Mean | SD | Linear slope per ps |
| --- | ---: | ---: | ---: |
| Temperature (K) | 300.413 | 1.345 | +0.0006 |
| Box volume (nm^3) | 457.471 | 0.966 | -0.0140 |
| Potential energy (kJ/mol) | -739284.3 | 853.0 | +5.47 |
| Protein CA centered RMSD (A) | 2.434 | 0.280 | +0.0070 |

Temperature is centered on 300 K. Box volume and potential energy fluctuate in
a narrow range relative to their means. The final 60 ps CA centered RMSD is
`2.681 +/- 0.130 A` with a reduced slope of approximately `+0.0020 A/ps`.
This metric removes translation but not overall rotation, so it is a gross
stability monitor rather than a clean internal-coordinate convergence test.

## Decision

The final NPT state is accepted as a candidate starting state for the 2 ns
technical production pilot. This decision means the short pipeline is
numerically stable and the late thermodynamic observables are suitable for a
pilot. It does not mean the receptor ensemble is converged or that all saved
production frames will be independent or useful for docking.

## Integrity and local-only outputs

- Equilibration manifest SHA-256:
  `C4A00A8B7728E6AFB913DEE85751C367AC2EBE226F4CCE7BD6EAB87EC2F37297`.
- Equilibrated PDB SHA-256:
  `C71CB8BF333BA2254B27023996CD233589E5684A593878D73DD02947C186EAEF`.
- Full progress JSON, metrics CSV, checkpoints, state XML, and PDB remain on
  the remote instance and are not committed to Git.

## Production gate

The next pilot remains NPT at 300 K and 1 bar for 2 ns with a 2 fs timestep.
Metrics are reported every 10 ps, coordinate frames every 20 ps, and a durable
checkpoint is completed every 100 ps. Production frames require aligned
backbone/pocket RMSD, RMSF, and clustering before any frame is treated as a
candidate receptor conformer.
