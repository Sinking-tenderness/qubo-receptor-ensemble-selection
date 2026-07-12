# CDK2 AF2 CUDA 2 ns Production Pilot Record

## Scope

This record summarizes the completed 2 ns NPT production pilot for the
AF2-derived apo-like CDK2 system. The trajectory is accepted for quality
control, alignment, and clustering experiments. It is not treated as a
converged or independent receptor conformer ensemble.

## Completion and integrity

- Experiment: `stage03-cdk2-af2-apo-cuda-production-2ns-v1`.
- OpenMM: `8.5.2.dev-36a30cb`.
- Platform: RTX 4090 CUDA, mixed precision.
- System: 46,206 atoms.
- Ensemble: NPT at 300 K and 1 bar.
- Duration: 2 ns; timestep: 2 fs; total: 1,000,000 steps.
- Runtime: 405.434 seconds, corresponding to approximately 426 ns/day with
  reporting and checkpoint overhead.
- Metrics: 200 records at 10 ps intervals.
- Coordinates: 20 completed 100 ps DCD chunks, five frames per chunk, 100
  frames total at 20 ps intervals.
- No resume was needed and no CUDA or non-finite-state failure was reported.

## Late-window assessment

Statistics below use the twelve reported points from 1890 to 2000 ps.

| Metric | Mean | SD | Linear slope per ps |
| --- | ---: | ---: | ---: |
| Temperature (K) | 300.211 | 1.049 | +0.0131 |
| Box volume (nm^3) | 457.338 | 0.724 | +0.0022 |
| Potential energy (kJ/mol) | -739555.9 | 833.4 | +1.40 |
| Protein CA centered RMSD (A) | 3.929 | 0.149 | +0.0030 |

Temperature, volume, and potential energy remain in narrow late-window
fluctuation ranges. The final centered CA RMSD is 4.089 A. This metric removes
translation but not overall rotation, so it must not be interpreted as a
4.089 A internal structural change. Aligned backbone and pocket RMSD are the
next required checks.

## Integrity identifiers

- Production manifest SHA-256:
  `2CE2FC2AFD7E5CF74FB31FAFAAAD07D8DE72DEC6B464E247A13BF7D3107443CB`.
- Final PDB SHA-256:
  `6B45E143B2D13CD1164E0F7DC32C68D68193F6A7D3148E7D011564856F824ADC`.
- Each DCD chunk is approximately 2.7 MB. DCD files, checkpoints, state XML,
  and complete run directories remain outside Git.

## Decision and limitations

The trajectory passes completeness and gross numerical-stability checks and is
accepted for trajectory quality control. The 100 frames are consecutive,
time-correlated observations from one short replicate. Before any frame is
used as a receptor conformer, the trajectory must be globally aligned, checked
for backbone and pocket stability, summarized by per-residue RMSF, and reduced
through structure-based clustering. Screening utility remains untested.
