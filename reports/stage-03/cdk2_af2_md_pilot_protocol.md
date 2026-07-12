# CDK2 AF2 Apo-like MD Pilot Protocol

## Purpose and boundary

This protocol is pre-registered before any trajectory is produced. The pilot
will verify system construction, estimate runtime, and test the frame-to-pocket
feature pipeline. Its one 2 ns trajectory is not sufficient evidence of
converged CDK2 conformational sampling.

## System choice

- Starting structure: `AF-P24941-F1` version 6, aligned to the 1AQ1 chain-A
  reference frame.
- Chain: A; 298 residues.
- System state: apo-like, with no ligand, crystal water, metal, or cofactor.
- Rationale: all local crystal structures currently used in the pool have
  missing-residue records. Starting from complete AF2 avoids silently treating
  a modeled crystal loop as experimental structure. It does not make AF2 an
  experimentally validated pocket state.

## Parameter choices

| Component | Fixed pilot choice |
| --- | --- |
| MD engine | OpenMM 8.5, separate Conda environment |
| Protonation assumption | OpenMM `Modeller.addHydrogens`, pH 7.0 |
| Protein force field | Amber14SB (`amber14-all.xml`) |
| Water parameters | TIP3P-FB (`amber14/tip3pfb.xml`) |
| Solvent geometry | TIP3P geometry used by OpenMM modeller |
| Solvent padding | 1.0 nm |
| Salt | neutralized plus 0.15 M NaCl |
| Long-range electrostatics | PME; 1.0 nm cutoff |
| Constraints / timestep | H-bonds constrained; 2 fs |
| Temperature / pressure | 300 K / 1 bar |
| NVT / NPT | 100 ps / 500 ps |
| Production | one 2 ns pilot; report each 20 ps |
| Seed | 20260712 |

NVT means constant particle number, volume, and temperature. It lets the
newly solvated system relax at a fixed box volume. NPT means constant particle
number, pressure, and temperature; its box volume can adjust toward an
appropriate solution density. Production frames are recorded only after these
two equilibration stages.

## Known limitations

- Standard pH-based hydrogen assignment does not prove every histidine or
  titratable pocket residue has the biologically dominant microstate.
- Force fields and water models are approximations.
- A single 2 ns trajectory is a technical pilot, not a population estimate.
- Successive frames are correlated; 100 saved frames are not 100 independent
  receptor conformers.
- The Stage 2 test set must not be used to select MD parameters, frame count,
  or later QUBO weights.

## Build gate

Before any dynamics, run `scripts/build_openmm_system.py`. It must successfully
produce a solvated topology and audit its atom, water, and ion counts. It does
not minimize or integrate coordinates. The resulting manifest becomes the
input record for the later minimization/equilibration module.

Official sources accessed 2026-07-12:

- https://docs.openmm.org/latest/userguide/
- https://docs.openmm.org/latest/userguide/application/02_running_sims.html
- https://docs.openmm.org/latest/api-python/generated/openmm.app.modeller.Modeller.html
