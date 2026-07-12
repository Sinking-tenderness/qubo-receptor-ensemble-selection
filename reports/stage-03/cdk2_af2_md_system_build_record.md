# CDK2 AF2 Apo-like MD System-Build Record

## Purpose

This record verifies the initial OpenMM system build for the pre-registered
CDK2 AF2 apo-like MD pilot. This operation built a protonated, solvated,
parameterized topology only. It did not run energy minimization, NVT, NPT, or
production molecular dynamics.

## Inputs

- Experiment ID: `stage03-cdk2-af2-apo-md-pilot-v1`
- Protocol: `configs/stage03_cdk2_af2_md_pilot.json`
- Input structure: aligned `AF-P24941-F1` version 6, chain A.
- Input structure SHA-256:
  `1AF07AC0294F8E505D233C67515E005BFBF81018704BE5AA129B28CCF27F12B2`.
- Environment: `environment/stage03_openmm.yml`.
- Runtime software: OpenMM `8.5.2.dev-36a30cb`, MDTraj `1.11.1`, PDBFixer
  `1.12` (import verified).

## Fixed build choices

- Protonation: OpenMM `Modeller.addHydrogens` at pH 7.0.
- Protein / water parameters: Amber14SB plus TIP3P-FB.
- Solvation: 1.0 nm padding, neutralization plus 0.15 M NaCl.
- Planned nonbonded settings: PME, 1.0 nm cutoff, H-bond constraints.

## System-build result

| Quantity | Input protein | Solvated system |
| --- | ---: | ---: |
| Chains | 1 | 3 |
| Residues | 298 | 14,091 |
| Atoms | 2,398 | 46,071 |
| Water residues | 0 | 13,715 |
| Sodium ions | 0 | 37 |
| Chloride ions | 0 | 41 |

The unequal sodium and chloride counts are expected: OpenMM first neutralizes
the protein's net charge, then adds ions to approximate the requested salt
concentration. They should not be interpreted as an error by themselves.

## Output integrity

- Solvated PDB: local and ignored by Git; SHA-256
  `378F38F83D085AD48CAC7473BD810BB854CC6B71739967B39379C6D9DBABC90C`.
- OpenMM System XML: local and ignored by Git; SHA-256
  `CA70FACCD89C880A12782F83EC21A3EFF49C21DA4CE04D21A9A64AC261604AFF`.
- Versioned manifest:
  `data/stage03_cdk2_af2_md_pilot_system_build.json`.

## Interpretation and gate

The parameterized system can now enter the separate minimization and
equilibration module. This does not demonstrate stable dynamics or conformer
sampling. The next module must first add trajectory reporters and explicit
minimization/NVT/NPT checks before any 2 ns pilot production trajectory is
started.
