# Stage 2 Redocking Record: CDK2 1AQ1-STU

## Target and Structure

- Target: CDK2
- PDB ID: 1AQ1
- Chain: A
- Co-crystal ligand: STU
- Ligand residue: STU A 299
- Experimental method: X-ray diffraction
- Resolution: 2.00 Å

## Input Files

- Raw receptor structure: receptors/raw/1AQ1.pdb
- Prepared receptor: receptors/prepared/1AQ1_A_receptor.pdbqt
- Raw ligand: ligands/raw/1AQ1_STU_A_299.sdf
- Explicit-H ligand SDF: ligands/prepared/1AQ1_STU_A_299_explicitH.sdf
- Prepared ligand: ligands/prepared/1AQ1_STU_A_299.pdbqt
- Vina config: configs/1AQ1_STU_redocking_vina.txt

## Docking Box

The docking box was derived from the co-crystal STU coordinates.

- center_x = 0.52
- center_y = 27.06
- center_z = 8.97
- size_x = 18
- size_y = 18
- size_z = 16

## Vina Parameters

- Vina version: 1.2.7
- scoring function: vina
- exhaustiveness = 16
- num_modes = 10
- seed = 20260708

## Redocking Output

- Redocked poses: results/docking/1AQ1_STU_redocked.pdbqt
- Vina log: logs/1AQ1_STU_redocking_vina.log
- Number of output poses: 10
- Best docking score: -13.87 kcal/mol

## Pose Validation

- Reference pose: co-crystal STU from 1AQ1.pdb
- Predicted pose: Vina mode 1
- RMSD type: heavy-atom RMSD
- Heavy atoms compared: 35
- Heavy-atom RMSD: 0.225 Å

## Interpretation

The top-ranked Vina pose closely reproduces the crystallographic STU pose.
The heavy-atom RMSD is far below the common 2 Å reference threshold, indicating that the current receptor preparation, ligand preparation, docking box, and Vina execution are suitable for this 1AQ1-STU redocking baseline.

This result validates pose reproduction for this co-crystal ligand, but it does not prove that docking scores are accurate binding free energies or that virtual screening enrichment will be high.

## Relevance to Ensemble/QUBO Stage

This redocking baseline confirms that a single receptor conformer can be prepared and docked reproducibly.
For future multi-conformer docking, the same preparation and evaluation rules must be applied consistently so that differences in docking score mainly reflect receptor conformational differences rather than protocol artifacts.
