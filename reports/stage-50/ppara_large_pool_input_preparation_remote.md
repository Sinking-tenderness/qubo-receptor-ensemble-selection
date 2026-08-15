# Stage 50 PPARA Large-Pool Input Preparation

This bundle prepares the frozen 64-receptor PPARA pool for cognate redocking.

The input audit reads no docking score or protected validation/test row. Every
frozen receptor is attempted. Technical failures are retained as failures and
cannot be replaced after outcomes are known. At least 24 successfully prepared
receptors are required to proceed.

Standard-residue missing heavy atoms are completed with PDBFixer. Missing
residues are not added, nonstandard residues are not replaced, PDBFixer does not
add hydrogens, and Meeko is not allowed to delete bad residues.

Run with `--resume` through the packaged shell runner. Set `AUTO_POWEROFF=1` to
request instance shutdown after either success or failure.

Amendment01 normalizes PDBFixer 1.12 missing-atom entries that may be returned as
strings instead of atom objects. It changes no receptor identity, completion
rule, coordinate threshold, or docking parameter. Existing successful case
checkpoints are reused and only the same failed cases are retried.
