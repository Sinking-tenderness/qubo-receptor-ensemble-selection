# Stage 3 CDK2 AF2 2 ns Trajectory QC Record

## Scope

This record quality-controls the 100 saved frames from the short CDK2 AF2 apo
production pilot. It does not establish conformational convergence, binding
competence, or virtual-screening enrichment.

## Input Integrity

- Production experiment: `stage03-cdk2-af2-apo-cuda-production-2ns-v1`
- Equilibrated topology atoms: 46,206
- Protein atoms: 4,848
- Protein C-alpha atoms: 298
- Reference pocket C-alpha atoms: 21
- DCD chunks: 20
- Frames per chunk: 5
- Total frames: 100
- Frame interval: 20 ps
- Sampled production period: 2,000 ps (2 ns)
- Equilibrated topology SHA-256:
  `C71CB8BF333BA2254B27023996CD233589E5684A593878D73DD02947C186EAEF`

## Alignment-Based RMSD

All frames were least-squares aligned to the equilibrated reference using
protein backbone atoms. The values below therefore exclude overall translation
and rotation.

| Metric | Mean (A) | SD (A) | Minimum (A) | Maximum (A) | Final (A) |
|---|---:|---:|---:|---:|---:|
| Backbone RMSD | 1.247 | 0.185 | 0.749 | 1.609 | 1.387 |
| All C-alpha RMSD | 1.235 | 0.190 | 0.723 | 1.600 | 1.373 |
| Pocket C-alpha RMSD | 1.145 | 0.259 | 0.586 | 1.749 | 1.095 |

For the final 10 saved frames (1.82-2.00 ns), backbone RMSD averaged 1.416 A
with a 1.297-1.516 A range. Pocket C-alpha RMSD averaged 1.333 A with a
1.095-1.526 A range. These late values fluctuate within a bounded range and do
not show an immediate structural failure. A 25-frame late-window trend is added
to the revised QC output for a more reproducible drift check.

## Residue Flexibility

- Mean C-alpha RMSF across all residues: 0.715 A
- Maximum C-alpha RMSF: 2.667 A at terminal residue LEU 298
- Mean pocket C-alpha RMSF: 0.693 A
- Maximum pocket C-alpha RMSF: 1.762 A at HIS 84

The largest whole-protein fluctuations occur mainly at terminal residues and
the 159-163 region. Within the reference pocket, HIS 84 is the clearest flexible
site, followed by the 10-13 glycine-rich region. These motions are potential
sources of pocket diversity and should be retained for structure-only
clustering rather than interpreted as docking improvement by themselves.

The original QC JSON used a generic summary helper that included a `final`
field for RMSF arrays. RMSF is indexed by residue, not time, so that field meant
the last residue in the array and had no temporal interpretation. The revised
QC output removes it and reports median and 95th-percentile RMSF instead.

## Output Integrity

- QC summary SHA-256:
  `6D1E3D51E89C0490697870ABDE617CD8ADDAD2994E3972FB5E5061419842106C`
- Frame metrics SHA-256:
  `B6CF68111E82A63FFA0B27A27C9A91E3BAFDE6472FF60392CDB6FC8BBE6FEC92`
- Residue RMSF SHA-256:
  `8B6C317A7C10E56689422623C55619F56E67D7742796FC003F928E86BFEAABAD`

These hashes describe the first successful QC output. The summary and CSV
hashes will change after regeneration with the corrected RMSF schema and late
window statistics; the underlying DCD coordinates remain unchanged.

## Decision

The trajectory passes the limited numerical and structural checks required for
the MD-to-clustering MVP. No claim of a converged CDK2 ensemble is made, and the
100 frames remain correlated observations from one short trajectory. The next
step is structure-only pocket clustering and medoid selection. Selected medoids
will be candidate receptor conformers, not validated screening conformers, until
their docking and early-enrichment performance is measured.
