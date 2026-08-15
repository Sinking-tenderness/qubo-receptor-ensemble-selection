# Stage102A receipt audit

- Core archive SHA-256: `3ADC65BB2637188D8879D3A6AF9B405CC06BA74DE6EE549CBAE3FEC08DE763A9`
- Diagnostics archive SHA-256: `0AD64337534BAFC29A08E999EBB4DAA41F082DC081FF0B97535398DFEC6B6400`
- Both archives were fully listed and extracted without tar errors.
- EGFR: `36/36` batches, `21,600/21,600` seed-receptor-ligand scores, all status `ok`.
- FA10: `39/39` batches, `23,400/23,400` seed-receptor-ligand scores, all status `ok`.
- Combined: `75/75` batches and `45,000/45,000` expected raw scores.
- Pose integrity: `45,000/45,000` status `ok`; zero unresolved warning events.
- Median matrices: EGFR `600 x 12`; FA10 `600 x 13`; all `15,000` values finite.
- FA10 logs contain 13 known `add_to_output_container` warnings. The batch summaries classify all 13 as known and zero as unresolved; all affected runs still have 600 scores and valid single poses.
- EGFR has one seed-specific high score: `119.719` for `EGFR_decoy_L015100` against `EGFR_3W2S_aligned` in seed1. The same case scores `-7.525` and `-7.715` in seed0/seed2, so the primary three-seed median is finite and negative. The outlier does not survive median aggregation.
- Development boundary: no PARP1 fresh-validation rows, locked-test rows, or quantum-hardware jobs were used.

Result: the Stage102A docking output is technically complete and suitable for Phase-A development analysis. It is not by itself a passed biological gate.
