# Stage 07c MAPK14 Uni-Dock Warning Adjudication

## Boundary

This confirmation uses consumed Train-160 rows only. Fresh validation
and the locked test remain unread. Historical CPU Vina evidence remains
separate from the Uni-Dock evidence stream.

## Four-seed stability

- Minimum Spearman: 0.9358
- Median Top 5% overlap: 1.0000
- Maximum BEDROC delta: 0.0427

## Warning replay

- Exact score matches: 160/160
- Exact pose hash matches: 160/160
- Known warning events: 2
- Unresolved warning events: 0
- Pose integrity failures: 0

## Gate

- complete_new_seed: true
- complete_warning_replay: true
- zero_unresolved_warning_events: true
- zero_pose_integrity_failures: true
- exact_replay_scores: true
- exact_replay_pose_hashes: true
- minimum_seed_pair_spearman: true
- maximum_seed_pair_bedroc_delta: true

## Decision

Selected profile: `enhanced`.

A pass freezes only the Uni-Dock development protocol for a
separately preregistered training matrix. It does not authorize
validation/test access or establish QUBO or quantum advantage.

Authorization: `stage07c-mk14-unidock113-enhanced-warning-adjudication-v1`.
