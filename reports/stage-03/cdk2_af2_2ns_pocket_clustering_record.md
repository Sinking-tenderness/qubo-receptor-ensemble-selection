# Stage 3 CDK2 AF2 2 ns Pocket Clustering Record

## Scope

This record reduces 100 correlated frames from one short CDK2 AF2 apo
trajectory to a fixed-budget set of eight structural medoids. The clustering
uses no ligand labels, docking scores, or virtual-screening outcomes.

## Inputs and Features

- Frames: 100 at 20 ps intervals
- Protein atoms: 4,848
- Pocket residues: 21
- C-alpha pair-distance features: 210
- Side-chain heavy-atom centroid pair-distance features: 210
- Total retained features: 420 of 420
- Low-variance threshold: 0.01 A
- Method: per-feature standardization followed by Ward agglomerative clustering
- Clustering config SHA-256:
  `D41E9968231FB19C6487A3B5B9E70ECCB36107C88FB63088CA71D1381A5549ED`
- Aligned trajectory SHA-256:
  `487813FED691403BD5574EE422BAAEA96A2791F985FF6A14328ADAC5F17B885E`

## Cluster-Count Diagnostics

| k | Silhouette | Smallest cluster | Largest cluster | Singletons |
|---:|---:|---:|---:|---:|
| 2 | 0.158101 | 33 | 67 | 0 |
| 3 | 0.141226 | 25 | 42 | 0 |
| 4 | 0.135988 | 14 | 33 | 0 |
| 5 | 0.118231 | 14 | 28 | 0 |
| 6 | 0.092193 | 7 | 25 | 0 |
| 7 | 0.095862 | 2 | 23 | 0 |
| 8 | 0.099781 | 2 | 23 | 0 |
| 9 | 0.087487 | 2 | 17 | 0 |
| 10 | 0.085998 | 2 | 17 | 0 |

The low silhouette values indicate overlapping, continuously varying pocket
geometries rather than well-separated metastable states. Eight clusters were
selected as a downstream computational budget, not because k=8 is a natural or
biologically optimal cluster count.

## Selected Medoids and Temporal Support

| Cluster | Medoid frame | Time (ps) | Members | Runs | Longest run (frames) | Revisited | Role |
|---:|---:|---:|---:|---:|---:|:---:|---|
| 0 | 1 | 40 | 2 | 1 | 2 | No | exploratory low support |
| 1 | 5 | 120 | 23 | 3 | 21 | Yes | primary candidate |
| 2 | 31 | 640 | 16 | 2 | 10 | Yes | primary candidate |
| 3 | 41 | 840 | 17 | 4 | 11 | Yes | primary candidate |
| 4 | 66 | 1340 | 14 | 3 | 6 | Yes | primary candidate |
| 5 | 74 | 1500 | 7 | 3 | 5 | Yes | primary candidate |
| 6 | 77 | 1560 | 7 | 2 | 4 | Yes | primary candidate |
| 7 | 95 | 1920 | 14 | 4 | 10 | Yes | primary candidate |

There are 21 adjacent-frame cluster transitions, and adjacent frames share a
cluster 78.8% of the time. Seven of eight clusters are revisited after the
trajectory leaves them. This supports recurrent geometry for those seven
clusters, although the observations remain correlated and do not establish
equilibrium populations.

Cluster 0 occurs only at 40 and 60 ps and is never revisited. It is retained for
the small docking sensitivity test rather than silently deleted, but it must be
reported separately as a low-temporal-support exploratory receptor.

## Output Integrity

- Frame assignments SHA-256:
  `5AFA5A5D282871F02AEFE3AFCCF0E9D1B1C18BBB9EDABB45CA003D2D8E57CB21`
- Pocket feature matrix SHA-256:
  `8A160172E11C587865297E4861D5DAE3E226E5CF6FE5BB04EA2733DDFC81138E`
- Cluster diagnostics SHA-256:
  `5E267D47139996950A5CD7E487B93FE152008B40B06373E6762D8A3EF7627A81`
- Medoid manifest SHA-256:
  `AFEDEDDEE6CA5AF03F1E217AD01DD878BABA2E77915494ACBF744A72C7D81769`
- Final clustering summary SHA-256:
  `FAE3A81C34EAEF12011ACF987176EACFB98EBDBF101511CC75CAC3D6016B10EE`

## Decision

All eight medoids proceed to coordinate alignment and a four-ligand docking
gate because the total test is only 32 receptor-ligand pairs. Cluster 0 is
retained as an explicitly flagged sensitivity candidate. The other seven are
the primary structure-supported candidates. None is considered screening-useful
until standardized receptor preparation and ligand-ranking evaluation are
complete.
