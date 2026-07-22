# State-of-the-Art Docking Assessment for the MAPK14 Ensemble-QUBO Study

Date: 2026-07-22

## Scope

The project does not need only a plausible pose for one ligand. It needs a
reproducible ligand-by-receptor score matrix whose columns are comparable
protein conformations, followed by leakage-controlled active/decoy ranking and
selection of a small receptor subset. The primary endpoint is early enrichment,
not crystallographic RMSD or affinity regression.

The current confirmatory input is already frozen:

- target: MAPK14;
- development data: 696 ligands, eight receptors, and three Vina seeds;
- fresh validation: 1,576 ligands, five required receptor columns, and three
  seeds;
- engine: official AutoDock Vina 1.2.7, exhaustiveness 32; and
- missing work: 23,640 fresh-validation receptor-ligand-seed jobs.

## What Recent Literature Actually Uses

| Method or workflow | Main role in recent work | Relevance here |
|---|---|---|
| AutoDock Vina or smina | Conventional pose search and score-matrix generation | Best fit for the already frozen primary evidence |
| DOCK3.7 plus a docking-score surrogate | Prospective billion-scale screening after target-specific grid and enrichment calibration | Strong precedent for staged screening, but a full new preparation and scoring protocol |
| Glide SP | Strong physical validity and enrichment in recent benchmarks | Useful commercial reference; not a drop-in replacement |
| GNINA 1.3 | Vina-style search followed by CNN pose scoring; includes a distilled fast model | Strong secondary scoring branch after target-specific validation |
| Vina-GPU 2.1 | GPU-accelerated Vina-family search using `thread` and `search_depth` | Closest GPU candidate, but its search controls do not reproduce Vina `exhaustiveness` directly |
| Uni-Dock | Batched GPU docking with vina, vinardo, and AD4 scoring modes | Very fast, but the completed MAPK14 equivalence gate failed |
| DiffDock-L, SurfDock, DynamicBind, AF3-like cofolding | Pose generation and, for some methods, ligand-specific receptor movement | Useful for discovering candidate conformations; not yet a reliable substitute for a comparable screening matrix |
| KarmaDock, CarsiDock, and RTMScore hierarchy | Fast pose generation, accurate pose generation, then learned rescoring | Promising future screening branch, but vulnerable to training-domain and physical-validity effects |
| Boltz-2 | Cofolding and affinity-based hit rescoring | Candidate for late hit triage, not the current exhaustive receptor-matrix stage |

The 2025 Nature Machine Intelligence virtual-screening benchmark found no
universal winner. AI methods could improve pose recovery or performance on
random decoys, while physics-based methods produced more physically rational
poses and Glide-based workflows led on the TrueDecoy enrichment task. The 2025
Chemical Science benchmark similarly found that hybrid search-plus-learned-
scoring methods provided a better balance than direct coordinate regression.
These results argue for target-specific gates rather than choosing a tool from
headline RMSD or speed.

## The Closest Published Precedent: EnOpt

The 2024 EnOpt study is directly aligned with this project. It accepts an
`n x m` matrix containing compounds as rows and receptor conformations as
columns, then uses XGBoost or random forest models to rank compounds and infer
which conformations are useful. The authors generated the matrices with smina
for DUD-E and AutoDock Vina for LIT-PCBA, used the top pose, and evaluated
AUROC, PR-AUC, BEDROC, and enrichment factors.

This establishes three important precedents:

1. A Vina-family score matrix remains academically defensible for modern
   ensemble-selection research.
2. A nonlinear tree model is a necessary strong classical baseline for a QUBO
   receptor-selection claim.
3. Receptor importance and the final activity ranking should both be evaluated;
structural diversity alone cannot establish screening utility.

The preregistered EnOpt-style comparator has now been implemented and frozen
on Train-696. Its all-five nested OOF BEDROC20 was 0.9312, while its
receptor-budget-matched three-column result was 0.9038. The corresponding QUBO
result was 0.9225. The all-five model was less docking-seed robust than QUBO,
and the three-column model selected different subsets across all four outer
fits. These training-only findings make fresh validation more important; they
do not establish a winner.

The present project can improve on EnOpt's validation design. EnOpt uses random
three-fold cross-validation and describes its predictions as a model trained on
the matrix itself. This project already has grouped scaffold/source-ID splits,
nested development-only tuning, three docking seeds, a fresh validation panel,
and a still-locked test split. An EnOpt-style baseline should reuse these
stricter boundaries rather than use random ligand folds.

## Tool-Specific Decision

### Official CPU Vina

Retain it for the frozen fresh validation. Changing only the validation engine
would compare a CPU-trained QUBO and normalization scheme against a different
score-generating process. All 16,704 decisive Train-696 jobs would otherwise
have to be regenerated before validation.

### Vina-GPU 2.1

This is the preferred future GPU equivalence candidate. It is peer reviewed
and substantially faster, but it exposes GPU docking lanes and heuristic search
depth rather than a numerical equivalent of Vina `exhaustiveness=32`. A frozen
consumed-Train-160 gate must compare completeness, score deltas, receptor-seed
rank correlations, top-tail overlap, active/decoy signed bias, and throughput.
Even a pass would authorize a full engine rebaseline, not mixing GPU validation
scores with the existing CPU training matrix.

### Uni-Dock

The local RTX 4090 experiment demonstrated the attraction and the risk. The
enhanced repaired profile was 30.81-fold faster than the 32-vCPU reference and
passed six of seven checks, but its minimum receptor-seed Spearman correlation
was 0.9305 rather than the frozen 0.95. Both tested profiles therefore remain
failed, and Uni-Dock must not generate current validation scores.

### GNINA 1.3

GNINA is the strongest open-source learned-scoring candidate for a secondary
branch. Its current release uses Vina-like conformational sampling followed by
CNN scoring and provides a distilled `fast` model for high-throughput work.
However, adding CNN scores changes the feature matrix and can introduce
training-domain bias on DUD-E-like targets. It should be benchmarked on consumed
training rows and evaluated as a separately frozen score branch.

### Dynamic or Cofolding Models

DynamicBind and kinase-specific AlphaFold2 multi-state modeling are relevant to
future receptor-pool generation. The kinase study showed that state-specific
templates can reduce the DFG-state bias of standard AF2/AF3 structures and
improve ensemble screening. These methods should propose structurally distinct
DFGin, DFGout, and intermediate candidates; redocking and active/decoy evidence
must still determine whether a candidate enters the selectable pool.

### Boltz-2

Boltz-2 is promising for reranking a small list of docking hits. A 2026
independent assessment found strong true/false-positive separation but also
reported insensitivity to meaningful binding-site mutations and, in some cases,
target exchange. It is therefore a useful secondary hit-triage experiment, not
evidence that the receptor-ensemble matrix can be skipped.

## Literature-Guided Execution Route

1. Complete the preregistered fresh-validation matrix with official CPU Vina.
2. Before validation scores are opened, freeze a supplementary EnOpt-style
   tree baseline using only Train-696 and the same five receptor columns that
   validation already requires.
3. Include an all-five tree model and a receptor-budget-matched three-column
   tree model. Keep the existing QUBO acceptance gate unchanged.
4. Evaluate QUBO, greedy, exhaustive, matched linear, single best, and tree
   baselines once on the fresh panel.
5. Run PoseBusters-style physical checks and interaction inspection on a frozen
   top-ranked pose subset. Do not infer screening success from RMSD alone.
6. If fresh validation passes, preregister the locked-test release before any
   test docking. If it fails, diagnose the existing matrix instead of changing
   the validation protocol after seeing labels and scores.
7. Develop Vina-GPU 2.1, GNINA, and state-specific kinase conformer generation
   as separate consumed-data branches for a future full rebaseline.

This route follows the recent literature's hybrid principle: preserve a
physically grounded search baseline, add a strong learned comparator, validate
on the target before scale-up, and reserve more expensive or less interpretable
methods for later stages.

## CPU Allocation Implication

The available service charges approximately CNY 0.02 per selected CPU core-hour.
The measured median remaining workload is about 4,050 vCPU-hours, so idealized
compute cost is approximately CNY 81 whether the job uses 24 or 120 cores.
Increasing cores mainly reduces elapsed time:

| Selected vCPU | Median projected wall time | Idealized compute charge |
|---:|---:|---:|
| 24 | 7.0 days | about CNY 81 |
| 48 | 3.5 days | about CNY 81 |
| 96 | 1.8 days | about CNY 81 |
| 120 | 1.4 days | about CNY 81 |
| 144 | 1.2 days | about CNY 81 |

Actual billing and time will vary with host contention, storage, setup, and
parallel efficiency. Because receptor-ligand jobs are independent, 96 or 120
vCPU is a practical wall-time choice if the platform truly bills linearly and
provides sufficient memory. Otherwise, the authorized 32-vCPU resume-capable
bundle remains valid.

## Primary Sources

- Bhatt R, Wang A, Durrant JD. Teaching old docks new tricks with machine
  learning enhanced ensemble docking. Scientific Reports (2024).
  https://doi.org/10.1038/s41598-024-71699-3
- Song J et al. Improving docking and virtual screening performance using
  AlphaFold2 multi-state modeling for kinases. Scientific Reports (2024).
  https://doi.org/10.1038/s41598-024-75400-6
- Gu S et al. Benchmarking AI-powered docking methods from the perspective of
  virtual screening. Nature Machine Intelligence (2025).
  https://doi.org/10.1038/s42256-025-00993-0
- Buttenschoen M, Morris GM, Deane CM. PoseBusters. Chemical Science (2024).
  https://doi.org/10.1039/D3SC04185A
- Wang M et al. Decoding the limits of deep learning in molecular docking for
  drug discovery. Chemical Science (2025).
  https://doi.org/10.1039/D5SC05395A
- McNutt AT et al. GNINA 1.3. Journal of Cheminformatics (2025).
  https://doi.org/10.1186/s13321-025-00973-x
- Tang S et al. Vina-GPU 2.1. IEEE/ACM Transactions on Computational Biology
  and Bioinformatics (2024). https://doi.org/10.1109/TCBB.2024.3467127
- Yu Y et al. Uni-Dock. Journal of Chemical Theory and Computation (2023).
  https://doi.org/10.1021/acs.jctc.2c01145
- Lu W et al. DynamicBind. Nature Communications (2024).
  https://doi.org/10.1038/s41467-024-45461-2
- Lyu A et al. Rapid traversal of vast chemical space using machine
  learning-guided docking screens. Nature Computational Science (2025).
  https://doi.org/10.1038/s43588-025-00777-x
- Gu S et al. Facilitating structure-based drug discovery with an artificial
  intelligence-driven virtual screening platform. Nature Protocols (2026).
  https://doi.org/10.1038/s41596-026-01389-z
- Bret G, Sindt F, Rognan D. Assessing Boltz-2 performance for the binding
  classification of docking hits. Journal of Chemical Information and Modeling
  (2026). https://doi.org/10.1021/acs.jcim.5c02630

## Interpretation Boundary

This is a targeted literature and implementation assessment, not a formal
systematic review. Tool performance is target-, dataset-, preparation-, and
metric-dependent. No published aggregate benchmark can replace the frozen
MAPK14 target-specific validation gate.
