# Session 3 — Decision Gate: Within-Fold vs Pooled AUC

**Question:** the within-fold AUC says how well the model ranks points *inside* a
held-out block. A susceptibility map is one ranking across the whole district.
Are those the same number?

They are not. Under spatial CV the gap is **~0.14 AUC**, in all three
configurations. Under a random split it is **~0.00**.

> **Superseded in part -- read section 6, then `session3_final.md`.** Sections
> 1-5 use the fold merge rule the brief specified (`balance="positives"`),
> which section 3.2 shows is defective. Under the corrected size-aware rule the terrain-only
> floor is **within-fold 0.8041 +/- 0.0580, pooled 0.7567**, a gap of
> **0.047**, not 0.137. The direction of every conclusion below holds; the
> magnitude of the penalty does not.

**Run:** 2026-09-20 · pandas 2.2.2 / numpy 1.26.4 / scikit-learn 1.8.0
**Scope:** decision gate. `src/splits.py` was written after this ran and now
carries the R2 scheme used here; `models.py` and `run.py` remain scaffolds.

---

## 1. What was held constant

| | |
|---|---|
| Rows | 496 (2 exact duplicate rows dropped), `data.clean`, shuffled seed 42 |
| Model | `RandomForestClassifier(n_estimators=500, random_state=42)` |
| Preprocessing | `features.build_preprocessor`, fit on the training fold only |
| Imputation | `KNNImputer(n_neighbors=5)`, `use_coords=False` |
| Coordinates | never a model feature in any configuration |
| Duplicate `SPI` | dropped, `TWI` kept |

### 1.1 The three configurations

| config | contents | n features |
|---|---|---:|
| **(a) terrain-only** | 10 conditioning factors | 21 |
| **(b) terrain + Drainage_Density** | (a) + `Drainage_Density` = 11 factors | 22 |
| **(c) terrain + Cluster** | (a) + DBSCAN cluster one-hot | 31 |

A note on the arithmetic, because the brief said "11 real factors, no
`Drainage_Density`" and those two clauses do not both hold. The 11 factors named
in `session2_floor_test.md` §1 *include* `Drainage_Density`. Removing it leaves
**10**. Configuration (a) is therefore the 10-factor set, and (b) is (a) plus
`Drainage_Density`, restoring the full 11. This reading makes (b) and (c) each
differ from (a) by exactly one thing, which is what an ablation table needs.

### 1.2 Fidelity of the reconstruction

The Session 2 floor test was run as a standalone diagnostic and **no script
survives**, so this session rebuilt it on top of `src/features.py` -- the code
that will actually be used -- rather than re-deriving the preprocessing by hand.

The rebuild reproduces the Session 2 five-block geometry **exactly**:
`KMeans(n_clusters=5, random_state=42, n_init=10)` on Lat/Long returns block
sizes `{21, 68, 80, 131, 196}` with positive counts `{0, 0, 0, 44, 152}`,
matching `session2_floor_test.md` §2 row for row, including the three
all-negative blocks.

It does **not** reproduce the per-fold AUCs bit-for-bit -- the two usable
five-block folds come back at 0.8113 / 0.7350 against the reported
0.8084 / 0.7593. The difference tracks two implementation details that a
standalone script fixes silently: the row order handed to the forest (which
determines its bootstrap draws) and whether the `*_was_missing` indicators enter
the KNN imputer's distance. Across those variants the five-block mean moves over
0.7714-0.7839, so the reported 0.7838 sits inside the range. **Treat agreement
with Session 2 as +/- 0.02, not as four decimals.** Every number below comes
from one pipeline, so the comparisons *within* this table are exact even where
the tie back to Session 2 is approximate.

On the balanced scheme the rebuild gives **0.7736 +/- 0.0715** against the
0.7816 +/- 0.0735 quoted in `session2_floor_test.md` §3.2 -- same central value,
same spread, all 5 folds usable.

---

## 2. The comparison table

Pooled AUC concatenates every point's out-of-fold prediction into a single
vector -- each of the 496 points scored exactly once, by the fold that held it
out -- and computes one ROC against the true labels.

| config | CV scheme | within-fold AUC | pooled AUC | gap |
|---|---|---:|---:|---:|
| **(a) terrain-only** | R2 balanced spatial | **0.7736 +/- 0.0715** | **0.6366** | **-0.137** |
| (a) terrain-only | R1 random stratified | 0.9493 +/- 0.0177 | 0.9475 | -0.002 |
| **(b) + Drainage_Density** | R2 balanced spatial | **0.7942 +/- 0.0610** | **0.6560** | **-0.138** |
| (b) + Drainage_Density | R1 random stratified | 0.9493 +/- 0.0201 | 0.9486 | -0.001 |
| **(c) + Cluster** | R2 balanced spatial | **0.7390 +/- 0.0576** | **0.5997** | **-0.139** |
| (c) + Cluster | R1 random stratified | 0.9632 +/- 0.0085 | 0.9627 | -0.001 |

Standard deviations are population (`ddof=0`) over the 5 fold AUCs.

### Per-fold detail, R2 balanced spatial

| fold | blocks | n_train | n_test | test pos | pos rate | AUC (a) | AUC (b) | AUC (c) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 351 | 145 | 122 | 0.841 | 0.8124 | 0.8223 | 0.7992 |
| 2 | 1 | 422 | 74 | 44 | 0.595 | 0.7803 | 0.7886 | 0.7723 |
| 3 | 0 | 455 | 41 | 20 | 0.488 | 0.7095 | 0.7167 | 0.6667 |
| 4 | 8 | 454 | 42 | 8 | 0.190 | 0.8824 | 0.8934 | 0.7849 |
| 5 | 2,3,4,6,7,9 | 302 | 194 | 2 | **0.010** | 0.6836 | 0.7500 | 0.6719 |

### Per-fold detail, R1 random stratified

| fold | n_test | test pos | AUC (a) | AUC (b) | AUC (c) |
|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 40 | 0.9408 | 0.9396 | 0.9598 |
| 2 | 99 | 39 | 0.9577 | 0.9564 | 0.9722 |
| 3 | 99 | 39 | 0.9588 | 0.9607 | 0.9650 |
| 4 | 99 | 39 | 0.9192 | 0.9156 | 0.9487 |
| 5 | 99 | 39 | 0.9701 | 0.9741 | 0.9705 |

---

## 3. Three things the table says

### 3.1 The pooled penalty is a property of the split, not of the feature set

The gap is -0.137, -0.138, -0.139 across three configurations that differ by up
to 10 features and by 0.055 AUC in level. Adding `Drainage_Density` lifts both
numbers by ~0.02 and changes the gap by 0.001. Adding the cluster feature
*lowers* the blocked numbers and changes the gap by 0.002.

Nothing in the feature set touches this. Whatever produces the gap is produced
by holding out contiguous ground, and it will not be engineered away by picking
better covariates.

### 3.2 Session 2's "pooled is at chance" was too strong -- it was measuring the dead folds

`session2_floor_test.md` §3.3 reported pooled AUC = 0.5094 and read it as the
ranking collapsing to chance. That was computed under the five-block scheme, in
which three of five blocks contain no landslides and contribute out-of-fold
predictions from models never scored on a positive.

Under the balanced scheme, with all five folds carrying both classes, pooled
AUC is **0.6366** -- well below the 0.7736 within-fold figure, but equally
clearly not chance. §3.3 flagged this risk itself ("part of this is a symptom of
3.1"), and it was right to. The corrected statement is that the global ranking
**degrades substantially, not that it fails**.

The magnitude is also sensitive to how the ten sub-blocks are merged. The
specified greedy balances on positive count alone, which sends all six
zero-positive blocks into one fold -- fold 5 carries 194 points and 2 positives
(§3.3 below). A size-aware greedy that balances positives *and* fold size splits
them differently and gives within 0.8041 +/- 0.0580, pooled 0.7567 -- a gap of
0.047 rather than 0.137. Dropping fold 5 from the pooled ROC entirely raises
pooled to 0.8025.

So the honest range on the pooled penalty is roughly **0.05 to 0.14 AUC**,
central estimate ~0.14 under the scheme as specified. The direction is stable
under every variant tried; the size is not.

### 3.3 The mechanism is fold-to-fold scale, and it is visible directly

Mean out-of-fold predicted probability, configuration (a):

| fold | pos rate | mean p (y=1) | mean p (y=0) | within-fold AUC |
|---:|---:|---:|---:|---:|
| 1 | 0.841 | 0.351 | 0.175 | 0.8124 |
| 2 | 0.595 | 0.503 | 0.282 | 0.7803 |
| 3 | 0.488 | 0.194 | 0.111 | 0.7095 |
| 4 | 0.190 | 0.333 | 0.156 | 0.8824 |
| 5 | 0.010 | 0.436 | 0.346 | 0.6836 |

Every row separates its own classes in the right direction -- that is the
within-fold AUC doing its job. But **a landslide in fold 3 scores 0.194, while a
non-landslide in fold 5 scores 0.346**. Pool those and the non-landslide
outranks the landslide. The per-fold models put their probabilities on
systematically different scales, so 0.3 in one block does not mean what 0.3
means in another.

Fold 5 is the clearest case and the most suspect: 194 points, 2 positives, and
the highest negative scores anywhere in the table. A model trained on the
positive-dense remainder and applied to a background-only region over-predicts
susceptibility across the board. With 2 positives its 0.6836 is close to
meaningless on its own, and it contributes 39% of the pooled vector. This is a
defect of the balancing rule, not a finding about the terrain; see §5.

---

## 4. What this implies for a district-wide susceptibility map

**A susceptibility map is the pooled quantity, not the within-fold one.** Every
cell in the district gets one score, and the map's meaning rests entirely on
those scores being comparable across the whole area -- a 0.7 in the north has to
denote the same hazard as a 0.7 in the south, because that is what a shared
colour ramp asserts. The within-fold AUC never tests this. It asks only whether
the model orders points correctly *against other points in the same block*, and
it can sit at 0.77 while the cross-block ordering is worth 0.64. The gap between
the two numbers is the part of the apparent performance that does not survive
being assembled into one map, and reporting the within-fold figure beside a
district-wide map claims a validation that was not performed. Practically this
means three things for the rest of the project: the headline table must carry
the pooled number next to the within-fold number, exactly as the blocked number
is carried next to the random one; any published map needs its scores put on a
common scale before they can be compared across the district, which is a
calibration step (fit inside the fold, on training data) that the pipeline does
not currently have; and until that exists, the defensible deliverable is a map
of *relative* susceptibility within delineated zones, with class breakpoints
justified per zone rather than a single district-wide ramp.

---

## 5. Carried into the next session

1. **The balancing rule needs deciding before R2 is used for anything headline.**
   Balancing on positive count alone concentrates all six zero-positive blocks
   into fold 5. **Decided, 2026-09-20:** `splits.spatial_block_cv` now defaults
   to the size-aware `balance="both"`, and `balance="positives"` is kept only
   to reproduce this table. See §6 and `session3_final.md`.
2. **Per-fold calibration curves**, as §3.3 of the floor test asked for. The
   mean-probability table above is the cheap version of that check and it
   already shows the effect; the curves would quantify it.
3. **A fold-internal calibration step** (Platt or isotonic, fit on training data
   inside the fold) is the obvious candidate fix. Whether it recovers the pooled
   AUC is an empirical question and should be measured, not assumed.
4. **Configuration (c) is now doubly disqualified.** The cluster feature was
   already a position proxy (`session2_feature_check.md` §3). This table adds
   that it *lowers* blocked performance at every fold while raising the random
   split to 0.9632 -- the exact signature of a feature that memorises location.
   It belongs in the ablation row and nowhere else.
5. **`Drainage_Density` is worth ~0.02 AUC blocked** and is now out of the
   default feature set (`drainage_mode="drop"`). That lift is carried by 27
   rows, all negatives, and must be read against `session2_feature_check.md`
   §1.3 before it is called a contribution.
6. **The `use_coords` imputer flag was inert, and that revises §1.6 of
   `session2_feature_check.md`.** Setting it up as an ablation this session
   exposed the problem: with the KNN imputer fed raw columns, switching
   `use_coords` on changed the design matrix by **exactly 0.0** across all
   11,408 cells. `Latitude` has sd 0.36 against `Elevation`'s 116, so in an
   unstandardised `nan_euclidean` distance the coordinates cannot move the
   neighbour ordering at all.

   Session 2 justified `use_coords=False` on the grounds that keying the
   imputer on position "would reconstruct 40% of the topographic index from its
   geographic neighbours". On unscaled inputs it would not — it would do
   nothing. The default is still correct, but the reason has changed, and the
   implication runs the other way: if the original notebook also imputed on
   unscaled columns, **its coordinate keying was inert too**, and the leak it
   introduced came from fitting the imputer on the full dataset before
   splitting rather than from the choice of distance key. Those are two
   separate defects and only the first is repaired by `use_coords=False`.

   `build_spatial_imputer` now standardises before imputing on the
   `use_coords=True` branch, which is the only arrangement in which position
   competes for neighbours. That makes the True arm differ from the default in
   two ways rather than one — noted in the docstring, and worth pricing in
   before the ablation is read.

---

## 6. Corrected fold balance -- the authoritative spatial-CV floor

Sections 1-5 were computed under `balance="positives"`, the rule specified in
the Session 3 brief. Section 3.2 flagged that rule as defective and section 5
item 1 left the choice open. **That item is now closed: the size-aware rule
`balance="both"` is the reported scheme, and the row below is the authoritative
spatial-CV floor number for the terrain-only configuration.**

Re-run 2026-09-20, configuration (a), 21 features, through
`splits.spatial_block_cv(df, balance="both")` -- the pipeline function, not a
standalone reimplementation. `balance="positives"` was re-run in the same
process as a control and reproduced sections 2-3 exactly, so the two rows below
differ in nothing but the merge rule.

| config | CV scheme | fold balance | within-fold AUC | pooled AUC | gap |
|---|---|---|---:|---:|---:|
| **(a) terrain-only** | R2 balanced spatial | **`both` (authoritative)** | **0.8041 +/- 0.0580** | **0.7567** | **-0.047** |
| (a) terrain-only | R2 balanced spatial | `positives` (as specified, superseded) | 0.7736 +/- 0.0715 | 0.6366 | -0.137 |
| (a) terrain-only | R1 random stratified | n/a | 0.9493 +/- 0.0177 | 0.9475 | -0.002 |

**No fold is single-class.** All five carry both classes, so all five AUCs are
defined and the mean is a genuine five-fold mean. `spatial_fold_labels` was
called with `require_both_classes=True` and did not raise; `fold_composition`
reports `single_class = False` on every row.

### 6.1 Fold composition, `balance="both"`

| fold | blocks | n_train | n_test | test pos | pos rate | single-class | lat range | AUC |
|---:|---|---:|---:|---:|---:|:---:|---|---:|
| 1 | 5 | 351 | 145 | 122 | 0.841 | False | 22.535-22.720 | 0.8124 |
| 2 | 1 | 422 | 74 | 44 | 0.595 | False | 22.345-22.541 | 0.7803 |
| 3 | 0 | 455 | 41 | 20 | 0.488 | False | 22.728-22.918 | 0.7095 |
| 4 | 3,8 | 421 | 75 | 8 | 0.107 | False | 22.229-23.415 | 0.8834 |
| 5 | 2,4,6,7,9 | 335 | 161 | 2 | 0.012 | False | 21.876-23.719 | 0.8349 |

### 6.2 What actually changed

One sub-block moved. Block 3 -- 33 points, no landslides -- goes from fold 5 to
fold 4. Folds 1, 2 and 3 are untouched and return identical AUCs under both
rules (0.8124 / 0.7803 / 0.7095, to four decimals). Fold 5 drops from 194 test
points to 161, and its training set grows from 302 rows to 335.

That single move is worth 0.12 pooled AUC, and the mechanism is the one section
3.3 identified:

| fold | rule | n_train | mean p (y=1) | mean p (y=0) | AUC |
|---:|---|---:|---:|---:|---:|
| 5 | `positives` | 302 | 0.436 | **0.346** | 0.6836 |
| 5 | `both` | 335 | 0.431 | **0.250** | 0.8349 |

Fold 5 is the background-only south-to-north sweep, and under the specified rule
the model scored its non-landslides at 0.346 -- higher than fold 3 scores its
actual landslides (0.194). Giving fold 5's training set block 3's 33 background
rows pulls its negatives down to 0.250 and stops them outranking other folds'
positives. The cross-fold scale mismatch is not eliminated, only reduced.

### 6.3 What this does and does not overturn

It does not overturn the finding. Pooled still sits below within-fold, and both
still sit far below the 0.9475 a random split reports. The direction of every
claim in section 4 is unchanged: a district-wide map is the pooled quantity, the
within-fold figure overstates it, and scores are not yet comparable across the
district.

What changes is the size of the penalty, from ~0.14 to **0.047**, and with it
the reading of section 3.2. The larger figure was substantially an artefact of
dumping all six empty sub-blocks into one 194-point fold. The honest statement
is now: **the pooled penalty on this data is about 0.05 AUC under a
well-conditioned spatial partition**, and the 0.137 figure should be cited only
as what a positives-only merge rule produces.

Three cautions attach to the new number:

1. **Section 3.1's invariance claim was measured under `positives`.**
   Configurations (b) and (c) have not been re-run under `both`. The claim that
   the gap is a property of the split rather than the feature set rests on
   three gaps of -0.137 / -0.138 / -0.139 that were all computed under the
   superseded rule. It is probably still true and it is no longer demonstrated.
2. **Fold 5's within-fold AUC rests on 2 positives** and moved 0.15 on the
   strength of 33 extra training rows. The within-fold mean inherits that
   fragility; the pooled figure, computed once over all 496 points, does not.
   That asymmetry is itself an argument for leading with the pooled number.
3. **The pooled figure still leans on the worst-conditioned fold.** Dropping
   fold 5 from the pooled ROC entirely gives 0.8288 over the remaining 335
   points (against 0.8025 over 302 under the old rule). The spread between
   0.7567 and 0.8288 is the range within which the terrain-only floor sits,
   depending on whether a near-empty block is counted.

### 6.4 Code status

**Resolved, 2026-09-20.** `splits.spatial_block_cv` now defaults to
`balance="both"`, as do `spatial_fold_labels` and `assign_blocks_to_folds` --
all three, because the parameter is threaded through them and flipping only the
outermost would have left fold *inspection* disagreeing with fold *execution*.
A bare `spatial_block_cv(df)` therefore now returns the authoritative partition.
`balance="positives"` still works and reproduces sections 1-5 exactly.
Configurations (b) and (c) have since been re-run under the new default; caution
1 above is discharged in `session3_final.md` §2.1.
