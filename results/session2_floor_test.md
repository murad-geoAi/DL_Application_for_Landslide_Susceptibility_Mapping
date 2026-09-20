# Session 2 — Terrain-Only Floor Test

**Question:** with the leak-prone inputs removed, how much of the label can the
real conditioning factors actually explain?

**Run:** 2026-09-20 · pandas 2.2.2 / numpy 1.26.4 / scikit-learn 1.8.0
**Scope:** diagnostic only — no pipeline code was written or used for this.

---

## 1. What was held constant

| | |
|---|---|
| Features | **11 conditioning factors only** |
| Excluded | `Latitude`, `Longitude`, any DBSCAN-derived feature |
| Duplicate `SPI` column | dropped; the column labelled `TWI` kept (they are byte-identical — audit §6.1) |
| Rows | 496 (the 2 exact duplicate rows dropped) |
| Model | `RandomForestClassifier(n_estimators=500, random_state=42)` |
| Imputation | `KNNImputer(n_neighbors=5)`, **fit on the training fold only**, coordinates not among its inputs |
| Categoricals | `LULC`, `Soil_Texture` one-hot, vocabulary fit on the training fold, `handle_unknown="ignore"` |
| `Aspect` | sin/cos encoded; the 20 flat cells (`Slope == 0`, confirmed identical to the 20 `Aspect == -1` rows) set to `(0, 0)` and flagged `is_flat` |

The 11 factors: `Elevation`, `Slope`, `Aspect`, `Curvature`, `TWI`, `TRI`,
`NDVI`, `LULC`, `Annual_Rainfall`, `Drainage_Density`, `Soil_Texture`.

---

## 2. Headline result

| CV scheme | mean AUC | std | folds used |
|---|---:|---:|---:|
| **Spatially blocked 5-fold** (KMeans on Lat/Long, whole blocks held out) | **0.7838** | **0.0347** | **2 of 5** |
| **Random stratified 5-fold** | **0.9440** | **0.0313** | 5 of 5 |

**The gap is ~0.16 AUC.** The random-split number roughly reproduces the
existing manuscript's regime; the blocked number is the one that estimates
generalisation to unseen ground. Everything below is about why the blocked
number carries less weight than its tidy `+/- 0.0347` suggests.

### Per-fold detail

**Spatially blocked** (fold *k* holds out KMeans block *k-1*):

| fold | n_train | n_test | test positives | test pos. rate | AUC |
|---:|---:|---:|---:|---:|---:|
| 1 | 365 | 131 | 44 | 0.336 | 0.8084 |
| 2 | 428 | 68 | 0 | 0.000 | **undefined** |
| 3 | 416 | 80 | 0 | 0.000 | **undefined** |
| 4 | 475 | 21 | 0 | 0.000 | **undefined** |
| 5 | 300 | 196 | 152 | 0.776 | 0.7593 |

**Random stratified:**

| fold | n_test | test positives | AUC |
|---:|---:|---:|---:|
| 1 | 100 | 40 | 0.9046 |
| 2 | 99 | 39 | 0.9806 |
| 3 | 99 | 39 | 0.9188 |
| 4 | 99 | 39 | 0.9598 |
| 5 | 99 | 39 | 0.9560 |

---

## 3. Three findings that qualify the headline

### 3.1 Three of the five spatial blocks contain no landslides at all

AUC is undefined on a single-class fold, so the reported `0.7838 +/- 0.0347` is
a **mean over two folds**, and its standard deviation is the spread of two
numbers. It should never be quoted as a five-fold result.

This is not a defect of KMeans. It is the geometry the audit already
documented (§6.3): positives are a tight inventory cluster, negatives are
dispersed background samples, and the two occupy different bounding boxes.
Partition that space by location and you partition it largely by class. Blocks
1-3 (169 points between them) are pure background.

### 3.2 The number moves with the blocking seed, far more than the fold spread implies

Re-running the identical procedure with a different KMeans `random_state`:

| KMeans seed | usable folds | mean AUC |
|---:|---:|---:|
| 0 | 3 / 5 | 0.8632 +/- 0.1235 |
| 1 | 2 / 5 | 0.7906 +/- 0.0403 |
| 2 | 2 / 5 | 0.7919 +/- 0.0312 |
| 7 | 3 / 5 | 0.7136 +/- 0.1269 |
| **42** (reported) | **2 / 5** | **0.7838 +/- 0.0347** |
| 123 | 3 / 5 | 0.6824 +/- 0.0988 |

The honest range for the terrain-only floor is roughly **0.68 to 0.86**, not
`0.78 +/- 0.03`. The blocked estimate is real but imprecise, and the imprecision
comes from having ~2-3 usable folds over 496 spatially clustered points.

A grouping that keeps whole blocks intact while forcing every fold to contain
both classes — 10 KMeans blocks assigned to 5 folds balanced on positive count —
gives **0.7816 +/- 0.0735 with all 5 folds usable**, which corroborates the
~0.78 central value while showing the wider spread. Recommended as the primary
blocked scheme for Session 3; the 5-block version is reported here because it is
what was specified.

### 3.3 Pooled across the whole study area, blocked predictions are at chance

Concatenating every point's out-of-block prediction into one ROC over all 496
points gives **AUC = 0.5094**.

Within a held-out block the model ranks points at ~0.78. Across blocks, that
ranking collapses. The per-block models put their probabilities on
systematically different scales, so a score of 0.6 in one block is not
comparable to a 0.6 in another — and once pooled, the ordering is chance.

This matters more than the headline AUC, because **a susceptibility map is a
single global ranking across the whole area**, which is exactly the quantity
this number measures. A model that discriminates locally but not globally
produces a map whose colours are not comparable between regions. Worth
confirming in Session 3 against per-fold calibration curves before drawing a
firm conclusion — with three all-negative blocks, part of this is a symptom of
3.1 rather than an independent result.

---

## 4. Reading

The terrain-only floor is **AUC ~0.78 under spatial blocking, against ~0.94
under a random split**. Both numbers are honest measurements of different
things: the second measures how well the model memorises location, the first how
well the conditioning factors generalise to new ground.

Two things follow for the pipeline:

1. **~0.78 is the bar.** Any architecture — CNN, BNN, anything deeper — has to
   beat a 500-tree random forest on 11 tabular features under the same blocked
   scheme before its complexity is doing work. The floor is high enough that
   most of the terrain signal is already accessible to a simple model.
2. **Coordinates and DBSCAN features must stay out, and be shown to stay out.**
   They were excluded here by construction, which is what makes ~0.78 a floor
   rather than a ceiling. Adding a cluster-ID feature derived from position
   would move the number back toward 0.94 and would be measuring the same leak
   in a new coat.

Reported alongside each other, the 0.78 / 0.94 pair *is* the result. The drop is
the correction working.
