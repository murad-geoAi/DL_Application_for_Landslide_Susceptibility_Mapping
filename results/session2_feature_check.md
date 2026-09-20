# Session 2 — Feature Pipeline: Choices and Leakage Proof

**Implemented:** `src/data.py`, `src/features.py`, `tests/test_feature_leakage.py`
**Run:** 2026-09-20 · pandas 2.2.2 / numpy 1.26.4 / scikit-learn 1.8.0
**Not yet implemented:** `splits.py`, `models.py`, `evaluate.py`, `run.py` — still scaffolds.

Companion to `results/session2_floor_test.md`, which measures what these features
are worth: **AUC ~0.78 spatially blocked, ~0.94 random**, terrain only.

---

## 1. Decisions

| # | Decision | Where |
|---|---|---|
| 1 | **2 exact duplicate rows dropped** (498 → 496), keeping the first of each pair | `data.drop_exact_duplicates` |
| 2 | **`SPI` dropped, `TWI` kept** | `data.resolve_spi_twi(strategy="drop_spi")` |
| 3 | **`Drainage_Density` KEPT**, flagged low-information *and* suspicious | `features.drainage_density_variant(mode="raw")` |
| 4 | `LULC` (6 codes) + `Soil_Texture` (3 codes) one-hot, **fit on train only**, unseen → all-zeros | `features.build_categorical_encoder` |
| 5 | `Aspect` sin/cos; flat cells kept as rows, mapped to `(0, 0)`; binary `is_flat` = `Slope == 0` | `features.encode_aspect` |
| 6 | `KNNImputer(n_neighbors=5)` **fit on train, transform-only on val/test** | `features.build_spatial_imputer` |
| 7 | `DBSCAN(eps=0.01, min_samples=5)` **fit on train coordinates only**; val/test assigned to nearest core point's cluster or `-1` | `features.DBSCANClusterAssigner` |
| 8 | Fixed-seed (42) shuffle applied at clean time, before anything splits | `data.shuffle_rows`, called by `data.clean` |

### 1.3 Drainage_Density — kept, and why the flag is stronger than "low-information"

**Kept**, at `mode="raw"`, with `"binary"` and `"drop"` variants one argument
away. Keeping it preserves comparability with the existing manuscript, and a
tree model mostly declines to split on a column that is constant across 95% of
rows, so the cost is one column.

But "near-constant" undersells what is in there. Of 494 non-null values, 467 are
exactly zero. The **27 non-zero rows are all negatives**:

| `Drainage_Density` | n | P(landslide) |
|---|---:|---:|
| `== 0` | 467 | 0.418 |
| `> 0` | 27 | **0.000** |

On 27 rows, that almost certainly records where the background sample happened
to be drawn rather than a protective effect of drainage. It is the same shape as
the `LULC == 7` problem in audit §6.5 — a rare level that separates the classes
nearly perfectly — and it should be read as a flag, not a finding. Any
importance this column registers in Session 3 must be reported against those 27
rows. The real fix is upstream: recompute at a kernel radius wide enough that
the column varies.

### 1.4 Missing categorical codes get an explicit level, not the mode

`Soil_Texture` is 10% missing, and the audit found missing-texture rows carry a
slightly *higher* landslide rate (0.440 vs 0.393). Mode-filling would assert the
majority texture for one row in ten and bury that. Missing codes go to an
explicit `"missing"` level instead, which appears as its own one-hot column.
Unseen codes — including `Soil_Texture == 6`, the singleton that is absent from
most training folds — map to all-zeros via `handle_unknown="ignore"`.

### 1.5 Aspect — what "keep the flat rows as-is" was taken to mean

The 20 `Aspect == -1` rows are kept, not dropped and not imputed. `is_flat` is
derived from `Slope == 0`, and `features.check_flat_cell_agreement` asserts the
two definitions pick the same 20 rows (they do, 0 disagreements) rather than
assuming it.

The `-1` sentinel is intercepted *before* the sin/cos transform, which would
otherwise read it as a bearing of 359°. Flat cells get `(0, 0)` — the origin of
the unit circle, on it nowhere — so `is_flat` is what carries their meaning:

| rows | radius `hypot(sin, cos)` |
|---|---|
| 20 flat cells | 0.0 (origin) |
| 472 measured | 1.0 (unit circle) |
| 4 genuinely missing `Aspect` | NaN → left to the imputer |

Those 4 rows are the only ones where an aspect is reconstructed, and the KNN
imputer fills them by searching the **full feature space**, not the two aspect
columns alone — which is why the encoding happens before the imputer in the
pipeline, not after.

### 1.6 A deviation worth stating: the imputer is not keyed on coordinates

Session 1 sketched `build_spatial_imputer` as coordinate-keyed. It is
implemented with `use_coords=False` as the default.

Keying a KNN imputer on latitude/longitude would reconstruct 40% of the
topographic index from its geographic neighbours — writing location back into
the covariates immediately after the coordinates were removed from them. That is
the notebook's leak reappearing one layer down. Feature-space neighbours carry
the same intuition without it. `use_coords=True` is retained to reproduce the
notebook for comparison.

Two audit caveats survive this and are recorded in the docstring: a fixed
`n_neighbors` reaches a wider geographic area for a dispersed negative than for
a clustered positive, and a positive's neighbours are overwhelmingly other
positives, so imputation can still write label information into the covariates.
Fitting inside the fold bounds that; it does not remove it. Check against a
median-fill baseline before trusting imputed columns in an importance ranking.

---

## 2. Proof: the fitted steps see training rows only

`python tests/test_feature_leakage.py` — 5 checks, all passing, also runnable
under pytest. Split is stratified 60/20/20, shuffled, seed 42: **297 train / 99
val / 100 test**.

### [1] DBSCAN fitted on train coordinates alone

```
DBSCAN fit on TRAIN only (eps=0.01, min_samples=5)
    train rows seen at fit : 297  (val=99, test=100 not seen)
    clusters found         : 6
    core points            : 57
    noise points in train  : 228

    cluster centres (train core points only):
 cluster  n_core_points  centre_lat  centre_lon
       0              9    22.57593    92.10657
       1              9    22.49903    92.18982
       2             14    22.60504    92.09388
       3             18    22.64918    92.13543
       4              5    22.63295    92.11484
       5              2    22.53869    92.12025
```

Asserted, not just printed:

- every fitted core point **is** a training coordinate (`core_coords ⊆ train_coords`);
- **no** val or test coordinate is a core point.

### [2] Val/test assigned against that frozen structure

```
    val : n= 99  assigned to a train cluster= 27  noise(-1)= 72
          label counts: {-1: 72, 0: 4, 1: 3, 2: 6, 3: 7, 4: 4, 5: 3}
    test: n=100  assigned to a train cluster= 25  noise(-1)= 75
          label counts: {-1: 75, 1: 10, 2: 5, 3: 7, 4: 2, 5: 1}
    fitted state after both transforms: byte-identical
    eps rule holds: all 27 assigned val points lie within 0.01 deg of a
                    train core point; the other 72 are noise
    agreement with DBSCAN's own train labels: 1.000
```

The test snapshots `core_points_`, `core_labels_`, `labels_`, `n_clusters_` and
`n_train_` after `fit`, calls `transform` on val and then test, and asserts every
one is unchanged. No label is invented that the training fit did not contain.
The assignment rule reproduces DBSCAN's own labels on the fitted data exactly
(agreement 1.000), so freezing the structure costs nothing in fidelity.

### [3] The counterfactual — proof the discipline is doing work

"Fit on train only" is an empty claim if the result would have been the same
either way. It would not:

| fit on | clusters | core points |
|---|---:|---:|
| train only | 6 | **57** |
| train + val | 6 | **97** |

The held-out points would have contributed **40 additional core points** to the
structure they are then scored against. That is the leak this rule closes.

### [4] Full preprocessor — 31 features, coordinates gone

```
    fit on train (297 rows) -> 31 features
    transform val  ( 99 rows) -> (99, 31)
    transform test (100 rows) -> (100, 31)
    Latitude/Longitude present in output: False
    no NaN survives imputation in train or val
```

One-hot vocabularies and the imputer are asserted unchanged across `transform`.

Feature names: `Elevation`, `Slope`, `Curvature`, `TWI`, `TRI`, `NDVI`,
`Annual_Rainfall`, `Drainage_Density`, `is_flat`, `TWI_was_missing`,
`Soil_Texture_was_missing`, `Aspect_sin`, `Aspect_cos`, `LULC_{1,11,2,4,5,7,missing}`,
`Soil_Texture_{4,6,7,missing}`, `Cluster_{-1,0,1,2,3,4,5}`.

### [5] Unseen categories map to all-zeros

Train vocabulary is `LULC=['1','11','2','4','5','7','missing']`,
`Soil_Texture=['4','6','7','missing']`. A probe row with `LULC=42`,
`Soil_Texture=99` encodes to **0 non-zero entries across 11 columns**.

---

## 3. The DBSCAN feature is procedurally clean and substantively a label copy

Built as specified, and it passes every leakage check above. It should still not
be in the headline model, and the reason is not a procedural objection.

Measured on held-out data, with the fit done on train only:

| assigned | P(landslide) | n (val) |
|---|---:|---:|
| to any train cluster | **0.963** | 27 |
| to noise (`-1`) | 0.181 | 72 |

A single bit — in a cluster or not — separates the classes on unseen data at
better than 0.96 precision.

This was structurally determined before any code ran. Audit §6.3: positives sit
a median **0.132 km** from their nearest same-class neighbour, negatives
**2.240 km**. `eps = 0.01` degrees is **~1.1 km** at this latitude. A 1.1 km
radius with `min_samples = 5` can only find the tight inventory clusters, so
"is in a DBSCAN cluster" is very close to a restatement of "is a landslide
point". In train, in-cluster P(landslide) is 0.986.

This is the audit's 0.969 coordinates-only AUC arriving through a different
door, laundered through a procedurally correct fit. Spatial blocking does not
defend against it either — the feature travels with the point.

**Recommendation for Session 3:** run `build_preprocessor(include_cluster=False)`
as the headline configuration and report `include_cluster=True` beside it as the
leakage baseline, exactly as the random-split AUC is reported beside the blocked
one. `include_cluster` defaults to `True` because that is the specified
pipeline; the default is not the recommendation. If the cluster feature is kept,
`eps` should be justified against the autocorrelation range rather than left at
a value that happens to match the inventory spacing.

---

## 4. Open items carried into Session 3

Unchanged from the audit, still blocking interpretation:

1. **`LULC` and `Soil_Texture` legends** — needed before `LULC == 7` (92.3%
   landslide rate) can be called a predictor rather than a scar proxy.
2. **Regenerate `SPI`/`TWI` from the DEM** — decision 2 is a naming choice, not
   a recovery. The pipeline carries 11 predictors, not 12.
3. **Recompute `Drainage_Density`** at a wider kernel; until then treat §1.3 as
   a flag.

New from this session:

4. **Ablate the cluster feature** (§3) — and report both numbers.
5. **Check the KNN imputer against median-fill** (§1.6) before reading any
   importance on an imputed column.
6. **`splits.py` must still shuffle explicitly.** `data.clean` now shuffles with
   seed 42, but that is a second line of defence, not a substitute — a splitter
   that relies on it is one refactor from breaking silently.
7. **Blocked folds go single-class.** The floor test found 3 of 5 KMeans blocks
   contain no positives at all, leaving a 2-fold mean. `spatial_block_cv` needs
   to either group blocks into class-balanced folds or report per-fold
   composition and refuse to average over undefined folds.
