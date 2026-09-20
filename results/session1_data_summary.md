# Session 1 — Data Audit: `Labeled Data Set.csv`

**File:** `Labeled Data Set.csv` (repo root, 58 KB, dated 2025-02-14)
**Audited:** 2026-09-19 · pandas 2.2.2 / numpy 1.26.4 / scikit-learn 1.8.0
**Status:** audit only — no cleaning, imputation, or modelling performed.

---

## 1. Shape

| | |
|---|---|
| Rows | **498** |
| Columns | **15** |
| Target | `Landslide` |
| Coordinates | `Latitude`, `Longitude` |
| Candidate predictors | 12 (but see §6.1 — only **11 are distinct**) |
| Complete rows (no NaN anywhere) | **280** (56.2%) |
| Rows with at least one NaN | **218** (43.8%) |
| Exact duplicate rows | **2** |

## 2. Exact column names

Verified by `repr()` — no leading/trailing whitespace, no hidden characters, no BOM.

```
 0  'Latitude'
 1  'Longitude'
 2  'Elevation'
 3  'Slope'
 4  'Aspect'
 5  'Curvature'
 6  'SPI'
 7  'TWI'
 8  'TRI'
 9  'NDVI'
10  'LULC'
11  'Annual_Rainfall'
12  'Drainage_Density'
13  'Soil_Texture'
14  'Landslide'
```

## 3. Dtypes

| Column | dtype | Column | dtype |
|---|---|---|---|
| Latitude | float64 | TRI | float64 |
| Longitude | float64 | NDVI | float64 |
| Elevation | int64 | LULC | float64 |
| Slope | float64 | Annual_Rainfall | float64 |
| Aspect | float64 | Drainage_Density | float64 |
| Curvature | float64 | Soil_Texture | float64 |
| SPI | float64 | Landslide | int64 |
| TWI | float64 | | |

Everything loaded as numeric; pandas inferred no `object` columns. Note that
`LULC` and `Soil_Texture` are **integer class codes stored as float64** — the
float dtype is an artefact of NaN presence, not evidence of continuity.

## 4. Missingness

| Column | n missing | % missing | n unique |
|---|---:|---:|---:|
| `SPI` | **200** | **40.16** | 288 |
| `TWI` | **200** | **40.16** | 288 |
| `Soil_Texture` | **50** | **10.04** | 3 |
| `Slope` | 4 | 0.80 | 366 |
| `Aspect` | 4 | 0.80 | 411 |
| `Annual_Rainfall` | 2 | 0.40 | 436 |
| `Drainage_Density` | 2 | 0.40 | 28 |
| `LULC` | 1 | 0.20 | 6 |
| `Latitude` | 0 | 0.00 | 495 |
| `Longitude` | 0 | 0.00 | 496 |
| `Elevation` | 0 | 0.00 | 208 |
| `Curvature` | 0 | 0.00 | 45 |
| `TRI` | 0 | 0.00 | 409 |
| `NDVI` | 0 | 0.00 | 496 |
| `Landslide` | 0 | 0.00 | 2 |

**Is missingness informative?** Weakly, and not enough to panic over.
P(Landslide = 1) among present vs. missing rows:

| Column | present | missing | difference |
|---|---:|---:|---:|
| `SPI` / `TWI` | 0.386 | 0.415 | +0.029 |
| `Soil_Texture` | 0.393 | 0.440 | +0.047 |
| `Slope` / `Aspect` | 0.397 | 0.500 | (n = 4, noise) |

The gaps are small and, for `Slope`/`Aspect`/`Annual_Rainfall`/`LULC`, rest on
1-4 rows. Treating missingness as approximately MAR is defensible, but a binary
missing-indicator for `SPI`/`TWI` and `Soil_Texture` costs nothing and should be
carried as a robustness check.

The `SPI`/`TWI` gaps are **not** a contiguous block — they scatter across 114
separate runs through the file, consistent with per-pixel raster extraction
failures (NoData cells) rather than a truncated join.

## 5. Class balance — `Landslide`

| Class | n | proportion |
|---|---:|---:|
| 0 (non-landslide) | **300** | 0.6024 |
| 1 (landslide) | **198** | 0.3976 |

Mild imbalance at roughly **3:2**. This does not require SMOTE or class
weighting; it does require that **accuracy be dropped as a headline metric** —
a constant-zero predictor scores 60.2% — in favour of AUC, precision/recall,
and F1.

---

## 6. Findings that affect the modelling design

These emerged from the audit and should be settled before any model is fit.

### 6.1 `SPI` and `TWI` are the same column — blocking

`df['SPI'].equals(df['TWI'])` returns `True`. Identical values, identical NaN
masks, maximum absolute difference `0.0`, Pearson r = `1.0`.

They are also the only two columns on a **standardised scale** (mean -0.297,
sd 1.367, range -4.09 to +4.95) while every other variable is in raw physical
units. Real TWI is a positive quantity around 0-25; real SPI is positive and
strongly right-skewed. Neither matches what is in the file.

So one column has overwritten the other at some point in the upstream raster
extraction, and the surviving copy has been z-scored. **The dataset as it stands
contains only one of these two predictors, and we cannot tell which.** Both must
be regenerated from the DEM, or one dropped and the other relabelled honestly.
Until then the effective predictor count is 11, not 12.

### 6.2 The file is sorted by class — blocking

Rows 0-197 are **all** `Landslide = 1`; rows 198-497 are **all** `Landslide = 0`.

Any split, fold, or batch built without shuffling is degenerate. The existing
notebooks use `train_test_split(..., random_state=42)`, which shuffles by
default, so they escape this — but `splits.py` must shuffle explicitly rather
than inherit the behaviour by luck, and any sequential model treating row order
as meaningful is reading an artefact.

### 6.3 Coordinates alone reach AUC 0.97 under random CV — the central problem

A random forest given **nothing but `Latitude` and `Longitude`** scores
**0.969 +/- 0.011 AUC** under stratified 5-fold random CV.

The cause is visible in the geometry:

| | median within-class nearest-neighbour distance |
|---|---:|
| Landslide points | **0.132 km** |
| Non-landslide points | **2.240 km** |

Positives are tightly clustered inventory points; negatives are dispersed
background samples. They also occupy different bounding boxes — positives
22.35-22.87 N, negatives 21.88-23.72 N. Under a random split, a positive's
near-duplicate neighbours land in both train and test, so the model can
memorise *where* rather than learn *why*.

Because every terrain covariate is a smooth function of position, this leakage
does not go away by dropping `Latitude`/`Longitude` from the feature matrix —
it merely becomes harder to see. **Any headline metric from a random split is
optimistically biased and cannot be reported as a susceptibility-model
performance estimate.** Spatial blocking (see `src/splits.py`) is not a
refinement here; it is the correction.

### 6.4 Imputation in the existing pipeline leaks

`notebooks/landslide_susceptibility_mapping.ipynb` cell 6 fits `KNNImputer`
(k = 5, keyed on geographic coordinates) and `SimpleImputer` on the **entire
dataset** before any split. With 40% of `SPI`/`TWI` missing and imputation keyed
on position, test-fold values are being reconstructed from train-fold
neighbours. All imputers and scalers must be fit inside the training fold only.

### 6.5 `LULC = 7` is close to a giveaway

| LULC code | n (class 0) | n (class 1) | P(landslide) |
|---:|---:|---:|---:|
| 1 | 27 | 7 | 0.206 |
| 2 | 242 | 133 | 0.355 |
| 4 | 5 | 0 | 0.000 |
| 5 | 10 | 1 | 0.091 |
| **7** | **4** | **48** | **0.923** |
| 11 | 11 | 9 | 0.450 |

`LULC = 7` carries a 92.3% landslide rate on 52 rows. Alone it yields 0.64 AUC.
This may be genuine — bare and degraded slope classes do concentrate failures —
or it may be circular: if the land-cover raster was interpreted *after* the
landslide inventory was known, class 7 may effectively encode "visible scar".
The class-code legend is needed before this feature can be trusted. Flagging,
not excluding, for now.

### 6.6 Smaller items

- **`Aspect` has a `-1` sentinel** on 20 rows, matching exactly the 20 rows with
  `Slope == 0`. That is the standard flat-cell encoding, not missing data. It
  must not be imputed, nor fed to a linear model as the number -1.
- **`Aspect` is circular.** 0 and 359 degrees are adjacent. Feeding degrees raw
  to any model imposes a false discontinuity at north; sin/cos encoding is
  required.
- **`Drainage_Density` is near-degenerate.** 469 of 496 non-null values are
  exactly `0.0`; only 27 rows carry signal (range 0.009-0.250). As a continuous
  predictor it is mostly a constant. Either recompute it at a sensible kernel
  radius or recast it as a binary presence flag.
- **2 exact duplicate rows** — index pairs (97, 98) and (182, 183), both
  `Landslide = 1`. These are also the only two duplicated `(Latitude, Longitude)`
  pairs. They inflate n and can place the same point in both train and test.
  Drop one of each.
- **`Soil_Texture` code 6 has n = 1.** It will not survive stratified splitting
  as its own level; fold it into an "other" level or drop the row.
- **`Elevation` is `int64` with range 14-810 m** and is the only integer
  covariate — consistent with a rounded DEM extraction. No action needed.

---

## 7. Categorical vs. continuous

### Categorical (integer class codes; `float64` dtype is a NaN artefact)

| Column | Observed codes (with counts) | Notes |
|---|---|---|
| **`Soil_Texture`** | 4 (n=263), 7 (n=184), 6 (n=1) | 3 levels. Code 6 is a singleton. Legend needed. |
| **`LULC`** | 2 (n=375), 7 (n=52), 1 (n=34), 11 (n=20), 5 (n=11), 4 (n=5) | 6 levels, non-contiguous codes — no 3, 6, 8, 9, 10 — so this is a subset of a standard scheme. Legend needed. See §6.5. |

Both must be **one-hot encoded**. Leaving them numeric asserts that
`Soil_Texture 7` is "more" than `4`, and that the gap from `4` to `7` is three
units of something — neither is true. Arithmetic on these codes is meaningless,
including the mean `Soil_Texture` of 5.24 that `.describe()` reports. Missing
values take **mode or an explicit "unknown" level**, never the median.

### Continuous

`Elevation`, `Slope`, `Curvature`, `SPI`/`TWI` (whichever survives §6.1), `TRI`,
`NDVI`, `Annual_Rainfall`, `Drainage_Density` (with the §6.6 caveat).

### Special handling

| Column | Type | Treatment |
|---|---|---|
| `Aspect` | **circular** | sin/cos encode; `-1` (flat) is its own indicator, not a value |
| `Latitude`, `Longitude` | **coordinates** | Use for *splitting and imputation geometry only* — never as model features (§6.3) |
| `Landslide` | **binary target** | 0/1, no missing values |

---

## 8. Carried into Session 2

**Blocking — must be resolved before any result is reportable:**

1. Regenerate `SPI` and `TWI` from the DEM, or drop one and relabel the other (§6.1).
2. Rebuild evaluation on **spatially blocked** splits; treat random-split numbers
   as a leakage baseline reported alongside, not as the result (§6.3).
3. Move all imputation and scaling inside the training fold (§6.4).

**Required before results are interpretable:**

4. Obtain the `LULC` and `Soil_Texture` code legends (§6.5, §7).
5. Decide the `Drainage_Density` treatment (§6.6).

**Mechanical fixes:**

6. Drop the 2 duplicate rows; handle the `Aspect = -1` sentinel; sin/cos encode
   `Aspect`; one-hot the two categoricals; drop accuracy as the headline metric.

**Expectation to set now:** honest spatially-blocked AUC will land well below
the numbers in the current manuscript. That drop is the correction working, not
a regression — and it is the finding worth writing up.
