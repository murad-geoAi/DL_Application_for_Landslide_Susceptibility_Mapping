# Rev-2 audit diagnostics (facts from code + recomputed numbers)

Script: `src/diagnostics_rev2.py`; raw numbers: `results/revision/extra/rev2_diagnostics.json`. Paths are relative to the analysis repo. No pipeline code was changed.

## A. Spatial statistics

**1. Clark-Evans** (`revision_analyses.py:86-102`). R = mean NN distance / r_exp, r_exp = 0.5/sqrt(n/A), SE = 0.26136/sqrt(n*density), z = (r_obs - r_exp)/SE. NN distance by cKDTree on projected km. No edge correction in the reported R/z. A = convex hull of all 496 points (7284.9 km2) or the Rangamati ADM2 polygon (5755.6 km2).
| set | n | A | r_obs km | r_exp km | R | z |
|---|---|---|---|---|---|---|
| landslides, hull | 196 | 7284.9 | 0.541 | 3.048 | 0.177 | -22.03 |
| landslides, district | 196 | 5755.6 | 0.541 | 2.709 | 0.200 | -21.43 |
| non-landslides, hull | 300 | 7284.9 | 2.334 | 2.464 | 0.947 | -1.75 |
| non-landslides, district | 300 | 5755.6 | 2.334 | 2.190 | 1.066 | +2.18 |
| pooled, hull | 496 | 7284.9 | 1.478 | 1.916 | 0.771 | -9.74 |
| pooled, district (extra) | 496 | 5755.6 | 1.478 | 1.703 | 0.868 | -5.64 |

Diagnostic only (not in pipeline): Donnelly edge correction on the district polygon (perimeter 741.7 km): landslides R = 0.186, z = -19.8; non-landslides R = 1.005, z = +0.14; pooled R = 0.829, z = -6.79. Note the non-landslide "regular" (R>1) vs "random" verdict flips with the choice of A / edge correction.

**2. Projection** (`features.py:100-106` project_km; `revision_analyses.py:70-75` to_km, same constants). Local equirectangular, NOT an EPSG CRS: x = lon*111.320*cos(phi0), y = lat*110.574 (km). phi0 = 22.73373 deg N (mean latitude of all 496 rows, a fixed constant, `features.py:95`). => 102.67 km/deg lon, 110.574 km/deg lat. eps 1.1 km = 0.00995 deg lat / 0.01071 deg lon; Gi* band 5.5 km = 0.0497 deg lat / 0.0536 deg lon. Both are defined in km in the code; degree values are nominal conversions only.

**3. Jitter** (`revision_analyses.py:114`, also 306): `default_rng(42).normal(0, 1e-6, xy.shape)` added to the km coordinates. Units km (so 1e-6 km = 1 mm); 1e-6 is the SD (numpy `scale`), not the variance. Not needed in practice: 0 exact duplicate coordinates among the 496 (0 in either class), minimum NN distance 0.0108 km; Voronoi without jitter gives 0 undefined cells for both the 196 and 496 sets. It is a harmless safeguard. (Different jitter draws for the 196 and 496 tessellations.)

**4. Voronoi** (cells clipped to the ADM2 polygon, `revision_analyses.py:170-171`, not the convex hull; far dummy points added to close cells).
| tessellation | clip | median | P10 | P90 | CV | mean | max |
|---|---|---|---|---|---|---|---|
| 196 landslides | district (pipeline) | 1.274 | 0.109 | 36.92 | 5.47 | 29.37 | 1914 |
| 196 landslides | hull (extra) | 1.558 | 0.109 | 51.31 | 5.35 | 37.17 | 2414 |
| 496 all | district (pipeline) | 9.966 | 0.274 | 26.73 | 0.935 | 11.60 | 61.2 |
| 496 all | hull (extra) | 10.29 | 0.284 | 32.42 | 1.250 | 14.69 | 132.8 |
(all km2.) Check the manuscript against the district-clipped rows; the 196-point result is dominated by a few huge outer cells (CV 5.5, 89% of area in the largest 10% of cells).

**5. Gi\*** (`revision_analyses.py:240-246`). libpysal `DistanceBand(xy_km, threshold=5.5, binary=True)` -> transform "r" -> `esda.getisord.G_Local(y, w, star=True, permutations=999, seed=42)` (esda 2.9.0, libpysal 4.14.1).
- Star: yes, self weight included. W has zero diagonal; esda (UserWarning) fills the diagonal with the row maximum of the row-standardised matrix (= 1/k_i) and re-standardises, i.e. self weight = neighbour weight. Verified: my manual Gi* with self-weight reproduces esda Gs to 1.6e-17.
- Row-standardised: yes. Neighbours within 5.5 km (excl. self): min 0, median 7, max 92.
- Islands: 8 points with no neighbour; their W row is all zero (Gs = 0, Zs = NaN); excluded by mask `& ~isl` before classification. x_bar and S are computed by esda over ALL 496 (x_bar = 0.3952, S = 0.4889, population variance, denominator n, ddof 0); islands not removed.
- Permutations 999 (conditional), p_sim<0.05, no multiple-testing correction. Counts: hot 208, cold 224, ns 56, islands 8 (sum 496).
- Questionable points: (i) esda default `alternative='directed'` gives the smaller-tail ONE-sided p (not doubled); using p<0.05 on both tails is effectively two-sided alpha = 0.10. With p_sim<0.025 (two-sided 0.05): hot 202, cold 222, ns 64. (ii) For a zero neighbourhood (Gs = 0) the ">= observed" tie-counting makes p_sim = 0.001 regardless of neighbourhood size: 217 of the 224 "cold" points have Gs = 0 and 90 of them have <= 3 neighbours (median 4, min 1). (iii) esda's analytic `Zs` is degenerate here (range -0.81 to +1.17); the code uses it only for the sign. (iv) Textbook Ord-Getis Gi\* z (binary incl. self, |z|>1.96): hot 208, cold 93, ns 187 (cold overlap with pipeline 93/224; hot overlap 203/208). So the hot count is robust, the "cold" count (224) is inflated by the permutation tie issue; cold spots are 98.7% non-landslides (hot are 84.6% landslides).

**6. Moran's I and join counts** (`revision_analyses.py:220-239`). Moran: KNN k=5 on km coords, row-standardised, NOT symmetrised, 999 perm, np.random.seed(42): I = 0.7653, E[I] = -0.0020, z_sim = 29.0, p = 0.001. Join counts: same KNN k=5 graph, BINARY, NOT symmetrised (directed arcs). 496*5 = 2480 directed arcs (1822 reciprocal; 1569 unique undirected pairs if symmetrised). esda sums over directed arcs and halves, so J = 1240 "joins": BB = 440, WW = 650.5, BW = 149.5 (recount: 880 / 1301 / 299 directed arcs, sum 2480). Expected BB under permutation = 193.26 (analytic J*n1(n1-1)/(n(n-1)) = 193.03): same edge count (J = 1240) as observed, so 440 vs 193 is consistent. If the graph were symmetrised (1569 pairs): observed BB 549 vs expected 244.2. The 1240 is arcs/2, not a count of distinct neighbour pairs; state the convention in the text.

## B. DBSCAN

**7.** Class-blind: `DBSCANClusterAssigner.fit(X, y=None)` (`features.py:361`) takes only the Lat/Lon of ALL training-fold rows of both classes; y unused. eps = 1.1 km, min_samples = 5 (`features.py:89-90`), sklearn default Euclidean metric on projected km (min_samples counts the point itself). Test points: nearest training CORE point within eps (<=, `features.py:378-392`) -> that cluster, else -1; no refit.

**8.** k-distance (`revision_analyses.py:369-370`): `cKDTree.query(xy, k=6)`, kd = column 5 => 5th nearest neighbour EXCLUDING the point (column 0 is self). This is not the sklearn `NearestNeighbors(n_neighbors=5)` convention (which includes self, equal to my 4th-NN-excl.-self; verified identical). DBSCAN min_samples = 5 includes self, so the matching k-dist is the 4th NN excl. self (median 3.64 km vs 4.17 km plotted). Only 25% (k5) / 28.6% (k4) of points have k-dist <= 1.1 km; the chord "knee" is 6.96 km, so eps = 1.1 km is far below the knee (it is justified elsewhere by the 1.1 km ~ 0.01 deg legacy value and the sensitivity grid).

**9.** One-hot: `OneHotEncoder(handle_unknown="ignore")`, no drop (`features.py:644`): noise is its own column `Cluster_-1`, not an all-zero reference (unseen labels at test time -> all zeros). Total input dim, terrain + Cluster (23 + clusters + 1 noise col):
| spatial fold | train n | clusters | noise share (train) | noise share of train landslides | dim |
|---|---|---|---|---|---|
| 1 | 351 | 3 | 0.880 | 0.432 | 27 |
| 2 | 422 | 7 | 0.716 | 0.230 | 31 |
| 3 | 455 | 8 | 0.670 | 0.165 | 32 |
| 4 | 421 | 9 | 0.629 | 0.186 | 33 |
| 5 | 335 | 9 | 0.534 | 0.211 | 33 |
Ranges: dim 27-33; clusters 3-9; noise share of training landslides 0.165-0.432 (all training points 0.534-0.880). Random reference: 60/20/20 split seed 42 (train 297): 6 clusters, dim 30, noise 0.768 (landslides 0.419); StratifiedKFold(5, shuffle, rs 42): clusters 6-8, dim 30-32. (Note: `splits.py` has no stored random-CV folds; the repo's random reference in `run_seed` is the 60/20/20 split.)

**10.** Yes, in the spatial CV literally ALL test points of all 5 folds are noise (test_share_noise = 1.0 each fold; also `cluster_test_noise_share.csv`). Minimum distance from any test point to any training CORE point: fold 1 3.997 km, fold 2 2.157, fold 3 4.083, fold 4 16.246, fold 5 10.179 (eps = 1.1; 0 test points within eps). To any training point (core or not): min 1.75-2.16 km. So the Cluster feature is a constant (Cluster_-1 = 1) at test time under spatial CV. Under random folds 72-77% of test points are noise (23-28 test points within eps of a core point per fold).

## C. SMOTE / preprocessing

**11.** `sampling.py` (`snap_synthetic` 26-46, `SnapSMOTE` 49-67): imblearn SMOTE (k = 5, random_state = seed), then only synthetic rows are snapped. One-hot groups: `argmax` (`:32`) with first-max tie-break = lowest position in CATEGORY_LEVELS order (LULC 1,11,2,4,5,7,missing; Soil 4,6,7,missing), deterministic, not random. Binary flags `>= 0.5` -> 1 (`:36`; a tie rounds up). Aspect: `flat = (is_flat == 1) | (norm < 1e-6)` (`:39`) -> is_flat = 1, sin = cos = 0; others divided by norm. So a row whose flat flag rounds to 1 gets (0,0); a near-zero-norm row is forced flat. Real rows untouched (note: the 4 real rows with KNN-imputed aspect are not renormalised). Counts on seed 42, summed over the 5 spatial training folds, terrain config (pre-snap values):
- Full training fold (RF/XGB/SVM/LR path): 522 synthetic rows; flat flag rounds to 1: 2; of those with nonzero aspect vector: 0; flag at exactly 0.5: 0; non-flat rows with norm < 1e-6: 0; with norm < 0.1: 0 (< 0.5: 7, min norm 0.44 in fold 1); one-hot argmax ties: 0. Post-snap invalid rows: 0.
- 80% sub-train (Keras path): 417 synthetic; flat flag 1: 1 (nonzero aspect 0); norm < 0.1: 0 (< 0.5: 3); ties 0.
The two flat-flag rows occur in fold 5 only: in folds 1-4 the training minority is the landslide class (train prevalence < 0.5) and no flat cell is a landslide (see item 13); in fold 5 (train prevalence 0.579) the minority is the non-landslide class, which contains the flat cells, and both parents were flat so aspect was already (0,0). So the flat/aspect issue is moot in practice.

**12.** Order (spatial CV path, `cross_validate_model` with default `fit_fn` -> `fit_pipeline`, `models.py:886-913`): with no X_val, `pipe.fit(X_train, y_train)` (`:906`) fits the preprocessor (KNN imputer, scaler, one-hot schema, DBSCAN if included) on 100% of the training fold. For Keras models `KerasBinaryClassifier.fit` then takes the 20% stratified early-stopping split (`models.py:469`, random_state = seed) from the already-preprocessed matrix, so imputer/scaler statistics include the 20% validation rows (features only, no labels). SMOTE is applied AFTER that split, on the 80% sub-train (`models.py:486`). Non-Keras models: `ResampledClassifier.fit` (`models.py:971`) applies SMOTE to the full 100% training fold (no validation set). Test fold is transform-only everywhere. (The random-split reference uses `fit_pipeline` with an explicit validation set.)

## D. Data facts

**13.** Flat cells: 20 (all have Slope = 0 and Aspect = -1), all 20 are non-landslides (0 landslides). 17 of the 20 have LULC = 1 (water; there is no separate water flag in the code, water = LULC code 1); the other 3 are LULC 4 (2) and 5 (1). Water pixels overall: 33 (27 non-landslide, 6 landslide).

**14.** Spatial 5-fold (test, train prevalence = landslide share):
| fold | n | landslides | non | test prev | train n | train prev |
|---|---|---|---|---|---|---|
| 1 | 145 | 122 | 23 | 0.841 | 351 | 0.211 |
| 2 | 74 | 44 | 30 | 0.595 | 422 | 0.360 |
| 3 | 41 | 20 | 21 | 0.488 | 455 | 0.387 |
| 4 | 75 | 8 | 67 | 0.107 | 421 | 0.447 |
| 5 | 161 | 2 | 159 | 0.012 | 335 | 0.579 |
Bounding-box design (n = 349: 196 landslides, 153 non-landslides; folds re-derived with `spatial_block_cv`): n per fold 75, 64, 52, 53, 105 (range 52-105); landslides 65, 58, 38, 20, 15; landslide share 0.867, 0.906, 0.731, 0.377, 0.143 (range 0.143-0.906). Merged 4-fold design (fold 5 merged into fold 4; nearest by block centroid 22.3 km): n 145, 74, 41, 236; landslides 122, 44, 20, 10; shares 0.841, 0.595, 0.488, 0.042; train prevalence 0.211, 0.360, 0.387, 0.715.

**15.** Latitude span (deg N) and distinct values of `Annual_Rainfall` per spatial fold (CRU proxy; the 0.5 deg grid cells column counts floor(lat/0.5) x floor(lon/0.5)):
| fold | lat min-max | distinct rainfall values | 0.5-deg cells |
|---|---|---|---|
| 1 | 22.535-22.720 | 101 | 2 |
| 2 | 22.345-22.541 | 61 | 2 |
| 3 | 22.728-22.918 | 38 | 2 |
| 4 | 22.229-23.415 | 75 | 3 |
| 5 | 21.876-23.719 | 161 | 7 |
Overall 436 distinct rainfall values (so rainfall is NOT a single value per 0.5 deg CRU cell; it is spatially interpolated/varying per point, e.g. fold 1 has 101 distinct values in 2 cells; only 8 0.5-deg cells cover all data). The distinct-value proxy therefore does not give the CRU cell count; use the geometry-based count (2, 2, 2, 3, 7; 8 overall) if the CRU cell grid is aligned to multiples of 0.5 deg.

**16.** Spatial CV, terrain, no SMOTE, seed 42 (saved OOF `results/revision/seeds/oof_seed42.csv`; threshold = training prevalence):
XGBoost
| fold | test prev | mean prob | thr | frac above thr | recall | specificity |
|---|---|---|---|---|---|---|
| 1 | 0.841 | 0.119 | 0.211 | 0.186 | 0.213 | 0.957 |
| 2 | 0.595 | 0.440 | 0.360 | 0.541 | 0.750 | 0.767 |
| 3 | 0.488 | 0.026 | 0.387 | 0.000 | 0.000 | 1.000 |
| 4 | 0.107 | 0.044 | 0.447 | 0.000 | 0.000 | 1.000 |
| 5 | 0.012 | 0.181 | 0.579 | 0.099 | 0.000 | 0.899 |
RandomForest
| fold | test prev | mean prob | thr | frac above thr | recall | specificity |
|---|---|---|---|---|---|---|
| 1 | 0.841 | 0.325 | 0.211 | 0.766 | 0.828 | 0.565 |
| 2 | 0.595 | 0.417 | 0.360 | 0.581 | 0.773 | 0.700 |
| 3 | 0.488 | 0.153 | 0.387 | 0.049 | 0.100 | 1.000 |
| 4 | 0.107 | 0.148 | 0.447 | 0.027 | 0.125 | 0.985 |
| 5 | 0.012 | 0.256 | 0.579 | 0.075 | 0.000 | 0.925 |
XGBoost is much more extreme (probabilities collapse towards 0 in folds 3-4, so almost nothing exceeds the prior-matched threshold); RF is better calibrated across folds but also well below threshold in folds 3-5. Fold 5 has only 2 landslides (recall 0 for both).

**17.** Fig. 15 (map_model.json: RF chosen; `oof_seed_ensemble.csv`, RF terrain+SMOTE, 10-seed mean OOF; breaks = np.quantile(p, [.2,.4,.6,.8]) = 0.1014, 0.2104, 0.3834, 0.5758; np.digitize, `revision_extra.py:899-900`):
| class | n | landslides | share of landslides | share of samples | FR |
|---|---|---|---|---|---|
| Very low | 99 | 4 | 0.0204 | 0.1996 | 0.1022 |
| Low | 99 | 15 | 0.0765 | 0.1996 | 0.3834 |
| Moderate | 99 | 35 | 0.1786 | 0.1996 | 0.8947 |
| High | 99 | 60 | 0.3061 | 0.1996 | 1.5337 |
| Very high | 100 | 82 | 0.4184 | 0.2016 | 2.0751 |
FR = (landslides in class / 196)/(n in class / 496). Theoretical maximum for the top bin (100 samples, all landslides) = (100/196)/(100/496) = 496/196 = 2.531 (= 1/prevalence). Observed 2.075 is 82% of that ceiling.

**18.** AOA/DI (`revision_extra.py:835-870, 885-905`). Features: the 7 continuous terrain variables only (Elevation, Slope, Curvature, TWI, TRI, NDVI, Annual_Rainfall); no aspect, flags, LULC or soil (16 of 23 encoded columns excluded). Weights: mean |SHAP| per factor of the RF model (`shap_importance_by_model.csv`, column RF = pooled out-of-fold TreeSHAP, 10-seed mean), NOT Gini, and NOT in-fold: the weights are global (every fold's test points contribute to the weights that define the DI of that same fold). Normalised to mean 1 (Elevation 1.36, Slope 0.30, Curvature 0.29, TWI 0.36, TRI 0.39, NDVI 2.71, Rainfall 1.59). Reference cloud: real samples only (the other four spatial folds, no SMOTE); TWI NaN -> training median; standardised by training mean/sd then weighted. Threshold: Q3 + 1.5 IQR of training-point DI, each training point's DI = distance to nearest training point of a different fold, divided by mean pairwise training distance; per outer fold (0.464, 0.492, 0.499, 0.485, 0.637); CV scheme = the same 5 spatial folds. Result reproduced: 36 outside AOA (by fold 0, 0, 1, 6, 29; 2 landslides = 1.0% and 34 non-landslides = 11.3%).
Variants with all 23 encoded columns (preprocessor fitted on each training fold):
| variant | outside AOA (of 496) | by fold | landslides outside |
|---|---|---|---|
| current (7 continuous, weighted) | 36 | 0,0,1,6,29 | 2 |
| 7 continuous through encoded matrix (check) | 38 | 0,0,1,7,30 | 2 |
| 23 encoded, per-column mean abs SHAP weights, standardised | 32 | 4,0,0,3,25 | 5 |
| 23 encoded, weighted, as preprocessed (binary/one-hot not restandardised) | 47 | 0,0,1,6,40 | 2 |
| 23 encoded, unweighted, standardised | 15 | 0,2,1,1,11 | 2 |
| 23 encoded, unweighted, as preprocessed | 19 | 0,2,0,2,15 | 2 |
(The 36 vs 38 difference is imputation: median fill vs the pipeline KNN imputer.) Conclusion: AOA size is 15-47 across encoding/weighting choices; fold 5 always dominates.

**19.** Yes, ungrouped per-encoded-column OOF SHAP values (23 columns, 496 rows) are saved, but only as the 10-seed average (seeds 42, 0-8): RF `results/revision/shap_values_random_forest.csv` (+ `shap_features_random_forest.csv` = the matching preprocessed OOF features; TreeExplainer tree_path_dependent); CNN `results/revision/shap_values_cnn1d.csv` (+ `shap_features_cnn1d.csv`; KernelExplainer, kmeans background 20, nsamples 200). Per-seed per-sample values are not saved, only per-seed factor shares (`shap_share_by_seed_*.csv`) and `shap_seeds_*.json`. Grouping (sum within factor, then abs) is in `revision_analyses.factor_importance`.
