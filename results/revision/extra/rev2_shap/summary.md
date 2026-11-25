# Round-2 SHAP checks (rev2_shap)

Fold sizes (n): F1=145, F2=74, F3=41, F4=75, F5=161

## 1. 1D-CNN SHAP with coalitions on the 10 parent factors (exact, 2^10 = 1024 coalitions)

Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 42] (10). Same folds/seeds/training as the existing CNN SHAP; retrained pooled AUC per seed matches the stored value (0: 0.8550, 1: 0.8367, 2: 0.8426, 3: 0.8610, 4: 0.8554, 42: 0.8555, 5: 0.8770, 6: 0.8377, 7: 0.8424, 8: 0.8366). Background = the 20 k-means centroids of the training fold (weighted by cluster size, as in KernelExplainer); 'off' groups take background values; model output averaged over the background; exact Shapley formula over all 1024 masks (efficiency check phi sum = f(x) - E f(bg) asserted).
Runtime: 308 s per seed (mean; 10- and 12-player runs together), 51 min total.

| Factor | RF share % (mean±sd) | CNN old (Kernel SHAP, 200 samples) % | CNN new (grouped exact) % | new - old (pp) | rank old | rank new |
|---|---|---|---|---|---|---|
| NDVI | 28.6 ± 0.6 | 28.9 ± 1.2 | 27.7 ± 1.0 | -1.1 | 1 | 1 |
| Annual rainfall | 16.7 ± 0.5 | 16.8 ± 1.0 | 16.4 ± 0.9 | -0.4 | 2 | 2 |
| Elevation | 14.4 ± 0.2 | 11.7 ± 1.1 | 11.7 ± 1.0 | +0.1 | 3 | 3 |
| Soil texture | 10.1 ± 0.1 | 7.8 ± 0.9 | 8.0 ± 0.9 | +0.2 | 5 | 5 |
| LULC | 9.7 ± 0.3 | 6.2 ± 0.8 | 6.1 ± 0.7 | -0.1 | 8 | 8 |
| Aspect | 6.3 ± 0.2 | 9.0 ± 0.8 | 8.8 ± 0.8 | -0.1 | 4 | 4 |
| TRI | 4.2 ± 0.2 | 7.5 ± 1.4 | 7.7 ± 1.3 | +0.2 | 6 | 6 |
| TWI | 3.8 ± 0.1 | 7.2 ± 1.0 | 7.2 ± 0.9 | +0.1 | 7 | 7 |
| Slope | 3.2 ± 0.1 | 2.2 ± 0.7 | 2.8 ± 0.6 | +0.6 | 10 | 10 |
| Curvature | 3.0 ± 0.1 | 2.7 ± 0.4 | 3.4 ± 0.4 | +0.6 | 9 | 9 |

Spearman rank correlation of factor shares: CNN old vs new = 1.000; RF vs CNN new = 0.891; RF vs CNN old = 0.891.

12-player variant (TWI_was_missing and Soil_Texture_was_missing as separate players; Aspect keeps is_flat):

| Player (12-player variant: both missing flags split out, 4096 coalitions) | share % (mean±sd) |
|---|---|
| NDVI | 26.7 ± 1.0 |
| Annual rainfall | 15.8 ± 0.9 |
| Elevation | 11.3 ± 1.0 |
| Aspect | 8.5 ± 0.7 |
| Soil texture | 7.6 ± 1.0 |
| TRI | 7.4 ± 1.2 |
| TWI missing flag | 6.3 ± 1.0 |
| LULC | 5.9 ± 0.7 |
| Curvature | 3.3 ± 0.4 |
| Slope | 2.7 ± 0.6 |
| TWI | 2.7 ± 0.4 |
| Soil missing flag | 1.8 ± 0.3 |

## 2a. Missingness flags alone

Share of total mean |SHAP| computed at column level (no factor summation; all 23 encoded columns in the denominator). The CNN column uses the stored seed-averaged ungrouped Kernel SHAP values (per-seed per-column values were not stored; seed-averaging damps noisy Kernel-SHAP magnitudes, so it is a lower bound).

| Column | RF per-column (10 seeds) % | RF+SMOTE per-column % | CNN old per-column (seed-averaged SHAP) % |
|---|---|---|---|
| TWI_was_missing | 1.85 ± 0.13 | 1.94 ± 0.17 | 5.96 |
| Soil_Texture_was_missing | 0.45 ± 0.05 | 0.39 ± 0.05 | 1.11 |
| TWI (continuous column) | 3.19 ± 0.12 | 3.47 ± 0.27 | 1.88 |
| is_flat | 0.84 ± 0.07 | 0.84 ± 0.07 | 0.61 |
| Soil_Texture_missing (one-hot level) | 0.44 ± 0.04 | 0.38 ± 0.03 | 0.14 |

CNN, grouped exact Shapley with the flags as their own players (12-player run), share of total mean |SHAP| %:

| Player | share % (mean±sd) |
|---|---|
| TWI missing flag | 6.25 ± 0.96 |
| Soil missing flag | 1.78 ± 0.28 |
| TWI | 2.71 ± 0.40 |
| Soil texture | 7.58 ± 1.03 |
| Aspect | 8.52 ± 0.74 |

## 2b. Pooled vs within-fold Spearman rho (factor value vs its SHAP value)

rho computed per spatial fold on out-of-fold SHAP (seed-averaged, as for the manuscript pooled rho), fold-size-weighted mean over folds where rho is defined; 'sign' = same sign in every fold with variation. Pooled values recomputed here equal the manuscript values (shap_direction.json).

### RF TreeSHAP (10 seeds)

| factor | pooled rho | manuscript pooled | per-fold rho (F1..F5) | fold-size-weighted mean | same sign in all folds | folds with sign = pooled |
|---|---|---|---|---|---|---|
| Elevation | -0.87 | -0.87 | -0.77, -0.63, -0.91, -0.94, -0.88 | -0.82 | yes | 5/5 |
| Slope | +0.18 | +0.18 | +0.50, +0.15, +0.71, -0.48, +0.09 | +0.18 | NO | 4/5 |
| Curvature | +0.34 | +0.34 | +0.83, +0.12, -0.37, -0.03, +0.03 | +0.23 | NO | 3/5 |
| TWI | -0.58 | -0.58 | -0.74, -0.65, -0.27, -0.78, -0.36 | -0.57 | yes | 5/5 |
| TRI | +0.40 | +0.40 | +0.52, +0.25, +0.77, -0.29, +0.57 | +0.40 | NO | 4/5 |
| NDVI | -0.73 | -0.73 | +0.21, -0.94, -0.79, -0.81, -0.79 | -0.52 | NO | 4/5 |
| Annual_Rainfall | +0.63 | +0.63 | +0.12, -0.19, +0.25, +0.71, +0.74 | +0.38 | NO | 4/5 |

### 1D-CNN, grouped exact Shapley (seed-averaged; note the TWI factor value includes the TWI-missing flag effect)

| factor | pooled rho | per-fold rho (F1..F5) | fold-size-weighted mean | same sign in all folds | folds with sign = pooled |
|---|---|---|---|---|---|
| Elevation | -0.91 | -0.87, -0.75, -0.87, -0.97, -0.97 | -0.90 | yes | 5/5 |
| Slope | +0.61 | +0.97, -0.44, +0.92, +0.15, +0.93 | +0.62 | NO | 4/5 |
| Curvature | +0.19 | +0.70, +0.50, +0.00, +0.06, -0.59 | +0.10 | NO | 4/5 |
| TWI | -0.19 | -0.46, +0.35, -0.21, -0.51, +0.05 | -0.16 | NO | 3/5 |
| TRI | +0.89 | +0.97, +0.98, +0.93, +0.79, +0.90 | +0.92 | yes | 5/5 |
| NDVI | -0.75 | +0.09, -0.94, -0.82, -0.89, -0.75 | -0.56 | NO | 4/5 |
| Annual_Rainfall | +0.67 | +0.75, -0.19, +0.85, +0.70, +0.73 | +0.61 | NO | 4/5 |

### 1D-CNN, old ungrouped Kernel SHAP (stored seed-averaged)

| factor | pooled rho | per-fold rho (F1..F5) | fold-size-weighted mean | same sign in all folds | folds with sign = pooled |
|---|---|---|---|---|---|
| Elevation | -0.88 | -0.87, -0.74, -0.71, -0.97, -0.95 | -0.88 | yes | 5/5 |
| Slope | +0.61 | +0.98, -0.43, +0.88, +0.20, +0.92 | +0.62 | NO | 4/5 |
| Curvature | +0.19 | +0.73, +0.56, +0.12, +0.00, -0.62 | +0.11 | NO | 4/5 |
| TWI | -0.32 | -0.43, +0.46, -0.23, -0.58, -0.22 | -0.23 | NO | 4/5 |
| TRI | +0.89 | +0.97, +0.98, +0.92, +0.77, +0.92 | +0.92 | yes | 5/5 |
| NDVI | -0.75 | +0.09, -0.94, -0.81, -0.89, -0.76 | -0.56 | NO | 4/5 |
| Annual_Rainfall | +0.67 | +0.76, -0.17, +0.86, +0.70, +0.73 | +0.61 | NO | 4/5 |

## 2c. RF + SMOTE (TreeSHAP on the forest fitted to SMOTE-augmented training folds, 10 seeds)

| factor | pooled rho | per-fold rho (F1..F5) | fold-size-weighted mean | same sign in all folds | folds with sign = pooled |
|---|---|---|---|---|---|
| Elevation | -0.84 | -0.74, -0.29, -0.88, -0.95, -0.89 | -0.77 | yes | 5/5 |
| Slope | +0.20 | +0.32, +0.01, +0.74, -0.51, +0.30 | +0.18 | NO | 4/5 |
| Curvature | +0.36 | +0.90, +0.38, -0.56, +0.06, -0.02 | +0.28 | NO | 3/5 |
| TWI | -0.34 | -0.23, -0.60, -0.27, -0.69, -0.29 | -0.38 | yes | 5/5 |
| TRI | +0.41 | +0.30, +0.24, +0.79, -0.34, +0.77 | +0.39 | NO | 4/5 |
| NDVI | -0.69 | +0.39, -0.94, -0.78, -0.80, -0.79 | -0.47 | NO | 4/5 |
| Annual_Rainfall | +0.46 | +0.59, -0.27, +0.24, +0.69, +0.74 | +0.50 | NO | 4/5 |

## Figures (new, originals untouched)

- reports/figures/revision/fig08_voronoi_quantified_v2.png
- reports/figures/revision/fig09_gistar_v2.png
- reports/figures/revision/fig11_dbscan_kdistance_v2.png
- reports/figures/revision/fig13a_spatial_folds_v2.png
- reports/figures/revision/fig14a_shap_importance_v2.png