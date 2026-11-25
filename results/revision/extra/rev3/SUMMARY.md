# Revision round 3 reanalyses (A1-A10)

All values mean ± SD over seeds (SD ddof=1), 3 decimals; seeds 42,0-8 unless stated. Code: src/revision_rev3.py; per-seed CSVs in results/revision/extra/rev3/.

## IMPORTANT device finding (affects DNN rows of A1 and A5)
The DNN gives a different within-fold/pooled AUC when trained on the Metal GPU (the device that produced the existing primary DNN results, within 0.609) than on CPU (within about 0.775). 1D-CNN, LSTM and the classical models do not show this. DNN rows in A1/A5 are therefore reported separately as 'DNN [Metal GPU]' (comparable to the existing primary) and 'DNN [CPU]' (own baseline = CPU standard run). Fixed-epoch DNN counts differ by device (GPU 50, CPU 20), see A5.

## A1. Single-factor ablations (primary spatial CV, 10 seeds)
no_ndvi = drop NDVI; no_lulc = drop LULC (all LULC one-hots); no_flags_flat = drop TWI_was_missing, Soil_Texture_was_missing, is_flat (flag is `is_flat`, created in FrameEngineer/encode_aspect). Aspect sin/cos remain (flat cells keep (0,0)). Baseline 'none' = existing metrics_all_seeds terrain values.

| Model | Ablation | within AUC | pooled AUC | delta within | delta pooled |
|---|---|---|---|---|---|
| DNN [Metal GPU] | none | 0.609 ± 0.012 | 0.598 ± 0.035 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| DNN [Metal GPU] | no_ndvi | 0.592 ± 0.017 | 0.593 ± 0.030 | -0.017 ± 0.021 | -0.005 ± 0.036 |
| DNN [Metal GPU] | no_lulc | 0.592 ± 0.010 | 0.577 ± 0.029 | -0.017 ± 0.014 | -0.021 ± 0.019 |
| DNN [Metal GPU] | no_flags_flat | 0.602 ± 0.014 | 0.595 ± 0.029 | -0.007 ± 0.017 | -0.002 ± 0.027 |
| DNN [CPU] | none | 0.775 ± 0.019 | 0.859 ± 0.017 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| DNN [CPU] | no_ndvi | 0.658 ± 0.019 | 0.744 ± 0.017 | -0.117 ± 0.020 | -0.115 ± 0.010 |
| DNN [CPU] | no_lulc | 0.767 ± 0.015 | 0.840 ± 0.012 | -0.008 ± 0.008 | -0.019 ± 0.006 |
| DNN [CPU] | no_flags_flat | 0.774 ± 0.019 | 0.860 ± 0.014 | -0.001 ± 0.013 | 0.002 ± 0.005 |
| 1D-CNN | none | 0.789 ± 0.015 | 0.850 ± 0.013 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 1D-CNN | no_ndvi | 0.635 ± 0.013 | 0.726 ± 0.013 | -0.154 ± 0.011 | -0.124 ± 0.015 |
| 1D-CNN | no_lulc | 0.779 ± 0.014 | 0.829 ± 0.018 | -0.010 ± 0.009 | -0.021 ± 0.011 |
| 1D-CNN | no_flags_flat | 0.774 ± 0.019 | 0.852 ± 0.013 | -0.015 ± 0.009 | 0.002 ± 0.009 |
| LSTM | none | 0.557 ± 0.025 | 0.462 ± 0.065 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| LSTM | no_ndvi | 0.538 ± 0.037 | 0.493 ± 0.045 | -0.018 ± 0.031 | 0.031 ± 0.080 |
| LSTM | no_lulc | 0.552 ± 0.032 | 0.472 ± 0.064 | -0.005 ± 0.031 | 0.010 ± 0.094 |
| LSTM | no_flags_flat | 0.555 ± 0.026 | 0.446 ± 0.073 | -0.002 ± 0.024 | -0.015 ± 0.093 |
| RandomForest | none | 0.811 ± 0.007 | 0.761 ± 0.005 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| RandomForest | no_ndvi | 0.619 ± 0.008 | 0.627 ± 0.005 | -0.192 ± 0.008 | -0.134 ± 0.006 |
| RandomForest | no_lulc | 0.779 ± 0.007 | 0.733 ± 0.005 | -0.031 ± 0.010 | -0.028 ± 0.006 |
| RandomForest | no_flags_flat | 0.808 ± 0.007 | 0.761 ± 0.005 | -0.003 ± 0.012 | 0.001 ± 0.008 |
| XGBoost | none | 0.778 ± 0.008 | 0.707 ± 0.009 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| XGBoost | no_ndvi | 0.544 ± 0.012 | 0.610 ± 0.009 | -0.234 ± 0.015 | -0.097 ± 0.011 |
| XGBoost | no_lulc | 0.766 ± 0.006 | 0.698 ± 0.008 | -0.011 ± 0.009 | -0.009 ± 0.013 |
| XGBoost | no_flags_flat | 0.769 ± 0.004 | 0.707 ± 0.006 | -0.009 ± 0.008 | 0.001 ± 0.007 |
| SVM-RBF | none | 0.754 ± 0.000 | 0.838 ± 0.003 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| SVM-RBF | no_ndvi | 0.661 ± 0.000 | 0.689 ± 0.004 | -0.092 ± 0.000 | -0.149 ± 0.004 |
| SVM-RBF | no_lulc | 0.732 ± 0.000 | 0.811 ± 0.003 | -0.022 ± 0.000 | -0.027 ± 0.003 |
| SVM-RBF | no_flags_flat | 0.761 ± 0.000 | 0.844 ± 0.003 | 0.008 ± 0.000 | 0.005 ± 0.002 |
| LogisticRegression | none | 0.638 ± 0.000 | 0.768 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| LogisticRegression | no_ndvi | 0.572 ± 0.000 | 0.707 ± 0.000 | -0.066 ± 0.000 | -0.061 ± 0.000 |
| LogisticRegression | no_lulc | 0.639 ± 0.000 | 0.709 ± 0.000 | 0.001 ± 0.000 | -0.059 ± 0.000 |
| LogisticRegression | no_flags_flat | 0.599 ± 0.000 | 0.722 ± 0.000 | -0.039 ± 0.000 | -0.046 ± 0.000 |

CSV: A1_ablation_summary.csv, A1_ablation_per_seed.csv

## A2. Pooled F1 at threshold 0.5, terrain, 10 seeds

| Model | F1 |
|---|---|
| DNN | 0.334 ± 0.028 |
| 1D-CNN | 0.613 ± 0.045 |
| LSTM | 0.199 ± 0.026 |
| RandomForest | 0.309 ± 0.010 |
| XGBoost | 0.228 ± 0.018 |
| SVM-RBF | 0.604 ± 0.009 |
| LogisticRegression | 0.520 ± 0.000 |

## A3a. Prior rescaling of plain terrain OOF probabilities (per-fold training-fold share pi_k)
pi_k = fold1: 0.2108, fold2: 0.3602, fold3: 0.3868, fold4: 0.4466, fold5: 0.5791. Within-fold AUC unchanged (asserted; max abs difference over all model x seed x fold = 0.0e+00).

| Model | within plain | within rescaled | pooled plain | pooled rescaled | pooled terrain+SMOTE (existing) |
|---|---|---|---|---|---|
| DNN | 0.609 ± 0.012 | 0.609 ± 0.012 | 0.598 ± 0.035 | 0.787 ± 0.018 | 0.782 ± 0.018 |
| 1D-CNN | 0.789 ± 0.015 | 0.789 ± 0.015 | 0.850 ± 0.013 | 0.907 ± 0.007 | 0.884 ± 0.016 |
| LSTM | 0.557 ± 0.025 | 0.557 ± 0.025 | 0.462 ± 0.065 | 0.696 ± 0.043 | 0.648 ± 0.039 |
| RandomForest | 0.811 ± 0.007 | 0.811 ± 0.007 | 0.761 ± 0.005 | 0.889 ± 0.003 | 0.843 ± 0.005 |
| XGBoost | 0.778 ± 0.008 | 0.778 ± 0.008 | 0.707 ± 0.009 | 0.803 ± 0.006 | 0.809 ± 0.009 |
| SVM-RBF | 0.754 ± 0.000 | 0.754 ± 0.000 | 0.838 ± 0.003 | 0.899 ± 0.001 | 0.876 ± 0.005 |
| LogisticRegression | 0.638 ± 0.000 | 0.638 ± 0.000 | 0.768 ± 0.000 | 0.870 ± 0.000 | 0.866 ± 0.004 |

## A3b. Class weighting (RF/LR/SVM class_weight='balanced'; XGBoost scale_pos_weight = n_neg/n_pos per training fold), 10 seeds

| Model | within (variant) | pooled (variant) | within terrain | pooled terrain | within terrain+SMOTE | pooled terrain+SMOTE |
|---|---|---|---|---|---|---|
| RandomForest | 0.806 ± 0.004 | 0.834 ± 0.004 | 0.811 ± 0.007 | 0.761 ± 0.005 | 0.800 ± 0.010 | 0.843 ± 0.005 |
| XGBoost | 0.777 ± 0.008 | 0.742 ± 0.009 | 0.778 ± 0.008 | 0.707 ± 0.009 | 0.759 ± 0.009 | 0.809 ± 0.009 |
| SVM-RBF | 0.759 ± 0.000 | 0.851 ± 0.002 | 0.754 ± 0.000 | 0.838 ± 0.003 | 0.760 ± 0.010 | 0.876 ± 0.005 |
| LogisticRegression | 0.644 ± 0.000 | 0.870 ± 0.000 | 0.638 ± 0.000 | 0.768 ± 0.000 | 0.642 ± 0.009 | 0.866 ± 0.004 |

## A4. SMOTE-NC (k=5) inside training folds before one-hot encoding, 10 seeds

| Model | within (variant) | pooled (variant) | within terrain | pooled terrain | within terrain+SMOTE | pooled terrain+SMOTE |
|---|---|---|---|---|---|---|
| RandomForest | 0.798 ± 0.006 | 0.839 ± 0.004 | 0.811 ± 0.007 | 0.761 ± 0.005 | 0.800 ± 0.010 | 0.843 ± 0.005 |
| XGBoost | 0.762 ± 0.011 | 0.807 ± 0.008 | 0.778 ± 0.008 | 0.707 ± 0.009 | 0.759 ± 0.009 | 0.809 ± 0.009 |
| SVM-RBF | 0.760 ± 0.009 | 0.875 ± 0.005 | 0.754 ± 0.000 | 0.838 ± 0.003 | 0.760 ± 0.010 | 0.876 ± 0.005 |
| LogisticRegression | 0.636 ± 0.005 | 0.863 ± 0.003 | 0.638 ± 0.000 | 0.768 ± 0.000 | 0.642 ± 0.009 | 0.866 ± 0.004 |

A4 method: the existing pipeline applies SnapSMOTE AFTER one-hot encoding. Here a single estimator (PreEncodingSmoteNC, in src/revision_rev3.py) is fit per training fold: FrameEngineer (aspect sin/cos, is_flat) -> KNN imputation + standardization of the 9 continuous columns (same transformers as the standard preprocessor) -> flags (is_flat, TWI_was_missing, Soil_Texture_was_missing) passed through -> LULC and Soil_Texture mapped to integer level codes (fixed schema incl. 'missing') -> imblearn SMOTENC(categorical_features = 3 flags + 2 code columns, k_neighbors=5, random_state=seed) -> one-hot encode the two code columns into the identical 23-column layout -> model. Prediction applies the fitted imputer/scaler/encoders without resampling. Aspect sin/cos are interpolated as continuous columns and NOT renormalized to the unit circle (SnapSMOTE does renormalize). Classes are balanced to 1:1 as in SMOTE.

## A5. DL sensitivity (terrain; terrain+Cluster for variant a), 10 seeds
(a) stratified 20% early-stopping subset split off the raw training fold FIRST (same subset as the standard run), imputer/scaler (and DBSCAN) fit on the remaining 80% only, the 20% transformed out of sample. (b) 100% of the training fold, imputer/scaler fit on all of it, no early stopping, fixed epochs E = median best epoch (epoch of minimum val_loss) of the standard runs (50 folds per model = 10 seeds x 5 folds; DNN E per device). 'standard' = rerun of the existing procedure (logs epochs); 'primary_*_existing' = values from metrics_all_seeds.csv.

Fixed epochs E: 1D-CNN 38 (median 37.5, rounded half-even), LSTM 15, DNN [Metal GPU] 50, DNN [CPU] 20 (mixed-device DNN median 28 not used).

| Model | Variant | within AUC | pooled AUC |
|---|---|---|---|
| DNN [Metal GPU] | primary_terrain_existing | 0.609 ± 0.012 | 0.598 ± 0.035 |
| DNN [Metal GPU] | standard | 0.609 ± 0.012 | 0.598 ± 0.035 |
| DNN [Metal GPU] | a_terrain | 0.606 ± 0.016 | 0.599 ± 0.035 |
| DNN [Metal GPU] | b_fixed_epochs | 0.609 ± 0.019 | 0.597 ± 0.033 |
| DNN [Metal GPU] | primary_terrain+Cluster_existing | 0.603 ± 0.024 | 0.715 ± 0.032 |
| DNN [Metal GPU] | a_terrain+Cluster | 0.603 ± 0.017 | 0.698 ± 0.039 |
| DNN [CPU] | standard | 0.775 ± 0.019 | 0.859 ± 0.017 |
| DNN [CPU] | a_terrain | 0.774 ± 0.021 | 0.859 ± 0.016 |
| DNN [CPU] | b_fixed_epochs | 0.785 ± 0.008 | 0.861 ± 0.005 |
| DNN [CPU] | a_terrain+Cluster | 0.710 ± 0.037 | 0.825 ± 0.023 |
| 1D-CNN | primary_terrain_existing | 0.789 ± 0.015 | 0.850 ± 0.013 |
| 1D-CNN | standard | 0.788 ± 0.015 | 0.849 ± 0.014 |
| 1D-CNN | a_terrain | 0.785 ± 0.017 | 0.850 ± 0.015 |
| 1D-CNN | b_fixed_epochs | 0.787 ± 0.004 | 0.863 ± 0.012 |
| 1D-CNN | primary_terrain+Cluster_existing | 0.725 ± 0.021 | 0.807 ± 0.030 |
| 1D-CNN | a_terrain+Cluster | 0.734 ± 0.021 | 0.818 ± 0.022 |
| LSTM | primary_terrain_existing | 0.557 ± 0.025 | 0.462 ± 0.065 |
| LSTM | standard | 0.557 ± 0.025 | 0.462 ± 0.065 |
| LSTM | a_terrain | 0.561 ± 0.028 | 0.454 ± 0.075 |
| LSTM | b_fixed_epochs | 0.571 ± 0.013 | 0.498 ± 0.021 |
| LSTM | primary_terrain+Cluster_existing | 0.570 ± 0.039 | 0.589 ± 0.088 |
| LSTM | a_terrain+Cluster | 0.548 ± 0.056 | 0.451 ± 0.111 |

Standard-run epochs (all devices mixed, 50 folds/model):

| model   |   n |   best_epoch_median |   best_epoch_mean |   best_epoch_min |   best_epoch_max |   stop_epoch_median |
|:--------|----:|--------------------:|------------------:|-----------------:|-----------------:|--------------------:|
| 1D-CNN  |  50 |                37.5 |             36.06 |               20 |               50 |                42.5 |
| DNN     |  50 |                27.5 |             32.7  |               15 |               50 |                32.5 |
| LSTM    |  50 |                15   |             19.4  |                2 |               46 |                20   |

## A6. SHAP factor shares (% of total mean |SHAP|; signed values summed within group per sample, then absolute value, then mean; 10 seeds)
RF: exact TreeSHAP OOF, recomputed per seed (the 10-factor grouping reproduces shap_share_by_seed_random_forest.csv). 1D-CNN: grouped exact Shapley (20 k-means background points, as rev2) recomputed with 11 and 13 players; retrained pooled AUCs reproduce the stored per-seed values exactly. 11 groups = 10 factors with a separate 'Missingness/flat flags' group (TWI missing + soil-texture missing + is_flat; Aspect = sin/cos only). 13 groups = the three flags as three separate players.


**10 factors (flags inside parents; Fig. 14a v2)**

| Group | RF | 1D-CNN |
|---|---|---|
| Annual rainfall | 16.75 ± 0.48 | 16.44 ± 0.94 |
| Aspect | 6.27 ± 0.16 | 8.83 ± 0.76 |
| Curvature | 3.04 ± 0.08 | 3.38 ± 0.38 |
| Elevation | 14.39 ± 0.25 | 11.74 ± 1.00 |
| LULC | 9.70 ± 0.25 | 6.11 ± 0.74 |
| NDVI | 28.62 ± 0.60 | 27.73 ± 1.02 |
| Slope | 3.18 ± 0.11 | 2.83 ± 0.62 |
| Soil texture | 10.08 ± 0.11 | 8.02 ± 0.94 |
| TRI | 4.16 ± 0.18 | 7.69 ± 1.28 |
| TWI | 3.81 ± 0.13 | 7.23 ± 0.90 |

**11 groups (flags separate)**

| Group | RF | 1D-CNN |
|---|---|---|
| Annual rainfall | 16.49 ± 0.47 | 16.13 ± 0.94 |
| Aspect | 5.73 ± 0.14 | 8.16 ± 0.70 |
| Curvature | 3.00 ± 0.08 | 3.31 ± 0.37 |
| Elevation | 14.16 ± 0.24 | 11.52 ± 0.95 |
| LULC | 9.55 ± 0.25 | 5.99 ± 0.72 |
| Missingness/flat flags | 2.49 ± 0.14 | 6.84 ± 0.82 |
| NDVI | 28.17 ± 0.61 | 27.23 ± 1.00 |
| Slope | 3.13 ± 0.11 | 2.78 ± 0.61 |
| Soil texture | 9.85 ± 0.11 | 7.71 ± 1.05 |
| TRI | 4.09 ± 0.18 | 7.56 ± 1.26 |
| TWI | 3.35 ± 0.13 | 2.76 ± 0.41 |

**13 groups (3 flags separate)**

| Group | RF | 1D-CNN |
|---|---|---|
| Annual rainfall | 16.35 ± 0.46 | 15.76 ± 0.92 |
| Aspect | 5.69 ± 0.14 | 7.96 ± 0.67 |
| Curvature | 2.97 ± 0.08 | 3.24 ± 0.36 |
| Elevation | 14.05 ± 0.23 | 11.25 ± 0.94 |
| Flat-cell flag | 0.87 ± 0.07 | 1.03 ± 0.16 |
| LULC | 9.47 ± 0.24 | 5.86 ± 0.71 |
| NDVI | 27.94 ± 0.61 | 26.58 ± 0.95 |
| Slope | 3.10 ± 0.11 | 2.72 ± 0.59 |
| Soil missing flag | 0.47 ± 0.05 | 1.77 ± 0.28 |
| Soil texture | 9.77 ± 0.11 | 7.53 ± 1.02 |
| TRI | 4.06 ± 0.18 | 7.38 ± 1.23 |
| TWI | 3.32 ± 0.12 | 2.69 ± 0.40 |
| TWI missing flag | 1.93 ± 0.13 | 6.23 ± 0.96 |

**RF per encoded column (flags only)**: Soil_Texture_missing 0.44 ± 0.04; Soil_Texture_was_missing 0.46 ± 0.05; TWI 3.19 ± 0.12; TWI_was_missing 1.85 ± 0.13; is_flat 0.84 ± 0.07

Flat-cell share previously inside Aspect (is_flat as its own player, 13-group): RF 0.87 ± 0.07 %, 1D-CNN 1.04 ± 0.16 %. Aspect (sin/cos) share: 10-factor RF 6.27 -> 11-group 5.73 (13-group 5.69); 1D-CNN 8.83 -> 8.16 (13-group 7.96). TWI-missing flag vs TWI value (13-group): RF 1.93 ± 0.13 vs 3.32 ± 0.12; 1D-CNN 6.23 ± 0.96 vs 2.69 ± 0.40. Soil-missing flag: RF 0.47 ± 0.05, 1D-CNN 1.77 ± 0.28. Flags group total (11-group): RF 2.49 ± 0.14, 1D-CNN 6.84 ± 0.82. Note that grouped (summed-then-abs) shares are not additive across groupings.

Figure: reports/figures/revision/rev3/fig14a_shap_importance_rev3.png (same style as v2, 11 bars incl. 'Missingness/flat flags', bars ordered by RF share).

## A7. AOA with in-fold SHAP weights (RF TreeSHAP on training points of the training folds, 10-seed mean, normalized to mean 1 per fold)
The variant with all 23 encoded predictors and in-fold weights was absent from aoa_variants.csv and was computed here (DI on the standardized fold-preprocessed encoded matrix, same threshold rule Q3+1.5 IQR). The '16 samples' = non-landslide samples in 22.55-22.80 N, 92.25-92.45 E (33 negatives) whose seed-averaged RF+SMOTE class in map_points.csv is High or Very high (9 + 7 = 16).

| Variant | n outside AOA | by fold (1..5) | landslides outside | non-landslides outside | of 33 east negatives outside | of 16 High/VH east negatives outside |
|---|---|---|---|---|---|---|
| infold_7continuous_factor_weights (adopted) | 41 | 0,0,0,4,37 | 2 | 39 | 1 | 1 |
| infold_all23_encoded_col_weights_standardised | 28 | 0,0,0,2,26 | 1 | 27 | 1 | 1 |
| infold_all23_encoded_col_weights_as_preprocessed | 45 | 0,0,0,5,40 | 2 | 43 | 1 | 1 |
| infold_all23_unweighted_standardised (no SHAP; reference) | 15 | 0,2,1,1,11 | 2 | 13 | 0 | 0 |

The adopted 7-continuous in-fold variant is reproduced (41; 0,0,0,4,37).

## A8. Per-fold within-fold AUC, terrain, primary spatial CV (mean over 10 seeds; SD over seeds in CSV)

| Model | Fold | n_pos | n_neg | pairs | AUC mean | AUC SD |
|---|---|---|---|---|---|---|
| DNN | 1 | 122 | 23 | 2806 | 0.580 | 0.051 |
| DNN | 2 | 44 | 30 | 1320 | 0.701 | 0.026 |
| DNN | 3 | 20 | 21 | 420 | 0.499 | 0.039 |
| DNN | 4 | 8 | 67 | 536 | 0.838 | 0.036 |
| DNN | 5 | 2 | 159 | 318 | 0.426 | 0.031 |
| 1D-CNN | 1 | 122 | 23 | 2806 | 0.767 | 0.023 |
| 1D-CNN | 2 | 44 | 30 | 1320 | 0.813 | 0.013 |
| 1D-CNN | 3 | 20 | 21 | 420 | 0.701 | 0.017 |
| 1D-CNN | 4 | 8 | 67 | 536 | 0.881 | 0.042 |
| 1D-CNN | 5 | 2 | 159 | 318 | 0.783 | 0.045 |
| LSTM | 1 | 122 | 23 | 2806 | 0.658 | 0.051 |
| LSTM | 2 | 44 | 30 | 1320 | 0.504 | 0.014 |
| LSTM | 3 | 20 | 21 | 420 | 0.623 | 0.024 |
| LSTM | 4 | 8 | 67 | 536 | 0.272 | 0.067 |
| LSTM | 5 | 2 | 159 | 318 | 0.728 | 0.139 |
| RandomForest | 1 | 122 | 23 | 2806 | 0.817 | 0.010 |
| RandomForest | 2 | 44 | 30 | 1320 | 0.789 | 0.005 |
| RandomForest | 3 | 20 | 21 | 420 | 0.719 | 0.016 |
| RandomForest | 4 | 8 | 67 | 536 | 0.896 | 0.017 |
| RandomForest | 5 | 2 | 159 | 318 | 0.833 | 0.007 |
| XGBoost | 1 | 122 | 23 | 2806 | 0.756 | 0.012 |
| XGBoost | 2 | 44 | 30 | 1320 | 0.789 | 0.011 |
| XGBoost | 3 | 20 | 21 | 420 | 0.741 | 0.015 |
| XGBoost | 4 | 8 | 67 | 536 | 0.780 | 0.014 |
| XGBoost | 5 | 2 | 159 | 318 | 0.822 | 0.019 |
| SVM-RBF | 1 | 122 | 23 | 2806 | 0.746 | 0.000 |
| SVM-RBF | 2 | 44 | 30 | 1320 | 0.788 | 0.000 |
| SVM-RBF | 3 | 20 | 21 | 420 | 0.676 | 0.000 |
| SVM-RBF | 4 | 8 | 67 | 536 | 0.920 | 0.000 |
| SVM-RBF | 5 | 2 | 159 | 318 | 0.638 | 0.000 |
| LogisticRegression | 1 | 122 | 23 | 2806 | 0.757 | 0.000 |
| LogisticRegression | 2 | 44 | 30 | 1320 | 0.714 | 0.000 |
| LogisticRegression | 3 | 20 | 21 | 420 | 0.579 | 0.000 |
| LogisticRegression | 4 | 8 | 67 | 536 | 0.744 | 0.000 |
| LogisticRegression | 5 | 2 | 159 | 318 | 0.396 | 0.000 |

CSV: A8_per_fold_AUC_terrain.csv

## A9. Block-to-fold merge (splits.assign_blocks_to_folds, balance='both', the default)
spatial_block_labels runs KMeans (10 clusters, random_state=42) on raw Latitude/Longitude. assign_blocks_to_folds then visits the blocks in descending order of positive count (ties: larger block first, then block id) and gives each to the fold minimizing (fold positives + fold size x size_weight), size_weight = (total_pos/5)/n_total, ties broken by fewer points then lower fold index; whole blocks stay intact. Result (fold numbers 1-5 = fold index + 1):

| Fold | Blocks (greedy order) | n | positives |
|---|---|---|---|
| 1 | 5 (n=145, pos=122) | 145 | 122 |
| 2 | 1 (n=74, pos=44) | 74 | 44 |
| 3 | 0 (n=41, pos=20) | 41 | 20 |
| 4 | 8 (n=42, pos=8), 3 (n=33, pos=0) | 75 | 8 |
| 5 | 6 (n=36, pos=1), 7 (n=30, pos=1), 2 (n=41, pos=0), 9 (n=37, pos=0), 4 (n=17, pos=0) | 161 | 2 |

Block centroids in A9_blocks_to_folds.csv.

## A10. Figures (300 dpi, reports/figures/revision/rev3/)

Manuscript media mapping (from word/media of 'gmurad-Thesis_Manuscript-24 November.docx'; pixel sizes of the existing image vs the new PNG):

| Figure | media file | old px | new PNG | new px |
|---|---|---|---|---|
| Fig. 8 (Voronoi) | word/media/image8.png | 2492x1686 | fig08_voronoi_quantified_rev3.png | 2492x1686 |
| Fig. 9 (Gi*) | word/media/image9.png | 1615x1825 | fig09_gistar_rev3.png | 1597x1825 |
| Fig. 10 (Moran) | word/media/image10.png | 1738x1448 | fig10_moran_rev3.png | 1738x1448 |
| Fig. 11 (k-distance + DBSCAN) | word/media/image11.png | 2974x1401 | fig11_dbscan_kdistance_rev3.png | 3280x1552 |
| Fig. 14a (SHAP importance) | word/media/image15.png | 1838x1401 | fig14a_shap_importance_rev3.png | 1865x1401 |