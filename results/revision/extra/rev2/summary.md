# rev2 sensitivity experiments (spatial 5-fold block CV, terrain 23 inputs, 10 seeds)

SD = across-seed SD (ddof=1), as in revision_extra. Seed 42 RF repro: within 0.8068 / pooled 0.7595, identical to main pipeline.

## Part 1: ablation_more (all 7 models, ablation columns + bounding box)

| model | ablation | within-fold AUC | pooled AUC | baseline (T4 spatial within) | n seeds |
|---|---|---|---|---|---|
| DNN | none | 0.609 +/- 0.012 | 0.598 +/- 0.035 | 0.609 | 10 |
| 1D-CNN | none | 0.789 +/- 0.015 | 0.850 +/- 0.013 | 0.789 +/- 0.015 | 10 |
| LSTM | none | 0.557 +/- 0.025 | 0.462 +/- 0.065 | 0.557 | 10 |
| RandomForest | none | 0.811 +/- 0.007 | 0.761 +/- 0.005 | 0.811 +/- 0.007 | 10 |
| XGBoost | none | 0.778 +/- 0.008 | 0.707 +/- 0.009 | 0.778 | 10 |
| SVM-RBF | none | 0.754 +/- 0.000 | 0.838 +/- 0.003 | 0.754 | 10 |
| LogisticRegression | none | 0.638 +/- 0.000 | 0.768 +/- 0.000 | 0.638 | 10 |
| 1D-CNN | no_ndvi_lulc | 0.623 +/- 0.014 | 0.669 +/- 0.020 | 0.789 +/- 0.015 | 10 |
| RandomForest | no_ndvi_lulc | 0.568 +/- 0.009 | 0.561 +/- 0.005 | 0.811 +/- 0.007 | 10 |
| XGBoost | no_ndvi_lulc | 0.535 +/- 0.007 | 0.583 +/- 0.005 | 0.778 | 10 |
| LogisticRegression | no_ndvi_lulc | 0.586 +/- 0.000 | 0.654 +/- 0.000 | 0.638 | 10 |
| 1D-CNN | no_rainfall | 0.720 +/- 0.023 | 0.809 +/- 0.021 | 0.789 +/- 0.015 | 10 |
| RandomForest | no_rainfall | 0.773 +/- 0.004 | 0.770 +/- 0.004 | 0.811 +/- 0.007 | 10 |
| XGBoost | no_rainfall | 0.747 +/- 0.005 | 0.767 +/- 0.003 | 0.778 | 10 |
| LogisticRegression | no_rainfall | 0.586 +/- 0.000 | 0.763 +/- 0.000 | 0.638 | 10 |
| 1D-CNN | no_flags | 0.774 +/- 0.020 | 0.846 +/- 0.018 | 0.789 +/- 0.015 | 10 |
| RandomForest | no_flags | 0.805 +/- 0.004 | 0.760 +/- 0.006 | 0.811 +/- 0.007 | 10 |
| XGBoost | no_flags | 0.767 +/- 0.006 | 0.703 +/- 0.007 | 0.778 | 10 |
| LogisticRegression | no_flags | 0.628 +/- 0.000 | 0.763 +/- 0.000 | 0.638 | 10 |
| SVM-RBF | no_ndvi_lulc | 0.632 +/- 0.000 | 0.631 +/- 0.006 | 0.754 | 10 |
| DNN | no_ndvi_lulc | 0.578 +/- 0.017 | 0.543 +/- 0.025 | 0.609 | 10 |
| LSTM | no_ndvi_lulc | 0.525 +/- 0.042 | 0.497 +/- 0.094 | 0.557 | 10 |
| SVM-RBF | no_rainfall | 0.713 +/- 0.000 | 0.810 +/- 0.005 | 0.754 | 10 |
| DNN | no_rainfall | 0.580 +/- 0.020 | 0.582 +/- 0.040 | 0.609 | 10 |
| LSTM | no_rainfall | 0.512 +/- 0.040 | 0.309 +/- 0.024 | 0.557 | 10 |
| SVM-RBF | no_flags | 0.761 +/- 0.000 | 0.842 +/- 0.003 | 0.754 | 10 |
| DNN | no_flags | 0.610 +/- 0.009 | 0.604 +/- 0.023 | 0.609 | 10 |
| LSTM | no_flags | 0.558 +/- 0.034 | 0.443 +/- 0.089 | 0.557 | 10 |

Bounding-box negatives (existing caches, spatial CV, 7 models)

| model | design | within-fold AUC | pooled AUC | baseline (T4 spatial within) | n seeds |
|---|---|---|---|---|---|
| DNN | bbox_spatial | 0.573 +/- 0.017 | 0.551 +/- 0.038 | 0.609 | 10 |
| 1D-CNN | bbox_spatial | 0.675 +/- 0.025 | 0.762 +/- 0.021 | 0.789 +/- 0.015 | 10 |
| LSTM | bbox_spatial | 0.535 +/- 0.047 | 0.427 +/- 0.036 | 0.557 | 10 |
| RandomForest | bbox_spatial | 0.637 +/- 0.008 | 0.673 +/- 0.005 | 0.811 +/- 0.007 | 10 |
| XGBoost | bbox_spatial | 0.584 +/- 0.008 | 0.596 +/- 0.009 | 0.778 | 10 |
| SVM-RBF | bbox_spatial | 0.686 +/- 0.000 | 0.764 +/- 0.004 | 0.754 | 10 |
| LogisticRegression | bbox_spatial | 0.655 +/- 0.000 | 0.709 +/- 0.000 | 0.638 | 10 |

Runtime ~370-470 s per seed (10 parallel procs, 1 thread each).

## Part 2: spatial_es

| model | design | within-fold AUC | pooled AUC | baseline (T4 spatial within) | n seeds |
|---|---|---|---|---|---|
| DNN | random_ES (main) | 0.609 +/- 0.012 | 0.598 +/- 0.035 | 0.609 | 10 |
| 1D-CNN | random_ES (main) | 0.789 +/- 0.015 | 0.850 +/- 0.013 | 0.789 +/- 0.015 | 10 |
| LSTM | random_ES (main) | 0.557 +/- 0.025 | 0.462 +/- 0.065 | 0.557 | 10 |
| DNN | spatial_ES | 0.572 +/- 0.052 | 0.545 +/- 0.112 | 0.609 | 10 |
| 1D-CNN | spatial_ES | 0.745 +/- 0.020 | 0.782 +/- 0.061 | 0.789 +/- 0.015 | 10 |
| LSTM | spatial_ES | 0.468 +/- 0.021 | 0.229 +/- 0.062 | 0.557 | 10 |

Validation set per fold (averaged over seeds; min/max across seeds)

| model   |   fold |   n_train |   n_val_mean |   n_val_min |   n_val_max |   pos_mean |   pos_min |   neg_mean |   neg_min |   frac_mean |   frac_max |   epochs_mean |
|:--------|-------:|----------:|-------------:|------------:|------------:|-----------:|----------:|-----------:|----------:|------------:|-----------:|--------------:|
| 1D-CNN  |      1 |       351 |        109.2 |          71 |         190 |       29.9 |         9 |       79.3 |        30 |        0.31 |       0.54 |          27.7 |
| 1D-CNN  |      2 |       422 |        107.5 |          71 |         265 |       37.5 |         8 |       70   |        23 |        0.25 |       0.63 |          35.2 |
| 1D-CNN  |      3 |       455 |        125.5 |          72 |         298 |       28.1 |         8 |       97.4 |        63 |        0.28 |       0.65 |          27.4 |
| 1D-CNN  |      4 |       421 |        122.1 |          74 |         265 |       77.5 |        44 |       44.6 |        23 |        0.29 |       0.63 |          37.1 |
| 1D-CNN  |      5 |       335 |         83.5 |          74 |         116 |       29.6 |         8 |       53.9 |        30 |        0.25 |       0.35 |          29.6 |
| DNN     |      1 |       351 |        109.2 |          71 |         190 |       29.9 |         9 |       79.3 |        30 |        0.31 |       0.54 |          32.3 |
| DNN     |      2 |       422 |        107.5 |          71 |         265 |       37.5 |         8 |       70   |        23 |        0.25 |       0.63 |          26.1 |
| DNN     |      3 |       455 |        125.5 |          72 |         298 |       28.1 |         8 |       97.4 |        63 |        0.28 |       0.65 |          27.8 |
| DNN     |      4 |       421 |        122.1 |          74 |         265 |       77.5 |        44 |       44.6 |        23 |        0.29 |       0.63 |          27   |
| DNN     |      5 |       335 |         83.5 |          74 |         116 |       29.6 |         8 |       53.9 |        30 |        0.25 |       0.35 |          31.1 |
| LSTM    |      1 |       351 |        109.2 |          71 |         190 |       29.9 |         9 |       79.3 |        30 |        0.31 |       0.54 |           7   |
| LSTM    |      2 |       422 |        107.5 |          71 |         265 |       37.5 |         8 |       70   |        23 |        0.25 |       0.63 |           9.8 |
| LSTM    |      3 |       455 |        125.5 |          72 |         298 |       28.1 |         8 |       97.4 |        63 |        0.28 |       0.65 |          13.8 |
| LSTM    |      4 |       421 |        122.1 |          74 |         265 |       77.5 |        44 |       44.6 |        23 |        0.29 |       0.63 |           8.8 |
| LSTM    |      5 |       335 |         83.5 |          74 |         116 |       29.6 |         8 |       53.9 |        30 |        0.25 |       0.35 |           7.3 |

Runtime ~230-300 s per seed (parallel).

## Part 3: lstm_reg

| model | design | within-fold AUC | pooled AUC | baseline (T4 spatial within) | n seeds |
|---|---|---|---|---|---|
| LSTM (unregularised) | spatial CV, random_ES (main) | 0.557 +/- 0.025 | 0.462 +/- 0.065 |  | 10 |
| LSTM (unregularised) | random CV, random_ES | 0.725 +/- 0.019 | 0.717 +/- 0.019 |  | 10 |
| LSTM (L2+dropout) | spatial CV, random_ES | 0.525 +/- 0.033 | 0.441 +/- 0.069 |  | 10 |
| LSTM (L2+dropout) | spatial CV, spatial_ES | 0.520 +/- 0.056 | 0.357 +/- 0.073 |  | 10 |
| LSTM (L2+dropout) | random CV, random_ES | 0.715 +/- 0.008 | 0.703 +/- 0.011 |  | 10 |
| LSTM (L2+dropout) | random CV, spatial_ES | 0.678 +/- 0.034 | 0.674 +/- 0.021 |  | 10 |

Runtime ~3150-3230 s per seed (parallel, CPU contended).

## Part 4: CatBoost / TabPFN

| model | design | within-fold AUC | pooled AUC | baseline (T4 spatial within) | n seeds |
|---|---|---|---|---|---|
| CatBoost | spatial CV | 0.813 +/- 0.009 | 0.774 +/- 0.010 |  | 10 |
| CatBoost | random CV | 0.954 +/- 0.001 | 0.952 +/- 0.001 |  | 10 |

TabPFN: skipped (user declined installation).

Runtime 10-40 s per seed.
