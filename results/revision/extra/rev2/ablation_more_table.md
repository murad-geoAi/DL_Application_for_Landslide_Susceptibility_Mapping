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
