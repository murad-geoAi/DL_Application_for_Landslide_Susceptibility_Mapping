| model | design | within-fold AUC | pooled AUC | baseline (T4 spatial within) | n seeds |
|---|---|---|---|---|---|
| LSTM (unregularised) | spatial CV, random_ES (main) | 0.557 +/- 0.025 | 0.462 +/- 0.065 |  | 10 |
| LSTM (unregularised) | random CV, random_ES | 0.725 +/- 0.019 | 0.717 +/- 0.019 |  | 10 |
| LSTM (L2+dropout) | spatial CV, random_ES | 0.525 +/- 0.033 | 0.441 +/- 0.069 |  | 10 |
| LSTM (L2+dropout) | spatial CV, spatial_ES | 0.520 +/- 0.056 | 0.357 +/- 0.073 |  | 10 |
| LSTM (L2+dropout) | random CV, random_ES | 0.715 +/- 0.008 | 0.703 +/- 0.011 |  | 10 |
| LSTM (L2+dropout) | random CV, spatial_ES | 0.678 +/- 0.034 | 0.674 +/- 0.021 |  | 10 |
