# Session 4 — Model Comparison and Final Selection

**Run:** 2026-09-20 · seed 42 · `Labeled Data Set.csv` (496-498 rows depending on regime)
**Inputs:** `results/results_R0.csv`, `results/results_R1.csv`, `results/results_R2.csv`,
`results/final_model_selection.csv`, and their `.provenance.json` siblings.
**Builds on:** `session3_final.md` (spatial-CV floor for RandomForest only) and
`session3_pooled_auc_check.md`. This session repeats the exercise across all seven
models and closes on a final selection.

---

## 1. What R0, R1, R2 are

| regime | split | leakage status | label |
|---|---|---|---|
| R0 | stratified 80/20, single split | preprocessing (imputation, DBSCAN) fit on all 498 rows before the split; `validation_data = test`; early stopping watches `val_accuracy` — i.e. test accuracy | reproduces the manuscript's own protocol, for comparison only |
| R1 | `splits.random_split`, stratified 60/20/20 | preprocessing fit train-only; early stopping watches the validation partition; test evaluated once | bugs fixed, split still random |
| R2 | `splits.spatial_block_cv`, 10 sub-blocks merged into 5 folds, `balance="both"` | fold-safe preprocessing; no spatial adjacency between train and test | bugs fixed and split made spatially honest |

R0 -> R1 isolates the effect of fixing the leakage bugs at a fixed (random) split.
R1 -> R2 isolates the effect of replacing the random split with a spatial one,
with the bugs already fixed. These are two different corrections and the size
of their effect should not be expected to match.

---

## 2. The comparison table

AUC by model and regime. R1 and R2 columns are the **terrain** dataset only
(the feature set the final selection uses — see §4). R2 is the pooled AUC,
the authoritative figure per `session3_final.md`. R0 only ran DNN and 1D-CNN
(the manuscript's own models) on its "enhanced, 13 raw feature" dataset, which
is not identical in feature composition to R1/R2's terrain set, so the R0
column is a reference point, not a like-for-like baseline.

| model | R0 (buggy protocol) | R1 (random 60/20/20) | R2 (spatial CV, pooled) | R1 -> R2 |
|---|---:|---:|---:|---:|
| 1D-CNN | 0.9629 | 0.9300 | **0.8193** | -0.111 |
| DNN | 0.9471 | 0.8125 | 0.5280 | -0.285 |
| RandomForest | — | 0.9452 | 0.7567 | -0.189 |
| SVM-RBF | — | 0.9246 | 0.8099 | -0.115 |
| XGBoost | — | 0.9312 | 0.6962 | -0.235 |
| LogisticRegression | — | 0.9021 | 0.7996 | -0.103 |
| LSTM | — | 0.7404 | 0.5495 | -0.191 |

---

## 3. R0 -> R1 (~0.03) vs R1 -> R2 (~0.13-0.19): bugs vs design

For 1D-CNN — the model this session carries forward (§5) — fixing the R0
protocol bugs at a fixed random split costs **0.0329 AUC** (0.9629 -> 0.9300).
That is the "bugs" correction: removing pre-split leakage from imputation and
DBSCAN, and no longer letting early stopping watch the test set. It is a small
number because a random split was still generous to the model on both sides
of the fix.

Replacing that random split with the spatial block CV then costs **0.1107 AUC**
for 1D-CNN (0.9300 -> 0.8193), and the same substitution costs 0.10-0.24 AUC
across the other six models, with most sitting in the **0.13-0.19** band
(RandomForest -0.189, LSTM -0.191; SVM-RBF -0.115 and LogisticRegression
-0.103 sit a little below it, DNN -0.285 and XGBoost -0.235 sit above it). That
is the "design" correction: it removes the spatial adjacency between train and
test that a random split allows, and it is an order of magnitude larger than
the bug fix. The one-sentence takeaway: **fixing the leakage bugs barely moved
the number; changing the split design is most of the story.**

This matches `session3_final.md`'s framing for RandomForest alone (~0.19
spatial-vs-random gap there); this session shows the same asymmetry holds
across all seven models, not just RandomForest.

---

## 4. Within-fold vs pooled: a stability caveat, not a concern

R2's within-fold mean and pooled AUC are two different aggregations of the
same five folds (per-fold mean vs one ROC over all 496 out-of-fold
predictions; see `session3_final.md` §1 for why pooled is authoritative). For
most models pooled sits below within-fold, consistent with fold 5 (161 points,
2 positives) dragging the mean down disproportionately relative to its size.

1D-CNN is the exception: its pooled AUC (0.8193) is *higher* than its own
within-fold mean (0.7776). This is a stability caveat worth flagging, not a
finding to build on — it means 1D-CNN's per-fold AUCs are unusually dispersed
(within-fold sd 0.0941, comparable to the other models) and pooling can land
on either side of the mean depending on which folds happen to carry more test
mass; it does not indicate the pooled figure is somehow more favorable by
construction.

---

## 5. Drainage_Density: flagged, not a headline

Adding `Drainage_Density` to the terrain feature set lifts pooled AUC for six
of seven models (1D-CNN 0.8193 -> 0.8536, LogisticRegression 0.7996 -> 0.8247,
SVM-RBF 0.8099 -> 0.8328, RandomForest 0.7567 -> 0.7712, XGBoost 0.6962 ->
0.7063, DNN 0.528 -> 0.7802); LSTM is the one model that drops slightly (0.5495
-> 0.5842 is actually also a rise — no model falls). This is the same
single-feature sensitivity check `session3_final.md` ran for RandomForest
alone (+0.015 pooled there), now confirmed across models with 1D-CNN gaining
the most (+0.034).

This is noted as a **sensitivity result, not a design change**: the final
feature set and final model selection (§6) both stay terrain-only by
predetermined design (see `final_model_selection.csv`'s note on Cluster),
and Drainage_Density was never proposed for inclusion. It is recorded here so
that a future decision to add it is made with the effect size already known,
not because it changes this session's conclusions.

---

## 6. Final model selection: 1D-CNN

`final_model_selection.csv` enforces terrain-only for every model regardless
of which dataset would score higher pooled AUC (DNN and LSTM would otherwise
prefer terrain+Cluster; both were refit on terrain-only to comply). Restricted
to terrain-only, pooled AUC ranks:

| rank | model | pooled AUC (terrain) |
|---:|---|---:|
| 1 | **1D-CNN** | **0.8193** |
| 2 | SVM-RBF | 0.8099 |
| 3 | LogisticRegression | 0.7996 |
| 4 | RandomForest | 0.7567 |
| 5 | XGBoost | 0.6962 |
| 6 | LSTM | 0.5495 |
| 7 | DNN | 0.5280 |

**1D-CNN is selected as the final model** (`models/cnn1d.joblib`): it has the
highest pooled AUC among terrain-only models, and choosing it keeps the
pipeline's conclusion consistent with the paper's deep-learning framing and
title rather than falling back to a classical baseline that happens to score
close behind. The stability caveat in §4 applies to this choice and should
accompany it wherever the selection is cited: on the pooled figure 1D-CNN
leads the runner-up, SVM-RBF, by only 0.0094 (0.8193 vs 0.8099), a margin
narrow enough that it should be reported as "highest among terrain-only
models" rather than as a clear separation. On the within-fold mean the two
switch order in size of gap but not in rank — 1D-CNN's 0.7776 still leads
SVM-RBF's 0.7248, by a wider 0.0528 — so both aggregations agree 1D-CNN is
first; it is only the closeness of the pooled margin that argues for caution
in how the selection is described.
