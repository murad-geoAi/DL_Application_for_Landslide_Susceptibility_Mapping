# Session 3 — Final: All Three Configurations Under the Authoritative Fold Rule

**Run:** 2026-09-20 · pandas 2.2.2 / numpy 1.26.4 / scikit-learn 1.8.0
**Supersedes:** sections 1-5 of `session3_pooled_auc_check.md`. Section 6 of
that file re-ran configuration (a) under the size-aware merge rule
`balance="both"`; this file re-runs (b) and (c) the same way and closes the
comparison.

`splits.spatial_block_cv` now **defaults** to `balance="both"`, so every number
here comes from a bare `spatial_block_cv(df)` call. `models.py` and `run.py`
were untouched as of this session; both were subsequently modified in Session 4
(`session4_models.md`) to run the full seven-model comparison and to fit and
save the final model selection, so that statement no longer holds as of this
writing — see §4.

---

## 1. The table

R2 balanced spatial CV, `balance="both"`, `RandomForestClassifier(500, seed 42)`,
preprocessing fit inside each training fold. Pooled AUC concatenates the 496
out-of-fold predictions into one vector and computes a single ROC. Standard
deviations are population (`ddof=0`) over the 5 fold AUCs.

| config | n feat | within-fold AUC | pooled AUC (authoritative) | gap | pooled, thin fold excluded (n=335) | range |
|---|---:|---:|---:|---:|---:|---|
| **(a) terrain-only** | 21 | 0.8041 +/- 0.0580 | **0.7567** | -0.047 | 0.8288 | 0.757 - 0.829 |
| **(b) + Drainage_Density** | 22 | 0.8210 +/- 0.0656 | **0.7712** | -0.050 | 0.8396 | 0.771 - 0.840 |
| **(c) + Cluster** | 31 | 0.7756 +/- 0.0572 | **0.7292** | -0.046 | 0.8136 | 0.729 - 0.814 |

The "thin fold" is fold 5: 161 points, **2** positives, positive rate 0.012. It
is the same 161 points in all three configurations, because the partition is
built from coordinates and labels only and never sees the feature set. The
right-hand range is therefore the same sensitivity check for each row -- lower
bound counts fold 5, upper bound drops it -- and it is a statement about the
partition, not about the covariates.

**No fold is single-class**, in any configuration. `spatial_fold_labels` was
called with `require_both_classes=True` and did not raise; `fold_composition`
reports `single_class = False` on all five rows. Composition: 145 / 74 / 41 / 75
/ 161 test points at positive rates 0.841 / 0.595 / 0.488 / 0.107 / 0.012.

### 1.1 Per-fold AUC

| fold | blocks | n_train | n_test | test pos | (a) | (b) | (c) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 351 | 145 | 122 | 0.8124 | 0.8223 | 0.7992 |
| 2 | 1 | 422 | 74 | 44 | 0.7803 | 0.7886 | 0.7723 |
| 3 | 0 | 455 | 41 | 20 | 0.7095 | 0.7167 | 0.6667 |
| 4 | 3,8 | 421 | 75 | 8 | 0.8834 | 0.9030 | 0.8190 |
| 5 | 2,4,6,7,9 | 335 | 161 | 2 | 0.8349 | 0.8742 | 0.8208 |

### 1.2 Against the superseded rule

The same three configurations under `balance="positives"`, from sections 2-3 of
`session3_pooled_auc_check.md`:

| config | within (`positives`) | pooled (`positives`) | gap | within (`both`) | pooled (`both`) | gap |
|---|---:|---:|---:|---:|---:|---:|
| (a) terrain-only | 0.7736 | 0.6366 | -0.137 | 0.8041 | 0.7567 | **-0.047** |
| (b) + Drainage_Density | 0.7942 | 0.6560 | -0.138 | 0.8210 | 0.7712 | **-0.050** |
| (c) + Cluster | 0.7390 | 0.5997 | -0.139 | 0.7756 | 0.7292 | **-0.046** |

The R1 random-stratified rows are unchanged and not repeated: a stratified
random split never calls the merge rule, so pooled 0.9475 (a), 0.9486 (b) and
0.9627 (c) still stand exactly as reported.

---

## 2. Do the two claims still hold?

### 2.1 "The gap tracks the split, not the feature set" -- yes, and more tightly

The three gaps are **-0.047 / -0.050 / -0.046**, a spread of **0.003** across
configurations that differ by up to 10 features and by 0.042 AUC in pooled
level. Under the superseded rule the same claim rested on -0.137 / -0.138 /
-0.139, a spread of 0.002.

The demonstration section 6.3 flagged as missing is now supplied, and it
survives the change of rule. Two things sharpen it:

- Adding `Drainage_Density` lifts within-fold by 0.017 and pooled by 0.015, and
  moves the gap by 0.002. Adding the cluster feature lowers within-fold by 0.028
  and pooled by 0.027, and moves the gap by 0.001. In both directions the level
  moves and the gap does not.
- The invariance is not an artefact of the pathological fold. With fold 5 dropped
  entirely the three pooled figures still sit 0.011 and 0.015 apart in the same
  order, so the feature effects are stable while the penalty stays common.

What did change is the **size** of the penalty, from ~0.138 to ~0.048 in every
configuration. The claim was always about invariance across feature sets, not
about the number 0.14, and in that form it holds. Any text quoting "~0.14" as
the pooled penalty should now quote **~0.05**.

### 2.2 "Cluster underperforms terrain-only under spatial CV" -- yes, unanimously

Configuration (c) is below configuration (a) on every measure computed:

| measure | (a) terrain-only | (c) + Cluster | delta |
|---|---:|---:|---:|
| within-fold mean | 0.8041 | 0.7756 | -0.029 |
| pooled | 0.7567 | 0.7292 | -0.027 |
| pooled, thin fold excluded | 0.8288 | 0.8136 | -0.015 |
| per fold (5 of 5) | see 1.1 | lower in every fold | -0.013 to -0.064 |

The cluster feature loses in **5 folds out of 5**, and it does so while raising
the random-split pooled AUC to 0.9627, the highest of the three. That divergence
-- best under a random split, worst under a spatial one -- is the signature
`session2_feature_check.md` §3 identified as location memorisation, and the
corrected fold rule does not soften it. Configuration (c) stays an ablation row.

In the other direction, for completeness: `Drainage_Density` wins in 5 folds out
of 5 and is worth +0.015 pooled, close to the ~0.02 quoted under the old rule.

---

## 3. What stands against the new numbers

1. **The pooled penalty is real but modest.** ~0.05 AUC, not ~0.14. The larger
   figure was substantially an artefact of merging on positive count alone,
   which dumped all six empty sub-blocks into one 194-point fold. Sections 3.2
   and 4 of `session3_pooled_auc_check.md` keep their direction; their magnitude
   should be read from this file.
2. **Pooled still sits far below the random split**, 0.757 against 0.948 for
   configuration (a). The spatial-vs-random gap of ~0.19 is roughly four times
   the pooled-vs-within gap of 0.05, and it remains the finding that matters.
3. **The within-fold mean remains the fragile number.** Fold 5's AUC rests on 2
   positives and moved 0.15 on 33 extra training rows when the merge rule
   changed. The pooled figure, one ROC over all 496 points, does not inherit
   that. Lead with pooled.
4. **The floor is a range, not a point.** 0.757 to 0.829 for terrain-only,
   depending on whether a near-empty fold is counted. Quote 0.757 as the floor
   and the range as the honest interval.
5. **Calibration is still not in the pipeline.** Nothing here addresses the
   fold-to-fold probability-scale mismatch; it only stops one fold dominating
   it. The fold-internal calibration step of §5 item 3 remains open.

---

## 4. Code status

`splits.spatial_block_cv`, `splits.spatial_fold_labels` and
`splits.assign_blocks_to_folds` now default to `balance="both"`. The parameter is
threaded through all three, so flipping only the outermost would have left
`spatial_fold_labels(df)` -- the function used to inspect and cache fold
membership -- reporting a different partition than `spatial_block_cv(df)` runs.
All three were flipped for that reason, and the docstrings were updated to match.

`balance="positives"` still works and reproduces sections 1-5 of
`session3_pooled_auc_check.md` exactly. Nothing else in the repository calls
these functions and `tests/` does not import `splits`, so no test asserted the
old default and none needed changing. `pytest tests/ -q`: **5 passed**.

**Update (Session 4):** `models.py` and `run.py` are no longer untouched.
Session 4 added the multi-model (DNN, 1D-CNN, LSTM, RandomForest, XGBoost,
SVM-RBF, LogisticRegression) comparison across R0/R1/R2 and the final-model
selection logic in `run.py` (dataset selection per model, `joblib.dump` of the
fitted pipelines to `models/`, and `results/final_model_selection.csv`). See
`session4_models.md` for the resulting comparison table and the final
selection (1D-CNN).

A note on the sensitivity check's sample size: the brief asked for the
fold-5-excluded pooled AUC at n=463. Under `balance="both"` fold 5 holds 161
points, not 33, so dropping it leaves **n=335**. That is the figure reported
above, and it matches section 6.3 of the earlier file.
