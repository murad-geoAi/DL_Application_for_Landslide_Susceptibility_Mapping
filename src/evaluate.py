"""Metrics, the within-fold / pooled convention, and the leakage diagnostics.

Accuracy is not reported as a headline number anywhere in this module. With a
300/196 class split, predicting "no landslide" everywhere scores 60.5%, so
accuracy mostly measures the base rate. The manuscript's accuracy figures should
be replaced with AUC, precision/recall at a stated threshold, and F1.

Two numbers per cross-validated cell
------------------------------------
Every cross-validated result here is reported twice, following the convention
``results/session3_final.md`` settled on:

**within-fold**
    Compute the metric separately on each test fold, then take the mean and
    the population standard deviation (``ddof=0``) across folds. This is what
    most papers report, and under spatial blocking it is the fragile number:
    Session 3's fold 5 holds 161 points and **two** positives, so its AUC rests
    on two observations and moved 0.15 when 33 training rows were added.

**pooled**
    Concatenate the out-of-fold predictions for all 496 points into one vector
    and compute a single metric against the full label vector. One ROC over the
    whole study area, with no fold able to dominate it by being small.

The two differ because AUC is not a mean of per-fold AUCs. A per-fold AUC only
ranks points against others *in the same fold*; the pooled figure also requires
the probabilities to be comparable **between** folds, so the gap between them is
a measure of fold-to-fold calibration drift. Session 3 measured that gap at
about -0.05 for the random forest and found it near-constant across feature
sets. Pooled is the authoritative number. Quote it first.

The module also owns the leakage diagnostics. A susceptibility model's headline
metric is only meaningful alongside evidence that the evaluation was honest, and
that evidence is quantitative: the coordinates-only probe score, the test-to-
train nearest-neighbour distances, and the gap between random-split and blocked
metrics.

Calibration deserves more attention than it usually receives in this literature.
The deliverable is a susceptibility *map*, which a planner will threshold into
zones. If the predicted probabilities are not calibrated, the zone boundaries do
not mean what the legend says they mean, and a well-discriminating model can
still produce a misleading map. ``session3_final.md`` section 3 item 5 left
fold-internal calibration open; it is still open, and the functions at the
bottom of this file are where it will land.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

#: Decision threshold for the hard-label metrics. Fixed at 0.5 throughout
#: Session 4 by the brief, which is the right call for a comparison table: a
#: per-model tuned threshold would make the precision/recall columns
#: incomparable across rows. It is a convention, not a decision -- see
#: ``choose_threshold`` for selecting one on data, and note that the operating
#: point for an actual susceptibility map is a policy question, not a
#: statistical one.
DEFAULT_THRESHOLD: float = 0.5

#: The five columns every results CSV carries, in order. ``save_results``
#: enforces this, so a table cannot silently grow a column between regimes.
METRIC_COLUMNS: list[str] = ["AUC", "accuracy", "precision", "recall", "F1"]

#: Column layout of ``results_R0.csv`` / ``_R1.csv`` / ``_R2.csv``.
RESULT_COLUMNS: list[str] = ["regime", "model", "dataset", "fold", *METRIC_COLUMNS]

#: Values the ``fold`` column takes for rows that are not a single fold.
#: ``"NA"`` marks a single-split regime (R0, R1); the three aggregate labels
#: appear only under cross-validation.
FOLD_NA: str = "NA"
FOLD_WITHIN_MEAN: str = "within_mean"
FOLD_WITHIN_SD: str = "within_sd"
FOLD_POOLED: str = "pooled"


# --------------------------------------------------------------------------
# Core metrics
# --------------------------------------------------------------------------


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    extended: bool = False,
) -> dict[str, float]:
    """Compute the metric set from probabilities, not hard labels.

    Returns ``AUC``, ``accuracy``, ``precision``, ``recall``, ``F1`` -- the five
    the Session 4 tables carry -- and, with ``extended=True``, average precision
    and the Brier score as well.

    Average precision is worth having because it is more informative than ROC
    AUC when the positive class is the one that matters and is in the minority:
    ROC AUC is computed against a large negative pool and can look strong while
    the high-susceptibility zone is mostly false positives. The Brier score is
    the cheapest available calibration summary, and calibration is the open item
    from Session 3.

    Parameters
    ----------
    threshold:
        Cut-point for the hard-label metrics. The 0.5 default is a convention,
        not a decision; ``choose_threshold`` selects it on data.

    Notes
    -----
    ``AUC`` is ``nan`` when the fold is single-class -- ROC AUC is undefined
    without both labels present. Returning ``nan`` rather than raising keeps a
    pathological fold visible in the table instead of aborting the run.
    ``spatial_fold_labels(require_both_classes=True)`` means this should not
    arise under R2, and if it does the partition, not the metric, is the problem.

    ``precision`` is ``0.0`` with ``zero_division=0`` when a model predicts no
    positives at all, which happens under spatial CV on the folds with a 0.012
    positive rate. That is a true statement about the model at this threshold,
    not a missing value.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    both_classes = len(np.unique(y_true)) > 1
    out = {
        "AUC": float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if extended:
        out["average_precision"] = (
            float(average_precision_score(y_true, y_prob)) if both_classes else float("nan")
        )
        out["brier"] = float(brier_score_loss(y_true, y_prob))
    return out


def choose_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    criterion: str = "youden",
    fn_fp_ratio: float = 10.0,
) -> float:
    """Select an operating threshold on validation data, never on test data.

    Parameters
    ----------
    criterion:
        ``"youden"``
            Maximises sensitivity + specificity - 1. The usual default.
        ``"f1"``
            Maximises F1, which favours the positive class more heavily.
        ``"cost"``
            Minimises ``fn_fp_ratio * FN + FP``. For landslide susceptibility
            this is the most defensible framing, since a missed landslide and a
            needlessly restricted parcel are not comparable harms -- but the
            ratio is a policy input and has to come from outside the data. The
            10:1 default is a placeholder that should be replaced by a number
            someone is willing to defend.

    Must be called on a validation fold carved from the training data. Choosing
    a threshold on the test set is a leak, and a surprisingly common one.
    Session 4 does not call this: every reported metric sits at 0.5 so that
    rows stay comparable. It is here for the mapping stage, where a single
    operating point has to be chosen and justified.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    if criterion == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        return float(thr[int(np.argmax(tpr - fpr))])

    candidates = np.unique(y_prob)
    if criterion == "f1":
        scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in candidates]
        return float(candidates[int(np.argmax(scores))])
    if criterion == "cost":
        costs = []
        for t in candidates:
            pred = (y_prob >= t).astype(int)
            fn = int(((y_true == 1) & (pred == 0)).sum())
            fp = int(((y_true == 0) & (pred == 1)).sum())
            costs.append(fn_fp_ratio * fn + fp)
        return float(candidates[int(np.argmin(costs))])

    raise ValueError(
        f"unknown criterion {criterion!r}; expected 'youden', 'f1', or 'cost'"
    )


# --------------------------------------------------------------------------
# The within-fold / pooled convention
# --------------------------------------------------------------------------


def within_fold_summary(
    per_fold: pd.DataFrame, metrics: Iterable[str] = METRIC_COLUMNS
) -> tuple[dict[str, float], dict[str, float]]:
    """Mean and population SD of each metric across folds.

    ``ddof=0``, matching ``session3_final.md``: the five folds are the whole
    population of folds this partition produces, not a sample from a larger set
    of them, so the population SD is the honest denominator. It also makes the
    numbers reproduce that file exactly, which matters more than the 0.006
    difference ``ddof=1`` would introduce.

    ``nan`` fold values propagate through ``np.nanmean``/``np.nanstd`` rather
    than poisoning the summary, so a single-class fold costs one fold's worth of
    evidence instead of the whole row.

    Returns
    -------
    ``(mean, sd)``, each a dict keyed by metric name.
    """
    mean, sd = {}, {}
    for m in metrics:
        vals = per_fold[m].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        mean[m] = float(np.mean(finite)) if finite.size else float("nan")
        sd[m] = float(np.std(finite, ddof=0)) if finite.size else float("nan")
    return mean, sd


def pooled_metrics(
    y_true: np.ndarray,
    oof_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Metrics over the concatenated out-of-fold predictions -- one ROC, all points.

    Every one of the 496 points was predicted by a model that never saw it, so
    the pooled vector is a legitimate held-out prediction for the whole study
    area. Unlike the within-fold mean it requires the probabilities to be
    comparable across folds, which is the extra demand that opens the gap
    between the two figures.
    """
    return classification_metrics(y_true, oof_prob, threshold=threshold)


def cross_validate_model(
    model_fn,
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    target: str = "Landslide",
    threshold: float = DEFAULT_THRESHOLD,
    fit_fn=None,
) -> dict[str, Any]:
    """Run cross-validation, returning per-fold results *and* the pooled vector.

    Per-fold output is deliberate. Under spatial blocking the folds are not
    exchangeable -- positives are geographically clustered, so some blocks are
    nearly all positive and others nearly all negative -- and a mean AUC across
    them hides a spread that may span 0.2 or more. That spread is a genuine
    finding about how far the model transfers across the study area, and
    collapsing it to a single number discards the most interesting result. The
    pooled vector is returned alongside for the same reason in reverse: the mean
    is fragile where one fold is tiny, and pooling is not.

    Parameters
    ----------
    model_fn:
        Zero-argument callable returning an **unfitted** pipeline. A callable
        rather than a pipeline instance, because each fold must start from a
        fresh model -- reusing one would carry a Keras model's weights from the
        previous fold into the next.
    folds:
        List of ``(train_idx, test_idx)`` positional index arrays, as
        ``splits.spatial_block_cv`` yields.
    fit_fn:
        Optional ``(pipe, X_train, y_train) -> fitted pipe``. Defaults to
        ``models.fit_pipeline`` with no validation set, which lets a Keras model
        carve its own stratified validation slice out of the training fold.

    Returns
    -------
    dict with keys ``per_fold`` (DataFrame, one row per fold), ``oof`` (the
    out-of-fold probability vector), ``pooled`` (dict of pooled metrics), and
    ``fitted`` (the per-fold fitted pipelines, for anything downstream that
    wants them).
    """
    if fit_fn is None:
        from .models import fit_pipeline as fit_fn  # local import: avoids a cycle

    y = df[target].to_numpy(dtype=int)
    oof = np.full(len(df), np.nan, dtype=float)
    rows: list[dict[str, Any]] = []
    fitted = []

    for k, (train_idx, test_idx) in enumerate(folds, start=1):
        pipe = model_fn()
        pipe = fit_fn(pipe, df.iloc[train_idx], y[train_idx])
        prob = pipe.predict_proba(df.iloc[test_idx])[:, 1]
        oof[test_idx] = prob

        row = {
            "fold": k,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "test_pos": int(y[test_idx].sum()),
            **classification_metrics(y[test_idx], prob, threshold=threshold),
        }
        rows.append(row)
        fitted.append(pipe)

    per_fold = pd.DataFrame(rows)
    if np.isnan(oof).any():  # a partition that does not cover every row
        raise ValueError(
            f"{int(np.isnan(oof).sum())} rows were never in a test fold; "
            "the folds do not partition the frame"
        )

    return {
        "per_fold": per_fold,
        "oof": oof,
        "pooled": pooled_metrics(y, oof, threshold=threshold),
        "fitted": fitted,
    }


def cv_result_rows(
    cv: dict[str, Any],
    regime: str,
    model: str,
    dataset: str,
) -> pd.DataFrame:
    """Turn a ``cross_validate_model`` result into ``RESULT_COLUMNS`` rows.

    Emits ``n_folds + 3`` rows: one per fold, then ``within_mean``,
    ``within_sd``, and ``pooled`` in the ``fold`` column. Keeping all three in
    the same nine-column table means a reader never has to reconcile two files,
    and a spreadsheet filter on ``fold == "pooled"`` reproduces the headline
    table exactly.
    """
    per_fold = cv["per_fold"]
    mean, sd = within_fold_summary(per_fold)

    rows = []
    for _, r in per_fold.iterrows():
        rows.append(
            {
                "regime": regime,
                "model": model,
                "dataset": dataset,
                "fold": int(r["fold"]),
                **{m: r[m] for m in METRIC_COLUMNS},
            }
        )
    for label, values in ((FOLD_WITHIN_MEAN, mean), (FOLD_WITHIN_SD, sd), (FOLD_POOLED, cv["pooled"])):
        rows.append(
            {
                "regime": regime,
                "model": model,
                "dataset": dataset,
                "fold": label,
                **{m: values[m] for m in METRIC_COLUMNS},
            }
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def single_split_row(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    regime: str,
    model: str,
    dataset: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """One ``RESULT_COLUMNS`` row for a regime with no cross-validation.

    ``fold`` is ``"NA"``: R0 and R1 each evaluate a single held-out test set, so
    there is no fold index to report and no within-fold spread to summarise.
    That absence is itself worth seeing in the table -- a single-split number
    carries no information about how far it would move on a different split,
    which is most of what the R2 rows are for.
    """
    return pd.DataFrame(
        [
            {
                "regime": regime,
                "model": model,
                "dataset": dataset,
                "fold": FOLD_NA,
                **classification_metrics(y_true, y_prob, threshold=threshold),
            }
        ],
        columns=RESULT_COLUMNS,
    )


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def _provenance(label: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Commit, seed, versions, platform, timestamp."""
    import sklearn

    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = "not-a-git-repository"

    versions = {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}
    for name in ("tensorflow", "xgboost"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = "unavailable"

    prov = {
        "label": label,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
    }
    if extra:
        prov.update(extra)
    return prov


def save_results(
    results: pd.DataFrame,
    path: str | Path,
    label: str = "",
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Write a results table to ``results/`` with its provenance beside it.

    Records the git commit, the random seed, the splitting strategy, the package
    versions, and the timestamp in a ``.provenance.json`` sidecar. A thesis
    correction is partly an argument that the new numbers are reproducible, and
    that argument is only as good as the metadata stored with them. The sidecar
    rather than a comment header because the CSV has to stay loadable by
    ``pd.read_csv`` with no arguments.

    The frame is reindexed to ``RESULT_COLUMNS`` -- extra columns are dropped
    and a missing one raises -- so every regime's file has the identical layout.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    missing = [c for c in RESULT_COLUMNS if c not in results.columns]
    if missing:
        raise ValueError(f"results frame is missing required columns: {missing}")

    out = results[RESULT_COLUMNS].copy()
    for m in METRIC_COLUMNS:
        out[m] = out[m].astype(float).round(4)
    out.to_csv(path, index=False)

    sidecar = path.with_suffix(".provenance.json")
    sidecar.write_text(
        json.dumps(_provenance(label or path.stem, provenance), indent=2), encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------
# Diagnostics deferred past Session 4
# --------------------------------------------------------------------------
#
# The brief for this session stops at the three regimes and their tables: no
# SHAP, no figures, no calibration work. The functions below are the designed
# surface for the sessions that follow, and are left unimplemented rather than
# half-implemented so that nothing downstream can call a stub and get a number
# that looks real.


def compare_split_strategies(
    model: Pipeline,
    df: pd.DataFrame,
    strategies: dict[str, Any],
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate one model under several splitting strategies -- the key table.

    Superseded in practice by ``run.py``, which runs the R0/R1/R2 grid directly
    and writes one CSV per regime. Kept as the signature for the buffered-split
    and leave-one-block-out comparisons that ``splits.py`` still has stubbed.
    """
    raise NotImplementedError("deferred past Session 4; see run.run_R1 / run.run_R2")


def leakage_report(df: pd.DataFrame, splitter: Any, random_state: int = 42) -> pd.DataFrame:
    """Quantify how much a splitting strategy leaks, before fitting any model.

    Reports, per fold: the coordinates-only probe AUC
    (``models.build_coordinates_only_probe``), the minimum and median distance
    from each test point to its nearest training point, the share of test points
    within 500 m of a training point, and the class balance on both sides.

    Worth running first. If the probe scores well above chance, the split is
    unsound and nothing computed on it is worth interpreting, so this check
    should gate the rest of the analysis rather than accompany it. Sessions 1-3
    have already established the numbers by hand (0.969 random, near chance
    blocked); this function is the mechanised version.
    """
    raise NotImplementedError("deferred past Session 4")


def calibration_report(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Assess whether predicted probabilities mean what they claim.

    Returns a binned reliability table -- predicted probability against observed
    frequency -- plus the Brier score and expected calibration error.

    This matters more here than the usual boilerplate reason. A susceptibility
    map is thresholded into "high", "moderate", and "low" zones, and those
    labels are read as probabilities by whoever uses the map. A model with
    excellent AUC but a systematic upward bias will place the "high" boundary in
    the wrong place, and the map will overstate risk everywhere while still
    ranking slopes correctly.

    It is also the direct explanation for the within-fold / pooled gap this
    module reports, which is ``session3_final.md`` section 3 item 5's open
    question. The gap is fold-to-fold probability-scale mismatch; a reliability
    curve per fold would show it rather than merely summarise it.
    """
    raise NotImplementedError("deferred past Session 4; open item from session3_final.md §3.5")


def permutation_importance_report(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_repeats: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Held-out permutation importance, with names recovered from the pipeline.

    Preferred over impurity-based importance, which is biased towards
    high-cardinality features and would systematically understate the one-hot
    ``LULC`` and ``Soil_Texture`` levels relative to the continuous terrain
    variables.

    Two cautions specific to this dataset:

    - Correlated predictors share importance and can both appear unimportant.
      ``SPI`` is already dropped as a duplicate of ``TWI``, so that particular
      trap is closed, but ``TRI``/``Slope`` remain closely related.
    - If ``LULC = 7`` dominates, that is the finding from audit section 6.5
      resurfacing, and it should be pursued rather than reported at face value.

    Explicitly out of scope for Session 4, whose brief says no SHAP and no
    figures yet.
    """
    raise NotImplementedError("deferred past Session 4")


def ablation_study(
    model: Pipeline,
    df: pd.DataFrame,
    splitter: Any,
    feature_groups: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Re-fit with feature groups removed, to test what the model actually uses.

    Partly answered already: the ``Cluster`` and ``Drainage_Density`` ablations
    are run directly by ``run.run_R2`` across all seven models, and their
    results are rows in ``results_R2.csv``. What remains for this function is
    the ``LULC`` question -- whether the 92.3% landslide rate on code 7 is doing
    the work -- and the ``TRI``/``Slope`` pair.
    """
    raise NotImplementedError("deferred past Session 4; Cluster and Drainage_Density are in run.run_R2")
