"""Spatially blocked train/test partitioning.

This module carries the central correction of the project.

A random forest given nothing but ``Latitude`` and ``Longitude`` scores
0.969 +/- 0.011 AUC on this dataset under stratified 5-fold random CV. That
number is not a result; it is a measurement of how much the split leaks.
Landslide points sit a median of 0.132 km from their nearest same-class
neighbour, while non-landslide points sit 2.240 km apart. The positives are a
tight inventory cluster, the negatives are dispersed background samples, and the
two classes occupy different bounding boxes. Under a random split, a positive's
near-duplicate neighbours fall on both sides of the partition, and a model can
score well by memorising location.

Dropping the coordinate columns does not fix this. Elevation, slope, curvature,
rainfall, and NDVI are all smooth functions of position, so a model can
reconstruct "roughly where" from the covariates alone. The leak has to be closed
at the split, by ensuring that training and test points are far enough apart
that spatial autocorrelation between them has decayed.

Every splitter here shuffles explicitly. The raw file is sorted by class -- rows
0-197 are all positives -- so any splitter that relies on a default is one
refactor away from producing a training set with a single label. ``data.clean``
also shuffles, but that is a second line of defence and not a substitute.

The two schemes
---------------
``random_split`` (**R1**)
    Stratified random 60/20/20 train/val/test. The leakage baseline.

``spatial_block_cv`` (**R2**)
    Ten KMeans sub-blocks on Lat/Long, merged into five folds that each contain
    both classes. The estimate that generalises.

What to report
--------------
Both numbers, side by side. The random-split metric is the leakage baseline and
roughly reproduces the existing manuscript; the spatially blocked metric is the
estimate that generalises. The gap between them is the finding.

And for R2, two numbers again: the within-fold AUC and the **pooled** AUC over
concatenated out-of-fold predictions. ``results/session3_pooled_auc_check.md``
measures a ~0.14 AUC gap between them under this scheme, present in every
feature configuration tried. A susceptibility map is the pooled quantity, so the
within-fold figure alone overstates what the map delivers.

Still scaffolds
---------------
``spatial_block_split``, ``spatial_buffer_split``,
``estimate_autocorrelation_range`` and ``summarise_split`` remain unimplemented.
The first three are alternative or supporting strategies, not the two schemes
this pipeline runs on; the last is the leak-distance diagnostic that belongs in
the supplementary material.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from .data import COORD_COLS, TARGET

RANDOM_STATE: int = 42

#: Sub-block count for R2. Deliberately larger than the fold count: blocking
#: straight into 5 KMeans blocks leaves 3 of them with no landslides at all
#: (``results/session2_floor_test.md`` section 3.1), which makes their AUC
#: undefined. Cutting finer and then merging is what buys five usable folds.
N_SUB_BLOCKS: int = 10
N_SPLITS: int = 5

#: ``n_init`` for the blocking KMeans. Pinned, not left to the sklearn default:
#: with ``n_init="auto"`` (1 initialisation) the partition of this study area is
#: a different and worse local optimum, and the Session 2 block geometry does
#: not reproduce. See ``spatial_block_labels``.
KMEANS_N_INIT: int = 10


# --------------------------------------------------------------------------
# R1 -- stratified random split (the leakage baseline)
# --------------------------------------------------------------------------


def random_split(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """**R1.** Stratified random 60/20/20 train/val/test -- the leakage baseline.

    Retained deliberately, not as a fallback. It reproduces what the existing
    notebooks do and gives the corrected pipeline something to be compared
    against, so that the manuscript can quantify the optimism rather than merely
    assert it.

    Results from this splitter must be labelled as such wherever they appear. On
    the terrain-only feature set it returns ~0.95 AUC against ~0.77 under R2;
    the difference is not an improvement, it is the leak.

    Implemented as two nested ``train_test_split`` calls -- 60/40, then the 40
    halved -- which is the recipe ``tests/test_feature_leakage.py`` already uses,
    so on the cleaned 496-row frame this reproduces its 297 / 99 / 100 split
    exactly.

    Parameters
    ----------
    val_size, test_size:
        Fractions of the **whole** frame, not of the remainder. They must sum to
        less than 1.
    stratify:
        Preserve the class ratio in all three partitions. Shuffling is always
        on, regardless of this flag, because the input file is class-sorted.

    Returns
    -------
    ``(train, val, test)``, each a copy carrying the original index, so a caller
    can trace a row back to the cleaned frame.
    """
    if not 0 < val_size + test_size < 1:
        raise ValueError(
            f"val_size + test_size must lie in (0, 1); got {val_size} + {test_size}"
        )
    if TARGET not in df.columns:
        raise ValueError(f"frame has no {TARGET!r} column to stratify on")

    rest_size = val_size + test_size
    strat = df[TARGET] if stratify else None
    train, rest = train_test_split(
        df, test_size=rest_size, random_state=random_state, shuffle=True, stratify=strat
    )

    # test_size is re-expressed as a share of ``rest`` so that both fractions
    # keep referring to the whole frame.
    strat_rest = rest[TARGET] if stratify else None
    val, test = train_test_split(
        rest,
        test_size=test_size / rest_size,
        random_state=random_state,
        shuffle=True,
        stratify=strat_rest,
    )
    return train, val, test


# --------------------------------------------------------------------------
# R2 -- balanced spatial blocking
# --------------------------------------------------------------------------


def spatial_block_labels(
    df: pd.DataFrame,
    n_sub_blocks: int = N_SUB_BLOCKS,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Partition the study area into KMeans sub-blocks on raw Lat/Long.

    Coordinates are passed unscaled. Over this study area's ~0.2 degree extent
    the two axes are close enough in span that standardising them would only
    stretch the geometry without making it more meaningful.

    On the cleaned 496-row frame with ``n_sub_blocks=5`` this reproduces the
    Session 2 block geometry exactly -- sizes ``{21, 68, 80, 131, 196}`` with
    positive counts ``{0, 0, 0, 44, 152}`` -- which is the check that this
    function is the same blocking the floor test measured.

    Returns
    -------
    Integer block label per row, positionally aligned with ``df``.
    """
    missing = [c for c in COORD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"spatial blocking needs {missing}; the frame has none")

    coords = df[COORD_COLS].to_numpy(dtype=float)
    if np.isnan(coords).any():
        raise ValueError("coordinates contain NaN; blocking geometry must be complete")

    km = KMeans(n_clusters=n_sub_blocks, random_state=random_state, n_init=KMEANS_N_INIT)
    return km.fit_predict(coords)


def assign_blocks_to_folds(
    block_sizes: dict[int, int],
    block_positives: dict[int, int],
    n_splits: int = N_SPLITS,
    balance: str = "both",
) -> dict[int, int]:
    """Merge whole sub-blocks into folds, balancing the positives across them.

    Greedy: take the blocks in descending order of positive count (ties broken
    by descending size) and give each one to the fold holding fewest positives
    so far. Whole blocks stay intact, so no fold boundary ever cuts through a
    cluster of near-duplicate inventory points.

    Parameters
    ----------
    balance:
        ``"positives"``
            Balance on positive count alone -- the rule the Session 3 brief
            specified, and the rule behind sections 1-5 of
            ``results/session3_pooled_auc_check.md``. **No longer the
            default.** Section 6 of that file superseded it in favour of
            ``"both"``, and ``results/session3_final.md`` confirmed the
            replacement across all three configurations. Kept as an option so
            those sections stay reproducible; pass it explicitly to do so.

            It has a known and fairly ugly failure mode on this dataset, and it
            is documented rather than silently patched: six of the ten
            sub-blocks contain no landslides, every one of them looks equally
            attractive to a rule that only counts positives, and so all six land
            in the same fold. Fold 5 ends up with 194 of 496 points and 2
            positives. Its AUC is close to meaningless on its own and it still
            contributes 39% of the pooled prediction vector.
        ``"both"``
            Add a penalty on fold size, so the empty blocks spread out
            instead of piling up. **The default, and the reported scheme.**
            Section 6 of ``results/session3_pooled_auc_check.md`` makes this
            the authoritative spatial-CV floor for the terrain-only
            configuration -- within-fold 0.8041 +/- 0.0580, pooled 0.7567, all
            five folds carrying both classes -- and
            ``results/session3_final.md`` extends it to configurations (b) and
            (c).

            On this data it moves exactly one empty sub-block out of the
            overloaded fold, which is worth 0.12 pooled AUC. The
            pooled-vs-within gap falls from ~0.138 to ~0.048, in all three
            configurations; the penalty does not disappear, it stops being
            dominated by one 194-point fold.

    Returns
    -------
    ``{block label: fold index}``.
    """
    if balance not in {"positives", "both"}:
        raise ValueError(f"unknown balance {balance!r}; expected 'positives' or 'both'")

    n_total = sum(block_sizes.values())
    total_pos = sum(block_positives.values())
    # Converts a fold's size into the units of a positive count, so the two
    # terms in the "both" objective can be added without a free weight.
    size_weight = (total_pos / n_splits) / n_total if n_total else 0.0

    order = sorted(
        block_sizes, key=lambda b: (-block_positives[b], -block_sizes[b], b)
    )

    fold_of_block: dict[int, int] = {}
    fold_pos = [0] * n_splits
    fold_n = [0] * n_splits

    for block in order:
        if balance == "positives":
            key = lambda i: (fold_pos[i], fold_n[i], i)  # noqa: E731
        else:
            key = lambda i: (fold_pos[i] + fold_n[i] * size_weight, fold_n[i], i)  # noqa: E731
        k = min(range(n_splits), key=key)
        fold_of_block[block] = k
        fold_pos[k] += block_positives[block]
        fold_n[k] += block_sizes[block]

    return fold_of_block


def spatial_fold_labels(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    n_sub_blocks: int = N_SUB_BLOCKS,
    random_state: int = RANDOM_STATE,
    balance: str = "both",
    require_both_classes: bool = True,
) -> np.ndarray:
    """Fold index per row for **R2**: blocks cut, then merged into folds.

    The step ``spatial_block_cv`` is built on, exposed separately so that fold
    membership can be inspected, cached, or written out alongside predictions
    without re-running the CV.

    Parameters
    ----------
    balance:
        Sub-block merge rule, passed through to ``assign_blocks_to_folds``.
        Defaults to the size-aware ``"both"``; pass ``"positives"`` only to
        reproduce sections 1-5 of ``results/session3_pooled_auc_check.md``.
    require_both_classes:
        Raise if any fold comes out single-class. ``True`` by default: a
        single-class fold has an undefined AUC, and Session 2 reported a
        five-fold mean that was in fact a mean over two folds. Refusing here is
        what stops that happening again silently.
    """
    if n_sub_blocks < n_splits:
        raise ValueError(
            f"n_sub_blocks ({n_sub_blocks}) must be at least n_splits ({n_splits}); "
            "merging is what buys class-balanced folds"
        )

    blocks = spatial_block_labels(df, n_sub_blocks=n_sub_blocks, random_state=random_state)
    y = df[TARGET].to_numpy(dtype=int)

    labels = np.unique(blocks)
    block_sizes = {int(b): int((blocks == b).sum()) for b in labels}
    block_positives = {int(b): int(y[blocks == b].sum()) for b in labels}

    fold_of_block = assign_blocks_to_folds(
        block_sizes, block_positives, n_splits=n_splits, balance=balance
    )
    folds = np.array([fold_of_block[int(b)] for b in blocks], dtype=int)

    if require_both_classes:
        bad = [
            k
            for k in range(n_splits)
            if len(np.unique(y[folds == k])) < 2 or not (folds == k).any()
        ]
        if bad:
            raise ValueError(
                f"folds {[k + 1 for k in bad]} are single-class, so their AUC is "
                "undefined. Raise n_sub_blocks, change balance=, or pass "
                "require_both_classes=False and report the per-fold composition "
                "instead of averaging over undefined folds."
            )
    return folds


def spatial_block_cv(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    n_sub_blocks: int = N_SUB_BLOCKS,
    random_state: int = RANDOM_STATE,
    balance: str = "both",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """**R2.** Yield spatially blocked cross-validation folds.

    Groups points into ``n_sub_blocks`` geographic blocks, merges whole blocks
    into ``n_splits`` folds balanced on class content, and yields each fold in
    turn as the test set. Yields positional index arrays, matching the
    scikit-learn splitter protocol, so this can be passed straight to
    ``cross_validate`` or ``GridSearchCV``.

    Blocking directly into five folds does not work on this data: three of the
    five blocks contain no landslides, and a mean over the two that remain is
    not a five-fold result. Cutting into ten and merging keeps whole blocks
    intact -- the property that closes the leak -- while giving every fold both
    classes.

    Class balance per fold still needs watching and reporting, and
    ``fold_composition`` exists to make that cheap. Because positives are
    geographically clustered, blocked folds cannot be stratified the way random
    ones can: even after merging, positive rates run from 0.010 to 0.841. A fold
    containing two positives will produce an unstable AUC, and that instability
    is a property of the data, not a bug to be smoothed away -- it should appear
    in the reported spread.

    Blocks are merged size-aware by default (``balance="both"``), which is the
    rule ``results/session3_final.md`` reports as authoritative. The earlier
    positives-only rule remains available as ``balance="positives"`` and
    reproduces sections 1-5 of ``results/session3_pooled_auc_check.md``.

    Examples
    --------
    >>> from sklearn.model_selection import cross_val_score
    >>> cross_val_score(pipe, df, df[TARGET], cv=list(spatial_block_cv(df)),
    ...                 scoring="roc_auc")  # doctest: +SKIP
    """
    folds = spatial_fold_labels(
        df,
        n_splits=n_splits,
        n_sub_blocks=n_sub_blocks,
        random_state=random_state,
        balance=balance,
    )
    all_idx = np.arange(len(df))
    for k in range(n_splits):
        test_idx = all_idx[folds == k]
        yield all_idx[folds != k], test_idx


def fold_composition(
    df: pd.DataFrame,
    folds: np.ndarray,
    blocks: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-fold row counts, class balance, and geographic extent.

    Session 2 closed with the requirement that ``spatial_block_cv`` either
    balance its folds or report their composition. It does both, and this is the
    reporting half. Print it beside any blocked metric: a fold's AUC cannot be
    read without knowing how many positives it was computed on.

    Parameters
    ----------
    folds:
        Fold index per row, as returned by ``spatial_fold_labels``.
    blocks:
        Optional sub-block labels, to record which blocks were merged into each
        fold.

    Returns
    -------
    One row per fold, ordered by fold index, with fold numbers 1-based to match
    how the results files report them.
    """
    y = df[TARGET].to_numpy(dtype=int)
    rows = []
    for k in sorted(set(folds.tolist())):
        m = folds == k
        row = {
            "fold": int(k) + 1,
            "n_test": int(m.sum()),
            "n_train": int((~m).sum()),
            "test_pos": int(y[m].sum()),
            "test_pos_rate": round(float(y[m].mean()), 4),
            "single_class": bool(len(np.unique(y[m])) < 2),
        }
        if blocks is not None:
            row["blocks"] = ",".join(str(b) for b in sorted(set(blocks[m].tolist())))
        for col, short in zip(COORD_COLS, ("lat", "lon")):
            if col in df.columns:
                v = df[col].to_numpy()[m]
                row[f"{short}_min"] = round(float(np.min(v)), 5)
                row[f"{short}_max"] = round(float(np.max(v)), 5)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Not yet implemented -- alternative strategies and supporting diagnostics
# --------------------------------------------------------------------------


def spatial_block_split(
    df: pd.DataFrame,
    n_blocks: int = 5,
    block_size_km: float | None = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition by contiguous geographic blocks rather than by point.

    A single-shot blocked holdout, as opposed to the cross-validation in
    ``spatial_block_cv``. Not needed by the current pipeline, which reports R2
    cross-validated, and left unimplemented for that reason.

    Choosing the block size is a judgement call with a real cost on both sides.
    Too small and clustered positives straddle a boundary, leaving the leak
    open; too large and, with 496 points across roughly 0.2 degrees of latitude,
    folds become small and imbalanced enough that the variance in the metric
    swamps the signal. The defensible procedure is to set the block size from
    the empirical range of spatial autocorrelation in the residuals -- see
    ``estimate_autocorrelation_range`` -- and to report the sensitivity of the
    headline metric to that choice.

    Parameters
    ----------
    n_blocks:
        Number of blocks per axis, used when ``block_size_km`` is ``None``.
    block_size_km:
        Explicit block edge length. Takes precedence over ``n_blocks``.
    """
    raise NotImplementedError


def spatial_buffer_split(
    df: pd.DataFrame,
    buffer_km: float = 1.0,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by point, then drop training points within ``buffer_km`` of a test point.

    An alternative to blocking, sometimes called buffered leave-one-out. It
    preserves a random split's class balance while excising the specific
    neighbours that carry the leak.

    The cost is sample size, and here that cost is asymmetric. Given a median
    within-class nearest-neighbour distance of 0.132 km for positives, even a
    modest buffer will delete a large share of the positive training data while
    barely touching the negatives -- the buffer effectively re-weights the
    classes. Worth running as a cross-check against blocking; not a good choice
    as the primary strategy on a dataset this small.

    Returns
    -------
    ``(train, test)``, where ``train`` has had buffered points removed. Callers
    should log how many rows were dropped and their class breakdown.
    """
    raise NotImplementedError


def estimate_autocorrelation_range(
    df: pd.DataFrame,
    value_col: str,
    max_lag_km: float = 20.0,
) -> float:
    """Estimate the distance at which spatial autocorrelation becomes negligible.

    Fits an empirical semivariogram to ``value_col`` and returns the range -- the
    lag beyond which the semivariance flattens. This is the principled way to set
    ``block_size_km``: blocks should be at least this wide, or training and test
    points remain correlated across the boundary and the split still leaks.

    Until this exists, R2's block count (10) is a pragmatic choice -- the
    smallest cut that yields five folds with both classes -- and not a
    justified one. That gap should be stated wherever R2 is reported.

    Should be run on each continuous covariate and on the model residuals. The
    largest range across them sets the binding constraint.

    Returns
    -------
    Estimated range in kilometres.
    """
    raise NotImplementedError


def summarise_split(
    train: pd.DataFrame, test: pd.DataFrame, name: str = ""
) -> pd.DataFrame:
    """Report the diagnostics that decide whether a split is trustworthy.

    Returns per-partition row counts, class balance, geographic extent, and --
    the number that matters -- the minimum and median nearest-neighbour distance
    from each test point to its closest training point. If that minimum is near
    zero, the split leaks no matter what it is called.

    ``fold_composition`` covers the counts and extents for R2 today; what is
    still missing is the train-to-test distance, which is the part that actually
    demonstrates separation rather than asserting it.

    This table belongs in the supplementary material. It is the evidence that
    the corrected metrics were computed on genuinely held-out ground, and it is
    exactly what a reviewer will ask for.
    """
    raise NotImplementedError
