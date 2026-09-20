"""Fold-safe feature construction: imputation, encoding, and scaling.

Everything in this module *learns* something from data -- an imputation
statistic, a category vocabulary, a set of DBSCAN core points, a mean and
standard deviation -- and therefore everything in this module must be fit on a
training fold alone and only applied to the corresponding validation or test
fold.

This is the single most consequential correction to the existing pipeline.
``notebooks/landslide_susceptibility_mapping.ipynb`` fits a ``KNNImputer``
keyed on latitude and longitude across the entire dataset before splitting.
With 40% of ``SPI``/``TWI`` missing and the imputer keyed on position, held-out
values are being reconstructed from their training-set neighbours, and the
reported metrics inherit that leak.

The public surface here is deliberately a scikit-learn ``Pipeline`` /
``ColumnTransformer`` rather than a set of frame-in, frame-out helpers, because
the pipeline object makes fold discipline structural instead of a rule someone
has to remember.

Encoding decisions from the audit
---------------------------------
- ``LULC`` and ``Soil_Texture`` are non-contiguous integer class codes and are
  one-hot encoded. Treating them as ordinal asserts an ordering and a spacing
  that the codes do not carry.
- ``Aspect`` is circular, so 0 and 359 degrees are neighbours. It is sin/cos
  encoded, with flat cells handled explicitly.
- ``Latitude`` and ``Longitude`` are *never* features. They are available to the
  DBSCAN cluster encoder as geometry and to ``splits.py`` as blocking keys, and
  they leave the matrix before the model sees it.

A warning about the cluster feature
-----------------------------------
``build_cluster_encoder`` derives a feature *from position*, which is precisely
the quantity the spatial blocking is designed to withhold. Assigning it without
refitting on held-out points, as implemented here, closes the procedural leak --
a validation point cannot influence the cluster structure it is scored against,
and ``tests/test_feature_leakage.py`` proves it.

It does not close the conceptual leak, and measurement says that one is severe.
With ``eps = 0.01`` degrees (~1.1 km), DBSCAN is separating the classes almost
perfectly, because it was always going to: positives sit a median 0.132 km from
their nearest same-class neighbour and negatives 2.240 km (audit section 6.3),
so a 1.1 km radius with ``min_samples = 5`` finds the inventory clusters and
nothing else. Measured on a stratified 60/20/20 split, fitting on train only:

===================================  =============  ============
Assigned                             P(landslide)   n (held-out)
===================================  =============  ============
to any train cluster                 0.963          27
to noise (-1)                        0.181          72
===================================  =============  ============

A single bit -- in a cluster or not -- separates the classes on held-out data
at better than 0.96 precision. That is the audit's 0.969 coordinates-only AUC
arriving through a different door, laundered through a procedurally clean fit.

The encoder is therefore built and available, but it is **off by default**
(``include_cluster=False``) as of Session 3, because a run that includes it is
measuring location, not susceptibility. ``results/session2_floor_test.md`` puts
the terrain-only floor at ~0.78 blocked / ~0.94 random with this feature
excluded by construction.

``results/session3_pooled_auc_check.md`` then measured the ablation directly
and settled the question: under balanced spatial CV the cluster feature *lowers*
performance at every fold (0.7390 against 0.7736 terrain-only) while raising the
random split to 0.9632. Helping only where the split leaks is the signature of a
feature that memorises position. It belongs in the ablation row, which is why
the default moved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import CATEGORICAL_COLS, COORD_COLS, TARGET

#: DBSCAN neighbourhood radius, in degrees of latitude/longitude. At this study
#: area's latitude 0.01 degrees is roughly 1.1 km.
DBSCAN_EPS: float = 0.01
DBSCAN_MIN_SAMPLES: int = 5

#: Level used for a categorical code that is missing in the training fold. An
#: explicit level rather than a mode-fill: 10% of ``Soil_Texture`` is missing,
#: and mode-filling would assert the majority texture for one row in ten.
MISSING_LEVEL: str = "missing"


# --------------------------------------------------------------------------
# Row-local derivations (stateless -- nothing is learned, so nothing can leak)
# --------------------------------------------------------------------------


def encode_aspect(df: pd.DataFrame, drop_original: bool = True) -> pd.DataFrame:
    """Convert ``Aspect`` from degrees to its sine and cosine components.

    An aspect of 359 degrees and one of 1 degree describe nearly the same slope
    face, but as raw numbers they sit at opposite ends of the range. Any model
    that measures distance or fits a coefficient on the raw column is told that
    north-facing slopes are maximally unlike other north-facing slopes.

    Flat cells -- the 20 rows with ``Slope == 0``, marked by ``is_flat`` and
    carrying the raw ``Aspect == -1`` sentinel -- receive ``0.0`` for both
    components, which places them at the origin of the unit circle rather than
    on any compass bearing. They are the only points not on the unit circle, and
    the indicator column is what lets the model distinguish that from a genuine
    measurement. The sentinel is intercepted here rather than being imputed:
    a flat cell has no aspect to reconstruct.

    Rows where ``Aspect`` is genuinely missing (4 of them) keep NaN in both
    components and are filled later by the KNN imputer, which searches the full
    feature space rather than the two aspect columns alone.

    Returns
    -------
    A copy of ``df`` with ``Aspect_sin`` and ``Aspect_cos`` appended.
    """
    out = df.copy()

    if "is_flat" in out.columns:
        flat = out["is_flat"].astype(bool).to_numpy()
    else:  # tolerate a frame that skipped data.decode_aspect_sentinel
        flat = (out["Slope"] == 0).fillna(False).to_numpy()

    radians = np.deg2rad(out["Aspect"].to_numpy(dtype=float))
    sin, cos = np.sin(radians), np.cos(radians)

    # Order matters: the sentinel is overwritten before it can be read as a
    # bearing of -1 degrees, which sin/cos would otherwise map to 359 degrees.
    sin[flat] = 0.0
    cos[flat] = 0.0

    out["Aspect_sin"] = sin
    out["Aspect_cos"] = cos
    if drop_original:
        out = out.drop(columns=["Aspect"])
    return out


def drainage_density_variant(df: pd.DataFrame, mode: str = "raw") -> pd.DataFrame:
    """Recast ``Drainage_Density``, which is near-constant as supplied.

    469 of its 496 non-null values are exactly zero; the 27 rows that carry
    signal span 0.009 to 0.250. As a continuous predictor it is mostly a
    constant with a thin tail, and any importance it registers is driven by a
    handful of points.

    **Session 2 decision: kept, at ``mode="raw"``, flagged as low-information
    and additionally flagged as suspicious.** Keeping it preserves the
    comparison with the existing manuscript, which includes it, and a tree-based
    model largely declines to split on a column that is constant across 95% of
    rows. But "low-information" understates the problem. Those 27 non-zero rows
    are **all negatives**: P(landslide) is 0.418 where the column is zero and
    0.000 where it is not. On 27 rows that is far more likely to record where
    the background sample happened to be drawn than a protective effect of
    drainage, and it is the same failure shape as ``LULC == 7`` (audit section
    6.5) -- a rare level that separates the classes almost perfectly.

    So the column is carried, but any importance it registers must be read
    against those 27 rows rather than reported as a finding, and the
    ``"binary"`` and ``"drop"`` variants should both appear in the ablation
    table. The real fix is upstream -- recomputing drainage density over a
    kernel wide enough that it varies across the study area -- and that is not a
    modelling decision.

    Parameters
    ----------
    mode:
        ``"raw"``
            Leave as-is. The default; see above.
        ``"binary"``
            Presence flag: ``density > 0``. Honest about what the column
            actually distinguishes, and the first sensitivity check to run.
        ``"drop"``
            Remove entirely.
    """
    out = df.copy()
    if "Drainage_Density" not in out.columns:
        return out

    if mode == "raw":
        return out
    if mode == "binary":
        out["Drainage_Density"] = (out["Drainage_Density"] > 0).astype(float)
        return out
    if mode == "drop":
        return out.drop(columns=["Drainage_Density"])
    raise ValueError(f"unknown mode {mode!r}; expected 'raw', 'binary', or 'drop'")


def check_flat_cell_agreement(df: pd.DataFrame) -> dict[str, int]:
    """Confirm that ``Slope == 0`` and ``Aspect == -1`` select the same rows.

    ``data.decode_aspect_sentinel`` builds ``is_flat`` from slope, on the
    assumption that the two definitions coincide. They do in this file, on all
    20 rows, but the assumption is upstream of everything the aspect encoding
    does, so it is asserted rather than trusted.

    Returns
    -------
    Counts for each definition and their disagreement.

    Raises
    ------
    ValueError
        If any row satisfies one definition but not the other.
    """
    by_slope = (df["Slope"] == 0).fillna(False)
    by_sentinel = (df["Aspect"] == -1).fillna(False)
    disagree = int((by_slope != by_sentinel).sum())
    if disagree:
        raise ValueError(
            f"{disagree} rows disagree between Slope == 0 and Aspect == -1; the "
            "flat-cell sentinel can no longer be derived from slope alone"
        )
    return {
        "n_flat_by_slope": int(by_slope.sum()),
        "n_flat_by_sentinel": int(by_sentinel.sum()),
        "n_disagree": disagree,
    }


class FrameEngineer(BaseEstimator, TransformerMixin):
    """Applies the stateless, row-local derivations ahead of the fitted steps.

    Aspect encoding and the drainage-density recast depend on nothing but the
    row in front of them, so they are safe to apply outside a fold. They live in
    a transformer anyway, so that the whole preprocessing chain is one object
    that can be pickled, cross-validated, and inspected as a unit.

    ``fit`` records the output column order and otherwise learns nothing.
    """

    def __init__(self, drainage_mode: str = "raw"):
        self.drainage_mode = drainage_mode

    def _derive(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if "is_flat" not in out.columns:
            out["is_flat"] = (out["Slope"] == 0).fillna(False).astype(int)
        if "Aspect" in out.columns:
            out = encode_aspect(out, drop_original=True)
        out = drainage_density_variant(out, mode=self.drainage_mode)
        return out.drop(columns=[TARGET], errors="ignore")

    def fit(self, X: pd.DataFrame, y=None) -> "FrameEngineer":
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.columns_ = list(self._derive(X).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._derive(X)
        # Reindex so the downstream ColumnTransformer sees a stable schema even
        # if a val/test frame arrives with columns in a different order.
        return out.reindex(columns=self.columns_)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.columns_, dtype=object)


# --------------------------------------------------------------------------
# DBSCAN cluster assignment
# --------------------------------------------------------------------------


class DBSCANClusterAssigner(BaseEstimator, TransformerMixin):
    """Fits DBSCAN on training coordinates and *assigns* held-out points to it.

    DBSCAN has no ``predict``. The usual workaround -- refitting on the union of
    train and test coordinates -- is exactly the leak this project exists to
    close: a held-out point would help define the cluster structure it is then
    scored against, and with positives sitting a median 0.132 km apart, a single
    test point can be what turns five neighbours into a cluster.

    So the structure is frozen at ``fit`` time. ``fit`` keeps the core points
    found in the training fold; ``transform`` assigns each incoming point to the
    cluster of its nearest *training* core point if that core point is within
    ``eps``, and to ``-1`` (noise) otherwise. This is the same rule DBSCAN
    itself uses to absorb border points into a cluster, applied without letting
    the new points alter what the clusters are.

    Two consequences worth knowing:

    - ``transform`` on the training data reproduces ``labels_`` for core and
      noise points. A border point equidistant from two clusters may land in the
      other one, because DBSCAN resolves those by visit order and this rule
      resolves them by distance. The disagreement is bounded by the number of
      border points and is reported by ``assignment_agreement``.
    - A validation point more than ``eps`` from every training core point is
      noise by construction. That is the honest answer -- it says the training
      data contains no cluster there -- and it is why the ``-1`` level must be
      carried through one-hot encoding like any other level rather than dropped.

    Parameters
    ----------
    eps:
        Neighbourhood radius in degrees, matching the units of the coordinate
        columns. 0.01 degrees is roughly 1.1 km here.
    min_samples:
        Minimum neighbourhood size for a core point.
    """

    def __init__(self, eps: float = DBSCAN_EPS, min_samples: int = DBSCAN_MIN_SAMPLES):
        self.eps = eps
        self.min_samples = min_samples

    @staticmethod
    def _as_array(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X[COORD_COLS].to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def fit(self, X, y=None) -> "DBSCANClusterAssigner":
        coords = self._as_array(X)
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(coords)

        self.labels_ = db.labels_
        self.core_sample_indices_ = db.core_sample_indices_
        self.core_points_ = coords[db.core_sample_indices_]
        self.core_labels_ = db.labels_[db.core_sample_indices_]
        self.n_train_ = len(coords)
        self.n_clusters_ = int(len(set(db.labels_) - {-1}))
        self.n_noise_ = int((db.labels_ == -1).sum())
        self.n_features_in_ = coords.shape[1]

        # One neighbour is enough: the nearest core point decides the label.
        self._nn = (
            NearestNeighbors(n_neighbors=1).fit(self.core_points_)
            if len(self.core_points_)
            else None
        )
        return self

    def transform(self, X) -> pd.DataFrame:
        coords = self._as_array(X)
        index = X.index if isinstance(X, pd.DataFrame) else None

        if self._nn is None:  # no cluster survived min_samples in this fold
            labels = np.full(len(coords), -1, dtype=int)
        else:
            dist, idx = self._nn.kneighbors(coords, n_neighbors=1)
            within = dist[:, 0] <= self.eps
            labels = np.where(within, self.core_labels_[idx[:, 0]], -1).astype(int)

        # Strings, so the one-hot columns read as Cluster_3 rather than
        # Cluster_3.0 and so -1 is unambiguously a level, not a number.
        return pd.DataFrame({"Cluster": labels.astype(str)}, index=index)

    def cluster_centres(self) -> pd.DataFrame:
        """Mean coordinate and core-point count for each fitted cluster.

        The evidence that ``fit`` saw only training rows: the centres are a
        function of the training coordinates alone, and they do not move when
        ``transform`` is called on anything else.
        """
        rows = []
        for lab in sorted(set(self.core_labels_.tolist())):
            pts = self.core_points_[self.core_labels_ == lab]
            rows.append(
                {
                    "cluster": int(lab),
                    "n_core_points": len(pts),
                    "centre_lat": pts[:, 0].mean(),
                    "centre_lon": pts[:, 1].mean(),
                }
            )
        return pd.DataFrame(rows)

    def assignment_agreement(self, X) -> dict[str, float]:
        """Compare ``transform`` labels against DBSCAN's own on the fitted data.

        Expected to be 1.0 or very close. Anything materially below it means
        border points are being resolved differently by distance than DBSCAN
        resolved them by visit order, which is benign but worth knowing before
        the cluster one-hot is read as if it reproduced DBSCAN exactly.
        """
        assigned = self.transform(X)["Cluster"].astype(int).to_numpy()
        return {
            "n": float(len(assigned)),
            "agreement": float((assigned == self.labels_).mean()),
            "n_disagree": float((assigned != self.labels_).sum()),
        }

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(["Cluster"], dtype=object)


# --------------------------------------------------------------------------
# Fitted transformer blocks
# --------------------------------------------------------------------------


def build_spatial_imputer(n_neighbors: int = 5, use_coords: bool = False) -> Pipeline:
    """Build the KNN imputer + scaler path for the continuous covariates.

    Nearest-neighbour imputation is the right instinct for raster-extracted
    terrain variables: a missing slope value is well predicted by the slopes of
    similar cells. The existing notebook's error is not the method but its
    placement -- fit on everything, before the split.

    The returned pipeline is fit on training rows only. At transform time it
    finds neighbours *within the training set*, so a held-out point is filled
    from data the model was always allowed to see.

    Parameters
    ----------
    n_neighbors:
        Neighbour count. Default 5 matches the existing notebook, for
        comparability; it is not independently justified.
    use_coords:
        Whether to include ``Latitude``/``Longitude`` among the imputer's
        distance inputs. **Default ``False``**, which is a change from the
        Session 1 sketch. Keying the imputer on position would reconstruct 40%
        of the topographic index from geographic neighbours, writing location
        back into the covariates after the coordinates were removed from them --
        the leak reappearing one layer down. Feature-space neighbours carry the
        same intuition without that. ``True`` is retained only to reproduce the
        notebook's behaviour for comparison, and when set, the coordinates are
        dropped again as soon as the imputer has used them -- they inform the
        distance and never become features, so the contrast against ``False``
        isolates the imputation choice alone.

    Notes
    -----
    **The scale at which this flag operates was measured, not assumed.** With
    the imputer fed raw columns -- the arrangement the default uses -- switching
    ``use_coords`` on changes the output by exactly 0.0 across all 11,408 cells
    of the design matrix. Coordinates span ~0.2 degrees while ``Elevation``
    spans 796 m, so in an unstandardised ``nan_euclidean`` distance the
    coordinates are invisible and the neighbour ordering never moves. The
    ``use_coords=True`` branch therefore standardises before imputing, which is
    the only arrangement in which position can influence anything.

    This is worth carrying into any comparison against the original notebook.
    If that notebook also imputed on unscaled columns, then its coordinate
    keying was inert too, and the leak it introduced was the *placement* of the
    fit -- on the full dataset, before the split -- rather than the choice of
    distance key. The two are separate defects and only the first is repaired by
    setting ``use_coords=False``.

    Two consequences of the audit still apply and are not fixed by this
    configuration. Positives sit 0.132 km apart at the median while negatives
    sit 2.240 km apart, so a fixed ``n_neighbors`` reaches a wider area for a
    negative than for a positive. And because positives are clustered, their
    neighbours are overwhelmingly other positives, so imputation can encode the
    label into the covariates. Fitting inside the training fold bounds this but
    does not eliminate it; the fitted result should be checked against a median
    fill baseline before the imputed columns are trusted in importance ranking.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)

    if use_coords:
        # Scaling moves *before* imputation on this branch, and it has to. On
        # the raw columns Latitude has sd 0.36 against Elevation's 116, so the
        # coordinates contribute ~1e-5 of the nan_euclidean distance and the
        # imputed values come back bit-identical to use_coords=False -- the flag
        # would be inert. Standardising first is what lets position actually
        # compete for neighbours, which is the behaviour this flag exists to
        # reproduce. The consequence is that the True arm differs from the
        # default in two ways, not one; see the Notes below.
        #
        # Coordinates are then discarded, so the model never sees them. pandas
        # output is what lets the drop happen by name rather than by a
        # positional index that would silently rot.
        steps: list[tuple[str, object]] = [
            # pandas output on the scaler too: it is what carries the column
            # names as far as drop_coords, which drops by name.
            ("scale", StandardScaler().set_output(transform="pandas")),
            ("impute", imputer.set_output(transform="pandas")),
            (
                "drop_coords",
                FunctionTransformer(
                    _drop_coord_columns, feature_names_out=_coord_dropped_names
                ),
            ),
        ]
    else:
        steps = [("impute", imputer), ("scale", StandardScaler())]

    pipe = Pipeline(steps)
    pipe.use_coords = use_coords  # read by build_preprocessor's column selector
    return pipe


def _drop_coord_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Remove the coordinate columns after they have served the imputer.

    Only reached when ``use_coords=True``. Without it, keying the imputer on
    position would also hand ``Latitude``/``Longitude`` straight to the model as
    features, which is a worse leak than the one the flag exists to reproduce --
    and it would make the ablation uninterpretable, because the contrast against
    ``use_coords=False`` would confound two changes at once.
    """
    return X.drop(columns=[c for c in COORD_COLS if c in X.columns])


def _coord_dropped_names(transformer, input_features) -> np.ndarray:
    """Feature names surviving ``_drop_coord_columns``."""
    dropped = set(COORD_COLS)
    return np.asarray([f for f in input_features if f not in dropped], dtype=object)


def _as_category_strings(X: pd.DataFrame) -> pd.DataFrame:
    """Coerce integer class codes to string levels, missing included.

    Module-level rather than a lambda so that the fitted pipeline pickles. The
    round-trip through ``Int64`` drops the spurious ``.0`` that ``float64``
    codes would otherwise carry into the one-hot column names.
    """
    out = pd.DataFrame(index=X.index)
    for col in X.columns:
        codes = pd.to_numeric(X[col], errors="coerce").round().astype("Int64")
        out[col] = codes.astype("object").where(codes.notna(), MISSING_LEVEL).astype(str)
    return out


def build_categorical_encoder() -> Pipeline:
    """Build the impute-then-one-hot path for ``LULC`` and ``Soil_Texture``.

    Missing codes are routed to an explicit ``"missing"`` level rather than
    mode-filled. For ``Soil_Texture``, where 10% of rows are missing, mode-fill
    would silently assert the majority texture for one row in ten, and the audit
    found missing-texture rows carry a slightly higher landslide rate (0.440 vs
    0.393) -- small, but the wrong direction to paper over.

    Unseen levels map to all-zeros via ``handle_unknown="ignore"``. Two details
    the audit forces:

    - ``Soil_Texture`` code 6 appears once in the whole dataset, so it is absent
      from most training folds by construction. When it lands in a validation
      fold it encodes as all-zeros, which is the correct behaviour: the model
      has never seen that texture and should not pretend otherwise.
    - ``LULC`` code 7 carries a 92.3% landslide rate. It stays in the matrix for
      now, but ``evaluate.py`` should report performance with it ablated,
      because its status as a legitimate predictor rather than a proxy for the
      inventory itself is unresolved.
    """
    return Pipeline(
        [
            ("levels", FunctionTransformer(_as_category_strings, feature_names_out="one-to-one")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def build_cluster_encoder(
    eps: float = DBSCAN_EPS, min_samples: int = DBSCAN_MIN_SAMPLES
) -> Pipeline:
    """Build the DBSCAN-assign-then-one-hot path for the coordinate columns.

    See ``DBSCANClusterAssigner`` for the assignment rule and the module
    docstring for why the resulting feature needs an ablation reported beside
    it. ``handle_unknown="ignore"`` covers a cluster label that exists in no
    training fold; ``-1`` itself is a level like any other whenever the training
    fold contained noise points, which it always does here.
    """
    return Pipeline(
        [
            ("dbscan", DBSCANClusterAssigner(eps=eps, min_samples=min_samples)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


class _ContinuousSelector:
    """Picklable column selector for the continuous block.

    Everything numeric that is not a coordinate, a categorical code, the target,
    or the raw ``Aspect`` column (which ``FrameEngineer`` has already replaced
    with its sin/cos pair).
    """

    def __init__(self, include_missing_indicators: bool = True, include_coords: bool = False):
        self.include_missing_indicators = include_missing_indicators
        self.include_coords = include_coords

    def __call__(self, df: pd.DataFrame) -> list[str]:
        excluded = set(CATEGORICAL_COLS) | {TARGET, "Aspect"}
        if not self.include_coords:
            excluded |= set(COORD_COLS)

        cols = []
        for c in df.columns:
            if c in excluded:
                continue
            if not self.include_missing_indicators and c.endswith("_was_missing"):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
        return cols


def build_preprocessor(
    use_spatial_imputer: bool = True,
    include_missing_indicators: bool = True,
    include_cluster: bool = False,
    n_neighbors: int = 5,
    use_coords: bool = False,
    drainage_mode: str = "drop",
) -> Pipeline:
    """Assemble the full fit-on-train-only preprocessing transformer.

    Routes each column group down its appropriate path -- continuous variables
    through KNN imputation and standardisation, categoricals through an explicit
    missing level and one-hot encoding, ``Aspect`` through the circular encoder,
    coordinates through DBSCAN cluster assignment -- and drops the raw
    coordinate columns before output.

    Dropping the coordinates is necessary but not sufficient. Every terrain
    covariate is a smooth function of position, so spatial information re-enters
    the matrix through ``Elevation``, ``Slope``, and the rest regardless. The
    defence against that is the splitting strategy, not the feature set; see
    ``splits.py``. And with ``include_cluster=True`` the matrix carries an
    explicitly position-derived feature, so the ablation is not optional.

    Returns a ``Pipeline``, not a bare ``ColumnTransformer``: the stateless
    row-local derivations in ``FrameEngineer`` have to run before the column
    routing can see ``Aspect_sin``/``Aspect_cos``. The final step is the
    ``ColumnTransformer``, reachable as ``preprocessor["columns"]``.

    Parameters
    ----------
    use_spatial_imputer:
        If ``False``, falls back to median-fill, which is the baseline the KNN
        imputer must beat to justify its extra assumptions.
    include_missing_indicators:
        Whether to pass through the ``*_was_missing`` flags added in
        ``data.add_missing_indicators``.
    include_cluster:
        Whether to carry the DBSCAN cluster one-hot. **Default ``False``** as of
        Session 3: the feature is a position proxy that helps only under a
        leaking split (module docstring, and ``session3_pooled_auc_check.md``
        section 5.4). ``True`` remains available and is the leakage-baseline
        ablation -- report it beside the default, never instead of it.
    use_coords:
        Whether ``Latitude``/``Longitude`` join the KNN imputer's distance
        computation. **Default ``False``**; see ``build_spatial_imputer``. When
        ``True`` the coordinates are dropped again immediately after imputation,
        so they never reach the model either way and the flag isolates one
        change.
    drainage_mode:
        Passed to ``drainage_density_variant``. **Default ``"drop"``** as of
        Session 3. The column is near-constant and its 27 non-zero rows are all
        negatives, so the ~0.02 blocked AUC it contributes
        (``session3_pooled_auc_check.md``, configuration b) is carried by an
        artefact of where the background sample was drawn rather than by
        drainage. ``"raw"`` restores it as an explicit ablation and ``"binary"``
        is the honest recast; see ``drainage_density_variant``.

    Returns
    -------
    An unfitted ``Pipeline``. Callers must fit it on training data only -- in
    practice by nesting it inside the model pipeline built in ``models.py``, so
    that cross-validation handles the discipline automatically.
    """
    if use_spatial_imputer:
        continuous = build_spatial_imputer(n_neighbors=n_neighbors, use_coords=use_coords)
    else:
        continuous = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )

    transformers: list[tuple[str, object, object]] = [
        (
            "continuous",
            continuous,
            _ContinuousSelector(
                include_missing_indicators=include_missing_indicators,
                include_coords=use_coords,
            ),
        ),
        ("categorical", build_categorical_encoder(), list(CATEGORICAL_COLS)),
    ]
    if include_cluster:
        transformers.append(("cluster", build_cluster_encoder(), list(COORD_COLS)))

    columns = ColumnTransformer(
        transformers,
        remainder="drop",  # this is what removes Latitude/Longitude themselves
        verbose_feature_names_out=False,
    )

    return Pipeline(
        [
            ("frame", FrameEngineer(drainage_mode=drainage_mode)),
            ("columns", columns),
        ]
    )


def feature_names(preprocessor: ColumnTransformer | Pipeline) -> list[str]:
    """Recover human-readable names for the columns a fitted preprocessor emits.

    One-hot expansion and circular encoding both change the column count, so the
    output matrix no longer lines up with the input frame. Feature-importance
    plots and SHAP summaries need this mapping to be truthful; without it,
    importances get attributed to whatever column happened to land at that
    index, which is a common and very quiet way for a results figure to be
    wrong.

    Accepts either the full pipeline from ``build_preprocessor`` or the inner
    ``ColumnTransformer``.
    """
    if isinstance(preprocessor, Pipeline):
        preprocessor = preprocessor.named_steps["columns"]
    return [str(n) for n in preprocessor.get_feature_names_out()]
