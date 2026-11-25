"""SMOTE with post-hoc snapping of synthetic rows to the encoded feature space.

Plain SMOTE interpolates every column of the *preprocessed* matrix, producing
one-hot groups with fractional entries, non-binary flags and (sin, cos) pairs off
the unit circle -- points that no real cell can occupy. ``SnapSMOTE`` runs
``imblearn.over_sampling.SMOTE`` (k=5) and then repairs ONLY the synthetic rows:

* each one-hot group (LULC_*, Soil_Texture_*) -> 1 at the argmax, 0 elsewhere;
* binary flags (is_flat, TWI_was_missing, Soil_Texture_was_missing) -> rounded to 0/1;
* (Aspect_sin, Aspect_cos) -> renormalised to unit length. Aspect components are
  deliberately NOT standardised (features._PartialStandardScaler), so this is done
  directly in the model matrix. A flat cell (is_flat = 1) is set to (0, 0). A
  synthetic row with is_flat = 0 but norm < 1e-6 is a flat cell by construction:
  it is kept at (0, 0) and is_flat is set to 1 for consistency.

Real (original) rows are returned untouched.
"""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE

from .features import BINARY_FLAG_COLS, CATEGORICAL_COLS, CATEGORY_LEVELS, terrain_feature_names


def snap_synthetic(Xs: np.ndarray, names: list[str]) -> np.ndarray:
    """Snap synthetic rows ``Xs`` (preprocessed space) to valid encodings."""
    Xs = np.array(Xs, dtype=float, copy=True)
    idx = {n: i for i, n in enumerate(names)}
    for col in CATEGORICAL_COLS:
        cols = [idx[f"{col}_{lvl}"] for lvl in CATEGORY_LEVELS[col]]
        arg = Xs[:, cols].argmax(axis=1)
        Xs[:, cols] = 0.0
        Xs[np.arange(len(Xs)), np.asarray(cols)[arg]] = 1.0
    for f in BINARY_FLAG_COLS:
        Xs[:, idx[f]] = (Xs[:, idx[f]] >= 0.5).astype(float)
    si, ci, fi = idx["Aspect_sin"], idx["Aspect_cos"], idx["is_flat"]
    norm = np.hypot(Xs[:, si], Xs[:, ci])
    flat = (Xs[:, fi] == 1) | (norm < 1e-6)
    Xs[flat, fi] = 1.0
    Xs[flat, si] = 0.0
    Xs[flat, ci] = 0.0
    ok = ~flat
    Xs[ok, si] /= norm[ok]
    Xs[ok, ci] /= norm[ok]
    return Xs


class SnapSMOTE:
    """SMOTE (k=5) whose synthetic rows are snapped to valid encoded values."""

    def __init__(self, random_state: int = 42, k_neighbors: int = 5, feature_names=None):
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.feature_names = feature_names

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        names = list(self.feature_names) if self.feature_names is not None else terrain_feature_names()
        if len(names) != X.shape[1]:
            raise ValueError(f"SnapSMOTE expects {len(names)} columns, got {X.shape[1]}")
        Xr, yr = SMOTE(random_state=self.random_state, k_neighbors=self.k_neighbors).fit_resample(X, y)
        n = len(X)
        assert np.array_equal(Xr[:n], X), "SMOTE reordered the original rows"
        Xr[n:] = snap_synthetic(Xr[n:], names)
        return Xr, yr
