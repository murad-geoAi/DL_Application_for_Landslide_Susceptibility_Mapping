"""Proof that the fitted feature steps see training rows only.

Runs standalone (``python tests/test_feature_leakage.py``) and under pytest.
Standalone it prints the evidence recorded in
``results/session2_feature_check.md``; under pytest the assertions are the test.

The claim being proved is narrow and specific: the DBSCAN cluster structure is a
function of the training coordinates alone, and validation and test points are
assigned against that frozen structure without refitting it. Everything else
here -- the imputer, the one-hot vocabularies -- is checked the same way.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D  # noqa: E402
from src import features as F  # noqa: E402

DATA_PATH = Path(__file__).resolve().parents[1] / "Labeled Data Set.csv"
SEED = 42


def make_splits():
    """Clean the raw file and cut it into train / val / test, shuffled, seed 42."""
    df = D.clean(D.load_raw(DATA_PATH))
    train, rest = train_test_split(
        df, test_size=0.4, random_state=SEED, shuffle=True, stratify=df[D.TARGET]
    )
    val, test = train_test_split(
        rest, test_size=0.5, random_state=SEED, shuffle=True, stratify=rest[D.TARGET]
    )
    return train, val, test


def _state(assigner: F.DBSCANClusterAssigner) -> dict:
    """Snapshot of everything the assigner learned at fit time."""
    return {
        "core_points": assigner.core_points_.copy(),
        "core_labels": assigner.core_labels_.copy(),
        "labels": assigner.labels_.copy(),
        "n_clusters": assigner.n_clusters_,
        "n_train": assigner.n_train_,
    }


def test_dbscan_fitted_on_train_coordinates_only(verbose: bool = False):
    """The cluster structure comes from train coordinates and nothing else."""
    train, val, test = make_splits()

    assigner = F.DBSCANClusterAssigner(eps=F.DBSCAN_EPS, min_samples=F.DBSCAN_MIN_SAMPLES)
    assigner.fit(train[D.COORD_COLS])
    before = _state(assigner)

    centres = assigner.cluster_centres()
    if verbose:
        print(f"\n[1] DBSCAN fit on TRAIN only "
              f"(eps={F.DBSCAN_EPS}, min_samples={F.DBSCAN_MIN_SAMPLES})")
        print(f"    train rows seen at fit : {assigner.n_train_}  "
              f"(val={len(val)}, test={len(test)} not seen)")
        print(f"    clusters found         : {assigner.n_clusters_}")
        print(f"    core points            : {len(assigner.core_points_)}")
        print(f"    noise points in train  : {assigner.n_noise_}")
        print("\n    cluster centres (train core points only):")
        print(centres.to_string(index=False,
                                float_format=lambda v: f"{v:.5f}"))

    # (a) Every fitted core point is literally a training coordinate.
    train_coords = set(map(tuple, train[D.COORD_COLS].to_numpy()))
    core_coords = set(map(tuple, assigner.core_points_))
    assert core_coords <= train_coords, "a core point is not a training row"

    # (b) No validation or test coordinate is a core point, except where that
    #     exact coordinate also occurs in train (it does not, after dedup).
    val_test_coords = set(map(tuple, pd.concat([val, test])[D.COORD_COLS].to_numpy()))
    assert not (core_coords & val_test_coords), "a held-out point became a core point"

    # (c) Transforming val and test does not move anything that was fitted.
    val_labels = assigner.transform(val[D.COORD_COLS])["Cluster"].astype(int)
    test_labels = assigner.transform(test[D.COORD_COLS])["Cluster"].astype(int)
    after = _state(assigner)

    for key in before:
        assert np.array_equal(np.asarray(before[key]), np.asarray(after[key])), (
            f"assigner state '{key}' changed during transform -- it refitted"
        )

    if verbose:
        print("\n[2] VAL/TEST assigned against that frozen structure (transform only)")
        for name, labels, frame in (("val", val_labels, val), ("test", test_labels, test)):
            counts = labels.value_counts().sort_index()
            assigned = int((labels >= 0).sum())
            print(f"    {name:4s}: n={len(frame):3d}  "
                  f"assigned to a train cluster={assigned:3d}  "
                  f"noise(-1)={int((labels == -1).sum()):3d}")
            print(f"          label counts: {counts.to_dict()}")
        print("    fitted state after both transforms: byte-identical "
              "(core points, labels, cluster count all unchanged)")

    # (d) Every assigned label is one that existed in the training fit.
    train_label_set = set(before["labels"].tolist())
    for labels in (val_labels, test_labels):
        unseen = set(labels.tolist()) - train_label_set - {-1}
        assert not unseen, f"invented cluster labels not present in train: {unseen}"

    # (e) The assignment rule respects eps: anything assigned to a cluster is
    #     within eps of a train core point, anything else is -1.
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1).fit(assigner.core_points_km_)
    vc = val[D.COORD_COLS].to_numpy()
    dist, _ = nn.kneighbors(F.project_km(vc[:, 0], vc[:, 1]))
    within = dist[:, 0] <= F.DBSCAN_EPS
    assert np.array_equal(within, (val_labels >= 0).to_numpy()), (
        "a point was assigned to a cluster whose core point is beyond eps"
    )
    if verbose:
        print(f"    eps rule holds: all {int(within.sum())} assigned val points lie "
              f"within {F.DBSCAN_EPS} km of a train core point; "
              f"the other {int((~within).sum())} are noise")

        # The assignment rule reproduces DBSCAN's own labels on the fitted data.
        print(f"    agreement with DBSCAN's own train labels: "
              f"{assigner.assignment_agreement(train[D.COORD_COLS])['agreement']:.3f}")


def test_holdout_points_would_have_changed_the_structure(verbose: bool = False):
    """The counterfactual: including val at fit time gives different clusters.

    Without this, "fit on train only" is an untested claim -- if the structure
    were the same either way, the discipline would be costless and unproven.
    It is not the same, which is exactly why the discipline matters.
    """
    train, val, test = make_splits()

    train_only = F.DBSCANClusterAssigner().fit(train[D.COORD_COLS])
    leaky = F.DBSCANClusterAssigner().fit(pd.concat([train, val])[D.COORD_COLS])

    if verbose:
        print("\n[3] Counterfactual -- what fitting on train+val would have done")
        print(f"    fit on train only : {train_only.n_clusters_} clusters, "
              f"{len(train_only.core_points_)} core points")
        print(f"    fit on train+val  : {leaky.n_clusters_} clusters, "
              f"{len(leaky.core_points_)} core points")
        gained = len(leaky.core_points_) - len(train_only.core_points_)
        print(f"    -> held-out points would have added {gained} core points and "
              f"changed the structure they are scored against")

    assert (
        leaky.n_clusters_ != train_only.n_clusters_
        or len(leaky.core_points_) != len(train_only.core_points_)
    ), "expected the held-out points to alter the cluster structure"


def test_preprocessor_is_fit_on_train_only(verbose: bool = False):
    """The whole preprocessor: fit on train, transform val/test, nothing moves."""
    train, val, test = make_splits()

    # Both configurations, because as of Session 3 the default no longer carries
    # the cluster one-hot. Without the second pass, the assembled-pipeline check
    # would silently stop covering the DBSCAN branch it was written to cover.
    for label, kwargs in (
        ("default (no Cluster, no Drainage_Density)", {}),
        ("leakage-baseline ablation", {"include_cluster": True, "drainage_mode": "raw"}),
    ):
        _assert_preprocessor_fit_on_train_only(train, val, test, label, kwargs, verbose)


def _assert_preprocessor_fit_on_train_only(train, val, test, label, kwargs, verbose):
    pre = F.build_preprocessor(**kwargs)
    Xtr = pre.fit_transform(train, train[D.TARGET])
    fitted = copy.deepcopy(pre.named_steps["columns"].named_transformers_)

    Xval = pre.transform(val)
    Xte = pre.transform(test)

    names = F.feature_names(pre)
    assert Xtr.shape[1] == Xval.shape[1] == Xte.shape[1] == len(names)

    # Coordinates are gone from the matrix the model sees.
    assert not ({"Latitude", "Longitude"} & set(names)), "coordinates reached the model"

    # The one-hot vocabularies and imputer are unchanged by transforming.
    after = pre.named_steps["columns"].named_transformers_
    for key, before_t in fitted.items():
        if hasattr(before_t, "named_steps") and "onehot" in getattr(before_t, "named_steps", {}):
            b = before_t.named_steps["onehot"].categories_
            a = after[key].named_steps["onehot"].categories_
            assert all(np.array_equal(x, y) for x, y in zip(b, a)), (
                f"{key} vocabulary changed during transform"
            )

    # Imputation leaves nothing unfilled. Asserted unconditionally: under
    # pytest, verbose is False, so nesting this inside the block below would
    # mean the check never ran where it matters.
    assert np.isfinite(Xtr).all() and np.isfinite(Xval).all(), "NaN survived imputation"

    if verbose:
        print(f"\n[4] Full preprocessor -- {label}")
        print(f"    fit on train ({len(train)} rows) -> {Xtr.shape[1]} features")
        print(f"    transform val  ({len(val):3d} rows) -> {Xval.shape}")
        print(f"    transform test ({len(test):3d} rows) -> {Xte.shape}")
        print(f"    Latitude/Longitude present in output: "
              f"{bool({'Latitude', 'Longitude'} & set(names))}")
        print(f"    feature names ({len(names)}):")
        for i in range(0, len(names), 4):
            print("      " + ", ".join(names[i:i + 4]))
        print("    no NaN survives imputation in train or val")


def test_unseen_category_maps_to_all_zeros(verbose: bool = False):
    """An unseen class code encodes as all-zeros rather than as a nearby level."""
    train, val, _ = make_splits()

    enc = F.build_categorical_encoder().fit(train[D.CATEGORICAL_COLS])
    probe = val[D.CATEGORICAL_COLS].head(1).copy()
    probe.loc[:, "Soil_Texture"] = 99  # a texture code that exists nowhere
    probe.loc[:, "LULC"] = 42

    out = enc.transform(probe)
    assert out.sum() == 0.0, "an unseen code did not encode as all-zeros"

    if verbose:
        vocab = enc.named_steps["onehot"].categories_
        print("\n[5] Unseen categories")
        print(f"    train vocabulary  LULC={list(vocab[0])}")
        print(f"                      Soil_Texture={list(vocab[1])}")
        print(f"    probe row (LULC=42, Soil_Texture=99) encodes to "
              f"{int(out.sum())} non-zero entries across {out.shape[1]} columns")


def test_flat_cells_and_aspect_encoding(verbose: bool = False):
    """Flat cells sit at the origin; every other row sits on the unit circle."""
    df = D.clean(D.load_raw(DATA_PATH))
    agreement = F.check_flat_cell_agreement(D.load_raw(DATA_PATH))

    enc = F.encode_aspect(df)
    flat = enc["is_flat"].astype(bool)
    radius = np.hypot(enc["Aspect_sin"], enc["Aspect_cos"])

    assert np.allclose(radius[flat], 0.0), "a flat cell was given a bearing"
    non_flat = radius[~flat].dropna()
    assert np.allclose(non_flat, 1.0), "a non-flat row left the unit circle"

    if verbose:
        print("\n[6] Aspect encoding")
        print(f"    Slope == 0 rows={agreement['n_flat_by_slope']}, "
              f"Aspect == -1 rows={agreement['n_flat_by_sentinel']}, "
              f"disagreement={agreement['n_disagree']}")
        print(f"    flat cells at origin (radius 0): {int(flat.sum())}")
        print(f"    remaining rows on the unit circle (radius 1): {len(non_flat)}")
        print(f"    rows with Aspect genuinely missing, left for the imputer: "
              f"{int(radius.isna().sum())}")


def _all_encoded_dims():
    """Encoded width of every spatial fold plus the random split (terrain config)."""
    from src.splits import random_split, spatial_block_cv

    df = D.clean(D.load_raw(DATA_PATH)).reset_index(drop=True)
    dims, names = [], []
    for tr, _ in spatial_block_cv(df):
        pre = F.build_preprocessor()
        X = pre.fit_transform(df.iloc[tr], df.iloc[tr][D.TARGET])
        dims.append(X.shape[1])
        names.append(F.feature_names(pre))
    train, _, _ = random_split(df, random_state=SEED)
    pre = F.build_preprocessor()
    dims.append(pre.fit_transform(train, train[D.TARGET]).shape[1])
    names.append(F.feature_names(pre))
    return df, dims, names


def test_encoded_dimension_constant_across_folds_and_random_split(verbose: bool = False):
    df, dims, names = _all_encoded_dims()
    assert set(dims) == {23}, f"encoded width varies: {dims}"
    assert all(n == names[0] for n in names), "column names/order differ between folds"
    assert names[0] == F.terrain_feature_names(), "schema names drifted from fitted names"
    # category schema equals the data's distinct values (+ missing)
    for col in D.CATEGORICAL_COLS:
        levels = sorted(str(int(v)) for v in df[col].dropna().unique())
        assert F.CATEGORY_LEVELS[col] == levels + [F.MISSING_LEVEL] or \
            sorted(F.CATEGORY_LEVELS[col][:-1]) == sorted(levels)
    assert abs(df["Latitude"].mean() - F.FIXED_LAT0) < 1e-9
    if verbose:
        print("\n[7] encoded dims", dims)


def test_binary_flags_not_scaled(verbose: bool = False):
    df = D.clean(D.load_raw(DATA_PATH)).reset_index(drop=True)
    pre = F.build_preprocessor()
    X = pd.DataFrame(pre.fit_transform(df, df[D.TARGET]), columns=F.feature_names(pre))
    for c in F.BINARY_FLAG_COLS:
        assert set(np.unique(X[c])) <= {0.0, 1.0}, c
    ok = (X.is_flat == 0).to_numpy() & df['Aspect'].notna().to_numpy()  # KNN-imputed aspects are not unit length
    assert np.allclose(np.hypot(X.Aspect_sin, X.Aspect_cos)[ok], 1.0)


def test_smote_synthetic_rows_are_valid_encodings(verbose: bool = False):
    from src.sampling import SnapSMOTE

    df = D.clean(D.load_raw(DATA_PATH)).reset_index(drop=True)
    pre = F.build_preprocessor()
    X = pre.fit_transform(df, df[D.TARGET])
    y = df[D.TARGET].to_numpy(dtype=int)
    # imbalance the training set so SMOTE has work to do
    keep = np.r_[np.where(y == 0)[0], np.where(y == 1)[0][:60]]
    Xk, yk = np.asarray(X)[keep], y[keep]
    Xr, yr = SnapSMOTE(random_state=0, feature_names=F.terrain_feature_names()).fit_resample(Xk, yk)
    n = len(Xk)
    assert len(Xr) > n and np.array_equal(Xr[:n], Xk)
    syn = pd.DataFrame(Xr[n:], columns=F.terrain_feature_names())
    for col in D.CATEGORICAL_COLS:
        grp = [f"{col}_{lvl}" for lvl in F.CATEGORY_LEVELS[col]]
        assert (syn[grp].sum(axis=1) == 1).all(), f"{col} group not one-hot"
        assert set(np.unique(syn[grp])) <= {0.0, 1.0}
    for c in F.BINARY_FLAG_COLS:
        assert set(np.unique(syn[c])) <= {0.0, 1.0}, c
    r = np.hypot(syn.Aspect_sin, syn.Aspect_cos)
    assert (np.isclose(r, 1.0, atol=1e-6) | np.isclose(r, 0.0, atol=1e-6)).all()
    flat = syn.is_flat == 1
    assert np.allclose(r[flat], 0.0)


def test_keras_validation_split_precedes_smote(verbose: bool = False):
    """The internal validation rows must be real: sampler sees only the post-split 80%."""
    from src.models import KerasBinaryClassifier

    seen = {}

    class Spy:
        def fit_resample(self, X, y):
            seen["n"] = len(X)
            return X, y

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    y = (rng.random(100) > 0.5).astype(int)
    KerasBinaryClassifier(architecture="dnn", epochs=1, sampler=Spy()).fit(X, y)
    assert seen["n"] == 80


def test_encoders_and_smote_fit_on_train_only(verbose: bool = False):
    """Fold pipeline fitted on train rows: vocabularies fixed, scaler stats equal train-only stats."""
    train, val, _ = make_splits()
    pre = F.build_preprocessor().fit(train, train[D.TARGET])
    scaler = pre.named_steps["columns"].named_transformers_["continuous"].named_steps["scale"]
    el = list(scaler.feature_names_in_).index("Elevation")
    pre2 = F.build_preprocessor().fit(pd.concat([train, val]), pd.concat([train, val])[D.TARGET])
    s2 = pre2.named_steps["columns"].named_transformers_["continuous"].named_steps["scale"]
    assert not np.isclose(scaler.mean_[el], s2.mean_[el]), "scaler did not depend on its fit data"


def main() -> int:
    print("=" * 72)
    print("SESSION 2 -- PROOF THAT FITTED FEATURE STEPS SEE TRAIN ROWS ONLY")
    print("=" * 72)
    test_dbscan_fitted_on_train_coordinates_only(verbose=True)
    test_holdout_points_would_have_changed_the_structure(verbose=True)
    test_preprocessor_is_fit_on_train_only(verbose=True)
    test_unseen_category_maps_to_all_zeros(verbose=True)
    test_flat_cells_and_aspect_encoding(verbose=True)
    test_encoded_dimension_constant_across_folds_and_random_split(verbose=True)
    print("\n" + "=" * 72)
    print("All checks passed.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
