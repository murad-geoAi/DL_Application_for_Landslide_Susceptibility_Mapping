"""Entry point: one command that reproduces every number in the corrected thesis.

Wires together ``data`` -> ``features`` -> ``splits`` -> ``models`` ->
``evaluate`` and writes the outputs to ``results/``. The goal is that a reviewer
can clone the repository, run a single command, and regenerate every table in
the manuscript with matching values.

The three regimes
-----------------
========  ====================================================================
**R0**    The original protocol, bugs and all. DBSCAN fit on the full dataset
          before splitting, KNN imputation fit on the full dataset, the test
          set passed as ``validation_data`` so that early stopping and
          ``restore_best_weights`` both select on it. A documentation
          spot-check, run for DNN and 1D-CNN on the enhanced dataset only, to
          show how close the corrected code lands to the manuscript's own
          reported table. **Not a result.**
**R1**    Stratified random 60/20/20. Preprocessing fit on train only, early
          stopping on a genuine validation partition, test touched exactly
          once. Honest code, leaking split: it is the manuscript's evaluation
          design with the implementation bugs removed, so the distance from R0
          to R1 measures the bugs and the distance from R1 to R2 measures the
          design.
**R2**    Spatially blocked 5-fold CV, ``splits.spatial_block_cv`` at its
          ``balance="both"`` default. The honest number, and the one the
          corrected manuscript should lead with.
========  ====================================================================

Every regime runs the same seven models over the same two feature sets
(terrain-only, 21 features; terrain + ``Cluster``, 31), except R0, which
reproduces the manuscript's own 13-column raw matrix because reproducing it is
the entire point.

Reading the output
------------------
``results/results_R0.csv``, ``_R1.csv``, ``_R2.csv``, all nine columns wide:
``regime, model, dataset, fold, AUC, accuracy, precision, recall, F1``. R0 and
R1 rows carry ``fold = "NA"`` -- a single held-out test set has no fold index.
R2 rows carry ``fold = 1..5`` plus three aggregate rows per cell:
``within_mean``, ``within_sd``, and ``pooled``. Filter on ``fold == "pooled"``
to get the headline table.

Usage
-----
    python -m src.run                       # all three regimes, then save models
    python -m src.run --regimes R2
    python -m src.run --regimes R1 R2 --no-save-models
    python -m src.run --seed 7 --output-dir results/seed7
"""

from __future__ import annotations

import os

# Set before TensorFlow is imported anywhere: the interpreter reads
# PYTHONHASHSEED at start-up, so this only takes full effect in a subprocess,
# but it is as early as a module can act and it is recorded in the provenance
# block either way. See ``models.set_global_seeds``.
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import TARGET, clean, load_raw  # noqa: E402
from .evaluate import (  # noqa: E402
    RESULT_COLUMNS,
    classification_metrics,
    cross_validate_model,
    cv_result_rows,
    save_results,
    single_split_row,
)
from .models import (  # noqa: E402
    MODEL_LABELS,
    MODEL_NAMES,
    fit_pipeline,
    get_model,
    set_global_seeds,
)
from .splits import random_split, spatial_block_cv  # noqa: E402

DEFAULT_DATA: str = "Labeled Data Set.csv"
DEFAULT_OUTPUT_DIR: str = "results"
DEFAULT_MODELS_DIR: str = "models"
SEED: int = 42

#: The two feature sets the R1 and R2 sweeps cross with the seven models, plus
#: the third that exists only as a one-line-per-model sensitivity check. Keys
#: are the ``dataset`` column of the results CSVs; values are the
#: ``build_preprocessor`` switches that produce them.
FEATURE_SETS: dict[str, dict[str, Any]] = {
    "terrain": {"include_cluster": False, "drainage_mode": "drop"},
    "terrain+Cluster": {"include_cluster": True, "drainage_mode": "drop"},
}

#: Session 3 configuration (b): terrain-only plus ``Drainage_Density``. Run
#: under R2 only, and reported as one line per model rather than as a third
#: column of the headline table -- it is a sensitivity check on a near-constant
#: column whose 27 non-zero rows are all negatives, not a candidate feature set.
SENSITIVITY_SET: dict[str, dict[str, Any]] = {
    "terrain+DrainageDensity": {"include_cluster": False, "drainage_mode": "raw"},
}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Flags
    -----
    ``--data``
        Path to the labelled CSV. Defaults to ``Labeled Data Set.csv``.
    ``--regimes``
        Which of ``R0``, ``R1``, ``R2`` to run. Defaults to all three.
    ``--models``
        Which models to fit under R1 and R2. Defaults to all seven. R0 always
        runs DNN and 1D-CNN only, because that is what it is for.
    ``--seed``
        Master seed, threaded through every stochastic component.
    ``--output-dir``
        Defaults to ``results/``.
    ``--models-dir``
        Where the final fitted pipelines are written. Defaults to ``models/``.
    ``--no-save-models``
        Skip the final refit-on-all-data step.
    ``--no-deterministic-ops``
        Skip ``tf.config.experimental.enable_op_determinism()``. Faster, and the
        only option on hardware whose kernels have no deterministic variant.

    A YAML run-configuration file was part of the original design and is
    deferred: with three regimes and two feature sets the flags cover the space,
    and a config file that nothing reads is one more thing to drift.
    """
    p = argparse.ArgumentParser(prog="python -m src.run", description=__doc__)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--regimes", nargs="+", default=["R0", "R1", "R2"], choices=["R0", "R1", "R2"])
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    p.add_argument("--no-save-models", action="store_true")
    p.add_argument("--no-deterministic-ops", action="store_true")
    return p.parse_args(argv)


def set_seeds(seed: int = SEED, deterministic: bool = True) -> None:
    """Seed every random source in the process.

    Thin wrapper over ``models.set_global_seeds``, kept here because ``run`` is
    where the seed enters the pipeline and a reader looking for "what is
    seeded?" will look in this file first. Covers ``PYTHONHASHSEED``,
    ``random``, ``numpy``, and TensorFlow, all set to ``seed``.

    GPU training is not bit-reproducible by default; this analysis runs on CPU,
    where ``enable_op_determinism`` is cheap. Which mode was used is recorded in
    the provenance sidecar either way.
    """
    set_global_seeds(seed, deterministic=deterministic)


# --------------------------------------------------------------------------
# R0 -- the original protocol, reproduced deliberately
# --------------------------------------------------------------------------


def _r0_matrix(data_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Rebuild the manuscript's enhanced feature matrix, bugs included.

    This function deliberately does **not** use ``data.clean`` or
    ``features.build_preprocessor``. Both of those are leak-proof by
    construction, and running R0 through them would defeat the purpose: the
    whole point is to fit the manuscript's own preprocessing on the full dataset
    before any split exists, so that the resulting numbers can be compared
    against the manuscript's Table 2.

    Reproduces ``notebooks/landslide_susceptibility_mapping.ipynb`` cells 6, 33
    and 48 in order:

    1. ``KNNImputer(n_neighbors=5)`` over ``[Latitude, Longitude, SPI, TWI]``,
       fit on all 498 rows. Held-out ``SPI``/``TWI`` values are reconstructed
       from their geographic neighbours, 40% of which will end up in the
       training set.
    2. Mode imputation for ``Soil_Texture`` and ``LULC``, median for ``Slope``,
       ``Aspect``, ``Annual_Rainfall`` and ``Drainage_Density``. Both fit on all
       498 rows.
    3. ``DBSCAN(eps=0.01, min_samples=5)`` on ``[Longitude, Latitude]``, fit on
       all 498 rows, producing the ``Cluster`` column. Every test point's
       cluster id is therefore determined partly by test points.
    4. The 13 columns as the notebook listed them, raw: ``LULC``,
       ``Soil_Texture`` and ``Cluster`` as integer codes rather than one-hot,
       ``Aspect`` in raw degrees rather than sin/cos, ``SPI`` and ``TWI`` both
       present despite being byte-identical, and the duplicate rows never
       dropped.

    Returns ``(X, y, feature_names)`` with 498 rows.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.impute import KNNImputer, SimpleImputer

    df = pd.read_csv(data_path)

    geo = KNNImputer(n_neighbors=5)
    spatial = ["Latitude", "Longitude", "SPI", "TWI"]
    df[spatial] = geo.fit_transform(df[spatial])

    mode = SimpleImputer(strategy="most_frequent")
    df[["Soil_Texture", "LULC"]] = mode.fit_transform(df[["Soil_Texture", "LULC"]])

    median = SimpleImputer(strategy="median")
    med_cols = ["Slope", "Aspect", "Annual_Rainfall", "Drainage_Density"]
    df[med_cols] = median.fit_transform(df[med_cols])

    dbscan = DBSCAN(eps=0.01, min_samples=5)
    df["Cluster"] = dbscan.fit_predict(df[["Longitude", "Latitude"]])

    features = [
        "Elevation",
        "Slope",
        "Aspect",
        "Curvature",
        "SPI",
        "TWI",
        "TRI",
        "NDVI",
        "Annual_Rainfall",
        "Drainage_Density",
        "LULC",
        "Soil_Texture",
        "Cluster",
    ]
    return df[features].to_numpy(dtype=float), df[TARGET].to_numpy(dtype=int), features


def run_R0(data_path: str | Path, seed: int = SEED) -> pd.DataFrame:
    """Reproduce the original buggy protocol. A spot-check, not a headline.

    DNN and 1D-CNN on the enhanced dataset only -- the two rows of the
    manuscript's Table 2 that this session needs a bearing on. The LSTM and the
    four classical models are not run here: R0 exists to calibrate how closely
    this codebase reproduces the manuscript, and two rows answer that as well as
    seven would while costing a third of the compute.

    Three leaks, all faithful to the original and all in plain sight:

    1. Preprocessing and clustering fit before the split (``_r0_matrix``).
    2. ``validation_data = (X_test, y_test)``. The test set *is* the validation
       set.
    3. ``EarlyStopping(monitor="val_accuracy", patience=5,
       restore_best_weights=True)`` on top of it, so training stops at the epoch
       that scored best on test and the weights from that epoch are restored.
       The reported test accuracy is therefore a maximum over up to 50 epochs,
       not an evaluation.

    The scaler is the one thing the original got right: ``StandardScaler`` is
    fit on the training split alone, and that is preserved here so the
    difference from R1 is attributable to the three leaks above rather than to
    four.

    Returns a frame in ``RESULT_COLUMNS`` layout, ``fold = "NA"``.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from .models import KerasBinaryClassifier

    X, y, features = _r0_matrix(data_path)
    print(f"[R0] enhanced matrix: {X.shape[0]} rows x {X.shape[1]} features {features}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rows = []
    for name in ("dnn", "cnn1d"):
        set_global_seeds(seed, deterministic=False)
        est = KerasBinaryClassifier(
            architecture=name,
            early_stopping=True,
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            random_state=seed,
        )
        t0 = time.time()
        # THE BUG, made explicit: the test set is handed in as the validation
        # set, and EarlyStopping selects the epoch on it.
        est.fit(X_train, y_train, validation_data=(X_test, y_test))
        prob = est.predict_proba(X_test)[:, 1]
        row = single_split_row(y_test, prob, "R0", MODEL_LABELS[name], "enhanced(13 raw feat)")
        rows.append(row)
        m = row.iloc[0]
        print(
            f"[R0] {MODEL_LABELS[name]:<8} AUC={m['AUC']:.4f} acc={m['accuracy']:.4f} "
            f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['F1']:.4f} "
            f"({est.epochs_run_} epochs, {time.time() - t0:.0f}s)"
        )
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# R1 -- stratified random 60/20/20
# --------------------------------------------------------------------------


def run_R1(
    df: pd.DataFrame,
    model_names: list[str],
    seed: int = SEED,
) -> pd.DataFrame:
    """Seven models x two feature sets on a stratified random 60/20/20 split.

    The split is ``splits.random_split``, which reproduces the 297/99/100
    partition ``tests/test_feature_leakage.py`` already uses. Everything the
    pipeline learns -- imputer, scaler, category vocabulary, DBSCAN clusters --
    is fit on the 297 training rows only.

    Early stopping looks at the 99-row validation partition and never at the
    100-row test partition, which is enforced structurally rather than by
    convention: ``models.fit_pipeline`` is the only path that supplies
    ``validation_data``, and it is handed ``val``, not ``test``. The test
    partition is touched exactly once per cell, after fitting, to produce the
    row below.

    The four classical models do not use the validation partition at all, so
    they are fitted on the same 297 rows the deep models train on rather than on
    train+val. Giving them the extra 99 rows would be defensible in isolation
    and would make the seven rows incomparable, which is the thing this table
    exists to avoid.

    Returns a frame in ``RESULT_COLUMNS`` layout, one row per (model, feature
    set), ``fold = "NA"``.
    """
    train, val, test = random_split(df, random_state=seed)
    y_train = train[TARGET].to_numpy(dtype=int)
    y_val = val[TARGET].to_numpy(dtype=int)
    y_test = test[TARGET].to_numpy(dtype=int)
    print(
        f"[R1] split {len(train)}/{len(val)}/{len(test)} "
        f"(pos {y_train.sum()}/{y_val.sum()}/{y_test.sum()})"
    )

    rows = []
    for dataset, kw in FEATURE_SETS.items():
        for name in model_names:
            set_global_seeds(seed, deterministic=False)
            t0 = time.time()
            pipe = get_model(name, random_state=seed, **kw)
            pipe = fit_pipeline(pipe, train, y_train, val, y_val)

            n_feat = pipe.named_steps["pre"].transform(train).shape[1]
            prob = pipe.predict_proba(test)[:, 1]
            row = single_split_row(y_test, prob, "R1", MODEL_LABELS[name], dataset)
            rows.append(row)
            m = row.iloc[0]
            print(
                f"[R1] {dataset:<16} {MODEL_LABELS[name]:<19} n_feat={n_feat:>2} "
                f"AUC={m['AUC']:.4f} acc={m['accuracy']:.4f} F1={m['F1']:.4f} "
                f"({time.time() - t0:.0f}s)"
            )
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# R2 -- spatially blocked cross-validation
# --------------------------------------------------------------------------


def run_R2(
    df: pd.DataFrame,
    model_names: list[str],
    seed: int = SEED,
    include_sensitivity: bool = True,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:
    """Seven models x two feature sets under ``spatial_block_cv``, plus the
    ``Drainage_Density`` sensitivity.

    The folds come from a bare ``spatial_block_cv(df)`` call, which since
    Session 3 defaults to ``balance="both"``: ten KMeans sub-blocks merged
    size-aware into five folds, whole blocks kept intact. The identical
    partition is used for every model and every feature set, because it is built
    from coordinates and labels alone and never sees the covariates. That is
    what makes the rows comparable -- fold 5's two positives are fold 5's two
    positives for all twenty-one cells.

    Each cell reports eight rows: five per-fold, then ``within_mean``,
    ``within_sd`` (population, ``ddof=0``) and ``pooled``. Pooled is the
    authoritative figure; see ``evaluate``'s module docstring for why the two
    differ and what the gap measures.

    Parameters
    ----------
    include_sensitivity:
        Also run the terrain + ``Drainage_Density`` set (Session 3
        configuration b). Written into the same CSV under
        ``dataset = "terrain+DrainageDensity"`` so the numbers are auditable,
        and reported in the summary as one line per model rather than as a third
        column of the headline table.

    Returns
    -------
    ``(results, cv_by_cell)`` where ``cv_by_cell`` is keyed by
    ``(model_name, dataset)`` and holds the full ``cross_validate_model``
    output, which ``save_final_models`` uses to pick each model's better feature
    set.
    """
    folds = list(spatial_block_cv(df))
    y = df[TARGET].to_numpy(dtype=int)
    print(
        "[R2] folds: "
        + " ".join(
            f"{k}:n={len(te)},pos={int(y[te].sum())}" for k, (_, te) in enumerate(folds, 1)
        )
    )

    sets = dict(FEATURE_SETS)
    if include_sensitivity:
        sets.update(SENSITIVITY_SET)

    rows = []
    cv_by_cell: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset, kw in sets.items():
        for name in model_names:
            set_global_seeds(seed, deterministic=False)
            t0 = time.time()
            cv = cross_validate_model(
                lambda name=name, kw=kw: get_model(name, random_state=seed, **kw),
                df,
                folds,
            )
            cv_by_cell[(name, dataset)] = cv
            rows.append(cv_result_rows(cv, "R2", MODEL_LABELS[name], dataset))

            per_fold_auc = cv["per_fold"]["AUC"].to_numpy(dtype=float)
            print(
                f"[R2] {dataset:<23} {MODEL_LABELS[name]:<19} "
                f"within={np.nanmean(per_fold_auc):.4f}+/-{np.nanstd(per_fold_auc):.4f} "
                f"pooled={cv['pooled']['AUC']:.4f} ({time.time() - t0:.0f}s)"
            )
    return pd.concat(rows, ignore_index=True), cv_by_cell


# --------------------------------------------------------------------------
# Final models
# --------------------------------------------------------------------------


def save_final_models(
    df: pd.DataFrame,
    cv_by_cell: dict[tuple[str, str], dict[str, Any]],
    model_names: list[str],
    models_dir: str | Path,
    seed: int = SEED,
) -> pd.DataFrame:
    """Refit each model on its better R2 feature set, over all 496 rows, and save.

    "Better" means higher **pooled** AUC under R2, between terrain-only and
    terrain + ``Cluster``. Pooled rather than within-fold because the within-fold
    mean is the fragile number -- one fold carries two positives -- and choosing
    a deliverable model on it would let that fold pick the feature set. The
    ``Drainage_Density`` sensitivity set is excluded from the comparison: it is a
    check, not a candidate.

    The saved pipeline is refit on the **entire** cleaned frame, which is the
    right thing for an artefact that will predict on new ground and the wrong
    thing to quote a number from. Nothing in ``results/`` comes from these
    objects; they exist for the mapping stage, and the score to attach to them
    is the R2 pooled AUC of the configuration they were selected on.

    Keras models survive ``joblib.dump`` through ``KerasBinaryClassifier``'s
    ``__getstate__``/``__setstate__``, which round-trip the network through the
    ``.keras`` archive format.

    Returns a selection table: model, chosen dataset, and both pooled AUCs.
    """
    import joblib

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    y = df[TARGET].to_numpy(dtype=int)

    rows = []
    for name in model_names:
        scored = {
            ds: cv_by_cell[(name, ds)]["pooled"]["AUC"]
            for ds in FEATURE_SETS
            if (name, ds) in cv_by_cell
        }
        if not scored:
            continue
        best = max(scored, key=scored.get)

        set_global_seeds(seed, deterministic=False)
        pipe = get_model(name, random_state=seed, **FEATURE_SETS[best])
        pipe = fit_pipeline(pipe, df, y)

        path = models_dir / f"{name}.joblib"
        joblib.dump({"pipeline": pipe, "dataset": best, "r2_pooled_auc": scored[best]}, path)
        rows.append(
            {
                "model": MODEL_LABELS[name],
                "selected_dataset": best,
                **{f"pooled_AUC[{ds}]": round(v, 4) for ds, v in scored.items()},
                "path": str(path).replace("\\", "/"),
            }
        )
        print(f"[save] {MODEL_LABELS[name]:<19} -> {best} (pooled AUC {scored[best]:.4f}) -> {path}")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Deferred
# --------------------------------------------------------------------------


def run_audit(data_path: str | Path, output_dir: str | Path) -> None:
    """Regenerate the data audit from the current input file.

    Would recompute shape, dtypes, missingness, class balance, the duplicate
    count, the SPI/TWI identity check, the coordinates-only probe, and the
    nearest-neighbour distance statistics, then write
    ``results/session1_data_summary.md``.

    Deferred: that file exists and its numbers are current. Mechanising it is
    worth doing before the rasters are re-extracted, not before.
    """
    raise NotImplementedError("deferred; results/session1_data_summary.md is current")


def build_susceptibility_map(
    model: Any,
    raster_dir: str | Path,
    output_path: str | Path,
    include_uncertainty: bool = True,
) -> None:
    """Predict across the study-area grid to produce the susceptibility surface.

    Requires the covariate rasters, which are not in this repository -- the CSV
    holds only the 498 sampled points. Left as a stub until they are located.

    Two requirements when it is implemented. First, the map must carry an
    uncertainty layer beside the mean prediction; the tree ensembles can supply
    a usable proxy from the spread across trees. Second, the extrapolation
    domain must be masked: the training points span a limited range of elevation
    and rainfall, and grid cells outside that envelope are extrapolations rather
    than predictions. A susceptibility map that renders them in the same colour
    scheme as well-supported cells overstates its own reach, which is precisely
    the failure the rest of this correction is addressing.
    """
    raise NotImplementedError("needs the covariate rasters; not in this repository")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    set_seeds(args.seed, deterministic=not args.no_deterministic_ops)

    df = clean(load_raw(args.data))
    y = df[TARGET].to_numpy(dtype=int)
    print(f"[data] cleaned frame: {df.shape[0]} rows, {int(y.sum())} positives")

    common = {
        "seed": args.seed,
        "deterministic_ops": not args.no_deterministic_ops,
        "data": str(args.data),
        "n_rows": int(df.shape[0]),
        "n_positives": int(y.sum()),
    }

    if "R0" in args.regimes:
        r0 = run_R0(args.data, seed=args.seed)
        save_results(
            r0,
            out_dir / "results_R0.csv",
            label="R0 original buggy protocol (documentation spot-check)",
            provenance={
                **common,
                "protocol": (
                    "KNN imputation, mode/median imputation and DBSCAN all fit on all 498 rows "
                    "before the split; stratified 80/20 train/test; validation_data = test; "
                    "EarlyStopping(val_accuracy, patience=5, restore_best_weights=True)"
                ),
                "n_rows": 498,
                "note": "not a result; reproduces the manuscript's own protocol for comparison",
            },
        )

    if "R1" in args.regimes:
        r1 = run_R1(df, args.models, seed=args.seed)
        save_results(
            r1,
            out_dir / "results_R1.csv",
            label="R1 stratified random 60/20/20",
            provenance={
                **common,
                "split": "splits.random_split, stratified, 60/20/20",
                "early_stopping": "validation partition only; test evaluated once",
            },
        )

    cv_by_cell: dict[tuple[str, str], dict[str, Any]] = {}
    if "R2" in args.regimes:
        r2, cv_by_cell = run_R2(df, args.models, seed=args.seed)
        save_results(
            r2,
            out_dir / "results_R2.csv",
            label="R2 spatial block CV (balance='both')",
            provenance={
                **common,
                "split": "splits.spatial_block_cv, 10 sub-blocks merged into 5 folds, balance='both'",
                "aggregates": "fold in {1..5, within_mean, within_sd(ddof=0), pooled}",
                "sensitivity": "dataset 'terrain+DrainageDensity' is session3 config (b)",
            },
        )

        if not args.no_save_models:
            selection = save_final_models(df, cv_by_cell, args.models, args.models_dir, seed=args.seed)
            sel_path = out_dir / "final_model_selection.csv"
            selection.to_csv(sel_path, index=False)
            print(f"[save] selection table -> {sel_path}")

    print(f"[done] {time.time() - t_start:.0f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
