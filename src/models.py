"""Model constructors, each wrapped in a fold-safe pipeline.

Every builder here returns a scikit-learn ``Pipeline`` whose first stage is the
preprocessor from ``features.py``. That is not a stylistic preference: it is what
makes leakage-free cross-validation automatic. A bare estimator handed a matrix
that was imputed and scaled beforehand will produce optimistic numbers no matter
how carefully the splitter was chosen.

Model selection notes
---------------------
With 496 rows, roughly 11 distinct predictors, and 196 positives, this is a
small-data problem. The existing work reaches for a 1D CNN and an LSTM over
tabular covariates, neither of which has a structure to exploit here -- there is
no sequence along the feature axis, so a convolution over column order is
convolving over an arbitrary ordering. They are retained below for
comparability with the manuscript, not because they are expected to win.

The honest expectation is that penalised logistic regression and the tree
ensembles will match or beat the deep models once the splits are corrected, and
that a well-specified baseline is the most useful thing the corrected analysis
can report. Susceptibility mapping also has a downstream requirement that
favours the simpler models: the output is a probability surface that a planner
will threshold, so calibration matters at least as much as discrimination.

The seven models of Session 4
-----------------------------
Three deep architectures carried over verbatim from
``notebooks/landslide_susceptibility_mapping.ipynb`` (cells 47-48) --

===========  ==========================================================
``dnn``      Dense 64-32-16, L2(0.01) on every hidden layer, Dropout(0.2)
``cnn1d``    Conv1D(64, k=3) - Dropout - Flatten - Dense 32 - Dropout - Dense 16
``lstm``     LSTM(128, return_sequences) - LSTM(64) - Dense(32)
===========  ==========================================================

-- all three compiled with ``Adam(learning_rate=1e-3)``,
``binary_crossentropy``, ``epochs=50``, ``batch_size=32``; and four classical
baselines the manuscript never ran: random forest, XGBoost, RBF-kernel SVM, and
penalised logistic regression. Their hyperparameters are stated in each
builder's docstring and collected in ``CLASSICAL_DEFAULTS``, so that "reasonable
defaults" is a checkable claim rather than an assertion.

Fidelity note. The original LSTM was written without L2 or dropout and is kept
that way. Revision 2 gives it the same stratified 20% internal validation split
and ``val_loss`` early stopping as the DNN and 1D-CNN.

Where early stopping looks
--------------------------
``KerasBinaryClassifier`` never sees the test set unless a caller hands it one
explicitly. It takes ``validation_data`` as a fit parameter, and ``fit_pipeline``
is the only sanctioned way to supply it: that helper fits the preprocessor on
the training rows, transforms the validation rows with the already-fitted
preprocessor, and passes the result down. When no validation set is available --
under spatial CV there is no natural one -- the wrapper carves a stratified
``val_fraction`` slice out of the *training fold* instead.

Passing the test set as ``validation_data`` is exactly the manuscript's bug, and
``run.run_R0`` does it deliberately, in one clearly-labelled place, so that the
leak is a visible line of code rather than an accident of plumbing.
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import build_preprocessor

RANDOM_STATE: int = 42

#: The seven models of the Session 4 sweep, in the order the results tables use:
#: the manuscript's own deep architectures first, classical baselines after.
MODEL_NAMES: list[str] = [
    "dnn",
    "cnn1d",
    "lstm",
    "random_forest",
    "xgboost",
    "svm_rbf",
    "logistic",
]

#: Display names for the results tables, so the CSV and the write-up agree.
MODEL_LABELS: dict[str, str] = {
    "dnn": "DNN",
    "cnn1d": "1D-CNN",
    "lstm": "LSTM",
    "random_forest": "RandomForest",
    "xgboost": "XGBoost",
    "svm_rbf": "SVM-RBF",
    "logistic": "LogisticRegression",
}

#: Shared training protocol for the three Keras models, from the notebook.
DEEP_DEFAULTS: dict[str, Any] = {
    "optimizer": "Adam",
    "learning_rate": 1e-3,
    "loss": "binary_crossentropy",
    "epochs": 50,
    "batch_size": 32,
}

#: Hyperparameters for the four classical baselines, with the reasoning kept
#: beside the numbers. Quoted directly by ``results/session4_summary.md``.
CLASSICAL_DEFAULTS: dict[str, dict[str, Any]] = {
    "random_forest": {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None,
        "random_state": RANDOM_STATE,
        "why": (
            "Session 3's forest, carried over unchanged, so the R2 terrain-only "
            "pooled AUC of 0.7567 in session3_final.md has to reappear in "
            "results_R2.csv or one of the two runs is wrong. Depth is "
            "unrestricted because bagging, not pruning, is what controls "
            "variance in a forest."
        ),
    },
    "xgboost": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "why": (
            "The standard small-data boosting recipe: depth 3 admits three-way "
            "interactions and no more, and 300 rounds at 0.05 is the usual "
            "pairing. Row and column subsampling at 0.8 adds the stochasticity "
            "boosting needs to avoid memorising 400 training rows."
        ),
    },
    "svm_rbf": {
        "C": 1.0,
        "gamma": "scale",
        "kernel": "rbf",
        "probability": True,
        "class_weight": None,
        "random_state": RANDOM_STATE,
        "why": (
            "Library defaults. C=1 and gamma='scale' are the only defensible "
            "choice without a nested tuning loop, and a nested loop inside "
            "spatial folds of 41-161 points would be fitting hyperparameters to "
            "noise. probability=True runs Platt scaling on an internal 5-fold "
            "CV of the training fold only, which is what makes an AUC available."
        ),
    },
    "logistic": {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": None,
        "random_state": RANDOM_STATE,
        "why": (
            "The thirty-year reference baseline for susceptibility mapping. L2 "
            "at C=1 on standardised inputs is mild shrinkage, enough to keep "
            "one-hot levels with two or three members from producing separated "
            "coefficients. max_iter is raised only to silence non-convergence "
            "on the wider cluster matrix."
        ),
    },
}

#: Class weighting is off everywhere. The 300/196 split is a 1.53:1 imbalance,
#: mild enough that reweighting mostly distorts the predicted probabilities --
#: and calibrated probabilities, not hard labels, are what a susceptibility map
#: is made of. Stated once here rather than repeated in four docstrings.
CLASS_WEIGHT: str | None = None


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------


def set_global_seeds(seed: int = RANDOM_STATE, deterministic: bool = True) -> None:
    """Seed ``PYTHONHASHSEED``, ``random``, ``numpy``, and TensorFlow.

    Parameters
    ----------
    deterministic:
        Also enable TensorFlow's deterministic op selection. On the CPU-only
        setup this analysis runs on it costs little; on a GPU it can be slow and
        some kernels have no deterministic implementation, so the call is
        guarded and a failure is reported rather than raised.

    A caveat that has to be recorded rather than fixed. ``PYTHONHASHSEED`` is
    read by the interpreter at start-up, so setting it inside a running process
    does not retroactively fix string hash ordering -- it only affects
    subprocesses. ``src/run.py`` sets it in its module preamble, before
    TensorFlow is imported, which is as early as a library can. For a guaranteed
    hash seed, export it in the shell before the run. Nothing in this pipeline is
    known to depend on dict or set iteration order, so this is a belt-and-braces
    measure; it is recorded in the provenance block either way.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as exc:  # pragma: no cover - platform dependent
            print(f"[seed] deterministic ops unavailable: {exc}")


# --------------------------------------------------------------------------
# Keras architectures
# --------------------------------------------------------------------------


def _keras():
    """Import Keras lazily, so that importing this module stays cheap.

    ``import tensorflow`` costs several seconds and emits oneDNN banners. The
    classical half of the sweep does not need it, and neither does anything that
    only wants ``MODEL_NAMES`` or ``CLASSICAL_DEFAULTS``.
    """
    from tensorflow import keras

    return keras


def _compile(model, learning_rate: float):
    keras = _keras()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _dnn_architecture(input_dim: int, dropout: float, l2: float, learning_rate: float):
    """64-32-16 dense stack -- ``build_dnn`` from notebook cell 47, verbatim."""
    keras = _keras()
    L = keras.layers
    reg = keras.regularizers.l2
    model = keras.Sequential(
        [
            L.Input(shape=(input_dim,)),
            L.Dense(64, activation="relu", kernel_regularizer=reg(l2)),
            L.Dropout(dropout),
            L.Dense(32, activation="relu", kernel_regularizer=reg(l2)),
            L.Dropout(dropout),
            L.Dense(16, activation="relu", kernel_regularizer=reg(l2)),
            L.Dense(1, activation="sigmoid"),
        ],
        name="dnn_64_32_16",
    )
    return _compile(model, learning_rate)


def _cnn1d_architecture(
    input_dim: int,
    filters: int,
    kernel_size: int,
    dropout: float,
    l2: float,
    learning_rate: float,
):
    """Conv1D(64, k=3) - Flatten - 32 - 16 -- ``build_1dcnn`` from cell 47."""
    keras = _keras()
    L = keras.layers
    reg = keras.regularizers.l2
    model = keras.Sequential(
        [
            L.Input(shape=(input_dim, 1)),
            L.Conv1D(
                filters,
                kernel_size=kernel_size,
                padding="valid",
                activation="relu",
                kernel_regularizer=reg(l2),
            ),
            L.Dropout(dropout),
            L.Flatten(),
            L.Dense(32, activation="relu", kernel_regularizer=reg(l2)),
            L.Dropout(dropout),
            L.Dense(16, activation="relu", kernel_regularizer=reg(l2)),
            L.Dense(1, activation="sigmoid"),
        ],
        name="cnn1d_64k3_32_16",
    )
    return _compile(model, learning_rate)


def _lstm_architecture(input_dim: int, learning_rate: float):
    """LSTM(128) - LSTM(64) - Dense(32) -- ``build_lstm`` from cell 47.

    Deliberately unregularised, matching the notebook.
    """
    keras = _keras()
    L = keras.layers
    model = keras.Sequential(
        [
            L.Input(shape=(input_dim, 1)),
            L.LSTM(128, return_sequences=True),
            L.LSTM(64),
            L.Dense(32, activation="relu"),
            L.Dense(1, activation="sigmoid"),
        ],
        name="lstm_128_64_32",
    )
    return _compile(model, learning_rate)


class KerasBinaryClassifier(BaseEstimator, ClassifierMixin):
    """scikit-learn adapter for the three Keras architectures.

    Written by hand rather than taken from ``scikeras``, which is not in the
    environment and would not earn its place for something this small. It does
    four things:

    1. Reshapes the preprocessed matrix to ``(n, d, 1)`` for ``cnn1d`` and
       ``lstm``, and leaves it flat for ``dnn``.
    2. Builds the architecture lazily at ``fit`` time, once the fold's feature
       count is known. That count varies between folds -- a one-hot level absent
       from a training fold produces no column -- so the input dimension cannot
       be fixed when the pipeline is constructed.
    3. Routes early stopping at a validation set that is never the test set. See
       ``validation_data`` and ``val_fraction`` below.
    4. Survives ``joblib.dump``. A Keras model is not picklable, so
       ``__getstate__`` serialises it through the ``.keras`` archive format and
       ``__setstate__`` reads it back.

    Parameters
    ----------
    architecture:
        ``"dnn"``, ``"cnn1d"``, or ``"lstm"``.
    early_stopping:
        Whether to fit an ``EarlyStopping`` callback at all. ``True`` for all
        three architectures (the LSTM gained early stopping in Revision 2).
    monitor, patience, restore_best_weights:
        Passed to ``EarlyStopping``. Revision 2: monitor ``val_loss`` (smoother
        than ``val_accuracy`` on a ~80-row validation slice), patience 5,
        restore best weights. ``restore_best_weights`` is
        only a leak when the validation set is the test set; pointed at a
        genuine held-out validation fold it is ordinary model selection.
    val_fraction:
        Used only when ``fit`` receives no ``validation_data``. A stratified
        slice of that size is taken out of the *training* rows and held back for
        early stopping. This is what makes early stopping possible under spatial
        CV, where there is no third partition to draw on.
    """

    def __init__(
        self,
        architecture: str = "dnn",
        dropout: float = 0.2,
        l2: float = 0.01,
        filters: int = 64,
        kernel_size: int = 3,
        learning_rate: float = DEEP_DEFAULTS["learning_rate"],
        epochs: int = DEEP_DEFAULTS["epochs"],
        batch_size: int = DEEP_DEFAULTS["batch_size"],
        early_stopping: bool = True,
        monitor: str = "val_loss",
        patience: int = 5,
        restore_best_weights: bool = True,
        val_fraction: float = 0.2,
        verbose: int = 0,
        random_state: int = RANDOM_STATE,
        sampler: Any = None,
    ) -> None:
        self.architecture = architecture
        self.dropout = dropout
        self.l2 = l2
        self.filters = filters
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.val_fraction = val_fraction
        self.verbose = verbose
        self.random_state = random_state
        self.sampler = sampler

    # -- shape handling ---------------------------------------------------

    def _reshape(self, X) -> np.ndarray:
        X = np.asarray(X, dtype="float32")
        if self.architecture == "dnn":
            return X
        return X.reshape((X.shape[0], X.shape[1], 1))

    def _build(self, input_dim: int):
        if self.architecture == "dnn":
            return _dnn_architecture(input_dim, self.dropout, self.l2, self.learning_rate)
        if self.architecture == "cnn1d":
            return _cnn1d_architecture(
                input_dim,
                self.filters,
                self.kernel_size,
                self.dropout,
                self.l2,
                self.learning_rate,
            )
        if self.architecture == "lstm":
            return _lstm_architecture(input_dim, self.learning_rate)
        raise ValueError(
            f"unknown architecture {self.architecture!r}; "
            "expected one of 'dnn', 'cnn1d', 'lstm'"
        )

    # -- sklearn API ------------------------------------------------------

    def fit(
        self,
        X,
        y,
        validation_data: tuple[Any, Any] | None = None,
    ) -> "KerasBinaryClassifier":
        """Fit the network.

        Parameters
        ----------
        validation_data:
            ``(X_val, y_val)`` **already passed through the fitted
            preprocessor**. ``fit_pipeline`` is the supported way to produce it;
            handing raw frames here would compare an unscaled validation set
            against a scaled training set.

            When ``None`` and ``early_stopping`` is on, a stratified
            ``val_fraction`` slice of ``X`` is used instead.
        """
        keras = _keras()
        keras.utils.set_random_seed(self.random_state)

        X = np.asarray(X, dtype="float32")
        y = np.asarray(y).astype("float32").ravel()
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        X_fit, y_fit = X, y
        val: tuple[np.ndarray, np.ndarray] | None = None

        if validation_data is not None:
            Xv, yv = validation_data
            val = (self._reshape(Xv), np.asarray(yv).astype("float32").ravel())
            self.validation_source_ = "explicit"
        elif self.early_stopping and self.val_fraction and self.val_fraction > 0:
            stratify = y if len(np.unique(y)) > 1 else None
            X_fit, Xv, y_fit, yv = train_test_split(
                X,
                y,
                test_size=self.val_fraction,
                random_state=self.random_state,
                shuffle=True,
                stratify=stratify,
            )
            val = (self._reshape(Xv), yv)
            self.validation_source_ = "inner_split"
        else:
            self.validation_source_ = "none"

        # Oversampling (if any) is applied AFTER the validation split, so the
        # validation set contains real training-fold points only and no
        # synthetic row (or neighbour of one) leaks into early stopping.
        if self.sampler is not None:
            X_fit, y_fit = self.sampler.fit_resample(X_fit, y_fit)
            X_fit = np.asarray(X_fit, dtype="float32")
            y_fit = np.asarray(y_fit).astype("float32").ravel()

        self.model_ = self._build(self.n_features_in_)

        callbacks = []
        if self.early_stopping and val is not None:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor=self.monitor,
                    patience=self.patience,
                    restore_best_weights=self.restore_best_weights,
                )
            )

        self.history_ = self.model_.fit(
            self._reshape(X_fit),
            y_fit,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=val,
            callbacks=callbacks,
            verbose=self.verbose,
        ).history
        self.epochs_run_ = len(self.history_.get("loss", []))
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = self.model_.predict(
            self._reshape(X), batch_size=self.batch_size, verbose=0
        ).ravel()
        return np.column_stack([1.0 - p, p])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    # -- persistence ------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        model = state.pop("model_", None)
        if model is not None:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.keras"
                model.save(path)
                state["_keras_bytes"] = path.read_bytes()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        blob = state.pop("_keras_bytes", None)
        self.__dict__.update(state)
        if blob is not None:
            keras = _keras()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.keras"
                path.write_bytes(blob)
                self.model_ = keras.models.load_model(path)


# --------------------------------------------------------------------------
# Builders -- deep
# --------------------------------------------------------------------------


def _pipeline(estimator, include_cluster: bool, drainage_mode: str) -> Pipeline:
    """Preprocessor + estimator, with the feature set selected by flags.

    ``include_cluster`` and ``drainage_mode`` are the only two feature-set
    switches Session 4 uses: terrain-only (both at their defaults, 23 encoded features),
    terrain + ``Cluster`` (23 + cluster one-hot columns), and the terrain + ``Drainage_Density``
    sensitivity (24 features).
    """
    return Pipeline(
        [
            (
                "pre",
                build_preprocessor(
                    include_cluster=include_cluster, drainage_mode=drainage_mode
                ),
            ),
            ("model", estimator),
        ]
    )


def build_dnn(
    dropout: float = 0.2,
    l2: float = 0.01,
    epochs: int = DEEP_DEFAULTS["epochs"],
    batch_size: int = DEEP_DEFAULTS["batch_size"],
    learning_rate: float = DEEP_DEFAULTS["learning_rate"],
    include_cluster: bool = False,
    drainage_mode: str = "drop",
    random_state: int = RANDOM_STATE,
    **kwargs: Any,
) -> Pipeline:
    """Feed-forward network matching the manuscript's dense architecture.

    Reproduces the 64-32-16 stack with ``Dropout(0.2)`` and ``L2(0.01)`` from
    ``notebooks/landslide_susceptibility_mapping.ipynb``, so that the corrected
    evaluation is measuring the same model the manuscript reports and the
    difference in results is attributable to the splitting and imputation fixes
    alone.

    Roughly 3,500 parameters against 300-450 training rows depending on the
    regime. Early stopping uses a validation slice taken from *within* the
    training data, never the test fold -- selecting the stopping epoch on test
    data is a second leak, distinct from the split leak and just as effective at
    inflating a score. That leak is reproduced only in ``run.run_R0``.
    """
    est = KerasBinaryClassifier(
        architecture="dnn",
        dropout=dropout,
        l2=l2,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping=True,
        random_state=random_state,
        **kwargs,
    )
    return _pipeline(est, include_cluster, drainage_mode)


def build_cnn1d(
    filters: int = 64,
    kernel_size: int = 3,
    dropout: float = 0.2,
    l2: float = 0.01,
    epochs: int = DEEP_DEFAULTS["epochs"],
    batch_size: int = DEEP_DEFAULTS["batch_size"],
    learning_rate: float = DEEP_DEFAULTS["learning_rate"],
    include_cluster: bool = False,
    drainage_mode: str = "drop",
    random_state: int = RANDOM_STATE,
    **kwargs: Any,
) -> Pipeline:
    """1D CNN over the feature vector, as in the manuscript.

    Included for comparability, with a caveat that belongs in the write-up: a 1D
    convolution slides a kernel across adjacent positions and assumes
    neighbouring positions are related. Applied to a tabular row, the
    "neighbours" are whatever columns happen to sit next to each other, so the
    architecture encodes a structure the data does not have. One-hot expansion
    makes it stranger still, scattering a single categorical across several
    adjacent channels -- and the column order here is set by
    ``ColumnTransformer``, which is to say by the order the transformers were
    declared in ``features.build_preprocessor``.

    A genuine CNN for this problem would convolve over **image patches** cut
    from the covariate rasters around each point, which is what the CNN
    susceptibility-mapping literature actually does and what would justify the
    architecture. That requires the source rasters, not this CSV. If they can be
    obtained, it is the single strongest improvement available to the thesis; if
    not, the limitation should be stated plainly rather than papered over.
    """
    est = KerasBinaryClassifier(
        architecture="cnn1d",
        filters=filters,
        kernel_size=kernel_size,
        dropout=dropout,
        l2=l2,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping=True,
        random_state=random_state,
        **kwargs,
    )
    return _pipeline(est, include_cluster, drainage_mode)


def build_lstm(
    epochs: int = DEEP_DEFAULTS["epochs"],
    batch_size: int = DEEP_DEFAULTS["batch_size"],
    learning_rate: float = DEEP_DEFAULTS["learning_rate"],
    include_cluster: bool = False,
    drainage_mode: str = "drop",
    random_state: int = RANDOM_STATE,
    **kwargs: Any,
) -> Pipeline:
    """LSTM 128-64-32 over the feature vector, as in the manuscript.

    The same objection as the CNN, one step stronger. An LSTM reads its input as
    an ordered sequence and carries state from one position to the next; here
    the "sequence" is the column order of a ``ColumnTransformer`` output, so the
    recurrence is modelling the order in which the preprocessing steps were
    declared. It is ~100,000 parameters -- by far the largest model in the sweep
    -- fitted to at most 455 training rows.

    Kept without L2 or dropout, as cell 47 wrote it. Revision 2: same stratified
    20% internal validation split and ``val_loss`` early stopping (patience 5,
    best weights restored) as the DNN and 1D-CNN.
    """
    est = KerasBinaryClassifier(
        architecture="lstm",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping=True,
        val_fraction=0.2,
        random_state=random_state,
        **kwargs,
    )
    return _pipeline(est, include_cluster, drainage_mode)


# --------------------------------------------------------------------------
# Builders -- classical
# --------------------------------------------------------------------------


def build_random_forest(
    n_estimators: int = 500,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    include_cluster: bool = False,
    drainage_mode: str = "drop",
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Random forest -- the strong tabular baseline. 500 trees, seed 42.

    Defaults: ``n_estimators=500``, ``max_depth=None``, ``min_samples_leaf=1``,
    ``max_features="sqrt"``, ``class_weight=None``. These are Session 3's
    settings unchanged, which is deliberate: the R2 terrain-only pooled AUC of
    0.7567 reported in ``session3_final.md`` has to reappear in
    ``results_R2.csv`` or one of the two runs is wrong.

    Handles the non-linear slope/elevation interactions that matter physically,
    needs little tuning, and gives permutation importances that can be compared
    against the logistic coefficients as a consistency check.

    Note for interpretation: impurity-based importances are biased towards
    high-cardinality features, which here would favour the continuous terrain
    variables over the one-hot ``LULC`` and ``Soil_Texture`` levels. Use
    permutation importance on held-out folds instead.
    """
    est = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight=CLASS_WEIGHT,
        random_state=random_state,
        n_jobs=-1,
    )
    return _pipeline(est, include_cluster, drainage_mode)


def build_xgboost(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    include_cluster: bool = False,
    drainage_mode: str = "drop",
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Gradient-boosted trees -- the likely top performer on data this size.

    Defaults: ``n_estimators=300``, ``learning_rate=0.05``, ``max_depth=3``,
    ``subsample=0.8``, ``colsample_bytree=0.8``, ``reg_lambda=1.0``,
    ``min_child_weight=1``, no ``scale_pos_weight``.

    Shallow trees with a low learning rate suit a few hundred rows better than
    any neural architecture. Included as the realistic upper bound on what the
    covariates support, so that the deep models can be assessed against
    something that has actually been specified rather than against a strawman.

    No early stopping and no validation set: the round count is fixed at 300 so
    that every fold fits the identical model, which keeps the fold-to-fold
    spread a property of the data rather than of a stopping rule.
    """
    from xgboost import XGBClassifier

    est = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        min_child_weight=1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )
    return _pipeline(est, include_cluster, drainage_mode)


def build_svm(
    C: float = 1.0,
    gamma: str | float = "scale",
    include_cluster: bool = False,
    drainage_mode: str = "drop",
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """RBF-kernel SVM -- the kernel baseline the susceptibility literature uses.

    Defaults: ``C=1.0``, ``gamma="scale"``, ``kernel="rbf"``,
    ``probability=True``, ``class_weight=None``.

    ``probability=True`` is what makes an AUC available, and it is not free: it
    runs Platt scaling over an internal 5-fold CV of the training fold. That CV
    is nested inside the outer fold, so it sees no held-out data, but it does
    mean the SVM's probabilities come from a different mechanism than every
    other model's and should be read as calibrated ranks rather than raw
    decision values.

    The RBF kernel is scale-sensitive, which the standardisation in
    ``build_preprocessor`` handles -- but only because that scaler is fit inside
    the fold. An SVM on an un-nested scaler is one of the more reliable ways to
    manufacture an optimistic result on data this size.
    """
    est = SVC(
        C=C,
        gamma=gamma,
        kernel="rbf",
        probability=True,
        class_weight=CLASS_WEIGHT,
        random_state=random_state,
    )
    return _pipeline(est, include_cluster, drainage_mode)


def build_logistic(
    C: float = 1.0,
    penalty: str = "l2",
    max_iter: int = 1000,
    include_cluster: bool = False,
    drainage_mode: str = "drop",
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Penalised logistic regression -- the reference baseline.

    Defaults: ``C=1.0``, ``penalty="l2"``, ``solver="lbfgs"``,
    ``max_iter=1000``, ``class_weight=None``.

    The model the literature has used for susceptibility mapping for thirty
    years, and the one any deep model in this manuscript has to beat before its
    complexity is justified. Coefficients are directly interpretable as log-odds
    contributions, which matters for a thesis that needs to say *why* a slope is
    susceptible, not only that it is.
    """
    est = LogisticRegression(
        C=C,
        penalty=penalty,
        solver="lbfgs",
        max_iter=max_iter,
        class_weight=CLASS_WEIGHT,
        random_state=random_state,
    )
    return _pipeline(est, include_cluster, drainage_mode)


# --------------------------------------------------------------------------
# Registry and helpers
# --------------------------------------------------------------------------

_BUILDERS = {
    "dnn": build_dnn,
    "cnn1d": build_cnn1d,
    "lstm": build_lstm,
    "random_forest": build_random_forest,
    "xgboost": build_xgboost,
    "svm_rbf": build_svm,
    "logistic": build_logistic,
}


def get_model(name: str, **kwargs: Any) -> Pipeline:
    """Look up a model builder by name.

    Recognised names are ``MODEL_NAMES``: ``"dnn"``, ``"cnn1d"``, ``"lstm"``,
    ``"random_forest"``, ``"xgboost"``, ``"svm_rbf"``, ``"logistic"``.

    Keyword arguments are forwarded to the builder; in practice the ones that
    matter are ``include_cluster`` and ``drainage_mode``, which select the
    feature set.

    Raises
    ------
    ValueError
        If ``name`` is unrecognised, listing the valid options.
    """
    if name not in _BUILDERS:
        raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
    return _BUILDERS[name](**kwargs)


def is_deep(name: str) -> bool:
    """Whether ``name`` is one of the three Keras architectures."""
    return name in ("dnn", "cnn1d", "lstm")


def fit_pipeline(pipe: Pipeline, X_train, y_train, X_val=None, y_val=None) -> Pipeline:
    """Fit a model pipeline, routing early stopping at ``X_val`` if given.

    This is the only supported way to give a Keras model a validation set, and
    the reason is the preprocessor. ``Pipeline.fit`` would forward a
    ``model__validation_data`` parameter untouched, so the network would see a
    raw, unimputed, unscaled validation frame and stop on a metric computed
    against it. Here the preprocessor is fit on the training rows, the
    validation rows are transformed with that *already-fitted* preprocessor, and
    only then does the estimator see them. Nothing about the validation rows
    influences the imputer, the scaler, the category vocabulary, or the DBSCAN
    clustering.

    For non-Keras estimators ``X_val`` is ignored -- none of the four classical
    models uses a validation set -- and the call reduces to ``pipe.fit``.

    Returns the fitted pipeline.
    """
    estimator = pipe.named_steps["model"]
    if X_val is None or not isinstance(estimator, KerasBinaryClassifier):
        pipe.fit(X_train, y_train)
        return pipe

    pre = pipe.named_steps["pre"]
    Xt = pre.fit_transform(X_train, y_train)
    Xv = pre.transform(X_val)
    estimator.fit(Xt, y_train, validation_data=(Xv, np.asarray(y_val)))
    return pipe


def build_coordinates_only_probe(random_state: int = RANDOM_STATE) -> Pipeline:
    """Fit a model on ``Latitude`` and ``Longitude`` alone -- the leakage detector.

    This is a diagnostic, not a candidate model. Under a sound split it should
    perform near chance, because position by itself carries no physical
    information about slope stability. Under the random split used in the
    manuscript it reaches 0.969 AUC (``session1_data_summary.md`` section 6.3).

    Run it against every splitting strategy and report the results as a table.
    It is the most compact possible demonstration that the blocked splits are
    doing their job, and it converts an argument about methodology into a number
    a reviewer can check.

    Takes the raw cleaned frame like every other pipeline here and selects the
    two coordinate columns itself, so it drops into the same cross-validation
    loop without special-casing.
    """
    from sklearn.compose import ColumnTransformer

    from .data import COORD_COLS

    return Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [("coords", StandardScaler(), list(COORD_COLS))],
                    remainder="drop",
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500, random_state=random_state, n_jobs=-1
                ),
            ),
        ]
    )


class ResampledClassifier(BaseEstimator, ClassifierMixin):
    """Fit ``estimator`` on ``sampler.fit_resample(X, y)``; predict on raw X.

    Used for the non-Keras models in the SMOTE configuration. Keras models take
    the sampler directly (``KerasBinaryClassifier(sampler=...)``) so that the
    internal validation split is drawn before oversampling.
    """

    def __init__(self, estimator=None, sampler=None):
        self.estimator = estimator
        self.sampler = sampler

    def fit(self, X, y):
        from sklearn.base import clone

        Xr, yr = self.sampler.fit_resample(np.asarray(X, dtype=float), np.asarray(y))
        self.estimator_ = clone(self.estimator).fit(Xr, yr)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X, dtype=float))


def count_keras_params(input_dim: int) -> dict[str, int]:
    """Keras ``count_params()`` of the three architectures for ``input_dim`` inputs."""
    return {
        "DNN": int(_dnn_architecture(input_dim, 0.2, 0.01, 1e-3).count_params()),
        "1D-CNN": int(_cnn1d_architecture(input_dim, 64, 3, 0.2, 0.01, 1e-3).count_params()),
        "LSTM": int(_lstm_architecture(input_dim, 1e-3).count_params()),
    }
