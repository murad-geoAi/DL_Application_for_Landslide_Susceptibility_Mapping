"""Third-round revision analyses (A1-A10).  New module; no existing module is modified.

Run from the repository root::

    python -m src.revision_rev3 --part <name> [--seeds 42 0 1 ...]

Heavy parts cache one CSV per (part, seed) in results/revision/extra/rev3/parts/, so a crash loses
at most one seed.  Outputs: results/revision/extra/rev3/, reports/figures/revision/rev3/.

Parts: abl, cw, smotenc, dl_a, dl_b, shap_cnn, light (A2, A3 rescale, A6 summary, A7, A8, A9), figs.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin, clone  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from .data import TARGET  # noqa: E402
from .evaluate import cross_validate_model  # noqa: E402
from .features import (  # noqa: E402
    BINARY_FLAG_COLS,
    CATEGORICAL_COLS,
    CATEGORY_LEVELS,
    FIXED_LAT0,
    FrameEngineer,
    _as_category_strings,
    _ContinuousSelector,
    build_preprocessor,
    build_spatial_imputer,
)
from .models import (  # noqa: E402
    MODEL_LABELS,
    MODEL_NAMES,
    KerasBinaryClassifier,
    fit_pipeline,
    get_model,
    set_global_seeds,
)
from .revision_extra import (  # noqa: E402
    SEEDS_ALL,
    TERRAIN_KW,
    ablated_model,
    cv_row,
    load_df,
    load_metrics,
    ms,
)
from .splits import N_SPLITS, spatial_block_cv, spatial_fold_labels  # noqa: E402

warnings.filterwarnings("ignore")

if os.environ.get("REV3_CPU") == "1":  # avoid several processes contending for the single Metal GPU
    import tensorflow as _tf

    _tf.config.set_visible_devices([], "GPU")
    _tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get("REV3_THREADS", "2")))
    _tf.config.threading.set_inter_op_parallelism_threads(1)

REV = Path("results/revision")
EXTRA = REV / "extra"
OUT = EXTRA / "rev3"
PARTS = OUT / "parts"
FIG = Path("reports/figures/revision/rev3")
LABEL = MODEL_LABELS
ALL7 = list(MODEL_NAMES)
DL3 = ["dnn", "cnn1d", "lstm"]


def cached(part: str, seed, fn):
    PARTS.mkdir(parents=True, exist_ok=True)
    p = PARTS / f"{part}_seed{seed}.csv"
    if p.exists():
        return pd.read_csv(p)
    t = time.time()
    d = fn()
    d.to_csv(p, index=False)
    print(f"[{part}] seed {seed} done in {time.time() - t:.0f}s", flush=True)
    return d


def load_cached(part: str, seeds) -> pd.DataFrame:
    fs = [PARTS / f"{part}_seed{s}.csv" for s in seeds if (PARTS / f"{part}_seed{s}.csv").exists()]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def load_multi(part: str, seeds, tags=("", "_rest", "_lstm")) -> pd.DataFrame:
    ds = [load_cached(part + t, seeds) for t in tags]
    ds = [d for d in ds if len(d)]
    return pd.concat(ds, ignore_index=True) if ds else pd.DataFrame()


TAGOVR = None  # explicit output tag (used for the device-specific DNN reruns)


def tag_of(models) -> str:
    if TAGOVR:
        return "_" + TAGOVR
    return "_lstm" if list(models) == ["lstm"] else "_rest"


def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fmt(m, s, d=3):
    return f"{m:.{d}f} ± {s:.{d}f}"


def baseline(seeds=SEEDS_ALL, dataset="terrain") -> pd.DataFrame:
    met = load_metrics()
    b = met[(met.split == "spatial") & (met.dataset == dataset) & met.seed.isin(seeds)]
    return pd.DataFrame({"seed": b.seed, "model": b.model, "within_AUC": b.within_mean_AUC,
                         "pooled_AUC": b.pooled_AUC})


def agg(d: pd.DataFrame, keys) -> pd.DataFrame:
    rows = []
    for k, g in d.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        r = dict(zip(keys, k))
        r["n_seeds"] = len(g)
        r["within_AUC_mean"], r["within_AUC_sd"] = ms(g.within_AUC)
        r["pooled_AUC_mean"], r["pooled_AUC_sd"] = ms(g.pooled_AUC)
        rows.append(r)
    return pd.DataFrame(rows)


# ==========================================================================
# A1. single-factor ablations (all 7 models x 10 seeds)
# ==========================================================================

ABL3 = {
    "no_ndvi": ["NDVI"],
    "no_lulc": ["LULC"],
    "no_flags_flat": ["TWI_was_missing", "Soil_Texture_was_missing", "is_flat"],
}


def abl_seed(df, seed, models):
    folds = list(spatial_block_cv(df))
    rows = []
    for abl, drop in ABL3.items():
        for name in models:
            set_global_seeds(seed, deterministic=False)
            cv = cross_validate_model(lambda name=name, drop=drop: ablated_model(name, seed, drop), df, folds)
            rows.append(cv_row(seed, name, cv, {"ablation": abl}))
            print(f"[abl seed {seed}] {abl:<14} {LABEL[name]:<18} within={rows[-1]['within_AUC']:.3f} "
                  f"pooled={rows[-1]['pooled_AUC']:.3f}", flush=True)
    return pd.DataFrame(rows)


def abl_summary(seeds):
    d = load_multi("abl", seeds)
    if d.empty:
        return
    d = d.drop_duplicates(["seed", "model", "ablation"])
    b = baseline(seeds).assign(ablation="none")
    d = pd.concat([b, d[b.columns]], ignore_index=True)
    d.to_csv(OUT / "A1_ablation_per_seed.csv", index=False)
    ref = d[d.ablation == "none"].set_index(["seed", "model"])
    rows = []
    for (m, a), g in d.groupby(["model", "ablation"], sort=False):
        g = g.set_index("seed")
        r = {"model": m, "ablation": a, "n_seeds": len(g)}
        r["within_AUC_mean"], r["within_AUC_sd"] = ms(g.within_AUC)
        r["pooled_AUC_mean"], r["pooled_AUC_sd"] = ms(g.pooled_AUC)
        r0 = ref.xs(m, level="model").reindex(g.index)
        for k in ("within", "pooled"):
            r[f"delta_{k}_mean"], r[f"delta_{k}_sd"] = ms((g[f"{k}_AUC"] - r0[f"{k}_AUC"]).dropna())
        rows.append(r)
    out = pd.DataFrame(rows)
    order = {a: i for i, a in enumerate(["none", *ABL3])}
    mo = {LABEL[m]: i for i, m in enumerate(ALL7)}
    out = out.assign(_o=out.ablation.map(order), _m=out.model.map(mo)).sort_values(["_m", "_o"]).drop(columns=["_o", "_m"])
    out.round(4).to_csv(OUT / "A1_ablation_summary.csv", index=False)
    print(out.round(3).to_string())


def part_abl(df, seeds, models):
    for s in seeds:
        if not TAGOVR and (PARTS / f"abl_seed{s}.csv").exists():
            continue
        cached("abl" + tag_of(models), s, lambda s=s: abl_seed(df, s, models))
    abl_summary(SEEDS_ALL)


# ==========================================================================
# A3. class weighting
# ==========================================================================

def _xgb_fit_fn(pipe, X, y):
    npos = float((y == 1).sum())
    pipe.named_steps["model"].set_params(scale_pos_weight=float((y == 0).sum()) / npos)
    return fit_pipeline(pipe, X, y)


def cw_model(name, seed):
    pipe = get_model(name, random_state=seed, **TERRAIN_KW)
    if name in ("random_forest", "logistic", "svm_rbf"):
        pipe.named_steps["model"].set_params(class_weight="balanced")
    return pipe


def cw_seed(df, seed):
    folds = list(spatial_block_cv(df))
    rows = []
    for name in ("random_forest", "xgboost", "svm_rbf", "logistic"):
        set_global_seeds(seed, deterministic=False)
        cv = cross_validate_model(lambda name=name: cw_model(name, seed), df, folds,
                                  fit_fn=_xgb_fit_fn if name == "xgboost" else None)
        rows.append(cv_row(seed, name, cv, {"variant": "class_weight"}))
        print(f"[cw seed {seed}] {LABEL[name]:<18} within={rows[-1]['within_AUC']:.3f} pooled={rows[-1]['pooled_AUC']:.3f}",
              flush=True)
    return pd.DataFrame(rows)


def part_cw(df, seeds):
    for s in seeds:
        cached("cw", s, lambda s=s: cw_seed(df, s))


# ==========================================================================
# A4. SMOTE-NC inside training folds, BEFORE one-hot encoding
# ==========================================================================

class PreEncodingSmoteNC(BaseEstimator, ClassifierMixin):
    """Whole preprocessing + SMOTE-NC + model in one estimator.

    fit(raw frame, y):
      1. FrameEngineer (aspect sin/cos, is_flat) -> KNN-impute + standardise the continuous block
         (same transformers/selector as ``build_preprocessor``), flags passed through, LULC and
         Soil_Texture mapped to integer level codes (``CATEGORY_LEVELS``; 'missing' is its own level).
      2. imblearn SMOTENC(k=5) on [continuous(9) | flags(3) | LULC code | Soil code]; the flags and the two
         code columns are declared categorical (nominal distance penalty, most-frequent-neighbour value).
      3. One-hot encode the two code columns with the fixed schema, giving the same 23-column layout and
         order as the standard terrain preprocessor, then fit the model on real + synthetic rows.
    predict_proba applies the fitted imputer/scaler/encoders (no resampling) and the same one-hot.
    """

    def __init__(self, estimator=None, random_state=42, k_neighbors=5):
        self.estimator = estimator
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def _build(self, X):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

        fe = FrameEngineer(drainage_mode="drop").fit(X)
        frame = fe.transform(X)
        cont = _ContinuousSelector(include_missing_indicators=True, include_coords=False)(frame)
        cont = [c for c in cont if c not in BINARY_FLAG_COLS]
        ct = ColumnTransformer([
            ("continuous", build_spatial_imputer(), cont),
            ("binary", "passthrough", list(BINARY_FLAG_COLS)),
            ("categorical", Pipeline([
                ("levels", FunctionTransformer(_as_category_strings, feature_names_out="one-to-one")),
                ("ord", OrdinalEncoder(categories=[CATEGORY_LEVELS[c] for c in CATEGORICAL_COLS],
                                       handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), list(CATEGORICAL_COLS)),
        ], remainder="drop", verbose_feature_names_out=False)
        return fe, ct, len(cont)

    def _onehot(self, M, n_cont):
        parts = [M[:, :n_cont + len(BINARY_FLAG_COLS)]]
        for j, c in enumerate(CATEGORICAL_COLS):
            codes = np.rint(M[:, n_cont + len(BINARY_FLAG_COLS) + j]).astype(int)
            lv = len(CATEGORY_LEVELS[c])
            oh = np.zeros((len(M), lv))
            ok = (codes >= 0) & (codes < lv)
            oh[np.where(ok)[0], codes[ok]] = 1.0
            parts.append(oh)
        return np.hstack(parts)

    def fit(self, X, y):
        from imblearn.over_sampling import SMOTENC

        self.fe_, self.ct_, self.n_cont_ = self._build(X)
        M = np.asarray(self.ct_.fit_transform(self.fe_.transform(X), y), dtype=float)
        ncat = list(range(self.n_cont_, M.shape[1]))
        Mr, yr = SMOTENC(categorical_features=ncat, k_neighbors=self.k_neighbors,
                         random_state=self.random_state).fit_resample(M, np.asarray(y))
        self.n_real_, self.n_synth_ = len(M), len(Mr) - len(M)
        self.est_ = clone(self.estimator).fit(self._onehot(np.asarray(Mr, float), self.n_cont_), yr)
        self.classes_ = self.est_.classes_
        return self

    def predict_proba(self, X):
        M = np.asarray(self.ct_.transform(self.fe_.transform(X)), dtype=float)
        return self.est_.predict_proba(self._onehot(M, self.n_cont_))


def smotenc_model(name, seed):
    est = get_model(name, random_state=seed, **TERRAIN_KW).named_steps["model"]
    return Pipeline([("model", PreEncodingSmoteNC(est, random_state=seed, k_neighbors=5))])


def smotenc_seed(df, seed):
    folds = list(spatial_block_cv(df))
    rows = []
    for name in ("random_forest", "xgboost", "svm_rbf", "logistic"):
        set_global_seeds(seed, deterministic=False)
        cv = cross_validate_model(lambda name=name: smotenc_model(name, seed), df, folds)
        rows.append(cv_row(seed, name, cv, {"variant": "smotenc"}))
        print(f"[smotenc seed {seed}] {LABEL[name]:<18} within={rows[-1]['within_AUC']:.3f} "
              f"pooled={rows[-1]['pooled_AUC']:.3f}", flush=True)
    return pd.DataFrame(rows)


def part_smotenc(df, seeds):
    for s in seeds:
        cached("smotenc", s, lambda s=s: smotenc_seed(df, s))


# ==========================================================================
# A5. DL sensitivity
# ==========================================================================

def _epochs_info(cv):
    out = []
    for k, pipe in enumerate(cv["fitted"], 1):
        m = pipe.named_steps["model"]
        vl = m.history_.get("val_loss", [])
        out.append({"fold": k, "epochs_run": int(m.epochs_run_),
                    "best_epoch": int(np.argmin(vl) + 1) if len(vl) else np.nan})
    return out


def make_fit_a(seed):
    """(a) hold out the stratified 20% FIRST; fit imputer/scaler/DBSCAN on the remaining 80%; transform the
    20% out of sample; early stopping on that 20% (same subset as the standard run: same y, same seed)."""
    def fit(pipe, X, y):
        y = np.asarray(y)
        i_fit, i_val = train_test_split(np.arange(len(X)), test_size=0.2, random_state=seed, shuffle=True,
                                        stratify=y)
        pre, est = pipe.named_steps["pre"], pipe.named_steps["model"]
        Xt = pre.fit_transform(X.iloc[i_fit], y[i_fit])
        Xv = pre.transform(X.iloc[i_val])
        est.fit(Xt, y[i_fit], validation_data=(Xv, y[i_val]))
        return pipe
    return fit


def make_fit_b(epochs):
    """(b) 100% of the training fold, fixed number of epochs, no early stopping / validation split."""
    def fit(pipe, X, y):
        pipe.named_steps["model"].set_params(early_stopping=False, epochs=int(epochs))
        pipe.fit(X, y)
        return pipe
    return fit


def dl_a_seed(df, seed, models=tuple(DL3)):
    folds = list(spatial_block_cv(df))
    rows, ep = [], []
    for variant, kw in (("standard", TERRAIN_KW), ("a_terrain", TERRAIN_KW),
                        ("a_terrain+Cluster", {"include_cluster": True, "drainage_mode": "drop"})):
        for name in [m for m in DL3 if m in models]:
            set_global_seeds(seed, deterministic=False)
            fit_fn = None if variant == "standard" else make_fit_a(seed)
            cv = cross_validate_model(lambda name=name, kw=kw: get_model(name, random_state=seed, **kw), df, folds,
                                      fit_fn=fit_fn)
            rows.append(cv_row(seed, name, cv, {"variant": variant}))
            for e in _epochs_info(cv):
                ep.append({"seed": seed, "model": LABEL[name], "variant": variant, **e})
            print(f"[dl_a seed {seed}] {variant:<18} {LABEL[name]:<7} within={rows[-1]['within_AUC']:.3f} "
                  f"pooled={rows[-1]['pooled_AUC']:.3f}", flush=True)
    pd.DataFrame(ep).to_csv(PARTS / f"dl_a_epochs{tag_of(models)}_seed{seed}.csv", index=False)
    return pd.DataFrame(rows)


def part_dl_a(df, seeds, models=tuple(DL3)):
    models = [m for m in models if m in DL3]
    for s in seeds:
        if not TAGOVR and (PARTS / f"dl_a_seed{s}.csv").exists():
            continue
        cached("dl_a" + tag_of(models), s, lambda s=s: dl_a_seed(df, s, models))


def fixed_epochs(seeds=SEEDS_ALL) -> dict:
    fs = [f for s in seeds for t in ("", "_rest", "_lstm") for f in [PARTS / f"dl_a_epochs{t}_seed{s}.csv"] if f.exists()]
    e = pd.concat([pd.read_csv(f) for f in fs])
    e = e[e.variant == "standard"]
    med = e.groupby("model").best_epoch.median()
    tab = e.groupby("model").agg(n=("best_epoch", "size"), best_epoch_median=("best_epoch", "median"),
                                 best_epoch_mean=("best_epoch", "mean"), best_epoch_min=("best_epoch", "min"),
                                 best_epoch_max=("best_epoch", "max"), stop_epoch_median=("epochs_run", "median"))
    tab.round(2).to_csv(OUT / "A5_epochs_standard_runs.csv")
    return {m: int(round(v)) for m, v in med.items()}


GPU_ORIGIN_SEEDS_DLA = [42, 0, 3, 4, 6, 7]  # seeds whose first-pass dl_a runs (all 3 models) used the Metal GPU


def fixed_epochs_dnn(device: str) -> int:
    """Median best epoch of the DNN standard runs made on the given device ('gpu' | 'cpu')."""
    fs = []
    if device == "gpu":
        fs += [PARTS / f"dl_a_epochs_seed{s}.csv" for s in GPU_ORIGIN_SEEDS_DLA]
        fs += sorted(PARTS.glob("dl_a_epochs_dnn_gpu_seed*.csv"))
    else:
        fs += sorted(PARTS.glob("dl_a_epochs_rest_seed*.csv")) + sorted(PARTS.glob("dl_a_epochs_dnn_cpu_seed*.csv"))
    e = pd.concat([pd.read_csv(f) for f in fs if f.exists()])
    e = e[(e.variant == "standard") & (e.model == "DNN")].drop_duplicates(["seed", "fold"])
    return int(round(float(e.best_epoch.median())))


def dl_b_seed(df, seed, epochs, models=tuple(DL3)):
    folds = list(spatial_block_cv(df))
    rows = []
    for name in [m for m in DL3 if m in models]:
        set_global_seeds(seed, deterministic=False)
        cv = cross_validate_model(lambda name=name: get_model(name, random_state=seed, **TERRAIN_KW), df, folds,
                                  fit_fn=make_fit_b(epochs[LABEL[name]]))
        rows.append(cv_row(seed, name, cv, {"variant": "b_fixed_epochs", "epochs": epochs[LABEL[name]]}))
        print(f"[dl_b seed {seed}] {LABEL[name]:<7} E={epochs[LABEL[name]]} within={rows[-1]['within_AUC']:.3f} "
              f"pooled={rows[-1]['pooled_AUC']:.3f}", flush=True)
    return pd.DataFrame(rows)


def part_dl_b(df, seeds, models=tuple(DL3)):
    models = [m for m in models if m in DL3]
    if TAGOVR in ("dnn_gpu", "dnn_cpu"):
        ep = {"DNN": fixed_epochs_dnn(TAGOVR[-3:])}
        (OUT / f"A5_fixed_epochs_{TAGOVR}.json").write_text(json.dumps(ep))
    else:
        ep = fixed_epochs()
        (OUT / "A5_fixed_epochs.json").write_text(json.dumps(ep))
    print("fixed epochs", ep)
    for s in seeds:
        cached("dl_b" + tag_of(models), s, lambda s=s: dl_b_seed(df, s, ep, models))


# ==========================================================================
# A6. CNN grouped exact Shapley with the flags as their own group
# ==========================================================================

def group11(name: str) -> str:
    from .revision_analyses import _factor

    if name in ("TWI_was_missing", "Soil_Texture_was_missing", "is_flat"):
        return "Missingness/flat flags"
    return _factor(name)


def group13(name: str) -> str:
    from .revision_analyses import _factor

    return {"TWI_was_missing": "TWI missing flag", "Soil_Texture_was_missing": "Soil missing flag",
            "is_flat": "Flat-cell flag"}.get(name) or _factor(name)


def cnn_shap_seed(df, seed):
    import shap

    from .revision_rev2_shap import grouped_kernel_shap

    set_global_seeds(seed, deterministic=False)
    folds = list(spatial_block_cv(df))
    cv = cross_validate_model(lambda: get_model("cnn1d", random_state=seed, **TERRAIN_KW), df, folds)
    p11, p13 = [], []
    for (tr, te), pipe in zip(folds, cv["fitted"]):
        pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
        Xtr = np.asarray(pre.transform(df.iloc[tr]), dtype=float)
        Xte = np.asarray(pre.transform(df.iloc[te]), dtype=float)
        names = [str(n) for n in pre.get_feature_names_out()]
        bg = shap.kmeans(Xtr, 20)
        for fn, acc in ((group11, p11), (group13, p13)):
            phi, _, _ = grouped_kernel_shap(model, Xte, np.asarray(bg.data, float), bg.weights,
                                            [fn(n) for n in names])
            phi.index = te
            acc.append(phi)
    return pd.concat(p11).sort_index(), pd.concat(p13).sort_index(), float(cv["pooled"]["AUC"])


def part_shap_cnn(df, seeds):
    d = EXTRA / "rev3" / "shap_cnn"
    d.mkdir(parents=True, exist_ok=True)
    for sd in seeds:
        f11, f13 = d / f"cnn_grouped11_seed{sd}.csv", d / f"cnn_grouped13_seed{sd}.csv"
        if f11.exists() and f13.exists():
            continue
        t0 = time.time()
        s11, s13, auc = cnn_shap_seed(df, sd)
        s11.to_csv(f11)
        s13.to_csv(f13)
        old = json.loads((EXTRA / "rev2_shap" / f"cnn_grouped_seed{sd}.json").read_text())["pooled_AUC"]
        (d / f"cnn_grouped_seed{sd}.json").write_text(json.dumps({"seed": sd, "seconds": time.time() - t0,
                                                                  "pooled_AUC": auc, "pooled_AUC_rev2": old}))
        print(f"[shap_cnn] seed {sd} {time.time() - t0:.0f}s AUC {auc:.4f} (rev2 {old:.4f})", flush=True)


# ==========================================================================
# A2 / A3 (rescaling) / A8 / A9
# ==========================================================================

def part_a2():
    t = pd.read_csv(EXTRA / "threshold_metrics.csv")
    t = t[(t.dataset == "terrain") & (t.threshold.astype(str) == "0.5")]
    t = t[["model", "n_seeds", "F1_mean", "F1_sd"]].copy()
    t["F1_mean_pm_sd"] = [fmt(a, b) for a, b in zip(t.F1_mean, t.F1_sd)]
    t.round(4).to_csv(OUT / "A2_pooled_F1_threshold05_terrain.csv", index=False)
    print(t.to_string())


def part_a3_rescale(df, seeds):
    y = df[TARGET].to_numpy(int)
    lab = spatial_fold_labels(df)
    pi = np.array([y[lab != k].mean() for k in range(N_SPLITS)])
    rows, maxdiff = [], 0.0
    for s in seeds:
        o = pd.read_csv(REV / "seeds" / f"oof_seed{s}.csv")
        for name in ALL7:
            p = o[f"{LABEL[name]}|terrain"].to_numpy(float)
            pc = np.clip(p, 1e-15, 1 - 1e-15)
            off = np.log((1 - pi) / pi)[lab]
            z = np.log(pc / (1 - pc)) + off  # logit of the rescaled probability (exact monotone map per fold)
            pt = 1 / (1 + np.exp(-z))
            pt_direct = (pc / pi[lab]) / ((pc / pi[lab]) + (1 - pc) / (1 - pi[lab]))
            assert np.allclose(pt, pt_direct, atol=1e-9)
            w0 = [roc_auc_score(y[lab == k], p[lab == k]) for k in range(N_SPLITS)]
            w1 = [roc_auc_score(y[lab == k], z[lab == k]) for k in range(N_SPLITS)]
            w2 = [roc_auc_score(y[lab == k], pt[lab == k]) for k in range(N_SPLITS)]
            d = max(np.max(np.abs(np.array(w0) - np.array(w1))), np.max(np.abs(np.array(w0) - np.array(w2))))
            maxdiff = max(maxdiff, d)
            assert d < 1e-6, (s, name, d)
            rows.append({"seed": s, "model": LABEL[name], "within_plain": np.mean(w0), "within_rescaled": np.mean(w2),
                         "pooled_plain": roc_auc_score(y, p), "pooled_rescaled": roc_auc_score(y, pt),
                         "pooled_rescaled_logit_exact": roc_auc_score(y, z)})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "A3_rescale_per_seed.csv", index=False)
    sm = baseline(seeds, "terrain+SMOTE").rename(columns={"within_AUC": "within_smote", "pooled_AUC": "pooled_smote"})
    d = d.merge(sm, on=["seed", "model"], how="left")
    mo = {LABEL[m]: i for i, m in enumerate(ALL7)}
    out = []
    for m, g in d.groupby("model", sort=False):
        r = {"model": m, "n_seeds": len(g)}
        for c in ("within_plain", "within_rescaled", "pooled_plain", "pooled_rescaled", "pooled_smote", "within_smote"):
            r[c + "_mean"], r[c + "_sd"] = ms(g[c])
        out.append(r)
    out = pd.DataFrame(out).assign(_m=lambda x: x.model.map(mo)).sort_values("_m").drop(columns="_m")
    out.round(4).to_csv(OUT / "A3_rescale_summary.csv", index=False)
    (OUT / "A3_rescale_meta.json").write_text(json.dumps({
        "pi_k_training_fold_share": {f"fold{k + 1}": float(pi[k]) for k in range(N_SPLITS)},
        "max_abs_within_fold_AUC_diff_over_all_model_seed_fold": float(maxdiff)}, indent=1))
    print(out.round(3).to_string(), "\nmax within-fold diff", maxdiff)


def part_summ_variants(seeds):
    """A3 class weights, A4 SMOTE-NC, A5 DL sensitivities: summaries vs the existing primary values."""
    ter, smo = baseline(seeds, "terrain"), baseline(seeds, "terrain+SMOTE")
    ter_c = baseline(seeds, "terrain+Cluster")
    for part, name, ref in (("cw", "A3_class_weight", True), ("smotenc", "A4_smotenc", True)):
        d = load_cached(part, seeds)
        if d.empty:
            continue
        d.to_csv(OUT / f"{name}_per_seed.csv", index=False)
        s = agg(d, ["model"]).rename(columns={c: c.replace("AUC_", "AUC_variant_") for c in []})
        t = agg(ter[ter.model.isin(s.model)], ["model"]).add_prefix("terrain_").rename(columns={"terrain_model": "model"})
        m = agg(smo[smo.model.isin(s.model)], ["model"]).add_prefix("smote_").rename(columns={"smote_model": "model"})
        o = s.merge(t.drop(columns="terrain_n_seeds"), on="model").merge(m.drop(columns="smote_n_seeds"), on="model")
        o.round(4).to_csv(OUT / f"{name}_summary.csv", index=False)
        print(name, "\n", o.round(3).to_string())
    d = load_multi("dl_a", seeds).drop_duplicates(["seed", "model", "variant"])
    if not d.empty:
        d.to_csv(OUT / "A5_dl_per_seed.csv", index=False)
        b = load_multi("dl_b", seeds)
        if not b.empty:
            d = pd.concat([d, b[d.columns.intersection(b.columns)]], ignore_index=True)
        s = agg(d, ["model", "variant"])
        pr = pd.concat([ter.assign(variant="primary_terrain_existing"),
                        ter_c.assign(variant="primary_terrain+Cluster_existing")])
        pr = pr[pr.model.isin(["DNN", "1D-CNN", "LSTM"])]
        s = pd.concat([s, agg(pr, ["model", "variant"])], ignore_index=True)
        mo = {"DNN": 0, "1D-CNN": 1, "LSTM": 2}
        s = s.assign(_m=s.model.map(mo)).sort_values(["_m", "variant"]).drop(columns="_m")
        s.round(4).to_csv(OUT / "A5_dl_summary.csv", index=False)
        print(s.round(3).to_string())


def part_a8(df):
    y = df[TARGET].to_numpy(int)
    lab = spatial_fold_labels(df)
    met = load_metrics()
    m = met[(met.split == "spatial") & (met.dataset == "terrain")]
    rows = []
    for name in ALL7:
        g = m[m.model == LABEL[name]]
        for k in range(N_SPLITS):
            npos, nneg = int(y[lab == k].sum()), int((lab == k).sum() - y[lab == k].sum())
            v = g[f"fold{k + 1}_AUC"]
            rows.append({"model": LABEL[name], "fold": k + 1, "n_pos": npos, "n_neg": nneg, "pairs": npos * nneg,
                         "AUC_mean_over_seeds": v.mean(), "AUC_sd_over_seeds": v.std(ddof=1), "n_seeds": len(v)})
    d = pd.DataFrame(rows)
    d.round(4).to_csv(OUT / "A8_per_fold_AUC_terrain.csv", index=False)
    print(d.pivot(index="model", columns="fold", values="AUC_mean_over_seeds").round(3).to_string())
    return d


def part_a9(df):
    from .splits import assign_blocks_to_folds, spatial_block_labels

    y = df[TARGET].to_numpy(int)
    blocks = spatial_block_labels(df, n_sub_blocks=10)
    sizes = {int(b): int((blocks == b).sum()) for b in np.unique(blocks)}
    pos = {int(b): int(y[blocks == b].sum()) for b in np.unique(blocks)}
    fob = assign_blocks_to_folds(sizes, pos, n_splits=N_SPLITS, balance="both")
    order = sorted(sizes, key=lambda b: (-pos[b], -sizes[b], b))
    rows = [{"block": b, "n": sizes[b], "n_pos": pos[b], "fold": fob[b] + 1,
             "greedy_order": order.index(b) + 1,
             "centroid_lat": float(df.Latitude[blocks == b].mean()), "centroid_lon": float(df.Longitude[blocks == b].mean())}
            for b in sorted(sizes)]
    t = pd.DataFrame(rows).sort_values(["fold", "greedy_order"])
    t.round(4).to_csv(OUT / "A9_blocks_to_folds.csv", index=False)
    lab = spatial_fold_labels(df)
    assert all(fob[int(b)] == lab[i] for i, b in enumerate(blocks))
    comp = t.groupby("fold").agg(blocks=("block", lambda s: ",".join(map(str, s))), n=("n", "sum"), n_pos=("n_pos", "sum"))
    print(t.to_string(), "\n", comp.to_string())
    return t, comp


# ==========================================================================
# A6. SHAP re-aggregation
# ==========================================================================

def _grp_share(S: pd.DataFrame, fn) -> pd.Series:
    g = S.T.groupby([fn(c) for c in S.columns]).sum().T
    imp = g.abs().mean()
    return imp / imp.sum()


def rf_shap_by_seed(df, seeds):
    """Per-seed RF OOF TreeSHAP shares for 10-group (old), 11-group (flags separate) and 13-group groupings."""
    from .revision_analyses import _factor
    from .revision_rev2_shap import _tree_oof

    f = OUT / "A6_rf_shares_by_seed.csv"
    if f.exists():
        return pd.read_csv(f)
    rows = []
    for sd in seeds:
        S, auc = _tree_oof(df, sd, False)
        for tag, fn in (("g10", _factor), ("g11", group11), ("g13", group13)):
            sh = _grp_share(S, fn)
            for k, v in sh.items():
                rows.append({"seed": sd, "grouping": tag, "group": k, "share": v})
        pc = S.abs().mean()
        pc = pc / pc.sum()
        for k, v in pc.items():
            rows.append({"seed": sd, "grouping": "per_column", "group": k, "share": v})
        print(f"[rf shap] seed {sd} AUC {auc:.4f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f, index=False)
    return d


def part_a6(df, seeds):
    rf = rf_shap_by_seed(df, seeds)
    old = pd.read_csv(REV / "shap_share_by_seed_random_forest.csv", index_col=0)
    g10 = rf[rf.grouping == "g10"].pivot(index="seed", columns="group", values="share")
    chk = float((g10.loc[old.index, old.columns] - old).abs().max().max())
    print("RF g10 recomputed vs stored per-seed shares, max abs diff:", chk)
    d = OUT / "shap_cnn"
    cn = {}
    for tag in ("11", "13"):
        r = {}
        for sd in seeds:
            f = d / f"cnn_grouped{tag}_seed{sd}.csv"
            if f.exists():
                imp = pd.read_csv(f, index_col=0).abs().mean()
                r[sd] = imp / imp.sum()
        cn[tag] = pd.DataFrame(r).T.sort_index() if r else pd.DataFrame()
    cn10 = {}
    for sd in seeds:
        f = EXTRA / "rev2_shap" / f"cnn_grouped10_seed{sd}.csv"
        imp = pd.read_csv(f, index_col=0).abs().mean()
        cn10[sd] = imp / imp.sum()
    cn10 = pd.DataFrame(cn10).T.sort_index()
    rows = []

    def add(model, tag, T):
        for c in T.columns:
            rows.append({"model": model, "grouping": tag, "group": c, "n_seeds": len(T),
                         "share_pct_mean": 100 * T[c].mean(), "share_pct_sd": 100 * T[c].std(ddof=1)})

    for tag, nm in (("g10", "10 factors (flags inside parents; Fig. 14a v2)"), ("g11", "11 groups (flags separate)"),
                    ("g13", "13 groups (3 flags separate)")):
        T = rf[rf.grouping == tag].pivot(index="seed", columns="group", values="share")
        add("RF", nm, T)
    add("1D-CNN", "10 factors (flags inside parents; Fig. 14a v2)", cn10)
    if len(cn["11"]):
        add("1D-CNN", "11 groups (flags separate)", cn["11"])
    if len(cn["13"]):
        add("1D-CNN", "13 groups (3 flags separate)", cn["13"])
    pc = rf[rf.grouping == "per_column"].pivot(index="seed", columns="group", values="share")
    add("RF", "per encoded column (sum of |SHAP| per column)", pc)
    t = pd.DataFrame(rows)
    t.round(4).to_csv(OUT / "A6_shap_shares.csv", index=False)
    return t, chk, rf, cn, cn10


def fig14a_rev3(rf, cn11):
    g11 = rf[rf.grouping == "g11"].pivot(index="seed", columns="group", values="share")
    order = g11.mean().sort_values(ascending=False).index
    imp = pd.DataFrame({"RF": g11.mean(), "CNN": cn11.mean()}).loc[order]
    sd = pd.DataFrame({"RF": g11.std(ddof=1), "CNN": cn11.std(ddof=1)}).loc[order]
    fa, aa = plt.subplots(figsize=(5.7, 5))
    ip, isd = imp.iloc[::-1] * 100, sd.iloc[::-1] * 100
    y = np.arange(len(ip))
    for off, m, col, nm in ((0.2, "RF", "#2171b5", f"RF ({len(g11)} seeds)"),
                            (-0.2, "CNN", "#fd8d3c", f"1D-CNN, grouped exact Shapley ({len(cn11)} seeds)")):
        v, e = ip[m], isd[m]
        aa.barh(y + off, v, 0.4, xerr=e, color=col, label=nm, error_kw={"lw": 0.7, "capsize": 1.5})
        for yi, vi, ei in zip(y + off, v, e):
            aa.text(vi + ei + 0.8, yi, f"{vi:.1f}%", va="center", fontsize=8)
    aa.set_xlim(0, float((ip.values + isd.values).max()) + 5)
    aa.set_yticks(y)
    aa.set_yticklabels(ip.index)
    aa.set_xlabel("Share of total mean |SHAP value| (%), mean ± SD over seeds")
    aa.set_title("(a) Factor importance, out-of-fold (spatial CV)", fontsize=10)
    aa.legend(fontsize=7.5, loc="lower right")
    savefig(fa, FIG / "fig14a_shap_importance_rev3.png")


# ==========================================================================
# A7. AOA variant with all encoded predictors, in-fold SHAP weights
# ==========================================================================

def infold_weights_cols(df, lab, seeds=SEEDS_ALL):
    import shap

    from .revision_analyses import factor_importance
    from .revision_extra import DI_FEATURES, IMP_NAME

    f1, f2 = OUT / "A7_infold_weights_23col.csv", OUT / "A7_infold_weights_7factor.csv"
    if f1.exists() and f2.exists():
        return pd.read_csv(f1, index_col=0), pd.read_csv(f2, index_col=0)
    y = df[TARGET].to_numpy(int)
    W23, W7 = {}, {}
    for k in range(N_SPLITS):
        tr = np.where(lab != k)[0]
        c23, c7 = [], []
        for sd in seeds:
            set_global_seeds(sd, deterministic=False)
            pipe = get_model("random_forest", random_state=sd, **TERRAIN_KW)
            pipe.fit(df.iloc[tr], y[tr])
            pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
            Xtr = np.asarray(pre.transform(df.iloc[tr]), dtype=float)
            names = [str(n) for n in pre.get_feature_names_out()]
            sv = shap.TreeExplainer(model).shap_values(Xtr)
            sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            S = pd.DataFrame(sv, columns=names)
            c23.append(S.abs().mean().to_numpy())
            imp = factor_importance(S)
            c7.append(np.array([float(imp[IMP_NAME.get(c, c)]) for c in DI_FEATURES]))
        w = np.mean(c23, axis=0)
        W23[f"fold{k + 1}"] = pd.Series(w / w.sum() * len(w), index=names)
        w = np.mean(c7, axis=0)
        W7[f"fold{k + 1}"] = pd.Series(w / w.sum() * len(w), index=DI_FEATURES)
        print(f"[aoa weights] fold {k + 1} done", flush=True)
    W23, W7 = pd.DataFrame(W23).T, pd.DataFrame(W7).T
    W23.to_csv(f1)
    W7.to_csv(f2)
    return W23, W7


def part_a7(df):
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import pdist

    from .revision_extra import DI_FEATURES, compute_di

    y = df[TARGET].to_numpy(int)
    lab = spatial_fold_labels(df)
    W23, W7 = infold_weights_cols(df, lab)
    mp = pd.read_csv(EXTRA / "map_points.csv")
    assert np.allclose(mp.Latitude, df.Latitude) and np.allclose(mp.Longitude, df.Longitude)
    east = ((y == 0) & df.Latitude.between(22.55, 22.80).to_numpy() & df.Longitude.between(92.25, 92.45).to_numpy())
    east16 = east & mp["class"].isin(["High", "Very high"]).to_numpy()
    assert east.sum() == 33 and east16.sum() == 16, (east.sum(), east16.sum())

    def di_enc(W, standardise):
        di, thr = np.full(len(df), np.nan), np.zeros(N_SPLITS)
        for k in range(N_SPLITS):
            tr, te = np.where(lab != k)[0], np.where(lab == k)[0]
            pre = build_preprocessor(include_cluster=False, drainage_mode="drop")
            pre.fit(df.iloc[tr], y[tr])
            Z = np.asarray(pre.transform(df), float)
            if standardise:
                sd_ = Z[tr].std(0)
                sd_[sd_ == 0] = 1.0
                Z = (Z - Z[tr].mean(0)) / sd_
            Z = Z * (W if np.ndim(W) == 1 else W[k])
            Ztr = Z[tr]
            dbar = pdist(Ztr).mean()
            di[te] = cKDTree(Ztr).query(Z[te], k=1)[0] / dbar
            dtr = np.empty(len(tr))
            for f in set(range(N_SPLITS)) - {k}:
                mm = lab[tr] == f
                dtr[mm] = cKDTree(Ztr[~mm]).query(Ztr[mm], k=1)[0] / dbar
            q1, q3 = np.percentile(dtr, [25, 75])
            thr[k] = q3 + 1.5 * (q3 - q1)
        return di > thr[lab]

    variants = {}
    di, thr, _ = compute_di(df, lab, W7.to_numpy())
    variants["infold_7continuous_factor_weights (adopted)"] = di > thr[lab]
    variants["infold_all23_encoded_col_weights_standardised"] = di_enc(W23.to_numpy(), True)
    variants["infold_all23_encoded_col_weights_as_preprocessed"] = di_enc(W23.to_numpy(), False)
    variants["infold_all23_unweighted_standardised (no SHAP; reference)"] = di_enc(np.ones(23), True)
    rows = []
    for nm, o in variants.items():
        rows.append({"variant": nm, "n_outside_AOA": int(o.sum()), "by_fold": ",".join(str(int(o[lab == k].sum())) for k in range(5)),
                     "landslides_outside": int(o[y == 1].sum()), "non_landslides_outside": int(o[y == 0].sum()),
                     "outside_of_33_east_negatives": int(o[east].sum()), "outside_of_16_high_VH_east_negatives": int(o[east16].sum()),
                     "idx_of_16_outside": ";".join(map(str, np.where(o & east16)[0]))})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "A7_aoa_infold_all_predictors.csv", index=False)
    print(t.drop(columns="idx_of_16_outside").to_string())
    return t


# ==========================================================================
# A10. figures (plotting code copied from revision_analyses / revision_rev2_shap; US spelling)
# ==========================================================================

def _scale_bar(ax, km, lat, x0, y0, fs=8):
    from .revision_rev2_shap import scale_bar

    scale_bar(ax, km, lat, x0, y0, fs=fs)


def fig08_rev3(df):
    from scipy.spatial import Voronoi
    from shapely.geometry import Polygon

    from .revision_analyses import SEED, district_polygon, to_km, voronoi_areas

    res = json.loads((REV / "sparsity_metrics.json").read_text())
    xy_all = to_km(df.Latitude.to_numpy(), df.Longitude.to_numpy(), FIXED_LAT0)
    pos = df[TARGET].to_numpy() == 1
    hull = district_polygon(FIXED_LAT0)
    a_pos = voronoi_areas(xy_all[pos], xy_all, hull)
    jitter = np.random.default_rng(SEED).normal(0, 1e-6, xy_all[pos].shape)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    span = 1000
    c = xy_all.mean(axis=0)
    far = np.array([[c[0] - span, c[1] - span], [c[0] + span, c[1] - span],
                    [c[0] + span, c[1] + span], [c[0] - span, c[1] + span]])
    vor = Voronoi(np.vstack([xy_all[pos] + jitter, far]))
    lg = np.log10(a_pos)
    norm = plt.Normalize(np.nanmin(lg), np.nanmax(lg))
    cmap = plt.get_cmap("viridis")
    for i in range(pos.sum()):
        region = vor.regions[vor.point_region[i]]
        if -1 in region or not region or not np.isfinite(lg[i]):
            continue
        poly = Polygon(vor.vertices[region]).intersection(hull)
        if poly.is_empty:
            continue
        for gg in getattr(poly, "geoms", [poly]):
            px, py = gg.exterior.xy
            ax.fill(px, py, color=cmap(norm(lg[i])), ec="k", lw=0.2)
    for gg in getattr(hull, "geoms", [hull]):
        ax.plot(*gg.exterior.xy, c="k", lw=0.6)
    ax.scatter(*xy_all[~pos].T, s=16, c="0.15", marker="x", lw=0.9, label="Non-landslide sample")
    ax.scatter(*xy_all[pos].T, s=4, c="r", label="Landslide inventory point")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.8)
    cb.set_label("log$_{10}$ Voronoi cell area (km$^2$)")
    ax.set_xlabel("Easting (km, local projection)")
    ax.set_ylabel("Northing (km, local projection)")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    ax.set_title("(a) Voronoi tessellation of the landslide inventory", fontsize=10)
    # added in rev3: metric scale bar and north arrow (axes are km, so the bar is exact)
    from matplotlib.patches import Rectangle

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    L = 25.0
    bx, by = x0 + 0.05 * (x1 - x0), y1 - 0.08 * (y1 - y0)
    h = 0.012 * (y1 - y0)
    ax.add_patch(Rectangle((bx, by), L / 2, h, fc="k", ec="k", lw=0.6, zorder=6))
    ax.add_patch(Rectangle((bx + L / 2, by), L / 2, h, fc="w", ec="k", lw=0.6, zorder=6))
    ax.text(bx + L / 2, by + 1.8 * h, "25 km", ha="center", va="bottom", fontsize=8, zorder=6,
            bbox=dict(fc="w", ec="none", alpha=0.7, pad=0.8))
    nx, ny0, ny1 = x1 - 0.07 * (x1 - x0), y1 - 0.14 * (y1 - y0), y1 - 0.05 * (y1 - y0)
    ax.annotate("", xy=(nx, ny1), xytext=(nx, ny0), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.3), zorder=6)
    ax.text(nx, ny1 + 0.008 * (y1 - y0), "N", ha="center", va="bottom", fontsize=10, fontweight="bold", zorder=6)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax = axes[1]
    a = a_pos[np.isfinite(a_pos)]
    ax.hist(np.log10(a), bins=30, color="#3b6fb6", ec="w")
    ax.set_xlabel("log$_{10}$ Voronoi cell area (km$^2$)")
    ax.set_ylabel("Number of cells")
    ce = res["clark_evans_landslides_district"]
    ax.set_title("(b) Cell-area distribution", fontsize=10)
    ax.text(0.03, 0.97,
            f"n = {len(a)}\nCV = {res['voronoi_landslides']['cv']:.2f}\n"
            f"P90/P10 = {res['voronoi_landslides']['p90_p10_ratio']:.0f}\n"
            f"Clark-Evans R = {ce['R']:.2f} (z = {ce['z']:.1f})",
            transform=ax.transAxes, va="top", fontsize=9, bbox=dict(fc="w", ec="0.7"))
    savefig(fig, FIG / "fig08_voronoi_quantified_rev3.png")


def fig09_rev3(df):
    from matplotlib.lines import Line2D

    from .revision_analyses import district_polygon
    from .revision_rev2_shap import gi_classification, km_to_deg, ref_circle, scale_bar

    BAND = 5.5
    cls = gi_classification(df, BAND)
    cat, isl = cls["cat"], cls["isl"]
    poly = district_polygon()
    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    for gg in getattr(poly, "geoms", [poly]):
        ax.plot(*gg.exterior.xy, c="k", lw=0.6)
    for c, col, lab in [(0, "#2c7bb6", f"Cold spot (Gi* < 0, {cls['crit']})"),
                        (1, "#bdbdbd", f"Not significant ({cls['ns']})"),
                        (2, "#d7191c", f"Hot spot (Gi* > 0, {cls['crit']})")]:
        m = (cat == c) & ~isl
        ax.scatter(df.Longitude[m], df.Latitude[m], s=12, c=col, label=f"{lab}, n = {m.sum()}", ec="k", lw=0.2)
    ax.scatter(df.Longitude[isl], df.Latitude[isl], s=30, facecolors="none", edgecolors="k", marker="s",
               label=f"No neighbor within {BAND} km (island), n = {isl.sum()}")
    ax.set_aspect(1 / np.cos(np.deg2rad(float(df.Latitude.mean()))))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Local Getis–Ord Gi* of the landslide label\n(distance band {BAND} km, {cls['subtitle']})", fontsize=10)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    scale_bar(ax, 25, 22.3, x0 + 0.10 * (x1 - x0), y0 + 0.05 * (y1 - y0))
    scale_bar(ax, 10, 22.3, x0 + 0.10 * (x1 - x0), y0 + 0.13 * (y1 - y0))
    _, rdl = km_to_deg(BAND, 22.3)
    ref_circle(ax, BAND, 22.3, x0 + 0.62 * (x1 - x0), y0 + 0.08 * (y1 - y0))
    ax.text(x0 + 0.62 * (x1 - x0), y0 + 0.08 * (y1 - y0) + 1.15 * rdl, f"{BAND} km", ha="center", fontsize=8,
            bbox=dict(fc="w", ec="none", alpha=0.7, pad=0.8))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], ls="--", c="k", lw=0.9, label=f"{BAND} km Gi* neighborhood radius (circle on map)"))
    ax.legend(handles=h, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0)
    savefig(fig, FIG / "fig09_gistar_rev3.png")


def fig10_rev3(df):
    from esda.moran import Moran
    from libpysal.weights import KNN

    from .revision_analyses import SEED, to_km

    xy_all = to_km(df.Latitude.to_numpy(), df.Longitude.to_numpy(), FIXED_LAT0)
    yv = df[TARGET].to_numpy(dtype=float)
    w = KNN.from_array(xy_all, k=5)
    w.transform = "r"
    np.random.seed(SEED)
    mor = Moran(yv, w, permutations=999)
    z = (yv - yv.mean()) / yv.std(ddof=0)
    lag = np.asarray(w.sparse @ z).ravel()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    jit = np.random.default_rng(SEED).normal(0, 0.03, len(z))
    ax.scatter(z + jit, lag, s=10, alpha=0.5, c="#3b6fb6")
    xs = np.linspace(z.min() - 0.2, z.max() + 0.2, 10)
    ax.plot(xs, mor.I * xs, c="#b2182b", lw=1.5)
    ax.axhline(0, c="0.4", ls="--", lw=0.7)
    ax.axvline(0, c="0.4", ls="--", lw=0.7)
    ax.set_xlabel("Standardized landslide label (0 = non-landslide, 1 = landslide)")
    ax.set_ylabel("Spatial lag of standardized label (k = 5 neighbors)")
    ax.set_title("Moran's I scatter plot of the landslide label", fontsize=10)
    ax.text(0.03, 0.97, f"Moran's I = {mor.I:.3f}\nE[I] = {mor.EI:.3f}\npseudo p = {mor.p_sim:.3f} (999 perm.)\nn = {len(z)}",
            transform=ax.transAxes, va="top", fontsize=8.5, bbox=dict(fc="w", ec="0.7"))
    savefig(fig, FIG / "fig10_moran_rev3.png")
    print("Moran", mor.I, mor.EI, mor.p_sim)


def fig11_rev3(df):
    import geopandas as gpd
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    from scipy.spatial import cKDTree
    from sklearn.cluster import DBSCAN

    from .features import DBSCAN_EPS, DBSCAN_MIN_SAMPLES
    from .revision_analyses import BOUNDARY_FILE, to_km
    from .revision_rev2_shap import km_to_deg, ref_circle, scale_bar

    xy = to_km(df["Latitude"].to_numpy(), df["Longitude"].to_numpy(), FIXED_LAT0)
    y = df[TARGET].to_numpy()
    d, _ = cKDTree(xy).query(xy, k=6)

    def knee_of(kd):
        xs = np.arange(len(kd))
        p1, p2 = np.array([0, kd[0]]), np.array([len(kd) - 1, kd[-1]])
        num = np.abs((p2[1] - p1[1]) * xs - (p2[0] - p1[0]) * kd + p2[0] * p1[1] - p2[1] * p1[0])
        return float(kd[int(np.argmax(num / np.linalg.norm(p2 - p1)))])

    col = 5
    try:
        target = json.loads((REV / "dbscan_rule_audit.json").read_text())["k_distance_knee_km"]
        errs = {c: abs(knee_of(np.sort(d[:, c])) - target) for c in (5, 4)}
        col = min(errs, key=errs.get)
    except Exception as e:
        print("[fig11] audit knee not readable:", repr(e))
    kd = np.sort(d[:, col])
    eps_knee_km = knee_of(kd)
    kth = "5th" if col == 5 else "4th"
    lab = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(xy)
    nc = int(lab.max()) + 1
    cmap = ListedColormap([plt.get_cmap("tab10")(i) for i in range(nc)])
    norm = BoundaryNorm(np.arange(-0.5, nc, 1.0), cmap.N)
    lon, lat = df["Longitude"].to_numpy(), df["Latitude"].to_numpy()
    bnd = gpd.read_file(BOUNDARY_FILE).geometry.union_all()

    fig, axes = plt.subplots(1, 2, figsize=(12.75, 5.65), gridspec_kw={"width_ratios": [1.0, 0.95]})
    ax = axes[0]
    ax.plot(kd, color="#3b6fb6")
    ax.axhline(DBSCAN_EPS, ls="--", c="r", lw=1, label=f"eps = {DBSCAN_EPS} km (adopted)")
    ax.axhline(eps_knee_km, ls=":", c="k", lw=1, label=f"knee = {eps_knee_km:.1f} km")
    ax.set_xlabel(f"Points sorted by {kth}-nearest-neighbor distance")
    ax.set_ylabel(f"{kth}-nearest-neighbor distance (km)")
    ax.set_title("(a) k-distance curve (4th nearest neighbor, k = 5 incl. self)" if col == 4 else
                 "(a) k-distance curve (5th nearest neighbor, k = 5 excl. self)", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]

    def draw(a, s_pt, s_ring, s_x, lw_x, label=False):
        sc = a.scatter(lon[lab >= 0], lat[lab >= 0], c=lab[lab >= 0], cmap=cmap, norm=norm, s=s_pt, zorder=3)
        a.scatter(lon[lab < 0], lat[lab < 0], c="0.2", s=s_x, marker="x", lw=lw_x, zorder=3,
                  label="Noise (label −1)" if label else None)
        a.scatter(lon[y == 1], lat[y == 1], facecolors="none", edgecolors="k", s=s_ring, lw=0.4, zorder=3,
                  label="Landslide inventory point" if label else None)
        return sc

    for poly in getattr(bnd, "geoms", [bnd]):
        ax.plot(*poly.exterior.xy, color="0.25", lw=0.8, zorder=1)
        for r in poly.interiors:
            ax.plot(*r.xy, color="0.25", lw=0.5, zorder=1)
    sc = draw(ax, 14, 22, 16, 0.9, label=True)
    cb = fig.colorbar(sc, ax=ax, ticks=range(nc), fraction=0.05, pad=0.02)
    cb.set_label("DBSCAN cluster ID")
    ax.set_aspect(1 / np.cos(np.deg2rad(FIXED_LAT0)))
    X0, X1, Y0, Y1 = 90.80, 92.70, 21.78, 23.82
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("(b) DBSCAN clusters (eps = 1.1 km, min_samples = 5)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    # 10 km scale bar for the main panel, correct at its own latitude (22.1 N)
    scale_bar(ax, 10, 22.1, 92.22, 21.90, fs=8)

    # inset: west-central cluster corridor, drawn in the empty western margin (outside the district polygon)
    zx, zy = (91.985, 92.235), (22.45, 22.90)
    wid_deg = 0.62                                    # inset width in longitude degrees (data units of the main axes)
    ratio = (zx[1] - zx[0]) / ((zy[1] - zy[0]) * (1 / np.cos(np.deg2rad(FIXED_LAT0))))   # display w/h of the zoom box
    hgt_deg = wid_deg / ratio / (1 / np.cos(np.deg2rad(FIXED_LAT0)))                         # in latitude degrees
    ix0, iy0 = 91.06, 22.18
    ins = ax.inset_axes([ix0, iy0, wid_deg, hgt_deg], transform=ax.transData, zorder=10)
    ins.set_facecolor("white")
    ins.patch.set_alpha(1.0)
    for poly in getattr(bnd, "geoms", [bnd]):
        ins.plot(*poly.exterior.xy, color="0.25", lw=0.6, zorder=1)
    draw(ins, 30, 50, 22, 0.9)
    ins.set_xlim(*zx)
    ins.set_ylim(*zy)
    ins.set_aspect("auto")  # the inset box was sized to the geographic aspect above
    ins.xaxis.set_major_locator(MultipleLocator(0.1))
    ins.yaxis.set_major_locator(MultipleLocator(0.1))
    ins.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ins.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ins.tick_params(labelsize=6.5, length=2.5, pad=1.5)
    ins.set_xticks([92.0, 92.1, 92.2])
    ins.set_yticks([22.5, 22.6, 22.7, 22.8, 22.9])
    ins.set_xlabel("Longitude (°E)", fontsize=6.5, labelpad=1)
    ins.set_ylabel("Latitude (°N)", fontsize=6.5, labelpad=1)
    for cid in range(nc):
        m = lab == cid
        cx, cy = lon[m].mean(), lat[m].mean()
        if zx[0] < cx < zx[1] and zy[0] < cy < zy[1]:
            ins.annotate(str(cid), (cx, cy), xytext=(4, 5), textcoords="offset points", fontsize=8,
                         fontweight="bold", color="k", bbox=dict(fc="w", ec="none", alpha=0.75, pad=0.6), zorder=5)
    cx_, cy_ = zx[1] - 0.035, zy[1] - 0.06
    ref_circle(ins, DBSCAN_EPS, FIXED_LAT0, cx_, cy_)
    ins.text(cx_ - km_to_deg(DBSCAN_EPS, FIXED_LAT0)[0] * 1.3, cy_, f"eps = {DBSCAN_EPS} km", fontsize=7,
             va="center", ha="right", bbox=dict(fc="w", ec="none", alpha=0.7, pad=0.6), zorder=6)
    scale_bar(ins, 5, 22.67, zx[0] + 0.012, zy[0] + 0.02, fs=7)
    ins.set_title("Zoom: west-central corridor", fontsize=7, pad=2)
    ax.indicate_inset_zoom(ins, edgecolor="k", lw=0.8, alpha=0.8)
    savefig(fig, FIG / "fig11_dbscan_kdistance_rev3.png")


def part_figs(df):
    fig08_rev3(df)
    fig09_rev3(df)
    fig10_rev3(df)
    fig11_rev3(df)



# ==========================================================================
# final assembly with the DNN device split (Metal GPU vs CPU)
# ==========================================================================

GPU_ABL_SEEDS = [42, 0, 4, 5]          # first-pass full-model ablation seeds (Metal GPU)
GPU_DLA_SEEDS = [42, 0, 3, 4, 6, 7]    # first-pass dl_a seeds (Metal GPU)
KEYS = ["seed", "model", "within_AUC", "pooled_AUC"]


def _dnn(d: pd.DataFrame, seeds) -> pd.DataFrame:
    return d[(d.model == "DNN") & d.seed.isin(seeds)]


def dnn_device_frames():
    """Return {'gpu': (abl, dla, dlb), 'cpu': (abl, dla, dlb)} for the DNN, all with the same column layout."""
    all_s = SEEDS_ALL
    g_abl = pd.concat([_dnn(load_cached("abl", GPU_ABL_SEEDS), GPU_ABL_SEEDS), load_cached("abl_dnn_gpu", all_s)])
    c_abl = pd.concat([_dnn(load_cached("abl_rest", all_s), all_s), load_cached("abl_dnn_cpu", all_s)])
    g_dla = pd.concat([_dnn(load_cached("dl_a", GPU_DLA_SEEDS), GPU_DLA_SEEDS), load_cached("dl_a_dnn_gpu", all_s)])
    c_dla = pd.concat([_dnn(load_cached("dl_a_rest", all_s), all_s), load_cached("dl_a_dnn_cpu", all_s)])
    g_dlb, c_dlb = load_cached("dl_b_dnn_gpu", all_s), load_cached("dl_b_dnn_cpu", all_s)
    return {"gpu": (g_abl, g_dla, g_dlb), "cpu": (c_abl, c_dla, c_dlb)}


def _delta_table(d: pd.DataFrame, ref_name: str, group: str) -> pd.DataFrame:
    ref = d[d[group] == ref_name].set_index(["seed", "model"])
    rows = []
    for (m, a), g in d.groupby(["model", group], sort=False):
        g = g.set_index("seed")
        r = {"model": m, group: a, "n_seeds": len(g)}
        r["within_AUC_mean"], r["within_AUC_sd"] = ms(g.within_AUC)
        r["pooled_AUC_mean"], r["pooled_AUC_sd"] = ms(g.pooled_AUC)
        r0 = ref.xs(m, level="model").reindex(g.index)
        for k in ("within", "pooled"):
            r[f"delta_{k}_mean"], r[f"delta_{k}_sd"] = ms((g[f"{k}_AUC"] - r0[f"{k}_AUC"]).dropna())
        rows.append(r)
    return pd.DataFrame(rows)


def part_final():
    S = SEEDS_ALL
    fr = dnn_device_frames()
    # ---------------- A1
    d = load_multi("abl", S).drop_duplicates(["seed", "model", "ablation"])
    d = d[d.model != "DNN"][KEYS + ["ablation"]]
    base = baseline(S).assign(ablation="none")
    parts = [base[base.model != "DNN"][KEYS + ["ablation"]], d]
    for dev, nm in (("gpu", "DNN [Metal GPU]"), ("cpu", "DNN [CPU]")):
        abl, dla, _ = fr[dev]
        if dev == "gpu":
            b = base[base.model == "DNN"][KEYS + ["ablation"]]
        else:
            b = dla[dla.variant == "standard"][KEYS].assign(ablation="none")
        a = abl[KEYS + ["ablation"]]
        x = pd.concat([b, a], ignore_index=True)
        x["model"] = nm
        parts.append(x)
    A1 = pd.concat(parts, ignore_index=True)
    A1.to_csv(OUT / "A1_ablation_per_seed.csv", index=False)
    order = {a: i for i, a in enumerate(["none", *ABL3])}
    mo = {LABEL[m]: i for i, m in enumerate(ALL7)}
    mo.update({"DNN [Metal GPU]": 0, "DNN [CPU]": 0.5})
    t = _delta_table(A1, "none", "ablation")
    t = t.assign(_o=t.ablation.map(order), _m=t.model.map(mo)).sort_values(["_m", "_o"]).drop(columns=["_o", "_m"])
    t.round(4).to_csv(OUT / "A1_ablation_summary.csv", index=False)
    # ---------------- A5
    d = load_multi("dl_a", S).drop_duplicates(["seed", "model", "variant"])
    d = d[d.model != "DNN"][KEYS + ["variant"]]
    b = load_multi("dl_b", S)
    b = b[b.model != "DNN"][KEYS].assign(variant="b_fixed_epochs")
    ter, terc = baseline(S, "terrain"), baseline(S, "terrain+Cluster")
    parts = [d, b]
    for dev, nm in (("gpu", "DNN [Metal GPU]"), ("cpu", "DNN [CPU]")):
        _, dla, dlb = fr[dev]
        parts.append(dla[KEYS + ["variant"]].assign(model=nm))
        parts.append(dlb[KEYS].assign(variant="b_fixed_epochs", model=nm))
    A5 = pd.concat(parts, ignore_index=True)
    A5.to_csv(OUT / "A5_dl_per_seed.csv", index=False)
    s = agg(A5, ["model", "variant"])
    pr = pd.concat([ter.assign(variant="primary_terrain_existing"), terc.assign(variant="primary_terrain+Cluster_existing")])
    pr = pr[pr.model.isin(["DNN", "1D-CNN", "LSTM"])]
    pr.loc[pr.model == "DNN", "model"] = "DNN [Metal GPU]"
    s = pd.concat([s, agg(pr, ["model", "variant"])], ignore_index=True)
    vo = {"primary_terrain_existing": 0, "standard": 1, "a_terrain": 2, "b_fixed_epochs": 3,
          "primary_terrain+Cluster_existing": 4, "a_terrain+Cluster": 5}
    s = s.assign(_v=s.variant.map(vo), _m=s.model.map({"DNN [Metal GPU]": 0, "DNN [CPU]": 1, "1D-CNN": 2, "LSTM": 3}))
    s = s.sort_values(["_m", "_v"]).drop(columns=["_v", "_m"])
    s.round(4).to_csv(OUT / "A5_dl_summary.csv", index=False)
    print(t.round(3).to_string())
    print(s.round(3).to_string())

# ==========================================================================
# main
# ==========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS_ALL)
    ap.add_argument("--models", nargs="*", default=ALL7)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    global TAGOVR
    TAGOVR = a.tag
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_df()
    p = a.part
    if p == "abl":
        part_abl(df, a.seeds, a.models)
    elif p == "cw":
        part_cw(df, a.seeds)
    elif p == "smotenc":
        part_smotenc(df, a.seeds)
    elif p == "dl_a":
        part_dl_a(df, a.seeds, a.models)
    elif p == "dl_b":
        part_dl_b(df, a.seeds, a.models)
    elif p == "shap_cnn":
        part_shap_cnn(df, a.seeds)
    elif p == "light":
        part_a2()
        part_a3_rescale(df, SEEDS_ALL)
        part_a8(df)
        part_a9(df)
        part_a7(df)
    elif p == "final":
        part_final()
    elif p == "summ":
        part_summ_variants(SEEDS_ALL)
        abl_summary(SEEDS_ALL)
    elif p == "a6":
        t, chk, rf, cn, cn10 = part_a6(df, SEEDS_ALL)
        print(t.round(2).to_string())
        if len(cn["11"]) == 10:
            fig14a_rev3(rf, cn["11"])
    elif p == "figs":
        part_figs(df)
    else:
        raise SystemExit(f"unknown part {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
