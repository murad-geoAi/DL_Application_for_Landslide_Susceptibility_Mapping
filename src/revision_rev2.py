"""Second-round sensitivity experiments (audit deepreview-2).

    python -m src.revision_rev2 --part <name> [--seeds ...] [--models ...]

Parts: repro, ablation_more, spatial_es, lstm_reg, extra_models.
Outputs: results/revision/extra/rev2/ ; per-seed caches in results/revision/extra/rev2/parts/.
Nothing in the existing pipeline outputs is modified.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from . import revision_extra as rx  # noqa: E402
from .data import TARGET  # noqa: E402
from .evaluate import classification_metrics, cross_validate_model, pooled_metrics  # noqa: E402
from .models import (  # noqa: E402
    DEEP_DEFAULTS,
    KerasBinaryClassifier,
    _compile,
    _keras,
    _pipeline,
    get_model,
    set_global_seeds,
)
from .splits import spatial_block_cv, spatial_block_labels, spatial_fold_labels  # noqa: E402

warnings.filterwarnings("ignore")

OUT = rx.OUT / "rev2"
PARTS = OUT / "parts"
LABEL = rx.LABEL
TERRAIN_KW = rx.TERRAIN_KW
SEEDS_ALL = rx.SEEDS_ALL
ALL7 = ["random_forest", "xgboost", "svm_rbf", "logistic", "cnn1d", "dnn", "lstm"]
BASE = {  # Table 4 spatial within-fold baselines (as quoted in the brief)
    "RandomForest": "0.811 +/- 0.007", "XGBoost": "0.778", "SVM-RBF": "0.754",
    "LogisticRegression": "0.638", "1D-CNN": "0.789 +/- 0.015", "DNN": "0.609", "LSTM": "0.557"}


def cached(part, seed, fn):
    PARTS.mkdir(parents=True, exist_ok=True)
    p = PARTS / f"{part}_seed{seed}.csv"
    if p.exists():
        return pd.read_csv(p)
    t = time.time()
    d = fn()
    d.to_csv(p, index=False)
    print(f"[{part}] seed {seed} done in {time.time() - t:.0f}s", flush=True)
    return d


def load_cached(part, seeds):
    fs = [PARTS / f"{part}_seed{s}.csv" for s in seeds if (PARTS / f"{part}_seed{s}.csv").exists()]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def fmt(m, s):
    return f"{m:.3f} +/- {s:.3f}"


def table(d, by, extra_cols=()):
    """Markdown table from per-seed rows (columns seed, within_AUC, pooled_AUC, + by)."""
    s = rx.summarize_cv(d, by)
    lines = ["| " + " | ".join(by) + " | within-fold AUC | pooled AUC | baseline (T4 spatial within) | n seeds |",
             "|" + "---|" * (len(by) + 4)]
    for _, r in s.iterrows():
        mod = r["model"] if "model" in by else ""
        lines.append("| " + " | ".join(str(r[b]) for b in by) + f" | {fmt(r.within_AUC_mean, r.within_AUC_sd)} | "
                     f"{fmt(r.pooled_AUC_mean, r.pooled_AUC_sd)} | {BASE.get(mod, '')} | {int(r.n_seeds)} |")
    return s, "\n".join(lines)


# ----------------------------------------------------------------------------
# repro
# ----------------------------------------------------------------------------
def part_repro(df):
    seed = 42
    folds = list(spatial_block_cv(df))
    set_global_seeds(seed, deterministic=False)
    cv = cross_validate_model(lambda: get_model("random_forest", random_state=seed, **TERRAIN_KW), df, folds)
    w, p = float(cv["per_fold"].AUC.mean()), float(cv["pooled"]["AUC"])
    m = pd.read_csv("results/revision/seeds/metrics_seed42.csv")
    ref = m[(m.model == "RandomForest") & (m.dataset == "terrain") & (m.split == "spatial")].iloc[0]
    r = {"repro_within": w, "repro_pooled": p, "main_within": float(ref.within_mean_AUC),
         "main_pooled": float(ref.pooled_AUC)}
    print(r)
    (OUT / "repro_rf_seed42.json").write_text(json.dumps(r, indent=1))
    return r


# ----------------------------------------------------------------------------
# part 1
# ----------------------------------------------------------------------------
def part_ablation_more(df, seeds, models):
    models = models or ["svm_rbf", "dnn", "lstm"]
    for s in seeds:
        cached("ablation_more", s, lambda s=s: rx.ablation_seed(df, s, models))
    new = load_cached("ablation_more", seeds)
    old = rx.load_cached("ablation", seeds)  # RF, XGB, LR, CNN (existing run)
    met = rx.load_metrics()
    met = met[(met.split == "spatial") & (met.dataset == "terrain") & met.seed.isin(seeds)]
    base = pd.DataFrame({"seed": met.seed, "model": met.model, "within_AUC": met.within_mean_AUC,
                         "pooled_AUC": met.pooled_AUC, "ablation": "none"})
    cols = ["seed", "model", "within_AUC", "pooled_AUC", "ablation"]
    parts = [base, old[cols] if len(old) else None, new[cols] if len(new) else None]
    d = pd.concat([p for p in parts if p is not None], ignore_index=True)
    d.to_csv(OUT / "ablation_all7_per_seed.csv", index=False)
    s, md = table(d.assign(model=d.model), ["model", "ablation"])
    s.round(4).to_csv(OUT / "ablation_all7_summary.csv", index=False)
    # bounding-box column, all 7, from existing caches
    bb = rx.load_cached("bbox", seeds)
    bs, bmd = table(bb.assign(design="bbox_spatial"), ["model", "design"])
    bs.round(4).to_csv(OUT / "bbox_all7_summary.csv", index=False)
    return md + "\n\nBounding-box negatives (existing caches, spatial CV, 7 models)\n\n" + bmd


# ----------------------------------------------------------------------------
# part 2: spatial early stopping
# ----------------------------------------------------------------------------
def choose_val_blocks(blk_tr, y_tr, seed, fold, lo=0.15, hi=0.35, min_cls=5):
    """Greedy whole-block validation selection (see brief). Returns list of block ids."""
    rng = np.random.default_rng(seed * 1000 + fold)
    blocks = list(rng.permutation(np.unique(blk_tr)))
    n = len(blk_tr)
    chosen, skipped = [], []
    nv = npos = 0

    def ok(nv, npos):
        return nv >= lo * n and npos >= min_cls and (nv - npos) >= min_cls

    for b in blocks:
        if ok(nv, npos):
            break
        m = blk_tr == b
        if nv + m.sum() > hi * n:
            skipped.append(b)
            continue
        chosen.append(b)
        nv += int(m.sum())
        npos += int(y_tr[m].sum())
    for b in skipped:  # fallback: criteria still unmet, accept exceeding the cap
        if ok(nv, npos):
            break
        m = blk_tr == b
        chosen.append(b)
        nv += int(m.sum())
        npos += int(y_tr[m].sum())
    return chosen


class SpatialESFit:
    """fit_fn for cross_validate_model: preprocessor fit on the whole training fold (as in the main
    pipeline), then whole training blocks are held out as the early-stopping validation set."""

    def __init__(self, blocks, seed):
        self.blocks, self.seed, self.info, self.k = blocks, seed, [], 0

    def __call__(self, pipe, Xtr, ytr):
        self.k += 1
        ytr = np.asarray(ytr)
        pos = Xtr.index.to_numpy()  # df has a RangeIndex => labels are row positions
        blk = self.blocks[pos]
        chosen = choose_val_blocks(blk, ytr, self.seed, self.k)
        vm = np.isin(blk, chosen)
        pre, est = pipe.named_steps["pre"], pipe.named_steps["model"]
        Xt = pre.fit_transform(Xtr, ytr)
        Xt = np.asarray(Xt)
        est.fit(Xt[~vm], ytr[~vm], validation_data=(Xt[vm], ytr[vm]))
        self.info.append({"fold": self.k, "n_train_fold": len(ytr), "n_val": int(vm.sum()),
                          "val_frac": float(vm.mean()), "val_pos": int(ytr[vm].sum()),
                          "val_neg": int((1 - ytr[vm]).sum()), "val_blocks": ";".join(str(int(b)) for b in chosen),
                          "n_fit": int((~vm).sum()), "epochs_run": int(est.epochs_run_)})
        return pipe


def spatial_es_seed(df, seed, models):
    blocks = spatial_block_labels(df)
    folds = list(spatial_block_cv(df))
    rows, infos = [], []
    for name in models:
        set_global_seeds(seed, deterministic=False)
        fit = SpatialESFit(blocks, seed)
        cv = cross_validate_model(lambda name=name: get_model(name, random_state=seed, **TERRAIN_KW), df, folds,
                                  fit_fn=fit)
        rows.append(rx.cv_row(seed, name, cv, {"design": "spatial_ES"}))
        for i in fit.info:
            infos.append({"seed": seed, "model": LABEL[name], **i})
        print(f"[spatial_es seed {seed}] {LABEL[name]} within={rows[-1]['within_AUC']:.3f} "
              f"pooled={rows[-1]['pooled_AUC']:.3f}", flush=True)
    d = pd.DataFrame(rows)
    pd.DataFrame(infos).to_csv(PARTS / f"spatial_es_valinfo_seed{seed}.csv", index=False)
    return d


def part_spatial_es(df, seeds, models):
    models = models or ["dnn", "cnn1d", "lstm"]
    for s in seeds:
        cached("spatial_es", s, lambda s=s: spatial_es_seed(df, s, models))
    d = load_cached("spatial_es", seeds)
    met = rx.load_metrics()
    met = met[(met.split == "spatial") & (met.dataset == "terrain") & met.seed.isin(seeds)
              & met.model.isin([LABEL[m] for m in models])]
    base = pd.DataFrame({"seed": met.seed, "model": met.model, "within_AUC": met.within_mean_AUC,
                         "pooled_AUC": met.pooled_AUC, "design": "random_ES (main)"})
    a = pd.concat([base, d[base.columns]], ignore_index=True)
    a.to_csv(OUT / "spatial_es_per_seed.csv", index=False)
    s, md = table(a, ["model", "design"])
    s.round(4).to_csv(OUT / "spatial_es_summary.csv", index=False)
    vi = pd.concat([pd.read_csv(PARTS / f"spatial_es_valinfo_seed{x}.csv") for x in seeds
                    if (PARTS / f"spatial_es_valinfo_seed{x}.csv").exists()], ignore_index=True)
    vi.to_csv(OUT / "spatial_es_valinfo.csv", index=False)
    g = vi.groupby(["model", "fold"]).agg(n_train=("n_train_fold", "mean"), n_val_mean=("n_val", "mean"),
                                           n_val_min=("n_val", "min"), n_val_max=("n_val", "max"),
                                           pos_mean=("val_pos", "mean"), pos_min=("val_pos", "min"),
                                           neg_mean=("val_neg", "mean"), neg_min=("val_neg", "min"),
                                           frac_mean=("val_frac", "mean"), frac_max=("val_frac", "max"),
                                           epochs_mean=("epochs_run", "mean")).round(2).reset_index()
    g.to_csv(OUT / "spatial_es_valinfo_summary.csv", index=False)
    return md + "\n\nValidation set per fold (averaged over seeds; min/max across seeds)\n\n" + g.to_markdown(index=False)


# ----------------------------------------------------------------------------
# part 3: regularised LSTM
# ----------------------------------------------------------------------------
class RegLSTM(KerasBinaryClassifier):
    def _build(self, input_dim):
        keras = _keras()
        L = keras.layers
        reg = keras.regularizers.l2(0.01)
        m = keras.Sequential([
            L.Input(shape=(input_dim, 1)),
            L.LSTM(128, return_sequences=True, kernel_regularizer=reg),
            L.Dropout(0.2),
            L.LSTM(64, kernel_regularizer=reg),
            L.Dropout(0.2),
            L.Dense(32, activation="relu"),
            L.Dense(1, activation="sigmoid"),
        ], name="lstm_reg")
        return _compile(m, self.learning_rate)


def reg_lstm_model(seed):
    est = RegLSTM(architecture="lstm", epochs=DEEP_DEFAULTS["epochs"], batch_size=DEEP_DEFAULTS["batch_size"],
                  early_stopping=True, val_fraction=0.2, random_state=seed)
    return _pipeline(est, False, "drop")


def rand_folds(df):
    y = df[TARGET].to_numpy(dtype=int)
    return [(a, b) for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)]


def lstm_reg_seed(df, seed):
    blocks = spatial_block_labels(df)
    sp, rnd = list(spatial_block_cv(df)), rand_folds(df)
    rows = []
    for cvname, folds in (("spatial", sp), ("random", rnd)):
        for es in ("random_ES", "spatial_ES"):
            fit = SpatialESFit(blocks, seed) if es == "spatial_ES" else None
            set_global_seeds(seed, deterministic=False)
            cv = cross_validate_model(lambda: reg_lstm_model(seed), df, folds, fit_fn=fit)
            rows.append(rx.cv_row(seed, "lstm", cv, {"design": f"{cvname} CV, {es}"}))
            print(f"[lstm_reg seed {seed}] {cvname} {es} within={rows[-1]['within_AUC']:.3f} "
                  f"pooled={rows[-1]['pooled_AUC']:.3f}", flush=True)
            if fit is not None:
                pd.DataFrame(fit.info).assign(seed=seed, cv=cvname).to_csv(
                    PARTS / f"lstm_reg_valinfo_{cvname}_seed{seed}.csv", index=False)
    return pd.DataFrame(rows)


def part_lstm_reg(df, seeds, models):
    for s in seeds:
        cached("lstm_reg", s, lambda s=s: lstm_reg_seed(df, s))
    d = load_cached("lstm_reg", seeds)
    d["model"] = "LSTM (L2+dropout)"
    # unregularised LSTM reference: spatial CV from main run; random CV not available -> computed below
    met = rx.load_metrics()
    met = met[(met.split == "spatial") & (met.dataset == "terrain") & (met.model == "LSTM") & met.seed.isin(seeds)]
    ref = pd.DataFrame({"seed": met.seed, "model": "LSTM (unregularised)", "within_AUC": met.within_mean_AUC,
                        "pooled_AUC": met.pooled_AUC, "design": "spatial CV, random_ES (main)"})
    for s in seeds:  # unregularised LSTM under random CV (10 seeds), cached
        r = cached("lstm_unreg_randomcv", s, lambda s=s: _unreg_random(df, s))
        ref = pd.concat([ref, r[ref.columns]], ignore_index=True)
    a = pd.concat([ref, d[ref.columns]], ignore_index=True)
    a.to_csv(OUT / "lstm_reg_per_seed.csv", index=False)
    s_, md = table(a, ["model", "design"])
    s_.round(4).to_csv(OUT / "lstm_reg_summary.csv", index=False)
    return md


def _unreg_random(df, seed):
    set_global_seeds(seed, deterministic=False)
    cv = cross_validate_model(lambda: get_model("lstm", random_state=seed, **TERRAIN_KW), df, rand_folds(df))
    r = rx.cv_row(seed, "lstm", cv, {"design": "random CV, random_ES"})
    r["model"] = "LSTM (unregularised)"
    return pd.DataFrame([r])


# ----------------------------------------------------------------------------
# part 4
# ----------------------------------------------------------------------------
class CatBoostWrap(BaseEstimator, ClassifierMixin):
    """CatBoost with native categorical LULC / Soil_Texture (string levels incl. 'missing').
    Input is the object matrix from the main preprocessor with the one-hot step replaced by string levels
    (last two columns = LULC, Soil_Texture)."""

    def __init__(self, seed):
        self.seed = seed
        self.classes_ = np.array([0, 1])

    def _df(self, X):
        X = np.asarray(X, dtype=object)
        d = pd.DataFrame(X)
        for c in d.columns[:-2]:
            d[c] = d[c].astype(float)
        for c in d.columns[-2:]:
            d[c] = d[c].astype(str)
        return d

    def fit(self, X, y):
        from catboost import CatBoostClassifier
        d = self._df(X)
        self.m_ = CatBoostClassifier(iterations=500, random_seed=self.seed, verbose=0, thread_count=1,
                                     cat_features=list(d.columns[-2:]))
        self.m_.fit(d, np.asarray(y))
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(self._df(X))


def catboost_model(seed):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    from .features import _as_category_strings, build_preprocessor
    pre = build_preprocessor(include_cluster=False, drainage_mode="drop")
    ct = pre.named_steps["columns"]
    ct.set_params(transformers=[(n, FunctionTransformer(_as_category_strings, feature_names_out="one-to-one")
                                 if n == "categorical" else t, c) for n, t, c in ct.transformers])
    return Pipeline([("pre", pre), ("model", CatBoostWrap(seed))])


def catboost_seed(df, seed):
    rows = []
    for cvname, folds in (("spatial", list(spatial_block_cv(df))), ("random", rand_folds(df))):
        cv = cross_validate_model(lambda: catboost_model(seed), df, folds)
        rows.append(rx.cv_row(seed, "xgboost", cv, {"design": f"{cvname} CV"}))
        rows[-1]["model"] = "CatBoost"
        print(f"[catboost seed {seed}] {cvname} within={rows[-1]['within_AUC']:.3f} "
              f"pooled={rows[-1]['pooled_AUC']:.3f}", flush=True)
    return pd.DataFrame(rows)


def part_extra_models(df, seeds=None):
    seeds = seeds or SEEDS_ALL
    for s in seeds:
        cached("catboost", s, lambda s=s: catboost_seed(df, s))
    d = load_cached("catboost", seeds)
    d.to_csv(OUT / "catboost_per_seed.csv", index=False)
    s_, md = table(d, ["model", "design"])
    s_.round(4).to_csv(OUT / "catboost_summary.csv", index=False)
    return md + "\n\nTabPFN: skipped (user declined installation)."


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["repro", "ablation_more", "spatial_es", "lstm_reg", "extra_models"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS_ALL)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--summarize", action="store_true", help="(summary only; same call, cached seeds skipped)")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    df = rx.load_df()
    t0 = time.time()
    fn = {"repro": lambda: part_repro(df), "ablation_more": lambda: part_ablation_more(df, a.seeds, a.models),
          "spatial_es": lambda: part_spatial_es(df, a.seeds, a.models),
          "lstm_reg": lambda: part_lstm_reg(df, a.seeds, a.models),
          "extra_models": lambda: part_extra_models(df, a.seeds)}[a.part]
    res = fn()
    dt = time.time() - t0
    if isinstance(res, str):
        (OUT / f"{a.part}_table.md").write_text(res + "\n")
        print(res)
    print(f"[{a.part}] total {dt:.0f}s", flush=True)
    with open(OUT / "runtime.log", "a") as f:
        f.write(f"{a.part} seeds={a.seeds} {dt:.0f}s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
