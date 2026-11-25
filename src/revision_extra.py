"""Extra analyses for the second revision round (fold weighting, 4-fold CV, bbox negatives,
feature-group ablation, CNN feature-order ablation, thresholds, ROC/map/fold figures,
geodistance diagnostic, external-inventory overlap).

Run from the repository root::

    python -m src.revision_extra --part <name> [--seeds 42 0 1 ...]

Parts: folds, bbox, ablation, cnnorder, thresholds, roc, geodist, map, foldmap, rabby.
Heavy parts cache one CSV per seed in results/revision/extra/parts/, so a part can be split
over several processes (disjoint --seeds) and then summarised by one final call that lists
every seed (cached seeds are skipped).  Reuses the pipelines of ``revision_analyses``.
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
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from .data import TARGET, clean, load_raw  # noqa: E402
from .evaluate import cross_validate_model  # noqa: E402
from .features import (  # noqa: E402
    CATEGORY_LEVELS,
    FIXED_LAT0,
    _as_category_strings,
    build_preprocessor,
    feature_names,
    project_km,
)
from .models import (  # noqa: E402
    MODEL_LABELS,
    MODEL_NAMES,
    build_coordinates_only_probe,
    get_model,
    set_global_seeds,
)
from .revision_analyses import (  # noqa: E402
    BOUNDARY_FILE,
    _factor,
    district_polygon,
)
from .splits import (  # noqa: E402
    N_SPLITS,
    random_split,
    spatial_block_cv,
    spatial_block_labels,
    spatial_fold_labels,
)

warnings.filterwarnings("ignore")

REV = Path("results/revision")
OUT = REV / "extra"
PARTS = OUT / "parts"
FIG = Path("reports/figures/revision")
SEEDS_ALL = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]
LABEL = MODEL_LABELS
TERRAIN_KW = {"include_cluster": False, "drainage_mode": "drop"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def load_df(path: str = "Labeled Data Set.csv") -> pd.DataFrame:
    return clean(load_raw(path)).reset_index(drop=True)


def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def ms(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)), (float(np.nanstd(x, ddof=1)) if len(x) > 1 else float("nan"))


def folds_from_labels(lab: np.ndarray):
    idx = np.arange(len(lab))
    return [(idx[lab != k], idx[lab == k]) for k in sorted(np.unique(lab))]


def load_metrics() -> pd.DataFrame:
    """Per-seed metrics of the main run (per-seed files if present, else the aggregate file)."""
    sd = sorted((REV / "seeds").glob("metrics_seed*.csv"))
    if sd:
        return pd.concat([pd.read_csv(p) for p in sd], ignore_index=True)
    return pd.read_csv(REV / "metrics_all_seeds.csv")


def cached(part: str, seed: int, fn):
    """Run ``fn() -> DataFrame`` once per (part, seed); cache as parts/<part>_seed<seed>.csv."""
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


def cv_row(seed, name, cv, extra=None) -> dict:
    pf = cv["per_fold"]
    r = {"seed": seed, "model": LABEL[name], "within_AUC": float(pf.AUC.mean()),
         "pooled_AUC": float(cv["pooled"]["AUC"])}
    for _, x in pf.iterrows():
        r[f"fold{int(x.fold)}_AUC"] = float(x.AUC)
    r.update(extra or {})
    return r


def summarize_cv(d: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in d.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        w, p = ms(g.within_AUC), ms(g.pooled_AUC)
        rows.append({**dict(zip(by, key)), "within_AUC_mean": w[0], "within_AUC_sd": w[1],
                     "pooled_AUC_mean": p[0], "pooled_AUC_sd": p[1], "n_seeds": g.seed.nunique()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# a. folds: weighted within-fold AUC and 4-fold variant
# --------------------------------------------------------------------------


def fold_counts(df: pd.DataFrame) -> pd.DataFrame:
    lab = spatial_fold_labels(df)
    y = df[TARGET].to_numpy(dtype=int)
    return pd.DataFrame([{"fold": k + 1, "n": int((lab == k).sum()), "n_pos": int(y[lab == k].sum()),
                          "n_neg": int(((lab == k) & (y == 0)).sum())} for k in range(N_SPLITS)])


def weighted_within_auc(df: pd.DataFrame) -> pd.DataFrame:
    fc = fold_counts(df)
    met = load_metrics()
    met = met[met.split == "spatial"].copy()
    A = met[[f"fold{k}_AUC" for k in range(1, 6)]].to_numpy(float)
    wp = (fc.n_pos * fc.n_neg).to_numpy(float)
    wn = fc.n.to_numpy(float)
    met["within_unweighted"] = np.nanmean(A, axis=1)
    met["within_pair_weighted"] = np.nansum(A * wp, axis=1) / wp.sum()
    met["within_n_weighted"] = np.nansum(A * wn, axis=1) / wn.sum()
    rows = []
    for (ds, m), g in met.groupby(["dataset", "model"], sort=False):
        r = {"dataset": ds, "model": m, "n_seeds": g.seed.nunique()}
        for c in ("within_unweighted", "within_pair_weighted", "within_n_weighted", "pooled_AUC"):
            mu, sd = ms(g[c])
            r[f"{c}_mean"], r[f"{c}_sd"] = mu, sd
        rows.append(r)
    res = pd.DataFrame(rows)
    res.round(4).to_csv(OUT / "within_auc_weighted.csv", index=False)
    fc["pair_weight"] = wp / wp.sum()
    fc["n_weight"] = wn / wn.sum()
    fc.round(4).to_csv(OUT / "within_auc_fold_weights.csv", index=False)
    return res


def merge_to_4(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Merge the current fold 5 with the spatially nearest fold (min block-centroid distance)."""
    lab = spatial_fold_labels(df)
    blocks = spatial_block_labels(df)
    y = df[TARGET].to_numpy(dtype=int)
    xy = project_km(df.Latitude, df.Longitude)
    cen = {int(b): xy[blocks == b].mean(0) for b in np.unique(blocks)}
    fold_of_block = {b: int(lab[blocks == b][0]) for b in cen}
    src = 4
    dist = {}
    for j in range(4):
        dist[j + 1] = min(np.linalg.norm(cen[a] - cen[b]) for a in cen for b in cen
                          if fold_of_block[a] == src and fold_of_block[b] == j)
    nearest = min(dist, key=dist.get) - 1
    new = lab.copy()
    new[lab == src] = nearest
    info = {"merged_fold": 5, "merged_into_fold": nearest + 1,
            "min_block_centroid_distance_km_from_fold5": {f"fold{k}": round(float(v), 2) for k, v in dist.items()},
            "n_pos_per_fold": [int(y[new == k].sum()) for k in range(4)],
            "n_per_fold": [int((new == k).sum()) for k in range(4)],
            "min_pos_ge_10": bool(min(int(y[new == k].sum()) for k in range(4)) >= 10)}
    return new, info


def cv4_seed(df, seed, models):
    lab, _ = merge_to_4(df)
    folds = folds_from_labels(lab)
    rows = []
    for name in models:
        set_global_seeds(seed, deterministic=False)
        cv = cross_validate_model(lambda name=name: get_model(name, random_state=seed, **TERRAIN_KW), df, folds)
        rows.append(cv_row(seed, name, cv))
        print(f"[cv4 seed {seed}] {LABEL[name]} within={rows[-1]['within_AUC']:.3f} "
              f"pooled={rows[-1]['pooled_AUC']:.3f}", flush=True)
    return pd.DataFrame(rows)


def part_folds(df, seeds, models):
    OUT.mkdir(parents=True, exist_ok=True)
    print(weighted_within_auc(df).round(4).to_string())
    lab, info = merge_to_4(df)
    y = df[TARGET].to_numpy(dtype=int)
    pd.DataFrame([{"fold": k + 1, "n": int((lab == k).sum()), "n_pos": int(y[lab == k].sum()),
                   "n_neg": int(((lab == k) & (y == 0)).sum())} for k in range(4)]).to_csv(
        OUT / "cv4_folds.csv", index=False)
    (OUT / "cv4_merge_info.json").write_text(json.dumps(info, indent=2))
    print(info)
    for s in seeds:
        cached("cv4", s, lambda s=s: cv4_seed(df, s, models))
    d = load_cached("cv4", seeds)
    if len(d):
        d.to_csv(OUT / "cv4_per_seed.csv", index=False)
        summarize_cv(d, ["model"]).round(4).to_csv(OUT / "cv4_summary.csv", index=False)
        print(summarize_cv(d, ["model"]).round(3).to_string())


# --------------------------------------------------------------------------
# b. bbox negatives
# --------------------------------------------------------------------------


def bbox_subset(df: pd.DataFrame):
    pos = df[df[TARGET] == 1]
    box = (pos.Latitude.min(), pos.Latitude.max(), pos.Longitude.min(), pos.Longitude.max())
    keep = (df[TARGET] == 1) | (df.Latitude.between(box[0], box[1]) & df.Longitude.between(box[2], box[3]))
    return df[keep].reset_index(drop=True), box


def bbox_seed(sub, seed, models):
    folds = list(spatial_block_cv(sub))
    rows = []
    for name in models:
        set_global_seeds(seed, deterministic=False)
        cv = cross_validate_model(lambda name=name: get_model(name, random_state=seed, **TERRAIN_KW), sub, folds)
        rows.append(cv_row(seed, name, cv))
        print(f"[bbox seed {seed}] {LABEL[name]} within={rows[-1]['within_AUC']:.3f} "
              f"pooled={rows[-1]['pooled_AUC']:.3f}", flush=True)
    return pd.DataFrame(rows)


def bbox_static(sub, box):
    y = sub[TARGET].to_numpy(dtype=int)
    folds = list(spatial_block_cv(sub))
    meta = {"n_samples": len(sub), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
            "bbox_latmin_latmax_lonmin_lonmax": [float(b) for b in box],
            "fold_sizes": [len(t) for _, t in folds], "fold_pos": [int(y[t].sum()) for _, t in folds]}
    tr, va, te = random_split(sub, random_state=42)
    trv = pd.concat([tr, va])
    pr = build_coordinates_only_probe(42).fit(trv, trv[TARGET].to_numpy(dtype=int))
    cvp = cross_validate_model(lambda: build_coordinates_only_probe(42), sub, folds)
    meta["probe_random_test_AUC"] = float(roc_auc_score(te[TARGET].to_numpy(dtype=int), pr.predict_proba(te)[:, 1]))
    meta["probe_spatial_within_AUC"] = float(cvp["per_fold"].AUC.mean())
    meta["probe_spatial_pooled_AUC"] = float(cvp["pooled"]["AUC"])
    pipe = get_model("logistic", random_state=42, **TERRAIN_KW).fit(sub, y)
    names = feature_names(pipe.named_steps["pre"])
    coef = pipe.named_steps["model"].coef_.ravel()
    t = pd.DataFrame({"encoded_column": names, "coef": coef, "abs_coef": np.abs(coef)})
    t["factor"] = [_factor(n) for n in names]
    sh = t.groupby("factor", as_index=False).abs_coef.sum().rename(columns={"abs_coef": "sum_abs_coef"})
    sh["share"] = sh.sum_abs_coef / sh.sum_abs_coef.sum()
    sh = sh.sort_values("share", ascending=False)
    sh.round(5).to_csv(OUT / "bbox_lr_coef_shares.csv", index=False)
    t.round(5).to_csv(OUT / "bbox_lr_coef_encoded.csv", index=False)
    (OUT / "bbox_meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)
    print(sh.round(3).to_string())


def part_bbox(df, seeds, models):
    sub, box = bbox_subset(df)
    for s in seeds:
        cached("bbox", s, lambda s=s: bbox_seed(sub, s, models))
    d = load_cached("bbox", seeds)
    if len(d):
        d.to_csv(OUT / "bbox_per_seed.csv", index=False)
        summarize_cv(d, ["model"]).round(4).to_csv(OUT / "bbox_summary.csv", index=False)
        print(summarize_cv(d, ["model"]).round(3).to_string())
    bbox_static(sub, box)


# --------------------------------------------------------------------------
# c. feature-group ablation + missingness diagnostics
# --------------------------------------------------------------------------

ABLATIONS = {
    "none": [],
    "no_ndvi_lulc": ["NDVI", "LULC"],
    "no_rainfall": ["Annual_Rainfall"],
    "no_flags": ["TWI_was_missing", "Soil_Texture_was_missing"],
}


class ColDropper(BaseEstimator, TransformerMixin):
    """Drop engineered columns after FrameEngineer (the *_was_missing flags come from data.clean)."""

    def __init__(self, cols=()):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.cols if c in X.columns])


def _cat_encoder(cols):
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

    return Pipeline([
        ("levels", FunctionTransformer(_as_category_strings, feature_names_out="one-to-one")),
        ("onehot", OneHotEncoder(categories=[CATEGORY_LEVELS[c] for c in cols],
                                 handle_unknown="ignore", sparse_output=False)),
    ])


def ablated_preprocessor(drop: list[str]) -> Pipeline:
    pre = build_preprocessor(include_cluster=False, drainage_mode="drop")
    ct = pre.named_steps["columns"]
    new = []
    for name, tr, cols in ct.transformers:
        if name == "categorical":
            cols = [c for c in cols if c not in drop]
            if not cols:
                continue
            tr = _cat_encoder(cols)
        new.append((name, tr, cols))
    ct.set_params(transformers=new)
    return Pipeline([("frame", pre.named_steps["frame"]), ("drop", ColDropper(cols=list(drop))),
                     ("columns", ct)])


def ablated_model(name: str, seed: int, drop: list[str]) -> Pipeline:
    pipe = get_model(name, random_state=seed, **TERRAIN_KW)
    pipe.steps[0] = ("pre", ablated_preprocessor(drop))
    return pipe


ABL_MODELS = ("random_forest", "xgboost", "logistic", "cnn1d")


def ablation_seed(df, seed, models):
    folds = list(spatial_block_cv(df))
    rows = []
    for abl, drop in ABLATIONS.items():
        if abl == "none":
            continue
        for name in models:
            set_global_seeds(seed, deterministic=False)
            cv = cross_validate_model(lambda name=name, drop=drop: ablated_model(name, seed, drop), df, folds)
            rows.append(cv_row(seed, name, cv, {"ablation": abl}))
            print(f"[ablation seed {seed}] {abl:<13} {LABEL[name]:<18} within={rows[-1]['within_AUC']:.3f} "
                  f"pooled={rows[-1]['pooled_AUC']:.3f}", flush=True)
    return pd.DataFrame(rows)


def ablation_summary(seeds, models):
    d = load_cached("ablation", seeds)
    if d.empty:
        return
    met = load_metrics()
    base = met[(met.split == "spatial") & (met.dataset == "terrain") & met.model.isin([LABEL[m] for m in models])
               & met.seed.isin(seeds)]
    b = pd.DataFrame({"seed": base.seed, "model": base.model, "within_AUC": base.within_mean_AUC,
                      "pooled_AUC": base.pooled_AUC, "ablation": "none"})
    d = pd.concat([b, d[b.columns]], ignore_index=True)
    d.to_csv(OUT / "ablation_per_seed.csv", index=False)
    ref = d[d.ablation == "none"].set_index(["seed", "model"])
    rows = []
    for (m, a), g in d.groupby(["model", "ablation"], sort=False):
        g = g.set_index("seed")
        r = {"model": m, "ablation": a, "n_seeds": len(g)}
        r["within_AUC_mean"], r["within_AUC_sd"] = ms(g.within_AUC)
        r["pooled_AUC_mean"], r["pooled_AUC_sd"] = ms(g.pooled_AUC)
        r0 = ref.xs(m, level="model").reindex(g.index) if m in ref.index.get_level_values("model") else pd.DataFrame(index=g.index, columns=["within_AUC", "pooled_AUC"], dtype=float)
        for k in ("within", "pooled"):
            dl = g[f"{k}_AUC"] - r0[f"{k}_AUC"]
            r[f"delta_{k}_mean"], r[f"delta_{k}_sd"] = ms(dl.dropna())
        rows.append(r)
    out = pd.DataFrame(rows)
    order = {a: i for i, a in enumerate(ABLATIONS)}
    out = out.assign(_o=out.ablation.map(order)).sort_values(["model", "_o"]).drop(columns="_o")
    out.round(4).to_csv(OUT / "ablation_summary.csv", index=False)
    print(out.round(3).to_string())


def missingness_diagnostics(df: pd.DataFrame) -> dict:
    from esda.moran import Moran
    from libpysal.weights import KNN
    from shapely.geometry import Point

    xy = project_km(df.Latitude, df.Longitude)
    y = df[TARGET].to_numpy(dtype=int)
    w = KNN.from_array(xy, k=5)
    w.transform = "r"
    res = {"projection": "project_km, FIXED_LAT0", "knn_k": 5, "permutations": 999}
    pos_idx = np.where(y == 1)[0]
    tree_pos = cKDTree(xy[pos_idx])
    d2, _ = tree_pos.query(xy, k=2)
    dnl = np.where(y == 1, d2[:, 1], d2[:, 0])  # nearest OTHER landslide for landslide points
    for flag in ("TWI_was_missing", "Soil_Texture_was_missing"):
        v = df[flag].to_numpy(dtype=float)
        np.random.seed(42)
        mi = Moran(v, w, permutations=999)
        m = v == 1
        d = {"n_missing": int(m.sum()), "share_missing": float(m.mean()),
             "morans_I": float(mi.I), "expected_I": float(mi.EI), "p_sim": float(mi.p_sim), "z_sim": float(mi.z_sim),
             "missing_among_landslides": int((m & (y == 1)).sum()), "n_landslides": int(y.sum()),
             "share_missing_among_landslides": float((m & (y == 1)).sum() / y.sum()),
             "missing_among_negatives": int((m & (y == 0)).sum()), "n_negatives": int((y == 0).sum()),
             "share_missing_among_negatives": float((m & (y == 0)).sum() / (y == 0).sum()),
             "mean_elevation_missing": float(df.Elevation[m].mean()),
             "mean_elevation_nonmissing": float(df.Elevation[~m].mean()),
             "mean_dist_to_nearest_landslide_km_missing": float(dnl[m].mean()),
             "mean_dist_to_nearest_landslide_km_nonmissing": float(dnl[~m].mean()),
             "median_dist_to_nearest_landslide_km_missing": float(np.median(dnl[m])),
             "median_dist_to_nearest_landslide_km_nonmissing": float(np.median(dnl[~m]))}
        from scipy.stats import fisher_exact

        tab = [[d["missing_among_landslides"], int(y.sum()) - d["missing_among_landslides"]],
               [d["missing_among_negatives"], int((y == 0).sum()) - d["missing_among_negatives"]]]
        d["fisher_exact_p_landslide_vs_negative"] = float(fisher_exact(tab)[1])
        for cls, nm in ((1, "landslides"), (0, "negatives")):
            mm, nn = m & (y == cls), (~m) & (y == cls)
            d[f"mean_elevation_{nm}_missing"] = float(df.Elevation[mm].mean()) if mm.any() else None
            d[f"mean_elevation_{nm}_nonmissing"] = float(df.Elevation[nn].mean())
            d[f"mean_dist_nn_landslide_km_{nm}_missing"] = float(dnl[mm].mean()) if mm.any() else None
            d[f"mean_dist_nn_landslide_km_{nm}_nonmissing"] = float(dnl[nn].mean())
        res[flag] = d
    poly_km = district_polygon(FIXED_LAT0)
    bound = poly_km.boundary
    rows = []
    for col in ("Slope", "Aspect", "Annual_Rainfall", "LULC"):
        for i in np.where(df[col].isna().to_numpy())[0]:
            p = Point(*xy[i])
            rows.append({"variable": col, "row": int(i), "lat": float(df.Latitude[i]),
                         "lon": float(df.Longitude[i]), "landslide": int(y[i]),
                         "inside_district": bool(poly_km.contains(p)),
                         "dist_to_boundary_km": float(bound.distance(p))})
    res["sparse_missing_points"] = rows
    res["sparse_missing_counts"] = {c: int(df[c].isna().sum()) for c in ("Slope", "Aspect", "Annual_Rainfall", "LULC")}
    di = np.array([bound.distance(Point(*xy[i])) for i in range(len(df))])
    res["all_points_dist_to_boundary_km_quantiles"] = {str(q): float(np.quantile(di, q)) for q in (0.05, 0.25, 0.5, 0.75)}
    (OUT / "missingness_diagnostics.json").write_text(json.dumps(res, indent=2))
    return res


def part_ablation(df, seeds, models):
    models = [m for m in models if m in ABL_MODELS] or list(ABL_MODELS)
    for s in seeds:
        cached("ablation", s, lambda s=s: ablation_seed(df, s, models))
    ablation_summary(seeds, models)
    r = missingness_diagnostics(df)
    print(json.dumps(r, indent=1)[:4000])


# --------------------------------------------------------------------------
# d. CNN feature-order ablation
# --------------------------------------------------------------------------


class ColPermute(BaseEstimator, TransformerMixin):
    def __init__(self, perm=None):
        self.perm = perm

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X if self.perm is None else X[:, list(self.perm)]


def permuted_model(name, seed, perm):
    pipe = get_model(name, random_state=seed, **TERRAIN_KW)
    pipe.steps[0] = ("pre", Pipeline([("inner", pipe.named_steps["pre"]), ("perm", ColPermute(perm=perm))]))
    return pipe


PERM_IDS = (100, 101, 102, 103, 104)
CNN_ORDER_SEEDS = (42, 0, 1)


def cnnorder_seed(df, seed):
    folds = list(spatial_block_cv(df))
    rows = []
    for pid in ("orig",) + PERM_IDS:
        perm = None if pid == "orig" else np.random.RandomState(pid).permutation(23).tolist()
        set_global_seeds(seed, deterministic=False)
        cv = cross_validate_model(lambda perm=perm: permuted_model("cnn1d", seed, perm), df, folds)
        rows.append(cv_row(seed, "cnn1d", cv, {"perm_id": pid, "perm": json.dumps(perm)}))
        print(f"[cnnorder seed {seed}] perm {pid} within={rows[-1]['within_AUC']:.3f} "
              f"pooled={rows[-1]['pooled_AUC']:.3f}", flush=True)
    return pd.DataFrame(rows)


def part_cnnorder(df, seeds):
    seeds = [s for s in seeds if s in CNN_ORDER_SEEDS] or list(CNN_ORDER_SEEDS)
    for s in seeds:
        cached("cnnorder", s, lambda s=s: cnnorder_seed(df, s))
    d = load_cached("cnnorder", seeds)
    if d.empty:
        return
    d["perm_id"] = d.perm_id.astype(str)
    d.to_csv(OUT / "cnn_order_per_seed.csv", index=False)
    g = d.groupby("perm_id", sort=False).agg(within_AUC_mean=("within_AUC", "mean"),
                                             pooled_AUC_mean=("pooled_AUC", "mean"), n_seeds=("seed", "nunique"))
    met = load_metrics()
    ref = met[(met.split == "spatial") & (met.dataset == "terrain") & met.seed.isin(seeds)]
    for m in ("DNN", "1D-CNN"):
        r = ref[ref.model == m]
        if len(r):
            g.loc[f"main_run_{m}"] = [r.within_mean_AUC.mean(), r.pooled_AUC.mean(), r.seed.nunique()]
    g = g.reset_index()
    g.round(4).to_csv(OUT / "cnn_order_summary.csv", index=False)
    pm = g[g.perm_id.isin([str(p) for p in PERM_IDS])]
    spread = {"seeds": list(seeds), "n_permutations": len(pm)}
    for k in ("within_AUC_mean", "pooled_AUC_mean"):
        spread[k] = {"min": float(pm[k].min()), "max": float(pm[k].max()), "range": float(pm[k].max() - pm[k].min()),
                     "sd_across_permutations": float(pm[k].std(ddof=1)),
                     "orig": float(g[g.perm_id == "orig"][k].iloc[0])}
    spread["sd_across_seeds_orig_within"] = float(d[d.perm_id == "orig"].within_AUC.std(ddof=1))
    (OUT / "cnn_order_spread.json").write_text(json.dumps(spread, indent=2))
    print(g.round(4).to_string())
    print(json.dumps(spread, indent=1))


# --------------------------------------------------------------------------
# e. thresholds
# --------------------------------------------------------------------------


def _prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    pre = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
    return {"accuracy": float((pred == y).mean()), "precision": pre, "recall": rec, "F1": f1}


def oof_files(seeds):
    return {s: REV / "seeds" / f"oof_seed{s}.csv" for s in seeds if (REV / "seeds" / f"oof_seed{s}.csv").exists()}


def part_thresholds(df, seeds):
    lab = spatial_fold_labels(df)
    y = df[TARGET].to_numpy(dtype=int)
    prior = np.array([y[lab != k].mean() for k in range(N_SPLITS)])  # training-fold prevalence
    thr = prior[lab]
    rows = []
    for s, f in oof_files(seeds).items():
        o = pd.read_csv(f)
        for c in o.columns:
            model, ds = c.split("|")
            p = o[c].to_numpy(float)
            variants = [("0.5", np.full(len(y), 0.5))]
            if ds == "terrain":
                variants.append(("fold_prior", thr))
            for vn, t in variants:
                rows.append({"seed": s, "model": model, "dataset": ds, "threshold": vn,
                             **_prf(y, (p >= t).astype(int)), "pred_pos_rate": float((p >= t).mean())})
    d = pd.DataFrame(rows)
    d = d[d.dataset.isin(["terrain", "terrain+SMOTE"])]
    d.to_csv(OUT / "threshold_metrics_per_seed.csv", index=False)
    rows = []
    for (ds, m, t), g in d.groupby(["dataset", "model", "threshold"], sort=False):
        r = {"dataset": ds, "model": m, "threshold": t, "n_seeds": g.seed.nunique()}
        for k in ("accuracy", "precision", "recall", "F1", "pred_pos_rate"):
            r[f"{k}_mean"], r[f"{k}_sd"] = ms(g[k])
        rows.append(r)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(OUT / "threshold_metrics.csv", index=False)
    (OUT / "threshold_values.json").write_text(json.dumps(
        {"fold_prior_thresholds": {f"fold{k + 1}": float(prior[k]) for k in range(N_SPLITS)}}, indent=1))
    print("training-fold prior thresholds", prior.round(3))
    print(out.round(3).to_string())


# --------------------------------------------------------------------------
# f. ROC figures
# --------------------------------------------------------------------------

GRID = np.linspace(0, 1, 101)
MODEL_ORDER = [LABEL[m] for m in MODEL_NAMES]
STYLE = {"1D-CNN": ("#1f77b4", "-"), "DNN": ("#ff7f0e", "--"), "LSTM": ("#2ca02c", "-."),
         "RandomForest": ("#d62728", "-"), "XGBoost": ("#9467bd", "--"), "SVM-RBF": ("#8c564b", "-."),
         "LogisticRegression": ("#7f7f7f", ":")}


def _interp_roc(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    t = np.interp(GRID, fpr, tpr)
    t[0] = 0.0
    return t


SHORT = {"RandomForest": "RF", "SVM-RBF": "SVM", "LogisticRegression": "LR"}


def part_roc(df, seeds):
    y = df[TARGET].to_numpy(dtype=int)
    lab = spatial_fold_labels(df)
    files = oof_files(seeds)
    summ = pd.read_csv(REV / "model_comparison_summary.csv")
    sp = summ[(summ.split == "spatial") & (summ.dataset == "terrain")].set_index("model")
    within, pooled = {}, {}
    for m in MODEL_ORDER:
        wf, pf = [], []
        for s, f in files.items():
            p = pd.read_csv(f)[f"{m}|terrain"].to_numpy(float)
            wf.append(np.mean([_interp_roc(y[lab == k], p[lab == k]) for k in range(N_SPLITS)], axis=0))
            pf.append(_interp_roc(y, p))
        within[m], pooled[m] = np.mean(wf, axis=0), np.mean(pf, axis=0)
    for curves, title, col, lab_txt, fn in [
        (within, "Macro-averaged within-fold ROC curves under spatial block CV (terrain features)",
         "within_mean_AUC_mean", "within-fold AUC", "fig12_roc_within.png"),
        (pooled, "Pooled out-of-fold ROC curves under spatial block CV (terrain features)\n"
         "pooled curves are affected by fold-wise prior shift (landslide share 0.01-0.84 across folds)",
         "pooled_AUC_mean", "pooled AUC", "figS_roc_pooled.png")]:
        fig, ax = plt.subplots(figsize=(6.4, 6.2))
        for m in sorted(curves, key=lambda m: -sp.loc[m, col]):
            c, ls = STYLE[m]
            ax.plot(GRID, curves[m], color=c, ls=ls, lw=1.8, label=f"{SHORT.get(m, m)} ({lab_txt} {sp.loc[m, col]:.3f})")
        ax.plot([0, 1], [0, 1], c="0.6", lw=0.8)
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.set_title(title, fontsize=8.5)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.set_aspect("equal")
        savefig(fig, FIG / fn)
    print("ROC figures written; seeds used:", list(files))


# --------------------------------------------------------------------------
# g. geodistance diagnostic
# --------------------------------------------------------------------------


def part_geodist(df):
    from scipy.stats import wasserstein_distance
    from shapely import contains_xy
    from sklearn.model_selection import StratifiedKFold

    xy = project_km(df.Latitude, df.Longitude)
    y = df[TARGET].to_numpy(dtype=int)
    poly = district_polygon(FIXED_LAT0)
    x0, y0, x1, y1 = poly.bounds
    gx, gy = np.meshgrid(np.arange(np.floor(x0), x1 + 1, 1.0) + 0.5, np.arange(np.floor(y0), y1 + 1, 1.0) + 0.5)
    gx, gy = gx.ravel(), gy.ravel()
    ins = contains_xy(poly, gx, gy)
    grid = np.column_stack([gx[ins], gy[ins]])
    d_pred = cKDTree(xy).query(grid, k=1)[0]

    def cv_dist(folds):
        d = np.empty(len(xy))
        for tr, te in folds:
            d[te] = cKDTree(xy[tr]).query(xy[te], k=1)[0]
        return d

    lab = spatial_fold_labels(df)
    sp_folds = folds_from_labels(lab)
    rnd_folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(xy, y))
    d_sp = cv_dist(sp_folds)
    d_rand = cv_dist(rnd_folds)
    qs = (0.05, 0.25, 0.5, 0.75, 0.95)

    def summ(d):
        return {"n": int(len(d)), "mean_km": float(d.mean()), "max_km": float(d.max()),
                **{f"q{int(q * 100):02d}_km": float(np.quantile(d, q)) for q in qs}}

    def cv_dist_cls(folds, test_cls):
        """Distance of test samples of class ``test_cls`` to the nearest *training landslide*."""
        out = []
        for tr, te in folds:
            tr_pos = tr[y[tr] == 1]
            te_c = te[y[te] == test_cls]
            if len(te_c):
                out.append(cKDTree(xy[tr_pos]).query(xy[te_c], k=1)[0])
        return np.concatenate(out)

    d_pred_ls = cKDTree(xy[y == 1]).query(grid, k=1)[0]
    d_pred_neg = cKDTree(xy[y == 0]).query(grid, k=1)[0]
    d_sp_ls, d_rand_ls = cv_dist_cls(sp_folds, 1), cv_dist_cls(rnd_folds, 1)
    d_sp_neg, d_rand_neg = cv_dist_cls(sp_folds, 0), cv_dist_cls(rnd_folds, 0)

    def summ_cc(d):
        return {"n": int(len(d)), **{f"q{int(q * 100):02d}_km": float(np.quantile(d, q)) for q in qs},
                "median_km": float(np.median(d))}

    cc = {"prediction_to_nearest_landslide": summ_cc(d_pred_ls),
          "prediction_to_nearest_negative": summ_cc(d_pred_neg),
          "random_cv_test_landslide_to_train_landslide": summ_cc(d_rand_ls),
          "spatial_cv_test_landslide_to_train_landslide": summ_cc(d_sp_ls),
          "random_cv_test_negative_to_train_landslide": summ_cc(d_rand_neg),
          "spatial_cv_test_negative_to_train_landslide": summ_cc(d_sp_neg),
          "_note": "median, q25, q75, q95 (and q05) in km; random CV = StratifiedKFold(5, shuffle, rs=42); "
                   "train landslide = training-fold samples with y==1"}

    res = {"prediction_to_sample_grid_1km": summ(d_pred), "spatial_block_cv": summ(d_sp), "random_cv": summ(d_rand),
           "wasserstein_to_prediction": {"spatial_block_cv": float(wasserstein_distance(d_pred, d_sp)),
                                         "random_cv": float(wasserstein_distance(d_pred, d_rand))},
           "n_grid_cells": int(len(d_pred)), "district_area_km2": float(poly.area),
           "class_conditional": cc}
    (OUT / "geodist_summary.json").write_text(json.dumps(res, indent=2))

    def ecdf(ax, d, lbl, c, ls):
        xs = np.sort(d)
        ax.step(np.r_[0, xs], np.r_[0, np.arange(1, len(xs) + 1) / len(xs)], where="post", color=c, ls=ls, lw=1.8,
                label=f"{lbl}, median {np.median(d):.2f} km" if np.median(d) < 1 else
                f"{lbl}, median {np.median(d):.1f} km")

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axs[0]
    for d, lbl, c, ls in [(d_pred, f"Prediction area (1-km grid, n={len(d_pred)})", "#222222", "-"),
                          (d_sp, "Spatial block CV (test-to-train)", "#d62728", "-"),
                          (d_rand, "Random 5-fold CV (test-to-train)", "#1f77b4", "--")]:
        ecdf(ax, d, lbl, c, ls)
    ax.set_xlabel("Nearest-neighbour distance to training samples, either class (km)")
    ax.set_xlim(0, float(max(d_pred.max(), d_sp.max(), d_rand.max())) * 1.02)
    ax.set_title("(a) All samples", fontsize=10)
    ax = axs[1]
    for d, lbl, c, ls in [(d_pred_ls, f"Prediction area (1-km grid) to landslide", "#222222", "-"),
                          (d_sp_ls, "Spatial CV: test landslide to training landslide", "#d62728", "-"),
                          (d_rand_ls, "Random CV: test landslide to training landslide", "#1f77b4", "--")]:
        ecdf(ax, d, lbl, c, ls)
    ax.set_xlabel("Nearest-neighbour distance to (training) landslides (km)")
    ax.set_xlim(0, float(max(d_pred_ls.max(), d_sp_ls.max(), d_rand_ls.max())) * 1.02)
    ax.set_title("(b) Landslides only", fontsize=10)
    for ax in axs:
        ax.set_ylabel("ECDF")
        ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Geodistance diagnostic (NNDM-style)", fontsize=11)
    fig.tight_layout()
    savefig(fig, FIG / "figS_geodist.png")
    print(json.dumps(res, indent=1))


# --------------------------------------------------------------------------
# i. sensitivity: drop negatives on LULC water pixels
# --------------------------------------------------------------------------


def water_seed(sub, folds, seed, models):
    rows = []
    for name in models:
        set_global_seeds(seed, deterministic=False)
        cv = cross_validate_model(lambda name=name: get_model(name, random_state=seed, **TERRAIN_KW), sub, folds)
        rows.append(cv_row(seed, name, cv))
        print(f"[water seed {seed}] {LABEL[name]} within={rows[-1]['within_AUC']:.3f} "
              f"pooled={rows[-1]['pooled_AUC']:.3f}", flush=True)
    return pd.DataFrame(rows)


def part_water(df, seeds, models):
    y = df[TARGET].to_numpy(dtype=int)
    water = (df.LULC == 1).fillna(False).to_numpy(dtype=bool)
    lab = spatial_fold_labels(df)
    drop = water & (y == 0)
    keep = ~drop
    sub = df[keep].reset_index(drop=True)
    sub_lab = lab[keep]
    ysub = sub[TARGET].to_numpy(dtype=int)
    folds = folds_from_labels(sub_lab)
    # alternative: re-derive spatial folds on the reduced frame (as the bbox part does); composition only
    alt = list(spatial_block_cv(sub))
    meta = {"fold_mode": "existing district-wide spatial fold labels kept (no re-clustering); "
                         "water negatives dropped; landslides on water pixels kept",
            "n_total_original": int(len(df)),
            "water_negatives": int((water & (y == 0)).sum()), "water_landslides": int((water & (y == 1)).sum()),
            "n_samples": int(len(sub)), "n_pos": int(ysub.sum()), "n_neg": int((1 - ysub).sum()),
            "lulc_class_counts_neg": {str(k): int(v) for k, v in df[y == 0].LULC.value_counts(dropna=False).items()},
            "fold_n_original": [int((lab == k).sum()) for k in sorted(np.unique(lab))],
            "fold_pos_original": [int(y[lab == k].sum()) for k in sorted(np.unique(lab))],
            "fold_n": [int((sub_lab == k).sum()) for k in sorted(np.unique(sub_lab))],
            "fold_pos": [int(ysub[sub_lab == k].sum()) for k in sorted(np.unique(sub_lab))],
            "fold_neg": [int(((sub_lab == k) & (ysub == 0)).sum()) for k in sorted(np.unique(sub_lab))],
            "fold_water_neg_dropped": [int((drop & (lab == k)).sum()) for k in sorted(np.unique(lab))],
            "alt_rederived_fold_n": [int(len(t)) for _, t in alt],
            "alt_rederived_fold_pos": [int(ysub[t].sum()) for _, t in alt],
            "alt_rederived_same_as_kept": bool(all(np.array_equal(np.sort(t), np.sort(np.where(sub_lab == k)[0]))
                                                   for k, (_, t) in enumerate(alt)))}
    (OUT / "water_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1))
    for s in seeds:
        cached("water", s, lambda s=s: water_seed(sub, folds, s, models))
    d = load_cached("water", seeds)
    if len(d):
        d["model"] = d.model.replace({"RandomForest": "RF", "LogisticRegression": "LR", "SVM-RBF": "SVM"})
        d.to_csv(OUT / "water_per_seed.csv", index=False)
        sm = summarize_cv(d, ["model"]).rename(columns={"within_AUC_sd": "within_AUC_std",
                                                         "pooled_AUC_sd": "pooled_AUC_std"})
        sm["n_samples"], sm["n_neg"] = meta["n_samples"], meta["n_neg"]
        sm = sm[["model", "within_AUC_mean", "within_AUC_std", "pooled_AUC_mean", "pooled_AUC_std",
                 "n_samples", "n_neg", "n_seeds"]]
        sm.round(4).to_csv(OUT / "water_sensitivity.csv", index=False)
        print(sm.round(3).to_string())


# --------------------------------------------------------------------------
# h. point-level susceptibility map with area of applicability
# --------------------------------------------------------------------------

CLASSES = ["Very low", "Low", "Moderate", "High", "Very high"]
CLASS_COLORS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
DI_FEATURES = ["Elevation", "Slope", "Curvature", "TWI", "TRI", "NDVI", "Annual_Rainfall"]
IMP_NAME = {"Annual_Rainfall": "Annual rainfall"}


def di_weights() -> np.ndarray:
    try:
        imp = pd.read_csv(REV / "shap_importance_by_model.csv", index_col=0)["RF"]
        w = np.array([float(imp[IMP_NAME.get(c, c)]) for c in DI_FEATURES])
        if np.all(np.isfinite(w)) and w.sum() > 0:
            return w / w.sum() * len(w)
    except Exception as e:  # noqa: BLE001
        print("DI weights unavailable, unweighted:", e)
    return np.ones(len(DI_FEATURES))


def infold_di_weights(df, lab, seeds=tuple(SEEDS_ALL)) -> np.ndarray:
    """Fold-specific DI weights that use NO test-fold information.

    For each spatial fold k: fit the map-model RF (terrain features, no SMOTE) on the training folds only,
    compute mean |TreeSHAP| per factor (same factor grouping as the manuscript SHAP analysis) on the TRAINING
    points, average over seeds, normalise to mean 1. Returns array (N_SPLITS, len(DI_FEATURES)).
    """
    import shap

    from .revision_analyses import factor_importance

    y = df[TARGET].to_numpy(int)
    W = np.zeros((N_SPLITS, len(DI_FEATURES)))
    for k in range(N_SPLITS):
        tr = np.where(lab != k)[0]
        imps = []
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
            imp = factor_importance(pd.DataFrame(sv, columns=names))
            imps.append(np.array([float(imp[IMP_NAME.get(c, c)]) for c in DI_FEATURES]))
        w = np.mean(imps, axis=0)
        W[k] = w / w.sum() * len(w)
    return W


def compute_di(df, lab, weights):
    """Meyer & Pebesma (2021) DI of each point w.r.t. its training folds (the other spatial folds).

    ``weights`` is either one vector (global) or an (N_SPLITS, n_features) array of fold-specific weights;
    the fold-k weights are used both for the training-DI threshold and for the fold-k test-point DI.
    """
    from scipy.spatial.distance import pdist

    weights = np.asarray(weights, float)
    X = df[DI_FEATURES].to_numpy(float)
    di = np.full(len(df), np.nan)
    thr, dbar_k = np.zeros(N_SPLITS), np.zeros(N_SPLITS)
    for k in range(N_SPLITS):
        wk = weights[k] if weights.ndim == 2 else weights
        tr, te = np.where(lab != k)[0], np.where(lab == k)[0]
        Z = np.where(np.isnan(X), np.nanmedian(X[tr], axis=0), X)
        Z = (Z - Z[tr].mean(0)) / Z[tr].std(0) * wk
        Ztr = Z[tr]
        dbar = pdist(Ztr).mean()
        dbar_k[k] = dbar
        di[te] = cKDTree(Ztr).query(Z[te], k=1)[0] / dbar
        dtr = np.empty(len(tr))
        for f in set(range(N_SPLITS)) - {k}:  # training DI: nearest training point in another fold
            m = lab[tr] == f
            dtr[m] = cKDTree(Ztr[~m]).query(Ztr[m], k=1)[0] / dbar
        q1, q3 = np.percentile(dtr, [25, 75])
        thr[k] = q3 + 1.5 * (q3 - q1)
    return di, thr, dbar_k


def aoa_variants(df, lab, y, w_global, w_infold) -> pd.DataFrame:
    """Sensitivity table of AOA size across encoding / weighting choices (diagnostics item 18 + in-fold)."""
    from scipy.spatial.distance import pdist

    shap_rf = pd.read_csv(REV / "shap_values_random_forest.csv", index_col=0)
    names23 = list(shap_rf.columns)
    w23 = shap_rf.abs().mean().to_numpy()
    w23n = w23 / w23.sum() * len(w23)

    def summ(name, desc, o):
        return {"variant": name, "description": desc, "n_outside_AOA": int(o.sum()),
                "by_fold": ",".join(str(int(o[lab == k].sum())) for k in range(N_SPLITS)),
                "landslides_outside": int(o[y == 1].sum()), "non_landslides_outside": int(o[y == 0].sum())}

    def di_matrix(weights, standardise):
        di2, thr2 = np.full(len(df), np.nan), np.zeros(N_SPLITS)
        for k in range(N_SPLITS):
            tr, te = np.where(lab != k)[0], np.where(lab == k)[0]
            pre = build_preprocessor(include_cluster=False, drainage_mode="drop")
            pre.fit(df.iloc[tr], y[tr])
            Z = np.asarray(pre.transform(df), float)
            if standardise:
                sd = Z[tr].std(0)
                sd[sd == 0] = 1.0
                Z = (Z - Z[tr].mean(0)) / sd
            Z = Z * weights
            Ztr = Z[tr]
            dbar_ = pdist(Ztr).mean()
            di2[te] = cKDTree(Ztr).query(Z[te], k=1)[0] / dbar_
            dtr = np.empty(len(tr))
            for f in set(range(N_SPLITS)) - {k}:
                mm = lab[tr] == f
                dtr[mm] = cKDTree(Ztr[~mm]).query(Ztr[mm], k=1)[0] / dbar_
            q1, q3 = np.percentile(dtr, [25, 75])
            thr2[k] = q3 + 1.5 * (q3 - q1)
        return di2 > thr2[lab]

    di, thr, _ = compute_di(df, lab, w_global)
    rows = [summ("current_global_weights", "7 continuous, global OOF-SHAP weights (uses test-fold information)",
                 di > thr[lab])]
    di, thr, _ = compute_di(df, lab, w_infold)
    rows.append(summ("infold_weights", "7 continuous, fold-specific TreeSHAP weights from training folds only (ADOPTED)",
                     di > thr[lab]))
    w7 = np.array([w_global[DI_FEATURES.index(c)] if c in DI_FEATURES else 0.0 for c in names23])
    rows += [
        summ("continuous7_via_encoded", "7 continuous (global weights) through encoded matrix, KNN-imputed",
             di_matrix(w7, True)),
        summ("all23_weighted_standardised", "23 encoded, per-column mean|SHAP| weights (global), standardised",
             di_matrix(w23n, True)),
        summ("all23_weighted_as_preprocessed", "23 encoded, weighted (global), not re-standardised",
             di_matrix(w23n, False)),
        summ("all23_unweighted_standardised", "23 encoded, unweighted, standardised", di_matrix(np.ones(23), True)),
        summ("all23_unweighted_as_preprocessed", "23 encoded, unweighted, not re-standardised",
             di_matrix(np.ones(23), False))]
    return pd.DataFrame(rows)


def draw_boundary(ax):
    import geopandas as gpd

    g = gpd.read_file(BOUNDARY_FILE).geometry.union_all()
    for poly in getattr(g, "geoms", [g]):
        ax.plot(*poly.exterior.xy, color="0.25", lw=0.8)
        for r in poly.interiors:
            ax.plot(*r.xy, color="0.25", lw=0.5)
    return g.bounds


def part_map(df):
    from matplotlib.lines import Line2D

    y = df[TARGET].to_numpy(dtype=int)
    lab = spatial_fold_labels(df)
    summ = pd.read_csv(REV / "model_comparison_summary.csv")
    sm = summ[(summ.split == "spatial") & (summ.dataset == "terrain+SMOTE")].sort_values(
        "within_mean_AUC_mean", ascending=False)
    chosen = sm.iloc[0].model
    (OUT / "map_model.json").write_text(json.dumps({
        "model": chosen, "dataset": "terrain+SMOTE", "rule": "highest 10-seed within-fold AUC among terrain+SMOTE",
        "candidates": {r.model: round(float(r.within_mean_AUC_mean), 4) for r in sm.itertuples()}}, indent=1))
    ens = pd.read_csv(REV / "oof_seed_ensemble.csv")
    p = ens[f"{chosen}|terrain+SMOTE"].to_numpy(float)
    br = np.quantile(p, [0.2, 0.4, 0.6, 0.8])
    cls = np.digitize(p, br)
    w_global = di_weights()  # old weights (global OOF SHAP; uses test-fold info) - kept for the sensitivity table only
    w_if = infold_di_weights(df, lab)
    di, thr, dbar = compute_di(df, lab, w_if)
    out_aoa = di > thr[lab]
    var = aoa_variants(df, lab, y, w_global, w_if)
    var.to_csv(OUT / "aoa_variants.csv", index=False)
    print(var.to_string())
    w = w_if
    edges = np.r_[p.min(), br, p.max()]
    rows = []
    for c in range(5):
        m = cls == c
        rows.append({"class": CLASSES[c], "prob_min": p[m].min(), "prob_max": p[m].max(), "n_samples": int(m.sum()),
                     "n_landslides": int(y[m].sum()), "share_of_landslides": y[m].sum() / y.sum(),
                     "share_of_samples": m.mean(), "frequency_ratio": (y[m].sum() / y.sum()) / m.mean(),
                     "n_outside_AOA": int(out_aoa[m].sum())})
    pd.DataFrame(rows).round(4).to_csv(OUT / "map_class_stats.csv", index=False)
    print(chosen, "\n", pd.DataFrame(rows).round(3).to_string())
    aoa = {"model": chosen, "di_features": DI_FEATURES, "di_weights": "fold-specific mean|TreeSHAP| (RF, terrain, no SMOTE, 10-seed mean) on TRAINING points of the training folds only; normalised to mean 1",
           "di_weights_normalised_mean1_by_fold": {f"fold{k + 1}": dict(zip(DI_FEATURES, map(float, w_if[k]))) for k in range(N_SPLITS)},
           "n_outside_by_fold_and_landslide": {f"fold{k + 1}": [int(out_aoa[(lab == k) & (y == 1)].sum()), int(out_aoa[(lab == k) & (y == 0)].sum())] for k in range(N_SPLITS)},
           "threshold_rule": "Q3 + 1.5*IQR of training-point DI (nearest training point in another fold), per training set",
           "threshold_per_fold": {f"fold{k + 1}": float(thr[k]) for k in range(N_SPLITS)},
           "mean_pairwise_dist_per_fold": {f"fold{k + 1}": float(dbar[k]) for k in range(N_SPLITS)},
           "n_outside_AOA": int(out_aoa.sum()), "share_outside_AOA": float(out_aoa.mean()),
           "share_outside_by_class": {CLASSES[c]: float(out_aoa[cls == c].mean()) for c in range(5)},
           "share_outside_by_fold": {f"fold{k + 1}": float(out_aoa[lab == k].mean()) for k in range(N_SPLITS)},
           "n_outside_by_fold": {f"fold{k + 1}": int(out_aoa[lab == k].sum()) for k in range(N_SPLITS)},
           "share_outside_landslides": float(out_aoa[y == 1].mean()),
           "share_outside_nonlandslides": float(out_aoa[y == 0].mean()),
           "DI_quantiles": {str(q): float(np.quantile(di, q)) for q in (0.5, 0.9, 0.99)}}
    (OUT / "aoa_summary.json").write_text(json.dumps(aoa, indent=2))
    pd.DataFrame({"Latitude": df.Latitude, "Longitude": df.Longitude, "Landslide": y, "fold": lab + 1, "prob": p,
                  "class": [CLASSES[c] for c in cls], "DI": di, "DI_threshold": thr[lab], "outside_AOA": out_aoa}
                 ).to_csv(OUT / "map_points.csv", index=False)
    print(json.dumps(aoa, indent=1))

    fig, ax = plt.subplots(figsize=(6.2, 8.4))
    bx0, by0, bx1, by1 = draw_boundary(ax)
    lon, lat = df.Longitude.to_numpy(), df.Latitude.to_numpy()
    for c in range(5):
        for lv, mk, s in ((0, "o", 22), (1, "^", 30)):
            m = (cls == c) & (y == lv)
            ax.scatter(lon[m], lat[m], s=s, marker=mk, color=CLASS_COLORS[c], edgecolor="0.25", lw=0.4, zorder=3)
    ax.scatter(lon[out_aoa], lat[out_aoa], s=80, facecolors="none", edgecolors="k", lw=0.9, zorder=4)
    ax.set_aspect(1 / np.cos(np.deg2rad(FIXED_LAT0)))
    ax.set_xlim(bx0 - 0.03, bx1 + 0.03)
    ax.set_ylim(by0 - 0.03, by1 + 0.03)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}°E")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}°N")
    km_deg = 1 / (111.320 * np.cos(np.deg2rad(FIXED_LAT0)))
    xs, ys = bx1 - 0.03 - 20 * km_deg, by0 + 0.01
    ax.plot([xs, xs + 20 * km_deg], [ys, ys], color="k", lw=3, zorder=5, solid_capstyle="butt")
    ax.text(xs + 10 * km_deg, ys + 0.012, "20 km", ha="center", va="bottom", fontsize=8)
    ax.annotate("N", xy=(bx1 - 0.02, by1 + 0.02), xytext=(bx1 - 0.02, by1 - 0.06), ha="center", fontsize=10,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.4))
    handles = [Line2D([], [], marker="s", ls="", color=CLASS_COLORS[c], mec="0.25",
                      label=f"{CLASSES[c]} ({edges[c]:.2f}–{edges[c + 1]:.2f})") for c in range(5)]
    handles += [Line2D([], [], marker="^", ls="", color="0.8", mec="0.25", label=f"Landslide (n={int(y.sum())})"),
                Line2D([], [], marker="o", ls="", color="0.8", mec="0.25", label=f"Non-landslide (n={int((1 - y).sum())})"),
                Line2D([], [], marker="o", ls="", mfc="none", mec="k", ms=9,
                       label=f"Outside AOA (n={int(out_aoa.sum())})")]
    ax.legend(handles=handles, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              title="Susceptibility class (quintiles)", title_fontsize=8, framealpha=0.9, borderaxespad=0)
    short = {"RandomForest": "RF", "SVM-RBF": "SVM", "LogisticRegression": "LR"}.get(chosen, chosen)
    ax.set_title(f"Point-level landslide susceptibility\n(out-of-fold, {short} + SMOTE, mean of 10 seeds)",
                 fontsize=9)
    savefig(fig, FIG / "fig15_susceptibility_points.png")


# --------------------------------------------------------------------------
# i. fold map
# --------------------------------------------------------------------------


def part_foldmap(df):
    from matplotlib.lines import Line2D

    y = df[TARGET].to_numpy(dtype=int)
    lab = spatial_fold_labels(df)
    fig, ax = plt.subplots(figsize=(6, 8))
    bx0, by0, bx1, by1 = draw_boundary(ax)
    pal = plt.get_cmap("Set1")
    cols = [pal(i) for i in (0, 1, 2, 4, 3)]
    handles = []
    for k in range(N_SPLITS):
        m = lab == k
        ax.scatter(df.Longitude[m & (y == 0)], df.Latitude[m & (y == 0)], s=16, marker="x", color=cols[k], lw=0.9)
        ax.scatter(df.Longitude[m & (y == 1)], df.Latitude[m & (y == 1)], s=16, marker="o", color=cols[k],
                   ec="k", lw=0.3)
        handles.append(Line2D([], [], marker="s", ls="", color=cols[k],
                              label=f"Fold {k + 1} (n={m.sum()}, landslides={int(y[m].sum())})"))
    handles += [Line2D([], [], marker="o", ls="", color="0.6", mec="k", label="Landslide"),
                Line2D([], [], marker="x", ls="", color="0.4", label="Non-landslide")]
    ax.set_aspect(1 / np.cos(np.deg2rad(FIXED_LAT0)))
    ax.set_xlim(bx0 - 0.03, bx1 + 0.03)
    ax.set_ylim(by0 - 0.03, by1 + 0.03)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend(handles=handles, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.9,
              borderaxespad=0)
    ax.set_title("(a) Spatial block cross-validation folds", fontsize=10)
    savefig(fig, FIG / "fig13a_spatial_folds.png")
    print("fold map written")


# --------------------------------------------------------------------------
# j. Rabby & Li overlap (reads an inventory downloaded into _external_rabby; data only)
# --------------------------------------------------------------------------

RABBY_DIR = Path("/Users/murad/Documents/Thesis_work/Revision/_external_rabby")


def part_rabby(df):
    import geopandas as gpd
    from shapely.geometry import Point

    files = [p for p in RABBY_DIR.rglob("*") if p.suffix.lower() in (".shp", ".geojson", ".gpkg", ".kml")]
    if not files:
        res = {"status": "not_obtained", "note": f"no readable inventory files in {RABBY_DIR}"}
        (OUT / "rabby_overlap.json").write_text(json.dumps(res, indent=2))
        print(res)
        return
    res = {"status": "ok", "layers": {}}
    poly = district_polygon(FIXED_LAT0)
    xy = project_km(df.Latitude, df.Longitude)
    y = df[TARGET].to_numpy(dtype=int)
    for p in files:
        g = gpd.read_file(p)
        if g.crs is None:
            g = g.set_crs(4326)
        g = g.to_crs(4326)
        c = g.geometry.representative_point()
        r = project_km(c.y.to_numpy(), c.x.to_numpy())
        inside = np.array([poly.contains(Point(*q)) for q in r])
        lay = {"n_records": int(len(g)), "n_in_rangamati_polygon": int(inside.sum()),
               "geometry_types": sorted(set(g.geom_type))}
        d = cKDTree(r).query(xy[y == 1], k=1)[0]
        lay["landslide_share_within_0.5km_of_record"] = float((d <= 0.5).mean())
        lay["landslide_share_within_1km_of_record"] = float((d <= 1.0).mean())
        lay["median_dist_km_landslides_to_nearest_record"] = float(np.median(d))
        res["layers"][str(p.relative_to(RABBY_DIR))] = lay
    (OUT / "rabby_overlap.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str))


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["folds", "bbox", "ablation", "cnnorder", "thresholds", "roc", "geodist", "map",
                             "foldmap", "rabby", "water"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS_ALL)
    ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    ap.add_argument("--data", default="Labeled Data Set.csv")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_df(a.data)
    t0 = time.time()
    if a.part == "folds":
        part_folds(df, a.seeds, a.models)
    elif a.part == "bbox":
        part_bbox(df, a.seeds, a.models)
    elif a.part == "ablation":
        part_ablation(df, a.seeds, a.models)
    elif a.part == "cnnorder":
        part_cnnorder(df, a.seeds)
    elif a.part == "thresholds":
        part_thresholds(df, a.seeds)
    elif a.part == "roc":
        part_roc(df, a.seeds)
    elif a.part == "geodist":
        part_geodist(df)
    elif a.part == "map":
        part_map(df)
    elif a.part == "foldmap":
        part_foldmap(df)
    elif a.part == "rabby":
        part_rabby(df)
    elif a.part == "water":
        part_water(df, a.seeds, a.models)
    print(f"[{a.part}] total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
