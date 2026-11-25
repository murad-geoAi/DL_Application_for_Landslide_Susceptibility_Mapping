"""Second-round SHAP checks and figure fixes (audit deepreview-2).

    python -m src.revision_rev2_shap --part cnn_grouped [--seeds ...]
    python -m src.revision_rev2_shap --part shap_checks
    python -m src.revision_rev2_shap --part figs
    python -m src.revision_rev2_shap --part summary

Writes only under results/revision/extra/rev2_shap/ and new ``*_v2.png`` figures.
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

from .data import TARGET  # noqa: E402
from .evaluate import cross_validate_model  # noqa: E402
from .models import get_model, set_global_seeds  # noqa: E402
from .revision_analyses import _factor, _smote_pipe, SHAP_SEEDS_ALL  # noqa: E402
from .revision_extra import FIG, REV, OUT as EXTRA, load_df, savefig  # noqa: E402
from .splits import N_SPLITS, spatial_block_cv, spatial_fold_labels  # noqa: E402

warnings.filterwarnings("ignore")
R2 = EXTRA / "rev2_shap"
CONT = ["Elevation", "Slope", "Curvature", "TWI", "TRI", "NDVI", "Annual_Rainfall"]


def group12(name: str) -> str:
    if name == "TWI_was_missing":
        return "TWI missing flag"
    if name == "Soil_Texture_was_missing":
        return "Soil missing flag"
    return _factor(name)


# --------------------------------------------------------------------------
# exact grouped Shapley (all 2^P coalitions)
# --------------------------------------------------------------------------


def coalition_masks(P: int) -> np.ndarray:
    m = np.arange(2 ** P)
    return ((m[:, None] >> np.arange(P)[None, :]) & 1).astype(bool)  # (2^P, P), bit i = player i


def shapley_from_values(V: np.ndarray, P: int) -> np.ndarray:
    """Exact Shapley values from coalition values V (n, 2^P); index bits = players."""
    from math import factorial

    masks = np.arange(2 ** P)
    size = np.array([bin(s).count("1") for s in masks])
    w = np.array([factorial(k) * factorial(P - k - 1) / factorial(P) for k in range(P)])
    phi = np.zeros((V.shape[0], P))
    for i in range(P):
        wo = masks[(masks >> i) & 1 == 0]
        phi[:, i] = ((V[:, wo | (1 << i)] - V[:, wo]) * w[size[wo]]).sum(1)
    return phi


def grouped_kernel_shap(model, Xte, bg, bgw, groups: list[str], chunk_rows=400_000):
    """Exact Shapley over parent-factor groups; 'off' groups take background values,
    model output averaged over the (weighted) background. Returns (phi, f(x), E[f(bg)])."""
    names = list(dict.fromkeys(groups))
    P = len(names)
    gi = np.array([names.index(g) for g in groups])
    M = coalition_masks(P)  # (2^P, P)
    C = M[:, gi]  # (2^P, d) column on/off
    B, d = bg.shape
    bgw = np.asarray(bgw, float) / np.sum(bgw)
    per_pt = (2 ** P) * B
    c = max(1, chunk_rows // per_pt)
    V = np.empty((len(Xte), 2 ** P))
    k = model.model_
    for s in range(0, len(Xte), c):
        xc = Xte[s:s + c]
        data = np.where(C[None, :, None, :], xc[:, None, None, :], bg[None, None, :, :])
        flat = data.reshape(-1, d)
        p = k.predict(model._reshape(flat), batch_size=16384, verbose=0).ravel()
        V[s:s + c] = (p.reshape(len(xc), 2 ** P, B) * bgw).sum(-1)
    phi = shapley_from_values(V, P)
    full, empty = V[:, -1], V[:, 0]
    assert np.allclose(phi.sum(1), full - empty, atol=1e-5)
    return pd.DataFrame(phi, columns=names), full, empty


def cnn_grouped_seed(df, seed):
    import shap

    set_global_seeds(seed, deterministic=False)
    folds = list(spatial_block_cv(df))
    cv = cross_validate_model(lambda: get_model("cnn1d", random_state=seed, include_cluster=False,
                                                drainage_mode="drop"), df, folds)
    parts10, parts12, oofs = [], [], []
    for (tr, te), pipe in zip(folds, cv["fitted"]):
        pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
        Xtr = np.asarray(pre.transform(df.iloc[tr]), dtype=float)
        Xte = np.asarray(pre.transform(df.iloc[te]), dtype=float)
        names = [str(n) for n in pre.get_feature_names_out()]
        bg = shap.kmeans(Xtr, 20)
        for groups, acc in (([_factor(n) for n in names], parts10), ([group12(n) for n in names], parts12)):
            phi, full, empty = grouped_kernel_shap(model, Xte, np.asarray(bg.data, float), bg.weights, groups)
            phi.index = te
            acc.append(phi)
    return (pd.concat(parts10).sort_index(), pd.concat(parts12).sort_index(), float(cv["pooled"]["AUC"]))


def part_cnn_grouped(df, seeds):
    R2.mkdir(parents=True, exist_ok=True)
    for sd in seeds:
        f10, f12 = R2 / f"cnn_grouped10_seed{sd}.csv", R2 / f"cnn_grouped12_seed{sd}.csv"
        if f10.exists() and f12.exists():
            continue
        t0 = time.time()
        s10, s12, auc = cnn_grouped_seed(df, sd)
        s10.to_csv(f10)
        s12.to_csv(f12)
        (R2 / f"cnn_grouped_seed{sd}.json").write_text(json.dumps({"seed": sd, "seconds": time.time() - t0,
                                                                   "pooled_AUC": auc}))
        print(f"[cnn_grouped] seed {sd} {time.time() - t0:.0f}s AUC {auc:.4f}", flush=True)


# --------------------------------------------------------------------------
# shap checks (RF TreeSHAP, RF+SMOTE TreeSHAP, flags, within-fold rho)
# --------------------------------------------------------------------------


def _tree_oof(df, seed, smote: bool):
    import shap

    set_global_seeds(seed, deterministic=False)
    folds = list(spatial_block_cv(df))
    kw = dict(random_state=seed, include_cluster=False, drainage_mode="drop")
    fn = (lambda: _smote_pipe("random_forest", seed, include_cluster=False, drainage_mode="drop")) if smote \
        else (lambda: get_model("random_forest", **kw))
    cv = cross_validate_model(fn, df, folds)
    vals = []
    for (tr, te), pipe in zip(folds, cv["fitted"]):
        pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
        est = model.estimator_ if smote else model
        Xte = np.asarray(pre.transform(df.iloc[te]), dtype=float)
        names = [str(n) for n in pre.get_feature_names_out()]
        sv = shap.TreeExplainer(est).shap_values(Xte)
        sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        vals.append(pd.DataFrame(sv, columns=names, index=te))
    return pd.concat(vals).sort_index(), float(cv["pooled"]["AUC"])


def _tree_run(df, seeds, tag, smote):
    f = R2 / f"{tag}_percol_by_seed.csv"
    fm = R2 / f"{tag}_shap_seedmean.csv"
    if f.exists() and fm.exists():
        return pd.read_csv(f, index_col=0), pd.read_csv(fm, index_col=0)
    rows, acc, aucs = {}, None, {}
    for sd in seeds:
        t0 = time.time()
        S, auc = _tree_oof(df, sd, smote)
        rows[sd] = S.abs().mean()
        acc = S if acc is None else acc + S
        aucs[sd] = auc
        print(f"[{tag}] seed {sd} {time.time() - t0:.0f}s AUC {auc:.4f}", flush=True)
    pc = pd.DataFrame(rows).T.rename_axis("seed")
    sm = acc / len(seeds)
    pc.to_csv(f)
    sm.to_csv(fm)
    (R2 / f"{tag}_aucs.json").write_text(json.dumps(aucs))
    return pc, sm


def rho_table(df, S: pd.DataFrame, lab: np.ndarray) -> pd.DataFrame:
    from scipy.stats import spearmanr

    rows = []
    for c in CONT:
        if c not in S:
            continue
        raw = df.loc[S.index, c].to_numpy(float)
        sv = S[c].to_numpy(float)
        ok = np.isfinite(raw)
        pooled = float(spearmanr(raw[ok], sv[ok]).correlation)
        r = {"factor": c, "pooled_rho": pooled}
        rhos, ns = [], []
        for k in range(N_SPLITS):
            m = ok & (lab[S.index.to_numpy()] == k)
            rho = np.nan
            if m.sum() > 2 and np.ptp(raw[m]) > 0 and np.ptp(sv[m]) > 0:
                rho = float(spearmanr(raw[m], sv[m]).correlation)
            r[f"rho_fold{k + 1}"] = rho
            r[f"n_fold{k + 1}"] = int(m.sum())
            rhos.append(rho)
            ns.append(m.sum())
        rhos, ns = np.array(rhos), np.array(ns, float)
        v = np.isfinite(rhos)
        r["weighted_mean_within_rho"] = float((rhos[v] * ns[v]).sum() / ns[v].sum())
        r["n_folds_defined"] = int(v.sum())
        sg = np.sign(rhos[v])
        r["same_sign_all_folds"] = bool((sg == sg[0]).all())
        r["n_folds_same_sign_as_pooled"] = int((sg == np.sign(pooled)).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def flag_shares(percol_mean: pd.Series) -> dict:
    tot = percol_mean.sum()
    out = {"TWI_was_missing": percol_mean["TWI_was_missing"] / tot,
           "Soil_Texture_was_missing": percol_mean["Soil_Texture_was_missing"] / tot,
           "TWI (continuous column)": percol_mean["TWI"] / tot,
           "Soil_Texture_missing (one-hot level)": percol_mean["Soil_Texture_missing"] / tot,
           "is_flat": percol_mean["is_flat"] / tot,
           "Soil_Texture_* one-hot incl. missing level": sum(v for k, v in percol_mean.items()
                                                             if k.startswith("Soil_Texture_") and k != "Soil_Texture_was_missing") / tot}
    return out


def part_shap_checks(df, seeds):
    lab = spatial_fold_labels(df)
    rf_pc, rf_sm = _tree_run(df, seeds, "rf", False)
    old = pd.read_csv(REV / "shap_values_random_forest.csv", index_col=0)
    print("RF recomputed vs stored seed-mean SHAP max abs diff:", float((rf_sm - old[rf_sm.columns]).abs().max().max()))
    # flags: RF per seed, CNN from stored seed-mean per-column file (per-seed per-column not stored)
    fl = pd.DataFrame({sd: flag_shares(rf_pc.loc[sd]) for sd in rf_pc.index}).T
    fl.to_csv(R2 / "flag_shares_rf_by_seed.csv")
    cnn_old = pd.read_csv(REV / "shap_values_cnn1d.csv", index_col=0)
    cnn_old_pc = cnn_old.abs().mean()
    pd.Series(flag_shares(cnn_old_pc)).to_csv(R2 / "flag_shares_cnn_ungrouped_seedmean.csv")
    # rho tables
    rho_table(df, rf_sm, lab).to_csv(R2 / "rho_rf.csv", index=False)
    try:
        sm_pc, sm_sm = _tree_run(df, seeds, "rf_smote", True)
        rho_table(df, sm_sm, lab).to_csv(R2 / "rho_rf_smote.csv", index=False)
        fl2 = pd.DataFrame({sd: flag_shares(sm_pc.loc[sd]) for sd in sm_pc.index}).T
        fl2.to_csv(R2 / "flag_shares_rf_smote_by_seed.csv")
    except Exception as e:  # pragma: no cover
        print("RF+SMOTE failed:", repr(e))
        (R2 / "rf_smote_error.txt").write_text(repr(e))
    g10 = sorted(R2.glob("cnn_grouped10_seed*.csv"))
    if g10:
        sm = sum(pd.read_csv(f, index_col=0) for f in g10) / len(g10)
        rho_table(df, sm.rename(columns={"Annual rainfall": "Annual_Rainfall"}), lab).to_csv(R2 / "rho_cnn_grouped.csv", index=False)
    rho_table(df, cnn_old, lab).to_csv(R2 / "rho_cnn_ungrouped.csv", index=False)


# --------------------------------------------------------------------------
# figures (v2)
# --------------------------------------------------------------------------

KM_LAT = 110.574
KM_LON0 = 111.320


def km_to_deg(km, lat):
    """(deg lon, deg lat) spanned by ``km`` at latitude ``lat`` (equirectangular, matches to_km)."""
    return km / (KM_LON0 * np.cos(np.deg2rad(lat))), km / KM_LAT


def scale_bar(ax, km, lat, x0, y0, fs=8, label=None):
    """Metric scale bar on a lon/lat axis, left end at (x0, y0) data coords; length is correct at ``lat``."""
    from matplotlib.patches import Rectangle

    dlon, _ = km_to_deg(km, lat)
    _, h = km_to_deg(km * 0.06, lat)
    ax.add_patch(Rectangle((x0, y0), dlon / 2, h, fc="k", ec="k", lw=0.6, zorder=6))
    ax.add_patch(Rectangle((x0 + dlon / 2, y0), dlon / 2, h, fc="w", ec="k", lw=0.6, zorder=6))
    ax.text(x0 + dlon / 2, y0 + 1.8 * h, label or f"{km:g} km", ha="center", va="bottom", fontsize=fs, zorder=6,
            bbox=dict(fc="w", ec="none", alpha=0.7, pad=0.8))


def ref_circle(ax, km, lat, cx, cy, **kw):
    from matplotlib.patches import Ellipse

    dlon, dlat = km_to_deg(km, lat)
    e = Ellipse((cx, cy), 2 * dlon, 2 * dlat, fc="none", ec="k", ls="--", lw=0.9, zorder=6, **kw)
    ax.add_patch(e)


def fig08_v2(df):
    from shapely.geometry import Polygon
    from scipy.spatial import Voronoi

    from .features import FIXED_LAT0
    from .revision_analyses import district_polygon, to_km, voronoi_areas, SEED

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
    savefig(fig, FIG / "fig08_voronoi_quantified_v2.png")


def gi_classification(df, band_km=5.5) -> dict:
    """Gi* hot/cold/ns classification recomputed at draw time (nothing hard-coded to a result table).

    Rule is chosen by ``GI_RULE`` ('auto' | 'perm' | 'analytic'): 'perm' = pseudo p < 0.05 (999 permutations,
    the original pipeline), 'analytic' = |Ord-Getis z| > 1.96. 'auto' picks 'analytic' if
    results/revision/sparsity_metrics.json describes the Gi* result as analytic, else 'perm'.
    If revision_analyses ever exposes ``gi_classify(df, band_km)`` returning {'cat','isl'} it is used instead.
    """
    from .revision_analyses import SEED, to_km
    from .features import FIXED_LAT0
    from . import revision_analyses as ra

    fn = getattr(ra, "gi_classify", None)
    if fn is not None:
        r = fn(df, band_km)
        return {"cat": np.asarray(r["cat"]), "isl": np.asarray(r["isl"]), "rule": "pipeline gi_classify",
                "subtitle": r.get("subtitle", "pipeline classification"), "crit": r.get("crit", "significant"),
                "ns": r.get("ns", "not significant")}
    from esda.getisord import G_Local
    from libpysal.weights import DistanceBand

    rule = os.environ.get("GI_RULE", "auto")
    if rule == "auto":
        try:
            txt = (REV / "sparsity_metrics.json").read_text().lower()
            rule = "analytic" if ("analytic" in txt or "ord_getis" in txt) else "perm"
        except Exception:
            rule = "perm"
    xy = to_km(df.Latitude.to_numpy(), df.Longitude.to_numpy(), FIXED_LAT0)
    yv = df[TARGET].to_numpy(dtype=float)
    wd = DistanceBand(xy, threshold=band_km, binary=True, silence_warnings=True)
    isl = np.zeros(len(df), bool)
    isl[sorted(wd.islands)] = True
    wd.transform = "r"
    g = G_Local(yv, wd, star=True, permutations=999, seed=SEED)
    if rule == "analytic":
        # use the pipeline's explicit Ord-Getis z (esda's Zs is degenerate for row-standardised star weights)
        gl = pd.read_csv(REV / "gistar_local.csv")
        assert np.allclose(gl.Latitude.to_numpy(), df.Latitude.to_numpy())
        z = gl.gi_star_z.to_numpy(dtype=float)
        sig = (np.abs(np.nan_to_num(z)) > 1.96) & ~isl
        g.Zs = np.nan_to_num(z)
        sub, crit, ns = "analytic Ord–Getis z", "|z| > 1.96", "|z| ≤ 1.96"
    else:
        sig = (g.p_sim < 0.05) & ~isl
        sub, crit, ns = "999 permutations", "p < 0.05", "p ≥ 0.05"
    cat = np.where(sig & (g.Zs > 0), 2, np.where(sig & (g.Zs < 0), 0, 1))
    return {"cat": cat, "isl": isl, "rule": rule, "subtitle": sub, "crit": crit, "ns": ns}


def fig09_v2(df):
    from matplotlib.lines import Line2D

    from .features import FIXED_LAT0
    from .revision_analyses import district_polygon, to_km

    BAND = 5.5
    cls = gi_classification(df, BAND)
    cat, isl = cls["cat"], cls["isl"]
    sig = (cat != 1) & ~isl
    ref = json.loads((REV / "sparsity_metrics.json").read_text()).get("gi_star_label_db005", {})
    got = {"n_hot_p05": int(((cat == 2) & ~isl).sum()), "n_cold_p05": int(((cat == 0) & ~isl).sum()),
           "islands": int(isl.sum())}
    mism = {k: (got[k], ref.get(k)) for k in got if k in ref and ref[k] != got[k]}
    print(f"[fig09] rule={cls['rule']} counts={got}" + (f"  WARNING differs from sparsity_metrics.json: {mism}" if mism else "  (matches sparsity_metrics.json)"))
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
               label=f"No neighbour within {BAND} km (island), n = {isl.sum()}")
    lat_m = float(df.Latitude.mean())
    ax.set_aspect(1 / np.cos(np.deg2rad(lat_m)))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Local Getis–Ord Gi* of the landslide label\n(distance band {BAND} km, {cls['subtitle']})",
                 fontsize=10)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    # scale bars (25 km and 10 km) and 5.5 km reference circle, lower-left
    scale_bar(ax, 25, 22.3, x0 + 0.10 * (x1 - x0), y0 + 0.05 * (y1 - y0))
    scale_bar(ax, 10, 22.3, x0 + 0.10 * (x1 - x0), y0 + 0.13 * (y1 - y0))
    _, rdl = km_to_deg(BAND, 22.3)
    ref_circle(ax, BAND, 22.3, x0 + 0.62 * (x1 - x0), y0 + 0.08 * (y1 - y0))
    ax.text(x0 + 0.62 * (x1 - x0), y0 + 0.08 * (y1 - y0) + 1.15 * rdl, f"{BAND} km", ha="center", fontsize=8,
            bbox=dict(fc="w", ec="none", alpha=0.7, pad=0.8))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], ls="--", c="k", lw=0.9, label=f"{BAND} km Gi* neighbourhood radius (circle on map)"))
    ax.legend(handles=h, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0)
    savefig(fig, FIG / "fig09_gistar_v2.png")


def fig11_v2(df):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from scipy.spatial import cKDTree
    from sklearn.cluster import DBSCAN

    from .features import DBSCAN_EPS, DBSCAN_MIN_SAMPLES, FIXED_LAT0
    from .revision_analyses import KM_PER_DEG_LAT, to_km

    xy = to_km(df["Latitude"].to_numpy(), df["Longitude"].to_numpy(), FIXED_LAT0)
    y = df[TARGET].to_numpy()
    d, _ = cKDTree(xy).query(xy, k=6)  # column 0 = self

    def knee_of(kd):
        xs = np.arange(len(kd))
        p1, p2 = np.array([0, kd[0]]), np.array([len(kd) - 1, kd[-1]])
        num = np.abs((p2[1] - p1[1]) * xs - (p2[0] - p1[0]) * kd + p2[0] * p1[1] - p2[1] * p1[0])
        return float(kd[int(np.argmax(num / np.linalg.norm(p2 - p1)))])

    # k-distance convention follows the pipeline: pick the neighbour rank whose knee equals the one stored in
    # dbscan_rule_audit.json (5th neighbour excluding self = old; 4th excluding self = n_neighbors=5 incl. self).
    col = 5
    try:
        target = json.loads((REV / "dbscan_rule_audit.json").read_text())["k_distance_knee_km"]
        errs = {c: abs(knee_of(np.sort(d[:, c])) - target) for c in (5, 4)}
        col = min(errs, key=errs.get)
        print(f"[fig11] k-distance column {col} (knee diffs {errs}; audit knee {target:.4f} km)")
    except Exception as e:
        print("[fig11] audit knee not readable, using 5th neighbour excl. self:", repr(e))
    kd = np.sort(d[:, col])
    eps_knee_km = knee_of(kd)
    kth = "5th" if col == 5 else "4th"
    kdesc = "k = min_samples = 5" if col == 5 else "4th neighbour excl. self = 5 incl. self"
    lab = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(xy)
    nc = int(lab.max()) + 1
    cmap = ListedColormap([plt.get_cmap("tab10")(i) for i in range(nc)])
    norm = BoundaryNorm(np.arange(-0.5, nc, 1.0), cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(kd, color="#3b6fb6")
    ax.axhline(DBSCAN_EPS, ls="--", c="r", lw=1, label=f"eps = {DBSCAN_EPS} km (adopted)")
    ax.axhline(eps_knee_km, ls=":", c="k", lw=1, label=f"knee = {eps_knee_km:.1f} km")
    ax.set_xlabel(f"Points sorted by {kth}-nearest-neighbour distance")
    ax.set_ylabel(f"{kth}-nearest-neighbour distance (km)")
    ax.set_title(f"(a) k-distance curve ({kdesc})", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    lon, lat = df["Longitude"].to_numpy(), df["Latitude"].to_numpy()

    def draw(a, s_pt, s_ring, s_x, lw_x):
        sc = a.scatter(lon[lab >= 0], lat[lab >= 0], c=lab[lab >= 0], cmap=cmap, norm=norm, s=s_pt)
        a.scatter(lon[lab < 0], lat[lab < 0], c="0.2", s=s_x, marker="x", lw=lw_x, label="Noise (label −1)")
        a.scatter(lon[y == 1], lat[y == 1], facecolors="none", edgecolors="k", s=s_ring, lw=0.4,
                  label="Landslide inventory point")
        return sc

    sc = draw(ax, 14, 22, 16, 0.9)
    cb = fig.colorbar(sc, ax=ax, ticks=range(nc))
    cb.set_label("DBSCAN cluster ID")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("(b) DBSCAN clusters (eps = 1.1 km, min_samples = 5)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(ax.get_xlim())
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    scale_bar(ax, 10, FIXED_LAT0, x1 - 0.30 * (x1 - x0), y0 + 0.2 * (y1 - y0))
    # inset: west-central cluster corridor
    zx, zy = (91.985, 92.235), (22.48, 22.90)
    ins = ax.inset_axes([0.60, 0.46, 0.39, 0.52])
    draw(ins, 30, 50, 22, 0.9)
    ins.set_xlim(*zx)
    ins.set_ylim(*zy)
    ins.set_aspect(1 / np.cos(np.deg2rad(FIXED_LAT0)), adjustable="box", anchor="NE")
    ins.tick_params(labelsize=6.5)
    for cid in range(nc):
        m = lab == cid
        cx, cy = lon[m].mean(), lat[m].mean()
        if zx[0] < cx < zx[1] and zy[0] < cy < zy[1]:
            ins.annotate(str(cid), (cx, cy), xytext=(4, 5), textcoords="offset points", fontsize=8,
                         fontweight="bold", color="k", bbox=dict(fc="w", ec="none", alpha=0.75, pad=0.6))
    cx_, cy_ = zx[1] - 0.04, zy[1] - 0.075
    ref_circle(ins, DBSCAN_EPS, FIXED_LAT0, cx_, cy_)
    ins.text(cx_ - km_to_deg(DBSCAN_EPS, FIXED_LAT0)[0] * 1.3, cy_, f"eps = {DBSCAN_EPS} km", fontsize=7,
             va="center", ha="right", bbox=dict(fc="w", ec="none", alpha=0.7, pad=0.6))
    scale_bar(ins, 5, FIXED_LAT0, zx[0] + 0.03, zy[0] + 0.025, fs=7)
    ins.text(0.5, 0.985, "Zoom: west-central corridor", transform=ins.transAxes, fontsize=7, ha="center", va="top", bbox=dict(fc="w", ec="none", alpha=0.8, pad=0.8))
    ax.indicate_inset_zoom(ins, edgecolor="k", lw=0.8, alpha=0.8)
    savefig(fig, FIG / "fig11_dbscan_kdistance_v2.png")


def fig13a_v2(df):
    from matplotlib.lines import Line2D

    from .features import FIXED_LAT0
    from .revision_extra import draw_boundary

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
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    scale_bar(ax, 25, FIXED_LAT0, x1 - 0.40 * (x1 - x0), y0 + 0.04 * (y1 - y0), fs=8)
    scale_bar(ax, 10, FIXED_LAT0, x1 - 0.40 * (x1 - x0), y0 + 0.10 * (y1 - y0), fs=8)
    savefig(fig, FIG / "fig13a_spatial_folds_v2.png")


def grouped_shares_by_seed():
    rows = {}
    for f in sorted(R2.glob("cnn_grouped10_seed*.csv")):
        sd = int(f.stem.split("seed")[1])
        imp = pd.read_csv(f, index_col=0).abs().mean()
        rows[sd] = imp / imp.sum()
    return pd.DataFrame(rows).T.sort_index()


def fig14a_v2():
    lab = {"random_forest": "RF"}
    rf = pd.read_csv(REV / "shap_share_by_seed_random_forest.csv", index_col=0)
    cn = grouped_shares_by_seed()
    order = rf.mean().sort_values(ascending=False).index
    imp = pd.DataFrame({"RF": rf.mean(), "CNN": cn.mean()}).loc[order]
    sd = pd.DataFrame({"RF": rf.std(ddof=1), "CNN": cn.std(ddof=1)}).loc[order]
    fa, aa = plt.subplots(figsize=(6.2, 5))
    ip, isd = imp.iloc[::-1] * 100, sd.iloc[::-1] * 100
    y = np.arange(len(ip))
    for off, m, col, nm in ((0.2, "RF", "#2171b5", f"RF ({len(rf)} seeds)"),
                            (-0.2, "CNN", "#fd8d3c", f"1D-CNN, grouped exact Shapley ({len(cn)} seeds)")):
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
    savefig(fa, FIG / "fig14a_shap_importance_v2.png")


def part_figs(df):
    fig08_v2(df)
    fig09_v2(df)
    fig11_v2(df)
    fig13a_v2(df)
    if len(list(R2.glob("cnn_grouped10_seed*.csv"))) >= 3:
        fig14a_v2()
    print("figures written")


# --------------------------------------------------------------------------
# summary.md
# --------------------------------------------------------------------------


def _md(df: pd.DataFrame, fmt="{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in ([df.index.name or ""] + cols)) + " |",
           "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                                  for v in r.values]) + " |")
    return "\n".join(out)


def _rho_md(t: pd.DataFrame, manuscript: dict | None = None) -> str:
    rows = []
    for _, r in t.iterrows():
        signs = ", ".join("n/a" if not np.isfinite(r[f"rho_fold{k}"]) else f"{r[f'rho_fold{k}']:+.2f}"
                          for k in range(1, N_SPLITS + 1))
        d = {"factor": r.factor, "pooled rho": f"{r.pooled_rho:+.2f}"}
        if manuscript is not None:
            d["manuscript pooled"] = f"{manuscript[r.factor]:+.2f}" if r.factor in manuscript else ""
        d.update({"per-fold rho (F1..F5)": signs, "fold-size-weighted mean": f"{r.weighted_mean_within_rho:+.2f}",
                  "same sign in all folds": "yes" if r.same_sign_all_folds else "NO",
                  "folds with sign = pooled": f"{r.n_folds_same_sign_as_pooled}/{r.n_folds_defined}"})
        rows.append(d)
    return _md(pd.DataFrame(rows).set_index("factor"))


def part_summary(df):
    L = ["# Round-2 SHAP checks (rev2_shap)", ""]
    lab = spatial_fold_labels(df)
    fc = pd.Series({f"F{k + 1}": int((lab == k).sum()) for k in range(N_SPLITS)})
    L += ["Fold sizes (n): " + ", ".join(f"{k}={v}" for k, v in fc.items()), ""]
    # 1. grouped CNN
    old = pd.read_csv(REV / "shap_share_by_seed_cnn1d.csv", index_col=0)
    rf = pd.read_csv(REV / "shap_share_by_seed_random_forest.csv", index_col=0)
    new = grouped_shares_by_seed()
    meta = [json.loads(f.read_text()) for f in sorted(R2.glob("cnn_grouped_seed*.json"))]
    AUCS = ', '.join('%d: %.4f' % (m['seed'], m['pooled_AUC']) for m in meta)
    L += ["## 1. 1D-CNN SHAP with coalitions on the 10 parent factors (exact, 2^10 = 1024 coalitions)", "",
          f"Seeds: {list(new.index)} ({len(new)}). Same folds/seeds/training as the existing CNN SHAP; "
          f"retrained pooled AUC per seed matches the stored value "
          f"({AUCS}). Background = the 20 k-means centroids of the "
          "training fold (weighted by cluster size, as in KernelExplainer); 'off' groups take background values; "
          "model output averaged over the background; exact Shapley formula over all 1024 masks "
          "(efficiency check phi sum = f(x) - E f(bg) asserted).",
          f"Runtime: {np.mean([m['seconds'] for m in meta]):.0f} s per seed (mean; 10- and 12-player runs together), "
          f"{sum(m['seconds'] for m in meta) / 60:.0f} min total.", ""]
    order = rf.mean().sort_values(ascending=False).index
    t = pd.DataFrame({"RF share % (mean±sd)": [f"{rf[c].mean() * 100:.1f} ± {rf[c].std() * 100:.1f}" for c in order],
                      "CNN old (Kernel SHAP, 200 samples) %": [f"{old[c].mean() * 100:.1f} ± {old[c].std() * 100:.1f}" for c in order],
                      "CNN new (grouped exact) %": [f"{new[c].mean() * 100:.1f} ± {new[c].std() * 100:.1f}" for c in order],
                      "new - old (pp)": [f"{(new[c].mean() - old[c].mean()) * 100:+.1f}" for c in order],
                      "rank old": [int((old.mean().rank(ascending=False))[c]) for c in order],
                      "rank new": [int((new.mean().rank(ascending=False))[c]) for c in order]}, index=order)
    t.index.name = "Factor"
    from scipy.stats import spearmanr
    L += [_md(t), "",
          f"Spearman rank correlation of factor shares: CNN old vs new = {spearmanr(old.mean()[order], new.mean()[order]).correlation:.3f}; "
          f"RF vs CNN new = {spearmanr(rf.mean()[order], new.mean()[order]).correlation:.3f}; "
          f"RF vs CNN old = {spearmanr(rf.mean()[order], old.mean()[order]).correlation:.3f}.", ""]
    # 12 player
    f12 = {}
    for f in sorted(R2.glob("cnn_grouped12_seed*.csv")):
        imp = pd.read_csv(f, index_col=0).abs().mean()
        f12[int(f.stem.split("seed")[1])] = imp / imp.sum()
    f12 = pd.DataFrame(f12).T
    t12 = pd.DataFrame({"share % (mean±sd)": [f"{f12[c].mean() * 100:.1f} ± {f12[c].std() * 100:.1f}" for c in f12.mean().sort_values(ascending=False).index]},
                       index=f12.mean().sort_values(ascending=False).index)
    t12.index.name = "Player (12-player variant: both missing flags split out, 4096 coalitions)"
    L += ["12-player variant (TWI_was_missing and Soil_Texture_was_missing as separate players; Aspect keeps is_flat):", "",
          _md(t12), ""]
    # 2a flags
    L += ["## 2a. Missingness flags alone", ""]
    fr = pd.read_csv(R2 / "flag_shares_rf_by_seed.csv", index_col=0)
    fs = pd.read_csv(R2 / "flag_shares_rf_smote_by_seed.csv", index_col=0) if (R2 / "flag_shares_rf_smote_by_seed.csv").exists() else None
    fc_ = pd.read_csv(R2 / "flag_shares_cnn_ungrouped_seedmean.csv", index_col=0).iloc[:, 0]
    rows = {}
    for k in ["TWI_was_missing", "Soil_Texture_was_missing", "TWI (continuous column)", "is_flat",
              "Soil_Texture_missing (one-hot level)"]:
        r = {"RF per-column (10 seeds) %": f"{fr[k].mean() * 100:.2f} ± {fr[k].std() * 100:.2f}"}
        if fs is not None:
            r["RF+SMOTE per-column %"] = f"{fs[k].mean() * 100:.2f} ± {fs[k].std() * 100:.2f}"
        r["CNN old per-column (seed-averaged SHAP) %"] = f"{fc_[k] * 100:.2f}"
        rows[k] = r
    t = pd.DataFrame(rows).T
    t.index.name = "Column"
    L += ["Share of total mean |SHAP| computed at column level (no factor summation; all 23 encoded columns in the denominator). "
          "The CNN column uses the stored seed-averaged ungrouped Kernel SHAP values (per-seed per-column values were not stored; "
          "seed-averaging damps noisy Kernel-SHAP magnitudes, so it is a lower bound).", "", _md(t), ""]
    tw = {c: f"{f12[c].mean() * 100:.2f} ± {f12[c].std() * 100:.2f}" for c in ["TWI missing flag", "Soil missing flag", "TWI", "Soil texture", "Aspect"]}
    L += ["CNN, grouped exact Shapley with the flags as their own players (12-player run), share of total mean |SHAP| %:",
          "", _md(pd.DataFrame({"share % (mean±sd)": tw}).rename_axis("Player")), ""]
    # 2b / 2c
    ms = json.loads((REV / "shap_direction.json").read_text())["spearman_value_vs_shap"]
    L += ["## 2b. Pooled vs within-fold Spearman rho (factor value vs its SHAP value)", "",
          "rho computed per spatial fold on out-of-fold SHAP (seed-averaged, as for the manuscript pooled rho), fold-size-weighted "
          "mean over folds where rho is defined; 'sign' = same sign in every fold with variation. Pooled values recomputed here equal the "
          "manuscript values (shap_direction.json).", "", "### RF TreeSHAP (10 seeds)", ""]
    L += [_rho_md(pd.read_csv(R2 / "rho_rf.csv"), ms), ""]
    if (R2 / "rho_cnn_grouped.csv").exists():
        L += ["### 1D-CNN, grouped exact Shapley (seed-averaged; note the TWI factor value includes the TWI-missing flag effect)", "", _rho_md(pd.read_csv(R2 / "rho_cnn_grouped.csv")), ""]
    L += ["### 1D-CNN, old ungrouped Kernel SHAP (stored seed-averaged)", "", _rho_md(pd.read_csv(R2 / "rho_cnn_ungrouped.csv")), ""]
    if (R2 / "rho_rf_smote.csv").exists():
        L += ["## 2c. RF + SMOTE (TreeSHAP on the forest fitted to SMOTE-augmented training folds, 10 seeds)", "",
              _rho_md(pd.read_csv(R2 / "rho_rf_smote.csv")), ""]
    L += ["## Figures (new, originals untouched)", ""]
    for n in ["fig08_voronoi_quantified_v2", "fig09_gistar_v2", "fig11_dbscan_kdistance_v2", "fig13a_spatial_folds_v2",
              "fig14a_shap_importance_v2"]:
        if (FIG / f"{n}.png").exists():
            L.append(f"- reports/figures/revision/{n}.png")
    (R2 / "summary.md").write_text("\n".join(L))
    print("summary written")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["cnn_grouped", "shap_checks", "figs", "summary"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SHAP_SEEDS_ALL))
    ap.add_argument("--data", default="Labeled Data Set.csv")
    a = ap.parse_args(argv)
    R2.mkdir(parents=True, exist_ok=True)
    df = load_df(a.data)
    t0 = time.time()
    if a.part == "cnn_grouped":
        part_cnn_grouped(df, a.seeds)
    elif a.part == "shap_checks":
        part_shap_checks(df, a.seeds)
    elif a.part == "figs":
        part_figs(df)
    elif a.part == "summary":
        part_summary(df)
    print(f"[{a.part}] total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
