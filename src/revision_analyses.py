"""Additional analyses requested in the BOEG-D-26-02116 review (major revision).

Run from the repository root:

    python -m src.revision_analyses --data "../Labeled Data Set.csv"

Outputs
-------
results/revision/*.csv, *.json  -- every number quoted in the revised manuscript
reports/figures/revision/*.png  -- regenerated figures

Sections (reviewer comment in brackets)
1. Quantitative sparsity metrics: Clark-Evans R, Voronoi cell-area statistics,
   grid occupancy, Moran's I / Gi* recomputed            [R1-3, R2-4]
2. DBSCAN parameter justification: k-distance curve, eps x min_samples grid,
   audit of the original Sparse_Cluster rule             [R1-4]
3. Non-landslide sample geometry                          [R2-3]
4. Spatial-CV comparison of seven models x two feature sets with stored
   out-of-fold predictions, ROC curves, coordinates-only probe [R1-2, R1-6, R2-6, R2-7]
5. SMOTE inside the training fold as an alternative sparsity remedy [R2-1]
6. Out-of-fold Kernel SHAP for the terrain-only 1D-CNN      [R2-9]
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
from scipy.spatial import ConvexHull, Voronoi, cKDTree  # noqa: E402
from sklearn.cluster import DBSCAN  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from .data import TARGET, clean, load_raw  # noqa: E402
from .evaluate import cross_validate_model, classification_metrics  # noqa: E402
from .models import (  # noqa: E402
    MODEL_LABELS,
    MODEL_NAMES,
    KerasBinaryClassifier,
    ResampledClassifier,
    build_coordinates_only_probe,
    count_keras_params,
    get_model,
    set_global_seeds,
)
from .features import (  # noqa: E402
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    FIXED_LAT0,
    DBSCANClusterAssigner,
    build_preprocessor,
    terrain_feature_names,
)
from .splits import random_split, spatial_block_cv, spatial_fold_labels  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42
KM_PER_DEG_LAT = 110.574


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def to_km(lat: np.ndarray, lon: np.ndarray, lat0: float) -> np.ndarray:
    """Local equirectangular projection (km); adequate over ~2 deg of latitude."""
    x = lon * 111.320 * np.cos(np.deg2rad(lat0))
    y = lat * KM_PER_DEG_LAT
    return np.column_stack([x, y])


def clark_evans(xy: np.ndarray, area: float) -> dict:
    n = len(xy)
    d, _ = cKDTree(xy).query(xy, k=2)
    r_obs = d[:, 1].mean()
    density = n / area
    r_exp = 0.5 / np.sqrt(density)
    se = 0.26136 / np.sqrt(n * density)
    z = (r_obs - r_exp) / se
    return {
        "n": n,
        "area_km2": area,
        "mean_nn_km": r_obs,
        "median_nn_km": float(np.median(d[:, 1])),
        "expected_nn_km_CSR": r_exp,
        "R": r_obs / r_exp,
        "z": z,
    }


def voronoi_areas(xy: np.ndarray, hull_xy: np.ndarray, clip=None) -> np.ndarray:
    """Voronoi cell areas clipped to ``clip`` (default: convex hull of ``hull_xy``), km^2."""
    from shapely.geometry import MultiPoint, Polygon

    hull = clip if clip is not None else MultiPoint([tuple(p) for p in hull_xy]).convex_hull
    span = np.ptp(hull_xy, axis=0).max() * 10
    c = hull_xy.mean(axis=0)
    far = np.array([[c[0] - span, c[1] - span], [c[0] + span, c[1] - span],
                    [c[0] + span, c[1] + span], [c[0] - span, c[1] + span]])
    jitter = np.random.default_rng(SEED).normal(0, 1e-6, xy.shape)
    vor = Voronoi(np.vstack([xy + jitter, far]))
    areas = []
    for i in range(len(xy)):
        region = vor.regions[vor.point_region[i]]
        if -1 in region or len(region) == 0:
            areas.append(np.nan)
            continue
        poly = Polygon(vor.vertices[region]).intersection(hull)
        areas.append(poly.area)
    return np.asarray(areas)


BOUNDARY_FILE = Path("data/external/rangamati_geoboundaries_adm2.geojson")


def district_polygon(lat0: float | None = None):
    """Rangamati ADM2 polygon (geoBoundaries gbOpen); in km if ``lat0`` is given."""
    import geopandas as gpd
    from shapely.ops import transform

    poly = gpd.read_file(BOUNDARY_FILE).geometry.union_all()
    if lat0 is None:
        return poly
    return transform(lambda x, y, z=None: (x * 111.320 * np.cos(np.deg2rad(lat0)), y * KM_PER_DEG_LAT), poly)


def savefig(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 1. sparsity metrics
# --------------------------------------------------------------------------


def sparsity_metrics(df: pd.DataFrame, out: Path, figdir: Path) -> dict:
    lat0 = FIXED_LAT0  # fixed from the full dataset (== df.Latitude.mean())
    xy_all = to_km(df["Latitude"].to_numpy(), df["Longitude"].to_numpy(), lat0)
    pos = df[TARGET].to_numpy() == 1
    hull_area = ConvexHull(xy_all).volume  # 2-D "volume" is area
    dist_poly = district_polygon(lat0)
    district_area = dist_poly.area  # km^2, geoBoundaries ADM2 polygon

    res = {
        "convex_hull_area_km2": hull_area,
        "district_area_km2": district_area,
        "clark_evans_landslides_hull": clark_evans(xy_all[pos], hull_area),
        "clark_evans_landslides_district": clark_evans(xy_all[pos], district_area),
        "clark_evans_nonlandslides_hull": clark_evans(xy_all[~pos], hull_area),
        "clark_evans_nonlandslides_district": clark_evans(xy_all[~pos], district_area),
        "clark_evans_all_hull": clark_evans(xy_all, hull_area),
    }

    # Voronoi cell areas (inventory only, clipped to the sampling hull)
    a_pos = voronoi_areas(xy_all[pos], xy_all, dist_poly)
    a_all = voronoi_areas(xy_all, xy_all, dist_poly)
    for key, a in [("voronoi_landslides", a_pos), ("voronoi_all", a_all)]:
        a = a[np.isfinite(a)]
        res[key] = {
            "n_cells": int(len(a)),
            "mean_km2": float(a.mean()),
            "median_km2": float(np.median(a)),
            "p10_km2": float(np.percentile(a, 10)),
            "p90_km2": float(np.percentile(a, 90)),
            "max_km2": float(a.max()),
            "cv": float(a.std() / a.mean()),
            "p90_p10_ratio": float(np.percentile(a, 90) / np.percentile(a, 10)),
            "share_area_in_largest_10pct_cells": float(
                np.sort(a)[::-1][: max(1, len(a) // 10)].sum() / a.sum()
            ),
        }

    # grid occupancy: share of 5 km cells inside the hull holding >= 1 landslide
    from shapely.geometry import MultiPoint, box

    hull = dist_poly
    occ = {}
    for cell in (2.0, 5.0, 10.0):
        x0, y0, x1, y1 = hull.bounds
        xs = np.arange(x0, x1, cell)
        ys = np.arange(y0, y1, cell)
        n_in, n_occ = 0, 0
        ix = np.floor((xy_all[pos, 0] - x0) / cell).astype(int)
        iy = np.floor((xy_all[pos, 1] - y0) / cell).astype(int)
        occupied = set(zip(ix, iy))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                b = box(x, y, x + cell, y + cell)
                if b.intersection(hull).area >= 0.5 * cell * cell:
                    n_in += 1
                    n_occ += (i, j) in occupied
        occ[f"{int(cell)}km"] = {"cells": n_in, "occupied": n_occ,
                                 "occupancy": n_occ / n_in}
    res["grid_occupancy_landslides"] = occ

    # Moran's I, join counts and Gi* on the projected (km) coordinates
    try:
        from esda.getisord import G_Local
        from esda.join_counts import Join_Counts
        from esda.moran import Moran
        from libpysal.weights import KNN, DistanceBand

        GI_BAND_KM = 5.5
        yv = df[TARGET].to_numpy(dtype=float)
        w = KNN.from_array(xy_all, k=5)
        w.transform = "r"
        np.random.seed(SEED)
        mor = Moran(yv, w, permutations=999)
        wb = KNN.from_array(xy_all, k=5)
        wb.transform = "B"  # binary, not row-standardised, for join counts
        np.random.seed(SEED)
        jc = Join_Counts(yv.astype(int), wb, permutations=999)
        res["join_counts"] = {
            "weights": "KNN k=5 on projected km coordinates, binary (asymmetric kNN graph)",
            "n_joins_total": float(jc.bb + jc.ww + jc.bw),
        }
        res["join_counts_bb"] = float(jc.bb)
        res["join_counts_ww"] = float(jc.ww)
        res["join_counts_bw"] = float(jc.bw)
        res["join_counts_mean_bb_sim"] = float(jc.mean_bb)
        res["join_counts_mean_bw_sim"] = float(jc.mean_bw)
        res["join_counts_p_sim_bb"] = float(jc.p_sim_bb)
        res["join_counts_p_sim_bw"] = float(jc.p_sim_bw)
        res["join_counts_J"] = float(jc.J)
        wd = DistanceBand(xy_all, threshold=GI_BAND_KM, binary=True, silence_warnings=True)
        islands = sorted(wd.islands)
        wd.transform = "r"
        g = G_Local(yv, wd, star=True, permutations=999, seed=SEED)
        isl = np.zeros(len(df), bool)
        isl[islands] = True
        # Analytic Ord-Getis Gi* z-score (self-weight included, binary weights; the z-score is invariant
        # to row-standardisation). xbar and S over all n points, S with denominator n (Getis & Ord 1992;
        # Ord & Getis 1995). esda's analytic Zs is degenerate here, so it is computed explicitly.
        from scipy.sparse import identity as sp_identity

        n_obs = len(yv)
        Wb = wd.sparse.copy()
        Wb.data = np.ones_like(Wb.data)
        Wb = (Wb + sp_identity(n_obs, format="csr")).tocsr()
        sw = np.asarray(Wb.sum(1)).ravel()          # sum of weights = k_i + 1
        sw2 = sw                                    # binary: sum of squared weights = sum of weights
        xbar, S = yv.mean(), yv.std(ddof=0)
        gi_z = (np.asarray(Wb @ yv).ravel() - xbar * sw) / (S * np.sqrt((n_obs * sw2 - sw ** 2) / (n_obs - 1)))
        gi_z[isl] = np.nan
        GI_Z_CRIT = 1.96  # two-sided alpha = 0.05
        hot_m = (gi_z > GI_Z_CRIT) & ~isl
        cold_m = (gi_z < -GI_Z_CRIT) & ~isl
        sig = hot_m | cold_m
        pd.DataFrame({"Latitude": df.Latitude, "Longitude": df.Longitude, TARGET: yv.astype(int),
                      "n_neighbours_5p5km": [wd.cardinalities[i] for i in range(n_obs)],
                      "island": isl, "gi_star_z": gi_z,
                      "class": np.where(isl, "island", np.where(hot_m, "hot", np.where(cold_m, "cold", "ns"))),
                      "p_sim_permutation_secondary": g.p_sim}).to_csv(out / "gistar_local.csv", index=False)
        res["moran_label_knn5"] = {"I": float(mor.I), "EI": float(mor.EI), "z": float(mor.z_sim),
                                   "p_sim": float(mor.p_sim)}
        # Fig. 9: Gi* significance map; Fig. 10: Moran scatter
        poly = district_polygon()
        cat = np.where(hot_m, 2, np.where(cold_m, 0, 1))
        fig, ax = plt.subplots(figsize=(7.0, 6.6))
        for gg in getattr(poly, "geoms", [poly]):
            ax.plot(*gg.exterior.xy, c="k", lw=0.6)
        for c, col, lab in [(0, "#2c7bb6", "Cold spot (z < −1.96)"),
                            (1, "#bdbdbd", "Not significant (|z| ≤ 1.96)"),
                            (2, "#d7191c", "Hot spot (z > 1.96)")]:
            m = (cat == c) & ~isl
            ax.scatter(df.Longitude[m], df.Latitude[m], s=12, c=col, label=f"{lab}, n = {m.sum()}",
                       ec="k", lw=0.2)
        ax.scatter(df.Longitude[isl], df.Latitude[isl], s=30, facecolors="none", edgecolors="k",
                   marker="s", label=f"No neighbour within {GI_BAND_KM} km (island), n = {isl.sum()}")
        ax.set_aspect(1 / np.cos(np.deg2rad(df.Latitude.mean())))
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(f"Local Getis–Ord Gi* of the landslide label\n(distance band {GI_BAND_KM} km, analytic z, |z| > 1.96)",
                     fontsize=10)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0)
        savefig(fig, figdir / "fig09_gistar.png")

        z = (df[TARGET].to_numpy(float) - df[TARGET].mean()) / df[TARGET].std(ddof=0)
        lag = np.asarray(w.sparse @ z).ravel()
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        jit = np.random.default_rng(SEED).normal(0, 0.03, len(z))
        ax.scatter(z + jit, lag, s=10, alpha=0.5, c="#3b6fb6")
        xs = np.linspace(z.min() - 0.2, z.max() + 0.2, 10)
        ax.plot(xs, mor.I * xs, c="#b2182b", lw=1.5)
        ax.axhline(0, c="0.4", ls="--", lw=0.7)
        ax.axvline(0, c="0.4", ls="--", lw=0.7)
        ax.set_xlabel("Standardised landslide label (0 = non-landslide, 1 = landslide)")
        ax.set_ylabel("Spatial lag of standardised label (k = 5 neighbours)")
        ax.set_title("Moran's I scatter plot of the landslide label", fontsize=10)
        ax.text(0.03, 0.97, f"Moran's I = {mor.I:.3f}\nE[I] = {mor.EI:.3f}\npseudo p = {mor.p_sim:.3f} (999 perm.)\nn = {len(z)}",
                transform=ax.transAxes, va="top", fontsize=8.5, bbox=dict(fc="w", ec="0.7"))
        savefig(fig, figdir / "fig10_moran.png")

        n_hot = int(hot_m.sum())
        n_cold = int(cold_m.sum())
        n_ns = int((~sig & ~isl).sum())
        res["gi_star_label_db005"] = {  # key name kept for lsnums; band is now 5.5 km
            "band_km": GI_BAND_KM,
            "islands": int(isl.sum()),
            "classification": "analytic Ord-Getis Gi* z, |z| > 1.96 (two-sided alpha = 0.05); islands excluded",
            "n_hot_p05": n_hot,  # key name kept for lsnums; now |z| > 1.96
            "share_landslides_hot": float(yv[hot_m].mean()),
            "share_landslides_cold": float(yv[cold_m].mean()),
            "share_landslides_ns": float(yv[~sig & ~isl].mean()),
            "n_hot_p_sim_lt_0.05_secondary": int(((g.p_sim < 0.05) & ~isl & (g.Zs > 0)).sum()),
            "n_cold_p_sim_lt_0.05_secondary": int(((g.p_sim < 0.05) & ~isl & (g.Zs < 0)).sum()),
            "n_cold_p05": n_cold,
            "n_not_sig": n_ns,
            "n_total_check": n_hot + n_cold + n_ns + int(isl.sum()),
        }
        assert res["gi_star_label_db005"]["n_total_check"] == len(df)
    except Exception as e:  # pragma: no cover
        res["moran_error"] = repr(e)
        raise

    # Figure: Voronoi of the inventory, cells coloured by log area
    from shapely.geometry import Polygon

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
        geoms = getattr(poly, "geoms", [poly])
        for gg in geoms:
            px, py = gg.exterior.xy
            ax.fill(px, py, color=cmap(norm(lg[i])), ec="k", lw=0.2)
    for gg in getattr(hull, "geoms", [hull]):
        ax.plot(*gg.exterior.xy, c="k", lw=0.6)
    ax.scatter(*xy_all[~pos].T, s=4, c="0.55", marker="x", lw=0.5, label="Non-landslide sample")
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
    ce = res["clark_evans_landslides_district"]  # district-area values, as quoted in the text
    ax.set_title("(b) Cell-area distribution", fontsize=10)
    ax.text(0.03, 0.97,
            f"n = {len(a)}\nCV = {res['voronoi_landslides']['cv']:.2f}\n"
            f"P90/P10 = {res['voronoi_landslides']['p90_p10_ratio']:.0f}\n"
            f"Clark-Evans R = {ce['R']:.2f} (z = {ce['z']:.1f})",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(fc="w", ec="0.7"))
    savefig(fig, figdir / "fig08_voronoi_quantified.png")

    (out / "sparsity_metrics.json").write_text(json.dumps(res, indent=2, default=float))
    return res


# --------------------------------------------------------------------------
# 2. DBSCAN justification
# --------------------------------------------------------------------------


def dbscan_analysis(df: pd.DataFrame, out: Path, figdir: Path) -> dict:
    """DBSCAN justification on the projected km plane (eps in km, fixed lat0)."""
    xy = to_km(df["Latitude"].to_numpy(), df["Longitude"].to_numpy(), FIXED_LAT0)
    y = df[TARGET].to_numpy()
    k = 5
    # 4th nearest neighbour excluding the point itself = sklearn NearestNeighbors(n_neighbors=5) last
    # column (self included), matching DBSCAN min_samples = 5 which counts the point itself.
    d, _ = cKDTree(xy).query(xy, k=k)
    kd = np.sort(d[:, k - 1])  # km
    # knee: max distance to the chord
    xs = np.arange(len(kd))
    p1, p2 = np.array([0, kd[0]]), np.array([len(kd) - 1, kd[-1]])
    num = np.abs((p2[1] - p1[1]) * xs - (p2[0] - p1[0]) * kd + p2[0] * p1[1] - p2[1] * p1[0])
    knee = int(np.argmax(num / np.linalg.norm(p2 - p1)))
    eps_knee_km = float(kd[knee])

    rows = []
    for eps_km in (0.55, 1.1, 2.2, 3.3, 5.5):
        for ms in (3, 5, 10):
            lab = DBSCAN(eps=eps_km, min_samples=ms).fit_predict(xy)
            inc = lab >= 0
            rows.append({
                # eps_deg is the nominal degree-equivalent (km / 110.574, 3 dp), kept as the
                # lookup key that lsnums.py / the text use; the clustering itself is in km.
                "eps_deg": round(eps_km / KM_PER_DEG_LAT, 3), "eps_km": eps_km, "min_samples": ms,
                "n_clusters": int(len(set(lab[inc]))),
                "noise_share": float((~inc).mean()),
                "P_landslide_in_cluster": float(y[inc].mean()) if inc.any() else np.nan,
                "P_landslide_noise": float(y[~inc].mean()) if (~inc).any() else np.nan,
                "share_landslides_in_clusters": float(inc[y == 1].mean()),
            })
    grid = pd.DataFrame(rows)
    grid.to_csv(out / "dbscan_sensitivity.csv", index=False)

    # Original configuration (now eps = 1.1 km) and the original Sparse_Cluster rule
    lab = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(xy)
    counts = pd.Series(lab).value_counts().sort_index()
    extent = np.ptp(df["Longitude"]) * np.ptp(df["Latitude"])  # deg^2, original rule
    avg_density = len(df) / extent
    threshold = 0.2 * avg_density * len(df)
    hull_area = ConvexHull(xy).volume
    per_cluster = []
    for cid, n in counts.items():
        m = lab == cid
        if cid >= 0 and m.sum() >= 3:
            try:
                a = ConvexHull(xy[m]).volume
            except Exception:
                a = np.nan
        else:
            a = np.nan
        per_cluster.append({
            "cluster": int(cid), "n_points": int(n), "n_landslides": int(y[m].sum()),
            "hull_area_km2": a,
            "density_per_km2": (n / a) if (a and np.isfinite(a) and a > 0) else np.nan,
        })
    pc = pd.DataFrame(per_cluster)
    pc["relative_density"] = pc["density_per_km2"] / (len(df) / hull_area)
    pc.to_csv(out / "dbscan_original_clusters.csv", index=False)
    rule = {
        "bbox_area_deg2": float(extent),
        "avg_density_pts_per_deg2": float(avg_density),
        "threshold_original_rule": float(threshold),
        "largest_cluster_count": int(counts.max()),
        "all_clusters_flagged_sparse": bool((counts < threshold).all()),
        "study_density_pts_per_km2": len(df) / hull_area,
        "k_distance_knee_deg": eps_knee_km / KM_PER_DEG_LAT,  # deg-equivalent of the km knee
        "k_distance_knee_km": eps_knee_km,
        "share_kdist_le_1p1km": float((kd <= 1.1).mean()),
        "eps_km_used": DBSCAN_EPS,
        "min_samples_used": DBSCAN_MIN_SAMPLES,
        "coordinate_system": "local equirectangular km, fixed lat0 = mean latitude of full data",
        "n_clusters_original": int((counts.index >= 0).sum()),
        "noise_share_original": float((lab < 0).mean()),
    }
    (out / "dbscan_rule_audit.json").write_text(json.dumps(rule, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(kd, color="#3b6fb6")
    ax.axhline(DBSCAN_EPS, ls="--", c="r", lw=1, label=f"eps = {DBSCAN_EPS} km (adopted)")
    ax.axhline(eps_knee_km, ls=":", c="k", lw=1,
               label=f"knee = {eps_knee_km:.1f} km")
    ax.set_xlabel("Points sorted by k-distance")
    ax.set_ylabel("k-distance, k = min_samples = 5 (km; 4th neighbour excluding the point)")
    ax.set_title("(a) k-distance curve (k = min_samples = 5)", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    sc = ax.scatter(df["Longitude"][lab >= 0], df["Latitude"][lab >= 0], c=lab[lab >= 0],
                    cmap="tab10", s=14, vmin=0, vmax=9)
    ax.scatter(df["Longitude"][lab < 0], df["Latitude"][lab < 0], c="0.6", s=6, marker="x",
               lw=0.6, label="Noise (label −1)")
    ax.scatter(df["Longitude"][y == 1], df["Latitude"][y == 1], facecolors="none", edgecolors="k",
               s=22, lw=0.4, label="Landslide inventory point")
    cb = fig.colorbar(sc, ax=ax, ticks=range(int(lab.max()) + 1))
    cb.set_label("DBSCAN cluster ID")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("(b) DBSCAN clusters (eps = 1.1 km, min_samples = 5)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    savefig(fig, figdir / "fig11_dbscan_kdistance.png")
    return {"grid": grid, "rule": rule, "clusters": pc}


# --------------------------------------------------------------------------
# 3. non-landslide sample geometry
# --------------------------------------------------------------------------


def negative_sampling(df: pd.DataFrame, out: Path) -> dict:
    lat0 = FIXED_LAT0
    xy = to_km(df["Latitude"].to_numpy(), df["Longitude"].to_numpy(), lat0)
    pos = df[TARGET].to_numpy() == 1
    d, _ = cKDTree(xy[pos]).query(xy[~pos], k=1)
    neg = df[~pos]
    res = {
        "n_negatives": int((~pos).sum()),
        "n_positives": int(pos.sum()),
        "ratio_neg_to_pos": float((~pos).sum() / pos.sum()),
        "min_dist_to_landslide_km": float(d.min()),
        "p05_dist_km": float(np.percentile(d, 5)),
        "median_dist_km": float(np.median(d)),
        "max_dist_km": float(d.max()),
        "share_within_1km": float((d < 1).mean()),
        "share_beyond_5km": float((d > 5).mean()),
        "lat_range_neg": [float(neg.Latitude.min()), float(neg.Latitude.max())],
        "lon_range_neg": [float(neg.Longitude.min()), float(neg.Longitude.max())],
        "lat_range_pos": [float(df[pos].Latitude.min()), float(df[pos].Latitude.max())],
        "lon_range_pos": [float(df[pos].Longitude.min()), float(df[pos].Longitude.max())],
        "slope_median_neg": float(neg.Slope.median()),
        "slope_median_pos": float(df[pos].Slope.median()),
    }
    (out / "negative_sampling.json").write_text(json.dumps(res, indent=2))
    return res


# --------------------------------------------------------------------------
# 4-5. spatial CV comparison with OOF storage, SMOTE, coordinates probe
# --------------------------------------------------------------------------


def _smote_pipe(name: str, seed: int, **kw):
    """Preprocessor -> SnapSMOTE (synthetic rows snapped to valid encodings) -> model.

    Keras models receive the sampler inside ``KerasBinaryClassifier.fit`` so that
    the internal 20% validation split is drawn from real training-fold points
    BEFORE oversampling; other models are wrapped in ``ResampledClassifier``.
    """
    from sklearn.pipeline import Pipeline

    from .sampling import SnapSMOTE

    base = get_model(name, random_state=seed, **kw)
    est = base.named_steps["model"]
    sampler = SnapSMOTE(random_state=seed, k_neighbors=5, feature_names=terrain_feature_names())
    if isinstance(est, KerasBinaryClassifier):
        est.set_params(sampler=sampler)
        model = est
    else:
        model = ResampledClassifier(est, sampler)
    return Pipeline([("pre", base.named_steps["pre"]), ("model", model)])


CONFIGS = [("terrain", {"include_cluster": False, "drainage_mode": "drop"}, False),
           ("terrain+Cluster", {"include_cluster": True, "drainage_mode": "drop"}, False),
           ("terrain+SMOTE", {"include_cluster": False, "drainage_mode": "drop"}, True)]
SMOTE_MODELS = ("cnn1d", "dnn", "lstm", "random_forest", "xgboost", "svm_rbf", "logistic")


def run_seed(df: pd.DataFrame, seed: int, models: list[str], out: Path) -> None:
    """One seed: spatial CV (3 configs) + random-split reference; writes CSVs."""
    from .models import fit_pipeline

    y = df[TARGET].to_numpy(dtype=int)
    folds = list(spatial_block_cv(df))  # partition is seed-independent (coords + labels)
    rows, oof = [], {}
    for dataset, kw, smote in CONFIGS:
        names = models if not smote else [m for m in SMOTE_MODELS if m in models]
        for name in names:
            set_global_seeds(seed, deterministic=False)
            fn = (lambda name=name, kw=kw: _smote_pipe(name, seed, **kw)) if smote else \
                 (lambda name=name, kw=kw: get_model(name, random_state=seed, **kw))
            cv = cross_validate_model(fn, df, folds)
            oof[f"{MODEL_LABELS[name]}|{dataset}"] = cv["oof"]
            pf = cv["per_fold"]
            rec = {"seed": seed, "model": MODEL_LABELS[name], "dataset": dataset, "split": "spatial",
                   "within_mean_AUC": float(pf.AUC.mean()),
                   **{f"pooled_{k}": float(v) for k, v in cv["pooled"].items()}}
            for _, r in pf.iterrows():
                rec[f"fold{int(r.fold)}_AUC"] = float(r.AUC)
            rows.append(rec)
            print(f"[seed {seed}] {dataset:<16} {MODEL_LABELS[name]:<19} pooled={cv['pooled']['AUC']:.4f}",
                  flush=True)
    sd = out / "seeds"
    sd.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(oof).to_csv(sd / f"oof_seed{seed}.csv", index=False)
    pd.DataFrame(rows).to_csv(sd / f"spatial_seed{seed}.csv", index=False)
    rows += random_split_rows(df, seed, models)
    pd.DataFrame(rows).to_csv(sd / f"metrics_seed{seed}.csv", index=False)


def random_split_rows(df: pd.DataFrame, seed: int, models: list[str]) -> list[dict]:
    """Stratified random 60/20/20 reference (split varies with the seed, as in run.run_R1)."""
    from .models import fit_pipeline

    train, val, test = random_split(df, random_state=seed)
    yt, yv, ys = (d[TARGET].to_numpy(dtype=int) for d in (train, val, test))
    rows = []
    for dataset, kw, _ in CONFIGS[:2]:
        for name in models:
            set_global_seeds(seed, deterministic=False)
            pipe = fit_pipeline(get_model(name, random_state=seed, **kw), train, yt, val, yv)
            m = classification_metrics(ys, pipe.predict_proba(test)[:, 1])
            rows.append({"seed": seed, "model": MODEL_LABELS[name], "dataset": dataset, "split": "random",
                         **{f"pooled_{k}": float(v) for k, v in m.items()}})
    return rows


def aggregate(df: pd.DataFrame, out: Path, figdir: Path) -> dict:
    y = df[TARGET].to_numpy(dtype=int)
    sd = out / "seeds"
    met = pd.concat([pd.read_csv(f) for f in sorted(sd.glob("metrics_seed*.csv"))], ignore_index=True)
    met.to_csv(out / "metrics_all_seeds.csv", index=False)
    seeds = sorted(met.seed.unique())
    oofs = {s: pd.read_csv(sd / f"oof_seed{s}.csv") for s in seeds}
    cols = list(oofs[seeds[0]].columns)
    ens = pd.DataFrame({c: np.mean([oofs[s][c].to_numpy() for s in seeds], axis=0) for c in cols})
    ens.insert(0, "Landslide", y)
    ens.insert(0, "Longitude", df.Longitude.to_numpy())
    ens.insert(0, "Latitude", df.Latitude.to_numpy())
    ens.to_csv(out / "oof_seed_ensemble.csv", index=False)

    metrics = ["AUC", "accuracy", "precision", "recall", "F1"]
    g = met.groupby(["split", "dataset", "model"])
    summ = g[[f"pooled_{m}" for m in metrics] + ["within_mean_AUC"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    ens_auc = {}
    for c in cols:
        model, dataset = c.split("|")
        ens_auc[(dataset, model)] = roc_auc_score(y, ens[c])
    summ["seed_ensemble_AUC"] = [ens_auc.get((d, m), np.nan) if s == "spatial" else np.nan
                                 for s, d, m in zip(summ.split, summ.dataset, summ.model)]
    summ["n_seeds"] = len(seeds)
    summ.to_csv(out / "model_comparison_summary.csv", index=False)

    # sensitivity: within-fold AUC over folds 1-4 only (terrain, spatial), 10-seed mean per model
    ft = met[(met.split == "spatial") & (met.dataset == "terrain")].copy()
    ft["within_AUC_excl_fold5"] = ft[[f"fold{k}_AUC" for k in range(1, 5)]].mean(axis=1)
    ex5 = ft.groupby("model", sort=False)["within_AUC_excl_fold5"].mean().reset_index()
    ex5.to_csv(out / "sensitivity_excl_fold5.csv", index=False)

    # paired comparison terrain vs terrain+Cluster under spatial CV (per seed)
    sp = met[met.split == "spatial"].pivot_table(index=["seed", "model"], columns="dataset",
                                                 values="pooled_AUC").reset_index()
    sp["delta_cluster"] = sp["terrain+Cluster"] - sp["terrain"]
    if "terrain+SMOTE" in sp:
        sp["delta_smote"] = sp["terrain+SMOTE"] - sp["terrain"]
    sp.to_csv(out / "paired_deltas_by_seed.csv", index=False)

    folds = list(spatial_block_cv(df))
    fold_id = spatial_fold_labels(df)
    pd.DataFrame({"Latitude": df.Latitude, "Longitude": df.Longitude, "Landslide": y,
                  "fold": fold_id + 1}).to_csv(out / "spatial_folds.csv", index=False)

    # coordinates-only probe
    train, val, test = random_split(df, random_state=SEED)
    cvp = cross_validate_model(lambda: build_coordinates_only_probe(SEED), df, folds)
    trv = pd.concat([train, val])
    pr = build_coordinates_only_probe(SEED).fit(trv, trv[TARGET].to_numpy(dtype=int))
    pd.DataFrame([{"split": "spatial block CV (pooled OOF)", "AUC": cvp["pooled"]["AUC"]},
                  {"split": "spatial block CV (within-fold mean)", "AUC": float(cvp["per_fold"].AUC.mean())},
                  {"split": "random 60/20/20 (test)",
                   "AUC": roc_auc_score(test[TARGET].to_numpy(dtype=int), pr.predict_proba(test)[:, 1])}]
                 ).to_csv(out / "coordinates_probe.csv", index=False)

    # ROC figure (terrain, seed-ensemble OOF)
    fig, ax = plt.subplots(figsize=(6.2, 6))
    t = summ[(summ.split == "spatial") & (summ.dataset == "terrain")].sort_values(
        "within_mean_AUC_mean", ascending=False)
    styles = {"1D-CNN": "-", "DNN": "--", "LSTM": "-."}
    for _, r in t.iterrows():
        fpr, tpr, _ = roc_curve(y, ens[f"{r.model}|terrain"])
        ax.plot(fpr, tpr, styles.get(r.model, ":"), lw=1.8 if r.model in styles else 1.4,
                label=f"{r.model} ({r.within_mean_AUC_mean:.3f} / {r.pooled_AUC_mean:.3f})")
    ax.plot([0, 1], [0, 1], c="0.6", lw=0.8)
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(f"ROC curves, pooled out-of-fold predictions\n(spatial block CV, terrain factors; "
                 f"curves = mean of {len(seeds)} seeds)", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right", title="Model (within-fold / pooled AUC)", title_fontsize=7.5)
    ax.set_aspect("equal")
    savefig(fig, figdir / "fig12_roc_spatialcv.png")

    # bar chart: random vs spatial, terrain vs +Cluster, mean ± SD
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    labs = list(t.model)
    w = 0.2
    sets = [("random", "terrain", "#9ecae1", "Random split, terrain"),
            ("random", "terrain+Cluster", "#fdae6b", "Random split, terrain + Cluster"),
            ("spatial", "terrain", "#2171b5", "Spatial CV (within-fold), terrain"),
            ("spatial", "terrain+Cluster", "#d94801", "Spatial CV (within-fold), terrain + Cluster")]
    for i, (spl, ds, col, lab) in enumerate(sets):
        sub = summ[(summ.split == spl) & (summ.dataset == ds)].set_index("model")
        col_m = "within_mean_AUC" if spl == "spatial" else "pooled_AUC"
        ax.bar(np.arange(len(labs)) + (i - 1.5) * w, sub.loc[labs, f"{col_m}_mean"], w,
               yerr=sub.loc[labs, f"{col_m}_std"], color=col, label=lab, capsize=2,
               error_kw={"lw": 0.8})
    ax.set_xticks(np.arange(len(labs)))
    ax.set_xticklabels([{"RandomForest": "RF", "LogisticRegression": "LR", "SVM-RBF": "SVM"}.get(m, m)
                        for m in labs], fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel(f"AUC (mean ± SD, {len(seeds)} seeds)")
    ax.axhline(0.5, c="0.5", lw=0.6)
    ax.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=False)
    ax.set_title("(b) Random-split vs spatial-CV AUC, with and without the DBSCAN Cluster feature",
                 fontsize=9)
    savefig(fig, figdir / "fig13_model_comparison.png")

    fig, ax = plt.subplots(figsize=(5.5, 7))
    pal = plt.get_cmap("Set2")
    for k in range(5):
        m = fold_id == k
        ax.scatter(df.Longitude[m & (y == 0)], df.Latitude[m & (y == 0)], s=10, marker="x",
                   color=pal(k), lw=0.8)
        ax.scatter(df.Longitude[m & (y == 1)], df.Latitude[m & (y == 1)], s=12, marker="o",
                   color=pal(k), ec="k", lw=0.3,
                   label=f"Fold {k + 1}: n = {m.sum()}, landslides = {int(y[m].sum())}")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Spatial block cross-validation folds\n(o landslide, x non-landslide)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    savefig(fig, figdir / "fig_spatial_folds.png")
    return {"summary": summ, "oof": ens, "folds": folds}


# --------------------------------------------------------------------------
# 6. SHAP for the terrain-only 1D-CNN (out-of-fold)
# --------------------------------------------------------------------------

SHAP_SEEDS_ALL = (42, 0, 1, 2, 3, 4, 5, 6, 7, 8)
SHAP_MODEL_LABEL = {"random_forest": "RF", "cnn1d": "1D-CNN", "svm_rbf": "SVM", "xgboost": "XGBoost"}


def _factor(name: str) -> str:
    """Map an encoded column to its conditioning factor.

    Binary flags are attributed to the factor they describe: ``TWI_was_missing`` ->
    TWI, ``Soil_Texture_was_missing`` -> Soil texture, ``is_flat`` -> Aspect. There is
    no separate "missing-value flags" group.
    """
    if name in ("Aspect_sin", "Aspect_cos", "is_flat"):
        return "Aspect"
    if name == "TWI_was_missing":
        return "TWI"
    if name == "Soil_Texture_was_missing" or name.startswith("Soil_Texture_"):
        return "Soil texture"
    if name.startswith("LULC_"):
        return "LULC"
    return {"Annual_Rainfall": "Annual rainfall"}.get(name, name)


def factor_importance(S: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per factor.

    Aggregation: within each sample the signed SHAP values of all columns belonging
    to a factor are SUMMED (so opposing one-hot/sin-cos components cancel as they do
    in the prediction), then the absolute value is taken, then averaged over samples.
    """
    grouped = S.T.groupby([_factor(c) for c in S.columns]).sum().T
    return grouped.abs().mean()


def _shap_one_seed(df: pd.DataFrame, name: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Out-of-fold SHAP values (terrain config, spatial block CV) for one seed.

    RF: exact TreeSHAP (``shap.TreeExplainer``, tree_path_dependent) on the fold's fitted
    forest. 1D-CNN: Kernel SHAP, k-means background of 20 training-fold points, 200 samples.
    Returns (SHAP values, preprocessed test features, pooled AUC), rows indexed by position.
    """
    import shap

    set_global_seeds(seed, deterministic=False)
    folds = list(spatial_block_cv(df))
    cv = cross_validate_model(lambda: get_model(name, random_state=seed, include_cluster=False,
                                                drainage_mode="drop"), df, folds)
    vals, feats = [], []
    for (tr, te), pipe in zip(folds, cv["fitted"]):
        pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
        Xtr = np.asarray(pre.transform(df.iloc[tr]), dtype=float)
        Xte = np.asarray(pre.transform(df.iloc[te]), dtype=float)
        names = [str(n) for n in pre.get_feature_names_out()]
        if name == "random_forest":
            sv = shap.TreeExplainer(model).shap_values(Xte)
            sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
        else:
            bg = shap.kmeans(Xtr, 20)
            f = lambda X, model=model: model.predict_proba(np.asarray(X, dtype=float))[:, 1]
            sv = shap.KernelExplainer(f, bg).shap_values(Xte, nsamples=200, silent=True)
        vals.append(pd.DataFrame(sv, columns=names, index=te))
        feats.append(pd.DataFrame(Xte, columns=names, index=te))
    return pd.concat(vals).sort_index(), pd.concat(feats).sort_index(), float(cv["pooled"]["AUC"])


def shap_analysis(df: pd.DataFrame, out: Path, name: str = "random_forest", seeds=SHAP_SEEDS_ALL,
                  time_limit_s: float | None = None) -> pd.DataFrame:
    """Seed-replicated out-of-fold SHAP for one model.

    Writes ``shap_values_{name}.csv`` (seed-averaged per-sample values),
    ``shap_share_by_seed_{name}.csv`` (per-seed factor shares of total mean |SHAP|) and
    ``shap_seeds_{name}.json`` (seed list, timings, per-seed pooled AUC). If
    ``time_limit_s`` is set and the first seed takes longer, only the first 5 seeds are run.
    """
    seeds = list(seeds)
    per_seed, shares, aucs, timings = [], {}, {}, {}
    X = None
    for i, sd in enumerate(seeds):
        t0 = time.time()
        S, X, auc = _shap_one_seed(df, name, sd)
        timings[sd] = time.time() - t0
        aucs[sd] = auc
        per_seed.append(S)
        imp = factor_importance(S)
        shares[sd] = imp / imp.sum()
        print(f"[shap {name}] seed {sd} done in {timings[sd]:.0f}s, pooled AUC {auc:.4f}", flush=True)
        if i == 0 and time_limit_s is not None and timings[sd] > time_limit_s and len(seeds) > 5:
            seeds = seeds[:5]
            print(f"[shap {name}] first seed took {timings[sd]:.0f}s > {time_limit_s:.0f}s -> using 5 seeds {seeds}",
                  flush=True)
    S_mean = sum(per_seed) / len(per_seed)
    S_mean.to_csv(out / f"shap_values_{name}.csv")
    X.to_csv(out / f"shap_features_{name}.csv")
    pd.DataFrame(shares).T.rename_axis("seed").to_csv(out / f"shap_share_by_seed_{name}.csv")
    (out / f"shap_seeds_{name}.json").write_text(json.dumps(
        {"seeds": seeds, "seconds_per_seed": timings, "pooled_AUC_per_seed": aucs,
         "explainer": "TreeExplainer(tree_path_dependent)" if name == "random_forest"
         else "KernelExplainer(kmeans bg=20, nsamples=200)"}, indent=2))
    return pd.DataFrame(shares).T.agg(["mean", "std"]).T


# --------------------------------------------------------------------------
# susceptibility surface (point-interpolated, as in the original manuscript)
# --------------------------------------------------------------------------


def susceptibility_surface(df: pd.DataFrame, oof, out: Path, figdir: Path,
                           model_key: str = "1D-CNN|terrain") -> None:
    from shapely.geometry import MultiPoint, Point

    p = np.asarray(oof[model_key])
    lon, lat = df.Longitude.to_numpy(), df.Latitude.to_numpy()
    hull = district_polygon()
    x0, y0, x1, y1 = hull.bounds
    gx, gy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 760))
    tree = cKDTree(np.column_stack([lon, lat]))
    d, idx = tree.query(np.column_stack([gx.ravel(), gy.ravel()]), k=8)
    w = 1 / np.maximum(d, 1e-6) ** 2
    z = (w * p[idx]).sum(1) / w.sum(1)
    from shapely import contains_xy
    inside = contains_xy(hull, gx.ravel(), gy.ravel())
    z[~inside] = np.nan
    z = z.reshape(gx.shape)
    # five classes by natural-breaks approximation (quantiles of OOF probability)
    breaks = np.nanquantile(p, [0.2, 0.4, 0.6, 0.8])
    pd.DataFrame({"class": ["Very low", "Low", "Moderate", "High", "Very high"],
                  "upper_bound": list(breaks) + [1.0]}).to_csv(out / "susceptibility_breaks.csv",
                                                                index=False)
    cls = np.digitize(z, breaks).astype(float)
    cls[np.isnan(z)] = np.nan
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"])
    fig, ax = plt.subplots(figsize=(5.8, 8))
    im = ax.pcolormesh(gx, gy, cls, cmap=cmap, vmin=-0.5, vmax=4.5, shading="auto")
    for g in getattr(hull, "geoms", [hull]):
        ax.plot(*g.exterior.xy, c="k", lw=0.6)
    ax.scatter(lon[df[TARGET] == 1], lat[df[TARGET] == 1], s=5, c="k", label="Landslide inventory point")
    ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    cb = fig.colorbar(im, ax=ax, ticks=range(5), shrink=0.7)
    cb.ax.set_yticklabels(["Very low", "Low", "Moderate", "High", "Very high"])
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Landslide susceptibility, {model_key.split('|')[0].replace('RandomForest', 'RF')} ({model_key.split('|')[1].replace('terrain+SMOTE', 'terrain factors, SMOTE')})\n(seed-averaged out-of-fold probabilities, IDW surface)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    savefig(fig, figdir / "fig15_susceptibility.png")
    # share of landslides falling in high + very-high classes
    pc = np.digitize(p, breaks)
    y = df[TARGET].to_numpy()
    share = pd.DataFrame({"class": pc, "y": y}).groupby("class").agg(n=("y", "size"), landslides=("y", "sum"))
    share["landslide_density_ratio"] = (share.landslides / y.sum()) / (share.n / len(y))
    share.to_csv(out / "susceptibility_class_stats.csv")


# --------------------------------------------------------------------------


def shap_figures(df: pd.DataFrame, out: Path, figdir: Path, primary: str = "random_forest",
                 secondary: str = "cnn1d") -> pd.DataFrame:
    """Fig. 14 from stored out-of-fold SHAP results.

    (a) factor shares (mean over seeds, error bar = SD over seeds) of two models;
    (b) beeswarm-style dependence of the primary model (seed-averaged SHAP values).
    Writes shap_importance_by_model.csv (index factor; columns RF, 1D-CNN = mean share;
    RF_sd, 1D-CNN_sd = SD over seeds), shap_importance_terrain.csv and shap_direction.json.
    """
    lab = SHAP_MODEL_LABEL
    mean, sd, S1 = {}, {}, None
    for m in (primary, secondary):
        sh = pd.read_csv(out / f"shap_share_by_seed_{m}.csv", index_col=0)
        mean[lab[m]] = sh.mean()
        sd[lab[m]] = sh.std(ddof=1)
        if m == primary:
            S1 = pd.read_csv(out / f"shap_values_{m}.csv", index_col=0)
    imp = pd.DataFrame(mean)
    imp_sd = pd.DataFrame(sd)
    order = imp[lab[primary]].sort_values(ascending=False).index
    imp, imp_sd = imp.loc[order], imp_sd.loc[order]
    both = imp.join(imp_sd.add_suffix("_sd"))
    both.to_csv(out / "shap_importance_by_model.csv")
    imp[lab[primary]].rename("mean_abs_SHAP_share").to_csv(out / "shap_importance_terrain.csv")

    from scipy.stats import spearmanr

    dirn = {}
    for c in ["Elevation", "Slope", "Curvature", "TWI", "TRI", "NDVI", "Annual_Rainfall"]:
        if c in S1:
            raw = df.loc[S1.index, c]
            ok = raw.notna()
            dirn[c] = float(spearmanr(raw[ok], S1.loc[ok, c]).correlation)
    lulc = {}
    for c in [c for c in S1.columns if c.startswith("LULC_")]:
        code = c.split("_", 1)[1]
        m = df.loc[S1.index, "LULC"].astype("string") == code
        if m.any():
            lulc[code] = float(S1.loc[m.values, c].mean())
    seeds = {lab[m]: json.loads((out / f"shap_seeds_{m}.json").read_text())["seeds"] for m in (primary, secondary)}
    (out / "shap_direction.json").write_text(json.dumps({"model": lab[primary], "seeds": seeds,
                                                         "spearman_value_vs_shap": dirn,
                                                         "mean_shap_when_LULC_class_present": lulc}, indent=2))

    fa, aa = plt.subplots(figsize=(6.2, 5))
    ip, isd = imp.iloc[::-1] * 100, imp_sd.iloc[::-1] * 100
    y = np.arange(len(ip))
    for off, m, col in ((0.2, primary, "#2171b5"), (-0.2, secondary, "#fd8d3c")):
        v, e = ip[lab[m]], isd[lab[m]]
        aa.barh(y + off, v, 0.4, xerr=e, color=col, label=f"{lab[m]} ({len(seeds[lab[m]])} seeds)",
                error_kw={"lw": 0.7, "capsize": 1.5})
        for yi, vi, ei in zip(y + off, v, e):
            aa.text(vi + ei + 0.8, yi, f"{vi:.1f}%", va="center", fontsize=8)
    aa.set_xlim(0, float((ip.values + isd.values).max()) + 5)
    aa.set_yticks(y)
    aa.set_yticklabels(ip.index)
    aa.set_xlabel("Share of total mean |SHAP value| (%), mean ± SD over seeds")
    aa.set_title("(a) Factor importance, out-of-fold (spatial CV)", fontsize=10)
    aa.legend(fontsize=8, loc="lower right")
    savefig(fa, figdir / "fig14a_shap_importance.png")

    fb, ab = plt.subplots(figsize=(6.2, 5))
    cont = [c for c in ["Slope", "Elevation", "NDVI", "Annual_Rainfall", "TRI", "TWI", "Curvature"] if c in S1]
    order_c = sorted(cont, key=lambda c: S1[c].abs().mean())
    rng = np.random.default_rng(SEED)
    for i, c in enumerate(order_c):
        raw = df.loc[S1.index, c].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(raw, [5, 95])
        colv = np.clip((raw - lo) / (hi - lo + 1e-12), 0, 1)
        ab.scatter(S1[c], i + rng.normal(0, 0.09, len(S1)), c=colv, cmap="coolwarm", s=6, alpha=0.8)
    ab.set_yticks(range(len(order_c)))
    ab.set_yticklabels([o.replace("Annual_Rainfall", "Annual rainfall") for o in order_c])
    ab.axvline(0, c="0.5", lw=0.6)
    ab.set_xlabel("SHAP value (impact on predicted probability)")
    ab.set_title(f"(b) Value dependence of continuous factors, {lab[primary]}", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    cb = fb.colorbar(sm, ax=ab, ticks=[0, 1])
    cb.ax.set_yticklabels(["Low", "High"])
    cb.set_label("Factor value")
    savefig(fb, figdir / "fig14b_shap_dependence.png")
    return both


def negative_extent_sensitivity(df: pd.DataFrame, out: Path, seeds=(42, 0, 1)) -> pd.DataFrame:
    """Sensitivity to negative sampling: keep only non-landslide samples inside the inventory's extent."""
    pos = df[df[TARGET] == 1]
    box = (pos.Latitude.min(), pos.Latitude.max(), pos.Longitude.min(), pos.Longitude.max())
    keep = (df[TARGET] == 1) | (df.Latitude.between(box[0], box[1]) & df.Longitude.between(box[2], box[3]))
    sub = df[keep].reset_index(drop=True)
    folds = list(spatial_block_cv(sub))
    ysub = sub[TARGET].to_numpy(dtype=int)
    rows = []
    for name in ("random_forest", "xgboost", "svm_rbf", "logistic", "cnn1d"):
        for seed in seeds:
            set_global_seeds(seed, deterministic=False)
            cv = cross_validate_model(lambda name=name, seed=seed: get_model(name, random_state=seed), sub, folds)
            rows.append({"model": MODEL_LABELS[name], "seed": seed, "within_AUC": float(cv["per_fold"].AUC.mean()),
                         "pooled_AUC": float(cv["pooled"]["AUC"])})
    tr, va, te = random_split(sub, random_state=SEED)
    trv = pd.concat([tr, va])
    pr = build_coordinates_only_probe(SEED).fit(trv, trv[TARGET].to_numpy(dtype=int))
    cvp = cross_validate_model(lambda: build_coordinates_only_probe(SEED), sub, folds)
    res = pd.DataFrame(rows).groupby("model")[["within_AUC", "pooled_AUC"]].agg(["mean", "std"]).round(3)
    res.columns = [f"{a}_{b}" for a, b in res.columns]
    res = res.reset_index()
    meta = {"n_samples": len(sub), "n_pos": int(ysub.sum()), "n_neg": int((1 - ysub).sum()), "bbox": box,
            "fold_sizes": [int(len(te_)) for _, te_ in folds], "fold_pos": [int(ysub[te_].sum()) for _, te_ in folds],
            "probe_random": float(roc_auc_score(te[TARGET].to_numpy(dtype=int), pr.predict_proba(te)[:, 1])),
            "probe_spatial_within": float(cvp["per_fold"].AUC.mean())}
    res.to_csv(out / "negative_extent_sensitivity.csv", index=False)
    (out / "negative_extent_sensitivity.json").write_text(json.dumps(meta, indent=2))
    return res, meta


def cluster_test_noise_share(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Share of each spatial-CV test fold assigned to DBSCAN noise (-1) by the train-fold fit.

    Seed-independent (DBSCAN and the partition are deterministic). Quantifies why the
    Cluster feature is uninformative under spatial CV: test points lie far from train clusters.
    """
    rows = []
    for k, (tr, te) in enumerate(spatial_block_cv(df), start=1):
        a = DBSCANClusterAssigner(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(df.iloc[tr][["Latitude", "Longitude"]])
        lab = a.transform(df.iloc[te][["Latitude", "Longitude"]])["Cluster"].astype(int).to_numpy()
        rows.append({"fold": k, "n_test": int(len(te)), "share_noise": float((lab == -1).mean()),
                     "n_train_clusters": int(a.n_clusters_)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "cluster_test_noise_share.csv", index=False)
    return res


def model_params(df: pd.DataFrame, out: Path) -> dict:
    """True encoded input width and Keras parameter counts -> model_params.json."""
    pre = build_preprocessor(include_cluster=False, drainage_mode="drop")
    X = pre.fit_transform(df, df[TARGET])
    names = [str(n) for n in pre.get_feature_names_out()]
    assert list(names) == terrain_feature_names(), names
    res = {"n_features": int(X.shape[1]), **count_keras_params(int(X.shape[1])), "feature_names": names}
    (out / "model_params.json").write_text(json.dumps(res, indent=2))
    return res


def architecture_figs(figdir: Path, n_features: int = 23, params: dict | None = None) -> None:
    """Figs. 2-4: schematic architectures that match the implemented models (Table 1)."""
    from matplotlib.patches import FancyBboxPatch

    nf = n_features
    specs = {
        "fig02_dnn.png": [(f"Input\n{nf} encoded\nfeatures", "#d9d9d9"), ("Dense 64\nReLU, L2 0.01", "#9ecae1"),
                          ("Dropout\n0.2", "#f0f0f0"), ("Dense 32\nReLU, L2 0.01", "#9ecae1"),
                          ("Dropout\n0.2", "#f0f0f0"), ("Dense 16\nReLU, L2 0.01", "#9ecae1"),
                          ("Output\n1 unit, sigmoid\nP(landslide)", "#fdd0a2")],
        "fig03_cnn1d.png": [(f"Input\n{nf} × 1 vector", "#d9d9d9"), ("Conv1D\n64 filters, k = 3\nReLU, L2 0.01", "#a1d99b"),
                            ("Dropout\n0.2", "#f0f0f0"), ("Flatten", "#f0f0f0"), ("Dense 32\nReLU, L2 0.01", "#9ecae1"),
                            ("Dropout\n0.2", "#f0f0f0"), ("Dense 16\nReLU, L2 0.01", "#9ecae1"),
                            ("Output\n1 unit, sigmoid\nP(landslide)", "#fdd0a2")],
        "fig04_lstm.png": [(f"Input\n{nf} steps × 1\n(one feature\nper step)", "#d9d9d9"),
                           ("LSTM 128\n(return\nsequences)", "#bcbddc"), ("LSTM 64", "#bcbddc"),
                           ("Dense 32\nReLU", "#9ecae1"), ("Output\n1 unit, sigmoid\nP(landslide)", "#fdd0a2")],
    }
    titles = {"fig02_dnn.png": "Deep neural network (DNN)", "fig03_cnn1d.png": "One-dimensional CNN (1D-CNN)",
              "fig04_lstm.png": "Stacked LSTM network"}
    for fname, blocks in specs.items():
        n = len(blocks)
        fig, ax = plt.subplots(figsize=(1.45 * 8 + 0.4, 2.3))
        ax.set_xlim(0, 1.45 * 8)
        ax.set_ylim(0, 2.3)
        ax.axis("off")
        for i, (txt, col) in enumerate(blocks):
            x = 0.2 + 1.45 * i
            ax.add_patch(FancyBboxPatch((x, 0.45), 1.05, 1.3, boxstyle="round,pad=0.03,rounding_size=0.08",
                                        fc=col, ec="#333333", lw=0.9))
            ax.text(x + 0.525, 1.1, txt, ha="center", va="center", fontsize=7.6)
            if i < n - 1:
                ax.annotate("", xy=(x + 1.42, 1.1), xytext=(x + 1.1, 1.1),
                            arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.9))
        key = {"fig02_dnn.png": "DNN", "fig03_cnn1d.png": "1D-CNN", "fig04_lstm.png": "LSTM"}[fname]
        ptxt = f" ({params[key]:,} trainable parameters)" if params and key in params else ""
        ax.text(0.2, 2.05, titles[fname] + ptxt, fontsize=9, fontweight="bold", va="center")
        ax.text(0.2, 0.18, "Adam (lr = 0.001), binary cross-entropy, batch 32, ≤ 50 epochs, early stopping (val. loss, patience 5) on a "
                "20% validation split of the training fold", fontsize=6.8, style="italic")
        savefig(fig, figdir / fname)


def flowchart(figdir: Path) -> None:
    """Fig. 6: revised methodological flowchart."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(8.8, 9.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13.4)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fb", fs=7.4, bold=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc=fc, ec="#2b4c7e", lw=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, wrap=True,
                fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#2b4c7e", lw=1.0))

    box(1.75, 12.6, 3.1, 1.0, "Landslide inventory\n198 points (196 unique)\nNASA, CEGIS, Bangladesh Army,\nand other secondary sources")
    box(5.0, 12.6, 3.0, 1.0, "Non-landslide samples\n300 random points\n(buffered from landslides)")
    box(8.25, 12.6, 3.1, 1.0, "12 conditioning factors\nSRTM DEM, Sentinel-2 NDVI,\nEsri LULC, CRU rainfall, SoilGrids")
    for x in (1.75, 5.0, 8.25):
        arrow(x, 12.08, 5.0, 11.42)
    box(5.0, 11.0, 7.6, 0.75, "Step 1 - Data audit: duplicate removal, SPI/TWI check, circular encoding of aspect,\none-hot encoding of LULC and soil texture (496 points, 10 terrain factors in the final set)")
    arrow(5.0, 10.6, 5.0, 10.1)
    box(5.0, 9.55, 7.6, 1.0, "Step 2 - Spatial sparsity diagnosis\nVoronoi cell-area statistics, Clark-Evans nearest-neighbour index,\ngrid occupancy, local Getis-Ord Gi* and global Moran's I", bold=False)
    arrow(5.0, 9.03, 5.0, 8.53)
    box(5.0, 8.0, 7.6, 0.95, "Step 3 - Spatial-context features and sparsity remedies\nDBSCAN cluster label (k-distance and eps x min_samples sensitivity);\nSMOTE oversampling inside each training fold")
    arrow(5.0, 7.5, 5.0, 7.0)
    box(5.0, 6.45, 7.6, 1.0, "Step 4 - Validation design\nReference: stratified random 60/20/20 split\nPrimary: spatial block cross-validation (10 k-means blocks merged into 5 folds);\nimputation, scaling, encoding, DBSCAN and SMOTE fitted inside each training fold")
    arrow(5.0, 5.93, 5.0, 5.43)
    box(2.75, 4.85, 4.1, 1.0, "Deep learning\nDNN, 1D-CNN, LSTM", fc="#fdf1e6")
    box(7.3, 4.85, 4.1, 1.0, "Machine-learning baselines\nRandom forest, XGBoost,\nSVM (RBF), logistic regression", fc="#fdf1e6")
    ax.text(5.0, 4.12, "3 feature configurations (terrain | terrain + Cluster | terrain + SMOTE) x 10 random seeds",
            ha="center", va="center", fontsize=7.2, style="italic")
    arrow(2.75, 3.95, 5.0, 3.55)
    arrow(7.3, 3.95, 5.0, 3.55)
    box(5.0, 3.05, 7.6, 0.9, "Step 5 - Evaluation\nwithin-fold and pooled out-of-fold AUC, accuracy, precision, recall, F1;\ncoordinates-only probe as a leakage check")
    arrow(5.0, 2.58, 5.0, 2.08)
    box(5.0, 1.6, 7.6, 0.85, "Step 6 - Interpretation and mapping\nout-of-fold SHAP with geomorphological reading;\npoint-level susceptibility map (seed-averaged out-of-fold) with area of applicability")
    savefig(fig, figdir / "fig06_flowchart.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Labeled Data Set.csv")
    ap.add_argument("--out", default="results/revision")
    ap.add_argument("--figdir", default="reports/figures/revision")
    ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    ap.add_argument("--part", choices=["descriptive", "seeds", "aggregate", "shap", "shapfig", "negsens"], default="descriptive")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--map-model", default="1D-CNN|terrain")
    ap.add_argument("--shap-model", default="random_forest", choices=["random_forest", "cnn1d"])
    ap.add_argument("--shap-seeds", nargs="+", type=int, default=list(SHAP_SEEDS_ALL))
    ap.add_argument("--shap-time-limit", type=float, default=None,
                    help="seconds; if the first seed exceeds it, fall back to 5 seeds")
    a = ap.parse_args(argv)
    out, figdir = Path(a.out), Path(a.figdir)
    out.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    df = clean(load_raw(a.data)).reset_index(drop=True)

    if a.part == "descriptive":
        print("rows", len(df), "positives", int(df[TARGET].sum()))
        sparsity_metrics(df, out, figdir)
        dbscan_analysis(df, out, figdir)
        print(negative_sampling(df, out))
        print(cluster_test_noise_share(df, out))
        mp = model_params(df, out)
        print({k: v for k, v in mp.items() if k != "feature_names"})
        flowchart(figdir)
        architecture_figs(figdir, mp["n_features"], mp)
    elif a.part == "seeds":
        for s in a.seeds:
            run_seed(df, s, a.models, out)
    elif a.part == "aggregate":
        agg = aggregate(df, out, figdir)
        susceptibility_surface(df, agg["oof"], out, figdir, a.map_model)
    elif a.part == "negsens":
        print(negative_extent_sensitivity(df, out))
    elif a.part == "shapfig":
        print(shap_figures(df, out, figdir))
    elif a.part == "shap":
        print(shap_analysis(df, out, a.shap_model, a.shap_seeds, a.shap_time_limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
