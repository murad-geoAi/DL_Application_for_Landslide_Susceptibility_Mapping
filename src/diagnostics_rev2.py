"""Second pre-submission audit: facts and small diagnostics (read-only on the pipeline).

Run from the repo root:  python -m src.diagnostics_rev2
Writes results/revision/extra/rev2_diagnostics.json and rev2_diagnostics.md.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import DBSCAN
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors

from .data import TARGET
from .features import (DBSCAN_EPS, DBSCAN_MIN_SAMPLES, FIXED_LAT0, DBSCANClusterAssigner,
                       build_preprocessor, project_km, terrain_feature_names)
from .revision_analyses import clark_evans, district_polygon, to_km, voronoi_areas
from .revision_extra import (DI_FEATURES, compute_di, di_weights, load_df, merge_to_4,
                             bbox_subset)
from .splits import N_SPLITS, random_split, spatial_block_cv, spatial_fold_labels

warnings.filterwarnings("ignore")
REV = Path("results/revision")
OUT = REV / "extra"
SEED = 42
R: dict = {}


def vsum(a):
    a = a[np.isfinite(a)]
    return {"n_cells": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "cv": float(a.std() / a.mean()), "max": float(a.max())}


def main():
    df = load_df()
    y = df[TARGET].to_numpy(int)
    pos = y == 1
    lab = spatial_fold_labels(df)
    lat0 = FIXED_LAT0
    xy = to_km(df.Latitude.to_numpy(), df.Longitude.to_numpy(), lat0)
    poly = district_polygon(lat0)
    hull_area = ConvexHull(xy).volume
    R["n_total"], R["n_pos"], R["n_neg"] = int(len(df)), int(pos.sum()), int((~pos).sum())

    # ---------------- A1 Clark-Evans
    a1 = {"formula": "R = mean NN dist / (0.5/sqrt(n/A)); SE = 0.26136/sqrt(n*density); z=(r_obs-r_exp)/SE; "
                     "revision_analyses.py:86-102; NN via cKDTree(k=2) on projected km; NO edge correction in the "
                     "reported R/z (A = convex hull of all 496 or Rangamati ADM2 polygon)",
          "district_area_km2": float(poly.area), "hull_area_km2": float(hull_area)}
    for nm, m in [("landslides", pos), ("non_landslides", ~pos), ("pooled", np.ones(len(df), bool))]:
        for an, A in [("hull", hull_area), ("district", poly.area)]:
            a1[f"{nm}_{an}"] = {k: float(v) for k, v in clark_evans(xy[m], A).items()}
        # Donnelly (1978) edge correction on the district polygon (diagnostic, NOT in the pipeline)
        n = int(m.sum())
        d, _ = cKDTree(xy[m]).query(xy[m], k=2)
        robs = d[:, 1].mean()
        B = poly.length
        A = poly.area
        e_don = 0.5 * np.sqrt(A / n) + (0.0514 + 0.041 / np.sqrt(n)) * B / n
        se_don = np.sqrt(0.070 * A / n ** 2 + 0.037 * B * np.sqrt(A / n ** 5))
        a1[f"{nm}_district_donnelly"] = {"perimeter_km": float(B), "r_exp": float(e_don), "R": float(robs / e_don),
                                         "z": float((robs - e_don) / se_don)}
    R["A1_clark_evans"] = a1

    # ---------------- A2 projection
    km_lon = 111.320 * np.cos(np.deg2rad(lat0))
    R["A2_projection"] = {
        "type": "local equirectangular (NOT an EPSG CRS): x = lon*111.320*cos(lat0) km, y = lat*110.574 km; "
                "features.py:100-106 (project_km) and revision_analyses.py:70-75 (to_km, same constants)",
        "lat0_deg": lat0, "lat0_source": "mean latitude of the full 496-row dataset, fixed constant (features.py:51)",
        "km_per_deg_lon_at_lat0": km_lon, "km_per_deg_lat": 110.574,
        "dbscan_eps_1.1km_in_deg": {"lat": 1.1 / 110.574, "lon": 1.1 / km_lon},
        "gi_band_5.5km_in_deg": {"lat": 5.5 / 110.574, "lon": 5.5 / km_lon},
        "note": "eps (1.1 km) and Gi* band (5.5 km) are defined directly in km on the projected plane; the degree "
                "values (~0.01 deg, ~0.05 deg) are nominal conversions only (sparsity code stores eps_deg=round(km/110.574,3))."}

    # ---------------- A3 jitter + duplicates
    d_all, _ = cKDTree(xy).query(xy, k=2)
    nn = d_all[:, 1]
    R["A3_jitter"] = {
        "code": "revision_analyses.py:114: np.random.default_rng(42).normal(0, 1e-6, xy.shape) added to km coordinates",
        "units": "km (xy is projected km), so SD = 1e-6 km = 1 mm; normal(loc, scale) -> 1e-6 is the SD, not the variance",
        "n_points_with_nn_dist_eq_0": int((nn == 0).sum()),
        "n_points_nn_below_1e-3_km": int((nn < 1e-3).sum()),
        "n_points_nn_below_1e-2_km": int((nn < 1e-2).sum()),
        "min_positive_nn_km": float(nn[nn > 0].min()),
        "n_exact_duplicate_coordinate_rows_in_496": int(df.duplicated(["Latitude", "Longitude"]).sum()),
        "n_exact_duplicate_coords_landslides": int(df[pos].duplicated(["Latitude", "Longitude"]).sum()),
        "n_exact_duplicate_coords_non_landslides": int(df[~pos].duplicated(["Latitude", "Longitude"]).sum()),
    }
    # Voronoi without jitter: how many cells come out undefined
    from scipy.spatial import Voronoi
    for nm, m in [("pos196", pos), ("all496", np.ones(len(df), bool))]:
        P = xy[m]
        span = np.ptp(xy, axis=0).max() * 10
        c = xy.mean(0)
        far = np.array([[c[0] - span, c[1] - span], [c[0] + span, c[1] - span], [c[0] + span, c[1] + span],
                        [c[0] - span, c[1] + span]])
        vor = Voronoi(np.vstack([P, far]))
        bad = sum(1 for i in range(len(P)) if (-1 in vor.regions[vor.point_region[i]]
                                               or len(vor.regions[vor.point_region[i]]) == 0))
        R["A3_jitter"][f"no_jitter_undefined_cells_{nm}"] = int(bad)

    # ---------------- A4 Voronoi
    from shapely.geometry import MultiPoint
    hull_poly = MultiPoint([tuple(p) for p in xy]).convex_hull
    a4 = {"clip_polygon_used_in_pipeline": "Rangamati ADM2 polygon (revision_analyses.py:170-171), not the convex hull"}
    for nm, m in [("landslides_196", pos), ("all_496", np.ones(len(df), bool))]:
        a4[f"{nm}_clip_district"] = vsum(voronoi_areas(xy[m], xy, poly))
        a4[f"{nm}_clip_hull"] = vsum(voronoi_areas(xy[m], xy, hull_poly))
    a4["note_tessellation_is_over_the_points_given_but_hull_for_far_points_uses_all_496_extent"] = True
    R["A4_voronoi"] = a4

    # ---------------- A5 Gi*, A6 Moran / joins
    from esda.getisord import G_Local
    from esda.join_counts import Join_Counts
    from esda.moran import Moran
    from libpysal.weights import KNN, DistanceBand
    yv = y.astype(float)
    wd = DistanceBand(xy, threshold=5.5, binary=True, silence_warnings=True)
    islands = sorted(wd.islands)
    ncount = np.asarray(wd.sparse.sum(axis=1)).ravel()
    wd.transform = "r"
    g = G_Local(yv, wd, star=True, permutations=999, seed=SEED)
    isl = np.zeros(len(df), bool)
    isl[islands] = True
    sig = (g.p_sim < 0.05) & ~isl
    Ab = (DistanceBand(xy, threshold=5.5, binary=True, silence_warnings=True).sparse > 0).astype(float).toarray()
    kk_nb = Ab.sum(1)
    np.fill_diagonal(Ab, 1.0)
    manual_star = (Ab @ yv / Ab.sum(1)) / yv.sum()
    swv, sw2v = Ab.sum(1), (Ab ** 2).sum(1)
    zord = ((Ab @ yv) - yv.mean() * swv) / (yv.std(ddof=0) * np.sqrt((len(yv) * sw2v - swv ** 2) / (len(yv) - 1)))
    cold_m = sig & (g.Zs < 0)
    rowsum = np.ones(len(yv))
    diag = np.ones(len(yv))
    sig2 = (g.p_sim < 0.025) & ~isl
    sigz = (g.p_norm * 2 < 0.05) & ~isl
    mean_all, sd0 = yv.mean(), yv.std(ddof=0)
    R["A5_gi_star"] = {
        "code": "revision_analyses.py:213-247: libpysal DistanceBand(xy_km, threshold=5.5, binary=True) -> transform 'r' "
                "-> esda.getisord.G_Local(y, w, star=True, permutations=999, seed=42), esda 2.9.0 / libpysal 4.14.1",
        "star_self_weight_included": bool(g.star) and bool(np.abs(g.Gs[~isl] - manual_star[~isl]).max() < 1e-9),
        "manual_gi_star_with_self_weight_matches_esda_maxabsdiff": float(np.abs(g.Gs[~isl] - manual_star[~isl]).max()),
        "mechanism": "input W has zero diagonal; star=True with transform 'R' -> esda fills diagonal with the row-maximum "
                     "of the row-standardised weights (= 1/k_i) then re-standardises, so self weight == neighbour "
                     "weight (equivalent to binary incl. self, divided by k_i+1); UserWarning emitted by esda",
        "diag_example_first_nonisland_row": float(diag[~isl][0]), "rowsum_min": float(rowsum.min()),
        "rowsum_nonisland_all_one": bool(np.allclose(rowsum[~isl], 1.0)),
        "neighbours_within_5.5km_min_median_max": [int(ncount.min()), float(np.median(ncount)), int(ncount.max())],
        "islands_n": int(isl.sum()), "island_rows_in_W": "zero weights (no self weight: row max = 0) -> Gs=0, Zs NaN; "
                                                          "islands are excluded from sig by mask `& ~isl` (pipeline code line ~226)",
        "island_Zs_nan": bool(np.isnan(g.Zs[isl]).all()) if isl.any() else None,
        "xbar_S_basis": "esda uses ALL 496 points (N = n for star), x_bar = mean(y)=%.4f, S^2 = mean(y^2)-xbar^2 "
                        "(population variance, denominator n, ddof=0), S=%.4f; islands not removed from x_bar/S" % (mean_all, sd0),
        "permutations": 999, "p_type": "esda default alternative='directed': p_sim = smaller-tail one-sided pseudo p "
                                       "(NOT doubled); pipeline tests p_sim<0.05 for both hot and cold",
        "counts_pipeline_p_sim_lt_0.05": {"hot": int((sig & (g.Zs > 0)).sum()), "cold": int((sig & (g.Zs < 0)).sum()),
                                           "ns": int((~sig & ~isl).sum()), "islands": int(isl.sum())},
        "counts_two_sided_equiv_p_sim_lt_0.025": {"hot": int((sig2 & (g.Zs > 0)).sum()), "cold": int((sig2 & (g.Zs < 0)).sum()),
                                                  "ns": int((~sig2 & ~isl).sum())},
        "esda_analytic_Zs_range_nonisland(degenerate, not a usable z)": [float(np.nanmin(g.Zs[~isl])), float(np.nanmax(g.Zs[~isl]))],
        "esda_Zs_used_only_for_sign": True,
        "textbook_Ord_Getis_Gi*_z(|z|>1.96, binary incl self, S ddof 0 or 1 same counts)": {
            "hot": int(((zord > 1.96) & ~isl).sum()), "cold": int(((zord < -1.96) & ~isl).sum()),
            "ns": int((np.abs(zord[~isl]) <= 1.96).sum())},
        "cold_pipeline_with_Gs_eq_0 (all-zero neighbourhood incl self)": int((cold_m & (g.Gs == 0)).sum()),
        "cold_pipeline_neighbour_count_excl_self_min_median_max": [int(kk_nb[cold_m].min()), float(np.median(kk_nb[cold_m])), int(kk_nb[cold_m].max())],
        "cold_pipeline_with_<=3_neighbours": int((cold_m & (kk_nb <= 3)).sum()),
        "cold_overlap_pipeline_and_textbook": int((cold_m & (zord < -1.96)).sum()),
        "hot_overlap_pipeline_and_textbook": int(((sig & (g.Zs > 0)) & (zord > 1.96)).sum()),
        "p_sim_tie_issue": "for observed G_i*=0 esda 'directed' p counts ties (sim>=obs) as extreme -> p_sim=0.001 irrespective of how "
                           "likely a zero neighbourhood is; cold set is therefore anti-conservative for small neighbourhoods",
        "no_multiple_testing_correction": True,
        "hot_share_landslides_among_hot": float(y[sig & (g.Zs > 0)].mean()),
        "cold_share_landslides_among_cold": float(y[sig & (g.Zs < 0)].mean()),
    }

    w5 = KNN.from_array(xy, k=5)
    S = w5.sparse.tocsr().copy()
    S.data[:] = 1.0
    arcs = int(S.nnz)
    Sb = S.multiply(S.T).tocsr()
    mutual_arcs = int(Sb.nnz)
    U = ((S + S.T) > 0).astype(int).tocsr()
    und_pairs = int((U.nnz) // 2)
    bb_arcs = int(S.multiply(np.outer(y, y)).sum()) if False else int(((S @ y) * y).sum())
    ww_arcs = int(((S @ (1 - y)) * (1 - y)).sum())
    bw_arcs = int(((S @ (1 - y)) * y).sum() + ((S @ y) * (1 - y)).sum())
    Uy = U
    bb_und = int(((Uy @ y) * y).sum() // 2)
    n1 = int(y.sum())
    n = len(y)
    np.random.seed(SEED)
    wr = KNN.from_array(xy, k=5)
    wr.transform = "r"
    mor = Moran(yv, wr, permutations=999)
    wb = KNN.from_array(xy, k=5)
    wb.transform = "B"
    np.random.seed(SEED)
    jc = Join_Counts(y.astype(int), wb, permutations=999)
    R["A6_moran_joins"] = {
        "code": "revision_analyses.py:218-239",
        "moran": {"weights": "KNN k=5 on projected km, row-standardised, asymmetric (not symmetrised), 999 perm, np.random.seed(42)",
                  "I": float(mor.I), "EI": float(mor.EI), "z_sim": float(mor.z_sim), "p_sim": float(mor.p_sim)},
        "join_counts_weights": "KNN k=5, binary ('B'), NOT symmetrised, directed kNN graph",
        "directed_arcs_total": arcs, "mutual_arcs_i_j_both_directions": mutual_arcs,
        "unique_undirected_pairs_if_symmetrised": und_pairs,
        "esda_join_counts_convention": "esda sums over all directed arcs and multiplies by 0.5, so n_joins_total = arcs/2",
        "esda_joins": {"bb": float(jc.bb), "ww": float(jc.ww), "bw": float(jc.bw), "J": float(jc.J),
                       "mean_bb_sim": float(jc.mean_bb), "p_sim_bb": float(jc.p_sim_bb)},
        "recount_directed_arcs": {"bb": bb_arcs, "ww": ww_arcs, "bw": bw_arcs, "sum": bb_arcs + ww_arcs + bw_arcs},
        "bb_equals_arcs_over_2": bool(abs(jc.bb - bb_arcs / 2) < 1e-9),
        "observed_and_expected_same_edge_count": "yes: both on J=%.0f (=arcs/2); expected analytic BB = J*n1(n1-1)/(n(n-1)) = %.2f; "
                                                 "permutation mean_bb=%.2f" % (jc.J, jc.J * n1 * (n1 - 1) / (n * (n - 1)), jc.mean_bb),
        "if_symmetrised_unique_pairs": {"pairs": und_pairs, "observed_bb": bb_und,
                                        "expected_bb": und_pairs * n1 * (n1 - 1) / (n * (n - 1))},
    }

    # ---------------- B DBSCAN
    folds = list(spatial_block_cv(df))
    b = {"B7": {"class_blind": "yes: DBSCANClusterAssigner.fit(X, y=None) takes only coordinates of ALL training-fold rows "
                               "(both classes pooled); y never used (features.py:350-372)",
                "eps_km": DBSCAN_EPS, "min_samples": DBSCAN_MIN_SAMPLES,
                "metric": "sklearn DBSCAN default euclidean on projected km (features.py:358-359); min_samples counts the point itself",
                "assignment_of_test_points": "nearest training CORE point within eps (<=) -> that cluster, else -1 (features.py:376-392); "
                                             "no refit on test"}}
    # B8 k-distance
    kd5 = np.sort(cKDTree(xy).query(xy, k=6)[0][:, 5])
    kd4 = np.sort(cKDTree(xy).query(xy, k=5)[0][:, 4])
    nn_sk = NearestNeighbors(n_neighbors=5).fit(xy).kneighbors(xy)[0][:, -1]
    b["B8"] = {"code": "revision_analyses.py:366-372: cKDTree.query(xy, k=6); kd = sort(d[:, 5])  (column 0 is the point itself)",
               "convention": "5th-nearest neighbour EXCLUDING self (= 6th incl. self). DBSCAN min_samples=5 INCLUDES self, so the "
                             "matching k-dist is the 4th NN excl. self (= sklearn NearestNeighbors(n_neighbors=5) last column which "
                             "includes self as neighbour 1)",
               "sorted_kdist_excl_self_k5_median_p90_max_km": [float(np.median(kd5)), float(np.percentile(kd5, 90)), float(kd5.max())],
               "sorted_kdist_matching_dbscan_k4_excl_self_median_p90_km": [float(np.median(kd4)), float(np.percentile(kd4, 90))],
               "sklearn_NN5_incl_self_equals_k4_excl_self": bool(np.allclose(np.sort(nn_sk), kd4)),
               "frac_points_kdist5_le_1.1": float((kd5 <= 1.1).mean()), "frac_points_kdist4_le_1.1": float((kd4 <= 1.1).mean()),
               "knee_km_in_dbscan_rule_audit": json.load(open(REV / "dbscan_rule_audit.json"))["k_distance_knee_km"],
               "eps_vs_knee": "adopted eps 1.1 km is far below the chord-knee 6.96 km (knee is at the large-distance tail)"}

    # B9/B10
    def cluster_stats(tr, te, tag):
        a = DBSCANClusterAssigner(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(df.iloc[tr][["Latitude", "Longitude"]])
        pre = build_preprocessor(include_cluster=True, drainage_mode="drop")
        Xt = pre.fit_transform(df.iloc[tr], y[tr])
        names = [str(c) for c in pre.get_feature_names_out()]
        tl = a.labels_
        ytr = y[tr]
        kmte = project_km(df.Latitude.to_numpy()[te], df.Longitude.to_numpy()[te])
        kmtr = project_km(df.Latitude.to_numpy()[tr], df.Longitude.to_numpy()[tr])
        dcore = cKDTree(a.core_points_km_).query(kmte)[0]
        dall = cKDTree(kmtr).query(kmte)[0]
        lab_te = a.transform(df.iloc[te][["Latitude", "Longitude"]])["Cluster"].astype(int).to_numpy()
        return {"fold": tag, "n_train": int(len(tr)), "n_test": int(len(te)),
                "n_clusters_train": int(a.n_clusters_), "n_noise_train": int(a.n_noise_),
                "noise_share_train": float((tl == -1).mean()),
                "noise_share_of_train_landslides": float((tl[ytr == 1] == -1).mean()),
                "n_train_landslides": int(ytr.sum()),
                "total_input_dim_terrain_plus_Cluster": int(Xt.shape[1]),
                "cluster_columns": [c for c in names if c.startswith("Cluster")],
                "noise_is_own_column(Cluster_-1)": "Cluster_-1" in names,
                "test_share_noise": float((lab_te == -1).mean()),
                "min_dist_test_to_train_core_km": float(dcore.min()), "median_dist_test_to_core_km": float(np.median(dcore)),
                "n_test_within_eps_of_core": int((dcore <= DBSCAN_EPS).sum()),
                "min_dist_test_to_any_train_point_km": float(dall.min())}

    sp = [cluster_stats(tr, te, k + 1) for k, (tr, te) in enumerate(folds)]
    b["B9_spatial_folds"] = sp
    b["B9_onehot"] = ("build_cluster_encoder uses OneHotEncoder(handle_unknown='ignore'), drop=None -> noise (-1) is its own column "
                      "'Cluster_-1', not an all-zero reference (features.py:630-646); unseen labels at test time -> all-zero")
    b["B9_dim_range_spatial"] = [min(s["total_input_dim_terrain_plus_Cluster"] for s in sp),
                                 max(s["total_input_dim_terrain_plus_Cluster"] for s in sp)]
    b["B9_n_clusters_range_spatial"] = [min(s["n_clusters_train"] for s in sp), max(s["n_clusters_train"] for s in sp)]
    b["B9_noise_landslide_share_range_spatial"] = [min(s["noise_share_of_train_landslides"] for s in sp),
                                                   max(s["noise_share_of_train_landslides"] for s in sp)]
    b["B9_noise_all_share_range_spatial"] = [min(s["noise_share_train"] for s in sp), max(s["noise_share_train"] for s in sp)]
    b["B10_min_dist_test_to_core_per_fold_km"] = {s["fold"]: s["min_dist_test_to_train_core_km"] for s in sp}
    b["B10_all_test_points_noise_all_folds"] = bool(all(s["test_share_noise"] == 1.0 for s in sp))
    tr0, va0, te0 = random_split(df, random_state=SEED)
    idx = {i: p for p, i in enumerate(df.index)}
    rtr = np.array([idx[i] for i in tr0.index])
    rte = np.array([idx[i] for i in te0.index])
    b["B9_random_60_20_20_seed42_train297"] = cluster_stats(rtr, rte, "random_split")
    rk = [cluster_stats(t, e, k + 1) for k, (t, e) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(xy, y))]
    b["B9_random_stratified_5fold_rs42"] = rk
    b["B9_dim_range_random_5fold"] = [min(s["total_input_dim_terrain_plus_Cluster"] for s in rk),
                                      max(s["total_input_dim_terrain_plus_Cluster"] for s in rk)]
    b["B9_random_5fold_test_share_noise"] = [s["test_share_noise"] for s in rk]
    R["B_dbscan"] = b

    # ---------------- C11 SMOTE snapping counts, C12 order
    from imblearn.over_sampling import SMOTE
    names = terrain_feature_names()
    ix = {nme: i for i, nme in enumerate(names)}
    from .features import CATEGORICAL_COLS, CATEGORY_LEVELS
    groups = {c: [ix[f"{c}_{l}"] for l in CATEGORY_LEVELS[c]] for c in CATEGORICAL_COLS}

    def count_syn(Xtr, ytr, seed=SEED):
        n0 = len(Xtr)
        Xr, yr = SMOTE(random_state=seed, k_neighbors=5).fit_resample(Xtr, ytr)
        Xs = Xr[n0:]
        flagraw = Xs[:, ix["is_flat"]]
        nrm = np.hypot(Xs[:, ix["Aspect_sin"]], Xs[:, ix["Aspect_cos"]])
        flag = flagraw >= 0.5
        ties = 0
        for c, cols in groups.items():
            v = Xs[:, cols]
            mx = v.max(1)
            ties += int(((np.isclose(v, mx[:, None], atol=1e-9)).sum(1) > 1).sum())
        return {"n_synthetic": int(len(Xs)),
                "flat_flag_rounds_to_1": int(flag.sum()),
                "flat_flag_eq_exactly_0.5": int((flagraw == 0.5).sum()),
                "flat1_with_nonzero_aspect_pre_snap(norm>1e-6)": int((flag & (nrm > 1e-6)).sum()),
                "flat0_with_norm_lt_1e-6(set_to_flat)": int((~flag & (nrm < 1e-6)).sum()),
                "norm_lt_0.1_nonflat": int((~flag & (nrm < 0.1)).sum()),
                "norm_lt_0.5_nonflat": int((~flag & (nrm < 0.5)).sum()),
                "min_norm_nonflat": float(nrm[~flag].min()) if (~flag).any() else None,
                "median_norm_nonflat": float(np.median(nrm[~flag])),
                "onehot_group_argmax_ties(LULC+Soil)": ties,
                "flat_flag_fractional_not_0_or_1": int(((flagraw > 0) & (flagraw < 1)).sum())}, Xs

    from .sampling import snap_synthetic
    c11 = {"code": "sampling.py:26-46 (snap_synthetic), 49-67 (SnapSMOTE); imblearn SMOTE k=5, random_state=seed",
           "projection": "one-hot groups: np.argmax over group columns (first max wins, i.e. level order in CATEGORY_LEVELS: "
                         "1,11,2,4,5,7,missing / 4,6,7,missing; exact tie at lambda=0.5 -> lowest-ordered level, not random); "
                         "binary flags: >=0.5 -> 1 (tie rounds up); aspect: if flat flag==1 or norm<1e-6 -> flag=1,sin=cos=0, "
                         "else divide by norm (unit circle); real rows untouched; ASPECT of REAL rows imputed by KNN for 4 missing "
                         "rows are not renormalised",
           "per_fold_full_trainfold(RF/XGB/SVM/LR path)": [], "per_fold_80pct_subtrain(Keras path)": []}
    tot_full, tot_80 = {}, {}
    snapped_bad = 0
    for k, (tr, te) in enumerate(folds, 1):
        pre = build_preprocessor(include_cluster=False, drainage_mode="drop")
        Xtr = np.asarray(pre.fit_transform(df.iloc[tr], y[tr]), float)
        ytr = y[tr]
        c, Xs = count_syn(Xtr, ytr)
        c["fold"] = k
        c11["per_fold_full_trainfold(RF/XGB/SVM/LR path)"].append(c)
        sn = snap_synthetic(Xs, names)
        nrm = np.hypot(sn[:, ix["Aspect_sin"]], sn[:, ix["Aspect_cos"]])
        snapped_bad += int(((sn[:, ix["is_flat"]] == 1) & (nrm > 0)).sum()) + int(((sn[:, ix["is_flat"]] == 0) & (np.abs(nrm - 1) > 1e-6)).sum())
        for kk, vv in c.items():
            if isinstance(vv, (int, float)) and kk != "fold" and not kk.startswith(("min_", "median_")):
                tot_full[kk] = tot_full.get(kk, 0) + vv
        Xf, Xv, yf, yvv = train_test_split(Xtr, ytr, test_size=0.2, random_state=SEED, shuffle=True, stratify=ytr)
        c2, _ = count_syn(Xf, yf)
        c2["fold"] = k
        c11["per_fold_80pct_subtrain(Keras path)"].append(c2)
        for kk, vv in c2.items():
            if isinstance(vv, (int, float)) and kk != "fold" and not kk.startswith(("min_", "median_")):
                tot_80[kk] = tot_80.get(kk, 0) + vv
    c11["totals_over_5_folds_full_trainfold"] = tot_full
    c11["totals_over_5_folds_80pct_subtrain"] = tot_80
    c11["post_snap_invalid_rows(flat with nonzero aspect, or nonflat norm!=1) full-train path"] = snapped_bad
    R["C11_smote"] = c11
    R["C12_order"] = {
        "fit_pipeline": "models.py:886-913: for any model with X_val=None (the CV path used by revision_analyses.run_seed via "
                        "cross_validate_model(fit_fn=None) -> pipe.fit(X_train,y_train)) the preprocessor (KNN imputer, scaler, "
                        "category schema, DBSCAN if included) is fitted on 100% of the training fold",
        "keras_inner_split": "models.py:466-478: KerasBinaryClassifier.fit then takes the 20% stratified early-stopping split from the "
                             "ALREADY PREPROCESSED training matrix (random_state=seed); so imputation/scaling statistics include the "
                             "20% validation rows (labels are not used by them, only features)",
        "smote": "models.py:~481-486: SnapSMOTE.fit_resample is applied AFTER the split, on the 80% sub-train only (Keras). "
                 "Non-Keras models: ResampledClassifier.fit applies SMOTE to the full 100% training fold (no validation set, models.py:956-975)",
        "test_fold": "never touches fit/transform statistics (transform only)"}

    # ---------------- D13 flat cells
    flat = (df.Slope == 0).fillna(False).to_numpy()
    water = (df.LULC == 1).fillna(False).to_numpy()
    R["D13_flat"] = {"n_flat": int(flat.sum()), "flat_landslides": int((flat & pos).sum()), "flat_non_landslides": int((flat & ~pos).sum()),
                     "flat_and_LULC1_water": int((flat & water).sum()),
                     "flat_and_water_by_class": {"landslide": int((flat & water & pos).sum()), "non_landslide": int((flat & water & ~pos).sum())},
                     "flat_LULC_value_counts": {str(k): int(v) for k, v in df[flat].LULC.value_counts(dropna=False).items()},
                     "water_pixels_total(LULC==1)": int(water.sum()),
                     "water_by_class": {"landslide": int((water & pos).sum()), "non_landslide": int((water & ~pos).sum())},
                     "water_flag_note": "no separate water flag in the code; water = LULC code 1 (revision_extra.py part_water: df.LULC==1)",
                     "flat_where_Aspect_eq_-1": int((df.Aspect == -1).sum()) if "Aspect" in df else None}

    # ---------------- D14 folds
    def fold_tab(labels, ylab, k):
        rows = []
        for f in range(k):
            m = labels == f
            rows.append({"fold": f + 1, "n": int(m.sum()), "pos": int(ylab[m].sum()), "neg": int((ylab[m] == 0).sum()),
                         "test_prev": float(ylab[m].mean()), "train_n": int((~m).sum()), "train_prev": float(ylab[~m].mean())})
        return rows
    d14 = {"spatial5": fold_tab(lab, y, 5)}
    sub, box = bbox_subset(df)
    ysub = sub[TARGET].to_numpy(int)
    labb = spatial_fold_labels(sub)
    d14["bbox5"] = fold_tab(labb, ysub, 5)
    d14["bbox_n_total_pos_neg"] = [int(len(sub)), int(ysub.sum()), int((ysub == 0).sum())]
    d14["bbox5_n_range"] = [min(r["n"] for r in d14["bbox5"]), max(r["n"] for r in d14["bbox5"])]
    d14["bbox5_pos_share_range"] = [min(r["test_prev"] for r in d14["bbox5"]), max(r["test_prev"] for r in d14["bbox5"])]
    lab4, info4 = merge_to_4(df)
    d14["cv4"] = fold_tab(lab4, y, 4)
    d14["cv4_merge_info"] = info4
    R["D14_folds"] = d14

    # ---------------- D15 lat span / rainfall cells
    rain = df.Annual_Rainfall
    d15 = {}
    for f in range(5):
        m = lab == f
        cells = set(zip(np.floor(df.Latitude[m] / 0.5).astype(int), np.floor(df.Longitude[m] / 0.5).astype(int)))
        d15[f"fold{f+1}"] = {"lat_min": float(df.Latitude[m].min()), "lat_max": float(df.Latitude[m].max()),
                             "lon_min": float(df.Longitude[m].min()), "lon_max": float(df.Longitude[m].max()),
                             "n_distinct_rainfall_values": int(rain[m].nunique()),
                             "n_rainfall_nan": int(rain[m].isna().sum()),
                             "n_distinct_0.5deg_grid_cells(floor lat/lon over 0.5)": len(cells)}
    d15["all"] = {"n_distinct_rainfall_values": int(rain.nunique()),
                  "n_distinct_0.5deg_grid_cells": len(set(zip(np.floor(df.Latitude / 0.5).astype(int), np.floor(df.Longitude / 0.5).astype(int))))}
    R["D15_lat_rain"] = d15

    # ---------------- D16 per-fold thresholding diagnostics
    oof = pd.read_csv(REV / "seeds" / "oof_seed42.csv")
    d16 = {"source": "results/revision/seeds/oof_seed42.csv (saved OOF, seed 42, terrain, no SMOTE); threshold = training-fold "
                     "prevalence (revision_extra.py part_thresholds, 574-580)"}
    for mname in ("XGBoost", "RandomForest"):
        p = oof[f"{mname}|terrain"].to_numpy(float)
        rows = []
        for f in range(5):
            m = lab == f
            thr = float(y[~m].mean())
            yp = p[m] >= thr
            rows.append({"fold": f + 1, "n_test": int(m.sum()), "test_prev": float(y[m].mean()), "mean_pred_prob": float(p[m].mean()),
                         "train_prev_threshold": thr, "frac_test_above_thr": float(yp.mean()),
                         "recall": float(yp[y[m] == 1].mean()) if (y[m] == 1).any() else None,
                         "specificity": float((~yp)[y[m] == 0].mean()) if (y[m] == 0).any() else None,
                         "frac_above_0.5": float((p[m] >= 0.5).mean()), "mean_prob_landslides": float(p[m][y[m] == 1].mean()),
                         "mean_prob_nonlandslides": float(p[m][y[m] == 0].mean())})
        d16[mname] = rows
    R["D16_threshold_by_fold"] = d16

    # ---------------- D17 quintiles
    ens = pd.read_csv(REV / "oof_seed_ensemble.csv")
    pr = ens["RandomForest|terrain+SMOTE"].to_numpy(float)
    br = np.quantile(pr, [0.2, 0.4, 0.6, 0.8])
    cls = np.digitize(pr, br)
    rows = []
    for c in range(5):
        m = cls == c
        rows.append({"class": c + 1, "n": int(m.sum()), "n_landslides": int(y[m].sum()), "prevalence": float(y[m].mean()),
                     "share_of_landslides": float(y[m].sum() / y.sum()), "share_of_samples": float(m.mean()),
                     "FR": float((y[m].sum() / y.sum()) / m.mean()), "prob_range": [float(pr[m].min()), float(pr[m].max())]})
    ntop = rows[-1]["n"]
    R["D17_quintiles"] = {"source": "oof_seed_ensemble.csv RandomForest|terrain+SMOTE (10-seed mean OOF); breaks=np.quantile(p,[.2,.4,.6,.8]), "
                                    "np.digitize (revision_extra.py:~895-897); map_model.json: RF chosen",
                          "breaks": [float(x) for x in br], "classes": rows,
                          "max_FR_top_bin_theoretical": float((min(ntop, 196) / 196) / (ntop / 496)),
                          "max_FR_note": "if all n_top points were landslides: (n_top/196)/(n_top/496)=496/196 = 1/prevalence"}

    # ---------------- D18 AOA
    w = di_weights()
    di, thr, dbar = compute_di(df, lab, w)
    out_base = di > thr[lab]
    shap_rf = pd.read_csv(REV / "shap_values_random_forest.csv", index_col=0)
    imp_col = shap_rf.abs().mean()
    names23 = list(shap_rf.columns)
    w23 = imp_col.to_numpy()
    w23n = w23 / w23.sum() * len(w23)

    def di_matrix(weights, standardise, per_fold_pre=True):
        di2 = np.full(len(df), np.nan)
        thr2 = np.zeros(N_SPLITS)
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
            from scipy.spatial.distance import pdist
            dbar_ = pdist(Ztr).mean()
            di2[te] = cKDTree(Ztr).query(Z[te], k=1)[0] / dbar_
            dtr = np.empty(len(tr))
            for f in set(range(N_SPLITS)) - {k}:
                mm = lab[tr] == f
                dtr[mm] = cKDTree(Ztr[~mm]).query(Ztr[mm], k=1)[0] / dbar_
            q1, q3 = np.percentile(dtr, [25, 75])
            thr2[k] = q3 + 1.5 * (q3 - q1)
        o = di2 > thr2[lab]
        return {"n_outside": int(o.sum()), "by_fold": [int(o[lab == k].sum()) for k in range(5)],
                "landslides_outside": int(o[y == 1].sum()), "non_landslides_outside": int(o[y == 0].sum())}
    R["D18_aoa"] = {
        "code": "revision_extra.py:835-870 (di_weights, compute_di), 885-905 (part_map)",
        "features": DI_FEATURES, "continuous_only": True,
        "excluded": "Aspect_sin/cos, is_flat, TWI/Soil missing flags, LULC and Soil_Texture one-hots (16 of 23 encoded columns)",
        "weights_source": "mean|SHAP| per factor of the RF terrain model from shap_importance_by_model.csv column 'RF' (TreeSHAP, "
                          "OUT-OF-FOLD SHAP values pooled over all 496 points, 10-seed mean) -> NOT Gini; NOT in-fold (the weights "
                          "are global: they used SHAP of every fold's test points, so each held-out fold contributes to the weights "
                          "that define its own DI) ; normalised to mean 1",
        "weights": dict(zip(DI_FEATURES, map(float, w))),
        "reference_cloud": "real samples only (other four spatial folds, no SMOTE); NaN (TWI) -> training-fold median; "
                           "standardised by training-fold mean/sd, then multiplied by weights",
        "threshold_rule": "Q3 + 1.5*IQR of training-point DI, where each training point's DI is its distance to the nearest training "
                          "point in a DIFFERENT fold (leave-fold-out within the 4 training folds), divided by mean pairwise distance; "
                          "per outer fold; CV scheme = the same 5 spatial folds",
        "reproduced_current_n_outside": int(out_base.sum()), "current_by_fold": [int(out_base[lab == k].sum()) for k in range(5)],
        "variant_all23_encoded": {
            "weights": "per encoded column mean|SHAP| from shap_values_random_forest.csv (seed-avg OOF), normalised to mean 1",
            "weights_by_column": dict(zip(names23, map(float, w23n))),
            "A_weighted_standardised_all23": di_matrix(w23n, True),
            "B_weighted_as_preprocessed_no_restandardise": di_matrix(w23n, False),
            "C_unweighted_standardised_all23": di_matrix(np.ones(23), True),
            "D_unweighted_as_preprocessed": di_matrix(np.ones(23), False)},
        "variant_continuous7_check_via_encoded_matrix": di_matrix(
            np.array([w[DI_FEATURES.index(c)] if c in DI_FEATURES else 0.0 for c in names23]), True)}

    # ---------------- D19 SHAP files
    d19 = {}
    for nm in ("random_forest", "cnn1d"):
        v = pd.read_csv(REV / f"shap_values_{nm}.csv", index_col=0)
        sj = json.load(open(REV / f"shap_seeds_{nm}.json"))
        d19[nm] = {"values_file": f"results/revision/shap_values_{nm}.csv", "features_file": f"results/revision/shap_features_{nm}.csv",
                   "shape": list(v.shape), "columns_are_encoded(ungrouped)": list(v.columns),
                   "seeds_averaged": sj["seeds"], "explainer": sj["explainer"],
                   "per_seed_values_saved": False,
                   "per_seed_files": "only per-seed factor shares (shap_share_by_seed_%s.csv); per-seed per-sample values not saved" % nm}
    R["D19_shap_files"] = d19

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rev2_diagnostics.json").write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("done")


def fold_contrasts():
    """Within-fold single-factor class contrasts (NDVI, built-area LULC 7, annual rainfall) and the landslide share
    among training-fold DBSCAN noise samples, for the five spatial folds. Writes
    results/revision/extra/rev2_fold_contrasts.json and .csv (used by Revision/scripts/lsnums.py)."""
    from sklearn.metrics import roc_auc_score
    df = load_df()
    y = df[TARGET].to_numpy(int)
    lab = spatial_fold_labels(df)
    rows = []
    lab = np.asarray(lab)
    for k, v in enumerate(sorted(set(lab.tolist())), start=1):
        te = np.where(lab == v)[0]
        tr = np.where(lab != v)[0]
        d = df.iloc[te]
        yt = y[te]
        ndvi, rain = d["NDVI"].to_numpy(float), d["Annual_Rainfall"].to_numpy(float)
        ok_n, ok_r = np.isfinite(ndvi), np.isfinite(rain)
        a = DBSCANClusterAssigner(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(df.iloc[tr][["Latitude", "Longitude"]])
        noise = a.labels_ == -1
        ytr = y[tr]
        rows.append({
            "fold": k, "n": int(len(te)), "n_pos": int(yt.sum()), "n_neg": int((1 - yt).sum()),
            "ndvi_median_pos": float(np.nanmedian(ndvi[yt == 1])), "ndvi_median_neg": float(np.nanmedian(ndvi[yt == 0])),
            "auc_low_ndvi_is_landslide": float(roc_auc_score(yt[ok_n], -ndvi[ok_n])),
            "built_pos": int(((d["LULC"].to_numpy() == 7) & (yt == 1)).sum()),
            "built_neg": int(((d["LULC"].to_numpy() == 7) & (yt == 0)).sum()),
            "p_rain_pos_gt_neg": float(roc_auc_score(yt[ok_r], rain[ok_r])),
            "train_noise_n": int(noise.sum()), "train_noise_landslide_share": float(ytr[noise].mean()),
            "train_landslides_noise_share": float((a.labels_[ytr == 1] == -1).mean())})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "rev2_fold_contrasts.csv", index=False)
    meta = {"annual_rainfall_min": float(df["Annual_Rainfall"].min()), "annual_rainfall_max": float(df["Annual_Rainfall"].max()),
            "annual_rainfall_median": float(df["Annual_Rainfall"].median()),
            "annual_rainfall_n_distinct": int(df["Annual_Rainfall"].nunique())}
    (OUT / "rev2_fold_contrasts_meta.json").write_text(json.dumps(meta, indent=1))
    print(out.round(3).to_string())
    print(meta)


if __name__ == "__main__":
    import sys
    if "--fold-contrasts" in sys.argv:
        fold_contrasts()
    else:
        main()
