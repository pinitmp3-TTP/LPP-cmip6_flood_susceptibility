#!/usr/bin/env python3
# ============================================================
# ROUND 2 - Is the spatial variation of lambda(x) a climate signal or an
#           EQM artefact?  (answers Xia R2 #1, options (a) and (b))
#
#   The daily CSVs already hold BOTH series:
#       Date, pr_gcm (raw GCM), pr_qm_corrected (after EQM)
#   so no new GEE run is needed.
#
#   (a) Which GCM cell is each gauge in?  Tested EMPIRICALLY: if two gauges get
#       an IDENTICAL raw pr_gcm series from the same model, they were read from
#       the same GCM grid cell. -> gauge_gcm_cell_groups.csv
#   (b) lambda from RAW GCM vs lambda after EQM, per gauge (same R95p definition
#       as the main pipeline: wet-day >=1 mm, 95th pct of the 1985-2014 baseline,
#       each series using ITS OWN baseline threshold).
#       If raw lambda is ~equal across gauges but EQM lambda varies -> the spatial
#       pattern of lambda(x) is produced by bias correction, not by the GCM.
#       -> lambda_raw_vs_eqm_by_station.csv, lambda_spatial_spread_raw_vs_eqm.csv
#
#   Input : GEE_LPP_CMIP6/03_QM_Corrected/   (same tree as the indices step)
#   Output: GEE_LPP_CMIP6/revision_outputs/round2_lambda_diag/
# ============================================================
try:
    import google.colab  # noqa: F401
    from google.colab import drive
    drive.flush_and_unmount(); drive.mount("/content/drive", force_remount=True)
    DRIVE = "/content/drive/MyDrive/GEE_LPP_CMIP6"
except ImportError:
    DRIVE = "."

import os, re, warnings
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = f"{DRIVE}/03_QM_Corrected"
OUT = Path(f"{DRIVE}/revision_outputs/round2_lambda_diag"); OUT.mkdir(parents=True, exist_ok=True)
GAUGES = ["250751", "250771", "250781", "250541", "250651", "250791"]   # the 6 lambda(x) gauges
SELECTED = {"MIROC6", "HadGEM3-GC31-LL", "ACCESS-CM2", "KACE-1-0-G", "MRI-ESM2-0"}
WET_DAY, PCTL = 1.0, 95
WINDOWS = {"near_2036-2065": (2036, 2065), "far_2071-2100": (2071, 2100)}   # same as the main pipeline
# NOTE: the folder terms are NOT these windows (Future_NearTerm = 2015-2054, Future_FarTerm = 2055-2100),
# so all future files of a model/scenario/gauge are concatenated and then cut by year.

# nominal atmospheric grid (deg lon x deg lat) - for the text; (a) is tested empirically below
GRID = {"MIROC6": (1.406, 1.401), "HadGEM3-GC31-LL": (1.875, 1.25), "ACCESS-CM2": (1.875, 1.25),
        "KACE-1-0-G": (1.875, 1.25), "MRI-ESM2-0": (1.125, 1.121)}
COORD = {"250751": (101.683889, 14.492778), "250771": (101.725, 14.355833),
         "250781": (101.651111, 14.476389), "250541": (101.566111, 14.861944),
         "250651": (101.478056, 14.592778), "250791": (102.001111, 14.942222)}


def meta(f):
    parts = os.path.relpath(f, ROOT).split(os.sep)
    term = parts[0]
    scen = "historical" if term.lower().startswith("hist") else parts[1]
    st = re.search(r"_(\d+)\.csv$", Path(f).name)
    return term, scen, Path(f).parent.name, (st.group(1) if st else "NA")


def read(f):
    d = pd.read_csv(f)
    dates = pd.to_datetime(d.iloc[:, 0], errors="coerce")
    years = dates.dt.year.values.astype("float64")
    yr = re.search(r"(\d{4})\s*-\s*(\d{4})", Path(f).name)
    if yr:                                   # map 2-digit / out-of-range years into the file's range
        y0, y1 = int(yr.group(1)), int(yr.group(2))
        def fix(y):
            if not np.isfinite(y): return np.nan
            if y0 <= y <= y1: return y
            for c in (1900, 2000, 2100):
                if y0 <= c + int(y) % 100 <= y1: return c + int(y) % 100
            return np.nan
        years = np.array([fix(y) for y in years])
    raw = pd.to_numeric(d.get("pr_gcm", d.iloc[:, 1]), errors="coerce").values
    qm = pd.to_numeric(d.get("pr_qm_corrected", d.iloc[:, 2]), errors="coerce").values
    return years, raw, qm


def r95p_mean(years, pr, thr):
    t = pd.DataFrame({"y": years, "p": pr}).dropna()
    return t.groupby("y")["p"].apply(lambda s: s[s > thr].sum()).mean()


def main():
    files = [str(p) for p in Path(ROOT).rglob("*.csv")]
    recs = [(*meta(f), f) for f in files]
    recs = [r for r in recs if r[3] in GAUGES]
    print(f"{len(recs)} files for the {len(GAUGES)} lambda(x) gauges")

    # ---------- (a) empirical GCM-cell grouping from identical raw series ----------
    rows_a = []
    hist = {(m, st): f for term, sc, m, st, f in recs if term.lower().startswith("hist")}
    for m in sorted({k[0] for k in hist}):
        series = {st: read(f)[1] for (mm, st), f in hist.items() if mm == m}
        sts = sorted(series); group, gid = {}, 0
        for s in sts:
            if s in group: continue
            gid += 1; group[s] = gid
            for o in sts:
                if o not in group and len(series[o]) == len(series[s]) and \
                        np.allclose(series[o], series[s], equal_nan=True, atol=1e-6):
                    group[o] = gid
        lon_r, lat_r = GRID.get(m, (np.nan, np.nan))
        # if raw series were INTERPOLATED to gauges they will not be identical; the
        # mean pairwise daily correlation then shows how similar the raw forcing is
        cors = [np.corrcoef(series[a], series[b])[0, 1] for a, b in combinations(sts, 2)
                if len(series[a]) == len(series[b])]
        for s in sts:
            rows_a.append({"model": m, "selected": m in SELECTED, "station": s,
                           "gcm_cell_group": group[s], "n_cells_spanned": gid,
                           "mean_pairwise_raw_corr": round(float(np.nanmean(cors)), 3) if cors else np.nan,
                           "grid_lon_deg": lon_r, "grid_lat_deg": lat_r})
    A = pd.DataFrame(rows_a); A.to_csv(OUT / "gauge_gcm_cell_groups.csv", index=False)
    lons = [c[0] for c in COORD.values()]; lats = [c[1] for c in COORD.values()]
    print(f"\n(a) gauge network span: {max(lons)-min(lons):.2f} deg lon x {max(lats)-min(lats):.2f} deg lat")
    print(A.groupby(["model", "selected"])[["n_cells_spanned", "mean_pairwise_raw_corr"]].first().to_string())
    print("    n_cells_spanned = number of DISTINCT raw series among the gauges (= GCM cells used)")
    print(f"    if n_cells_spanned = {len(GAUGES)} but corr > 0.95, raw was interpolated from the same 1-2 cells")

    # ---------- (b) lambda raw vs EQM per gauge ----------
    thr = {}
    for (m, st), f in hist.items():
        y, raw, qm = read(f)
        thr[(m, st)] = {k: float(np.percentile(v[(v >= WET_DAY) & np.isfinite(v)], PCTL))
                        for k, v in (("raw", raw), ("eqm", qm))}
    base = {}
    for (m, st), f in hist.items():
        y, raw, qm = read(f)
        base[(m, st)] = {"raw": r95p_mean(y, raw, thr[(m, st)]["raw"]),
                         "eqm": r95p_mean(y, qm, thr[(m, st)]["eqm"])}
    fut = {}
    for term, sc, m, st, f in recs:
        if term.lower().startswith("hist") or (m, st) not in base: continue
        y, raw, qm = read(f)
        k = (m, sc, st); a = fut.setdefault(k, [[], [], []])
        a[0].append(y); a[1].append(raw); a[2].append(qm)
    rows_b = []
    for (m, sc, st), (ys, rs, qs) in fut.items():
        y, raw, qm = np.concatenate(ys), np.concatenate(rs), np.concatenate(qs)
        # drop exact duplicate days if two files overlap
        df = pd.DataFrame({"y": y, "r": raw, "q": qm}).dropna(subset=["y"])
        for hz, (w0, w1) in WINDOWS.items():
            g = df[df.y.between(w0, w1)]
            if g.y.nunique() < 25: continue                  # incomplete window -> skip
            lr = r95p_mean(g.y.values, g.r.values, thr[(m, st)]["raw"]) / base[(m, st)]["raw"]
            le = r95p_mean(g.y.values, g.q.values, thr[(m, st)]["eqm"]) / base[(m, st)]["eqm"]
            rows_b.append({"model": m, "selected": m in SELECTED, "ssp": sc, "horizon": hz, "station": st,
                           "n_years": int(g.y.nunique()), "lam_raw": lr, "lam_eqm": le, "eqm_minus_raw": le - lr})
    B = pd.DataFrame(rows_b); B.to_csv(OUT / "lambda_raw_vs_eqm_per_model.csv", index=False)

    S = (B[B.selected].groupby(["ssp", "horizon", "station"])[["lam_raw", "lam_eqm"]]
         .mean().round(3).reset_index())
    S.to_csv(OUT / "lambda_raw_vs_eqm_by_station.csv", index=False)
    chk = S[(S.ssp == "ssp5_8_5") & (S.horizon == "far_2071-2100") & S.station.isin(["250751", "250771", "250781"])]
    print(f"\n[self-check] in-basin EQM lambda, far SSP5-8.5 = {chk.lam_eqm.mean():.3f} "
          f"(main pipeline / Table 4 = 1.116; should match to ~0.01)")
    print("[self-check] years per window:", sorted(B.n_years.unique().tolist()))

    sp = []
    for (sc, hz), g in S.groupby(["ssp", "horizon"]):
        sp.append({"ssp": sc, "horizon": hz,
                   "raw_min": g.lam_raw.min(), "raw_max": g.lam_raw.max(),
                   "raw_range": round(g.lam_raw.max() - g.lam_raw.min(), 3),
                   "eqm_min": g.lam_eqm.min(), "eqm_max": g.lam_eqm.max(),
                   "eqm_range": round(g.lam_eqm.max() - g.lam_eqm.min(), 3),
                   "share_of_eqm_range_already_in_raw":
                       round((g.lam_raw.max() - g.lam_raw.min()) /
                             max(g.lam_eqm.max() - g.lam_eqm.min(), 1e-9), 2),
                   "corr_raw_eqm_across_gauges": round(g[["lam_raw", "lam_eqm"]].corr().iloc[0, 1], 2)
                   if len(g) > 2 else np.nan})
    SP = pd.DataFrame(sp).sort_values(["horizon", "ssp"])
    SP.to_csv(OUT / "lambda_spatial_spread_raw_vs_eqm.csv", index=False)
    print("\n(b) between-gauge spread of the screened-5 ensemble-mean lambda")
    print(SP.to_string(index=False))
    print("\nREAD: share_of_eqm_range_already_in_raw ~1 and corr > 0.7 -> spatial pattern exists in the raw GCM")
    print("      share << 1 (e.g. < 0.3)                            -> pattern is mostly created by EQM")
    print(f"\nDone -> {OUT}/")


if __name__ == "__main__":
    main()
