#!/usr/bin/env python3
# ============================================================
# Paper #5 - compute R10mm / R95p from EQM-corrected daily rainfall
#
#   Input : GEE_LPP_CMIP6/03_QM_Corrected/  (CSV tree from pre-selection team)
#       03_QM_Corrected/
#         Historical/pr/<MODEL>/<MODEL>_Historical_1985-2014_<station>.csv
#         Future_NearTerm/<ssp>/<MODEL>/..._2036-2065_<station>.csv
#         Future_FarTerm/<ssp>/<MODEL>/..._2071-2100_<station>.csv
#       each CSV: Date, pr_gcm, pr_qm_corrected  (mm/day)
#
#   Output: GEE_LPP_CMIP6/indices_output/
#       indices_annual_long.csv     (one row per model x scenario x station x year)
#       indices_period_summary.csv  (period mean/sd per model x scenario x station)
#
#   ETCCDI definitions:
#     R10mm = annual count of days with pr >= 10 mm
#     R95p  = annual total pr on days above the WET-DAY 95th percentile of the
#             HISTORICAL baseline (threshold fixed on 1985-2014, applied to all)
#
#   ROBUSTNESS: the year is reconstructed from the filename's year range, NOT
#   trusted from the date column -- 2-digit years (e.g. "01-01-71") otherwise
#   silently corrupt the far-future period (2071-2100 -> 1971-2000).
#
# Paste into ONE Colab cell and Run.
# ============================================================

# --- Colab bootstrap ---
try:
    import google.colab  # noqa: F401
    if not __import__("os").path.exists("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
except ImportError:
    pass

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
ROOT    = "/content/drive/MyDrive/GEE_LPP_CMIP6/03_QM_Corrected"   # <<< EDIT if elsewhere
OUT_DIR = Path("/content/drive/MyDrive/GEE_LPP_CMIP6/indices_output")

WET_DAY   = 1.0     # mm, wet-day threshold for the percentile base
R10_LIM   = 10.0    # mm, heavy-rain-day threshold
PCTL      = 95      # R95p percentile
SELECTED  = {"MIROC6", "HadGEM3-GC31-LL", "ACCESS-CM2", "KACE-1-0-G", "MRI-ESM2-0"}

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# metadata from path + filename
# ------------------------------------------------------------
def parse_meta(fpath):
    rel = os.path.relpath(fpath, ROOT)
    parts = rel.split(os.sep)
    term = parts[0]                                   # Historical / Future_NearTerm / Future_FarTerm
    scenario = "historical" if term.lower().startswith("hist") else parts[1]
    model = Path(fpath).parent.name                   # immediate parent folder = model
    name = Path(fpath).name
    yr = re.search(r"(\d{4})\s*-\s*(\d{4})", name)
    yr_range = (int(yr.group(1)), int(yr.group(2))) if yr else (None, None)
    st = re.search(r"_(\d+)\.csv$", name, re.IGNORECASE)
    station = st.group(1) if st else "NA"
    return dict(term=term, scenario=scenario, model=model, station=station,
                yr_range=yr_range, path=fpath)


def reconstruct_year(yy, y0, y1):
    """Map a (possibly 2-digit) year into the filename's true range. Unique for <100-yr spans."""
    if y0 is None:
        return yy
    yy2 = int(yy) % 100
    for cent in (1900, 2000, 2100):
        cand = cent + yy2
        if y0 <= cand <= y1:
            return cand
    return None                                       # out of declared range -> dropped


def read_series(fpath, y0, y1):
    """Return (years, pr) with years reconstructed from the filename range."""
    df = pd.read_csv(fpath)
    datecol = df.columns[0]
    cand = [c for c in df.columns if "corrected" in c.lower()
            or ("qm" in c.lower() and "gcm" not in c.lower())]
    prcol = cand[0] if cand else df.columns[2]
    dates = pd.to_datetime(df[datecol], errors="coerce", dayfirst=True)
    if dates.isna().mean() > 0.5:
        dates = pd.to_datetime(df[datecol], errors="coerce")
    raw_year = dates.dt.year.values
    years = np.array([reconstruct_year(y, y0, y1) if np.isfinite(y) else None
                      for y in raw_year], dtype="object")
    pr = pd.to_numeric(df[prcol], errors="coerce").values
    return years, pr


def annual_indices(years, pr, thr):
    t = pd.DataFrame({"year": years, "pr": pr}).dropna()
    out = []
    for y, g in t.groupby("year"):
        s = g["pr"].values
        r10 = int(np.sum(s >= R10_LIM))
        r95 = float(np.sum(s[s > thr])) if np.isfinite(thr) else np.nan
        out.append((int(y), r10, r95))
    return out


def main():
    files = [str(p) for p in Path(ROOT).rglob("*.csv")]
    print(f"Found {len(files)} CSV files under {ROOT}")
    if not files:
        print("No CSVs found - check ROOT path."); return
    meta = [parse_meta(f) for f in files]

    # ---- PASS A: baseline wet-day 95th percentile per (model, station) ----
    thresholds = {}
    for m in meta:
        if not m["term"].lower().startswith("hist"):
            continue
        y0, y1 = m["yr_range"]
        _, pr = read_series(m["path"], y0, y1)
        wet = pr[(pr >= WET_DAY) & np.isfinite(pr)]
        thresholds[(m["model"], m["station"])] = float(np.percentile(wet, PCTL)) if wet.size else np.nan
    print(f"Baseline R95p thresholds computed for {len(thresholds)} (model, station) pairs")

    # ---- PASS B: annual indices for every file ----
    rows = []
    for m in meta:
        y0, y1 = m["yr_range"]
        years, pr = read_series(m["path"], y0, y1)
        thr = thresholds.get((m["model"], m["station"]), np.nan)
        for y, r10, r95 in annual_indices(years, pr, thr):
            rows.append({"model": m["model"], "scenario": m["scenario"], "term": m["term"],
                         "station": m["station"], "year": y, "R10mm": r10, "R95p": r95,
                         "selected": m["model"] in SELECTED})

    annual = pd.DataFrame(rows).sort_values(["model", "scenario", "station", "year"])
    annual.to_csv(OUT_DIR / "indices_annual_long.csv", index=False)

    # ---- period summary (what lambda uses) ----
    summ = (annual.groupby(["model", "scenario", "term", "station", "selected"])
            .agg(n_years=("year", "nunique"),
                 yr_min=("year", "min"), yr_max=("year", "max"),
                 R10mm_mean=("R10mm", "mean"), R10mm_std=("R10mm", "std"),
                 R95p_mean=("R95p", "mean"), R95p_std=("R95p", "std"))
            .round(2).reset_index())
    summ.to_csv(OUT_DIR / "indices_period_summary.csv", index=False)

    # ---- coverage + year-range check ----
    print("\nYear ranges actually computed (should match folder terms):")
    for term, g in annual.groupby("term"):
        print(f"  {term:16s}: {int(g['year'].min())}-{int(g['year'].max())}")

    print("\nCoverage (selected 5 models only):")
    sel = summ[summ["selected"]]
    for (term, scen), g in sel.groupby(["term", "scenario"]):
        print(f"  {term:16s} {scen:10s}: {g['model'].nunique()}/5 models, "
              f"{g['station'].nunique()} stations")
    missing = SELECTED - set(sel["model"].unique())
    if missing:
        print(f"  !! selected models with NO data: {missing}")

    print(f"\nDone. {len(annual)} annual rows -> {OUT_DIR}/")
    print("  indices_annual_long.csv     (interannual spread / uncertainty)")
    print("  indices_period_summary.csv  (period means -> lambda in Stage II)")


if __name__ == "__main__":
    main()
