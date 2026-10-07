#!/usr/bin/env python3
# ============================================================
# ROUND 2 - two quick consistency checks
#
#   A. Area-weighted lambda (answers Xia R2: Table 4 basin-mean 0.98 vs Table 5
#      +2.6% for SSP2-4.5 far). Table 4 averages the 3 in-basin gauges equally;
#      the projection uses lambda(x) by zone. Here lambda is weighted by
#        - domain area of each zone, and
#        - area of the present HIGH + VERY-HIGH class in each zone.
#   B. Built-up growth sanity check (answers Mohanty R2: cropland->built-up
#      58,201 px by 2045 looks too high). Compares observed built-up gain
#      2016->2024 (Dynamic World, remapped) with the projected 2024->2045 gain,
#      both as km2 per year.
#
#   Inputs (GEE_LPP_CMIP6/):
#     lambda_zones_station_id.tif, per_station_lambda.csv
#     stage2_outputs/futureII_spatial/present_susceptibility_recalibrated.tif
#     stage2_outputs/futureII_spatial/susceptibility_breaks_recalibrated.json
#     LULC/DW_lulc_2016.tif, LULC/DW_lulc_2024.tif, LULC/lulc_2045_camarkov.tif
#   Output: revision_outputs/round2_area_weighted_lambda.csv
#           revision_outputs/round2_builtup_growth_check.csv
# ============================================================
try:
    import google.colab  # noqa: F401
    from google.colab import drive
    drive.flush_and_unmount(); drive.mount("/content/drive", force_remount=True)
    for _p in ("rasterio",):
        try: __import__(_p)
        except ImportError:
            import subprocess, sys; subprocess.run([sys.executable, "-m", "pip", "install", "-q", _p])
    DRIVE = "/content/drive/MyDrive/GEE_LPP_CMIP6"
except ImportError:
    DRIVE = "."

import json
from pathlib import Path
import numpy as np, pandas as pd, rasterio
from rasterio.warp import reproject, Resampling

ZONES = f"{DRIVE}/lambda_zones_station_id.tif"
LAMCSV = f"{DRIVE}/per_station_lambda.csv"
PRESENT = f"{DRIVE}/stage2_outputs/futureII_spatial/present_susceptibility_recalibrated.tif"
BREAKS = f"{DRIVE}/stage2_outputs/futureII_spatial/susceptibility_breaks_recalibrated.json"
L16, L24, L45 = (f"{DRIVE}/LULC/DW_lulc_2016.tif", f"{DRIVE}/LULC/DW_lulc_2024.tif",
                 f"{DRIVE}/LULC/lulc_2045_camarkov.tif")
OUT = Path(f"{DRIVE}/revision_outputs"); OUT.mkdir(parents=True, exist_ok=True)
INBASIN = [250751, 250771, 250781]
BUILTUP = 50
PX_KM2 = 30 * 30 / 1e6


def aligned(path, t, crs, H, W):
    with rasterio.open(path) as s:
        if s.shape == (H, W) and s.transform == t and s.crs == crs:
            return s.read(1).astype("float64")
        d = np.zeros((H, W), "float64")
        reproject(rasterio.band(s, 1), d, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=t, dst_crs=crs, resampling=Resampling.nearest)
        return d


def main():
    with rasterio.open(PRESENT) as s:
        S = s.read(1).astype("float64"); H, W = s.height, s.width; t, crs = s.transform, s.crs
    dom = np.isfinite(S)
    try:
        br = json.load(open(BREAKS))["breaks"]
    except Exception:
        br = [float(np.quantile(S[dom], q)) for q in (0.2, 0.4, 0.6, 0.8)]
    cls = np.zeros(S.shape, "int16"); cls[dom] = np.digitize(S[dom], br) + 1
    high = dom & (cls >= 4)
    zones = aligned(ZONES, t, crs, H, W).astype("int64")

    # ---------- A ----------
    L = pd.read_csv(LAMCSV)
    st = sorted(L.station.unique())
    a_dom = {s: int((dom & (zones == s)).sum()) for s in st}
    a_hi = {s: int((high & (zones == s)).sum()) for s in st}
    print("zone share of domain / of high class (%):")
    for s in st:
        print(f"  {s}: {100*a_dom[s]/max(sum(a_dom.values()),1):5.1f} / {100*a_hi[s]/max(sum(a_hi.values()),1):5.1f}")
    rows = []
    for (sc, hz), g in L.groupby(["ssp", "horizon"]):
        lam = g.set_index("station").lam_mean
        wdom = sum(a_dom[s] * lam[s] for s in st) / max(sum(a_dom.values()), 1)
        whi = (sum(a_hi[s] * lam[s] for s in st) / sum(a_hi.values())) if sum(a_hi.values()) else np.nan
        rows.append({"ssp": sc, "horizon": hz, "lam_inbasin3_equal": round(lam[INBASIN].mean(), 3),
                     "lam_domain_area_weighted": round(wdom, 3),
                     "lam_highclass_area_weighted": round(whi, 3)})
    A = pd.DataFrame(rows).sort_values(["horizon", "ssp"])
    A.to_csv(OUT / "round2_area_weighted_lambda.csv", index=False)
    print("\n[A] lambda: equal in-basin mean vs area-weighted\n" + A.to_string(index=False))

    # ---------- B ----------
    l16, l24, l45 = (aligned(p, t, crs, H, W) for p in (L16, L24, L45))
    basin = np.isfinite(l24) & (l24 > 0)            # whole-basin grid (not only the lowland domain)
    def gain(a, b, m): return int((m & (b == BUILTUP) & (a != BUILTUP)).sum()) * PX_KM2
    rec = []
    for lab, m in (("whole_basin", basin), ("lowland_domain", dom & basin)):
        obs, proj = gain(l16, l24, m), gain(l24, l45, m)
        rec.append({"area": lab, "builtup_2016_km2": round(int((m & (l16 == BUILTUP)).sum()) * PX_KM2, 1),
                    "builtup_2024_km2": round(int((m & (l24 == BUILTUP)).sum()) * PX_KM2, 1),
                    "observed_gain_2016_2024_km2": round(obs, 1), "observed_km2_per_yr": round(obs / 8, 2),
                    "projected_gain_2024_2045_km2": round(proj, 1), "projected_km2_per_yr": round(proj / 21, 2),
                    "ratio_projected_over_observed_rate": round((proj / 21) / max(obs / 8, 1e-9), 2)})
    B = pd.DataFrame(rec); B.to_csv(OUT / "round2_builtup_growth_check.csv", index=False)
    print("\n[B] built-up growth: observed vs projected\n" + B.to_string(index=False))
    print("\nREAD: ratio ~1 -> projected rate matches observed trend (defensible);"
          "\n      ratio >> 1 -> CA-Markov over-projects built-up; say so and bound Table 6 accordingly.")


if __name__ == "__main__":
    main()
