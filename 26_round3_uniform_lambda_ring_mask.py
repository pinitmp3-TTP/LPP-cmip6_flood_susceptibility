#!/usr/bin/env python3
# ============================================================
# ROUND 3 checks (one run, three tables)
#
#   A. High-susceptibility area change (eta = 1) with
#        - lambda(x) by nearest-gauge zone (as in Table 5), and
#        - a UNIFORM lambda = mean of the 3 in-basin gauges (as in Table 4),
#      each on the full domain (self-check: must reproduce elasticity_sensitivity.csv)
#      and with the reservoir pool + drawdown ring REMOVED (reservoir level is set by
#      dam operation, not by R95p, so scaling its odds by lambda has no physical meaning).
#   A2. Elasticity sweep (eta = 0.5, 1, 2, 3) for lambda(x), pool + ring removed.
#   B. Cropland and population exposure (high + very-high) with pool + ring removed.
#   C. Cropland -> water transitions (2024 -> 2045): how many were water in an earlier
#      Dynamic World epoch (2016 or 2020) = class oscillation (seasonally inundated paddy),
#      and how many lie in the drawdown ring.
#
#   Inputs (GEE_LPP_CMIP6/):
#     stage2_outputs/futureII_spatial/present_susceptibility_recalibrated.tif
#     stage2_outputs/futureII_spatial/susceptibility_breaks_recalibrated.json
#     lambda_zones_station_id.tif, per_station_lambda.csv
#     reservoir_zone_hydro.tif (falls back to reservoir_zone.tif)   zones 3 pool, 4 drawdown
#     population.tif, LPP_paper5_predictor_stack_static.tif (band 11 = lulc)
#     LULC/DW_lulc_2016.tif, _2020.tif, _2024.tif, LULC/lulc_2045_camarkov.tif
#   Output: revision_outputs/round3_area_change_uniform_vs_spatial.csv
#           revision_outputs/round3_elasticity_ring_removed.csv
#           revision_outputs/round3_exposure_ring_excluded.csv
#           revision_outputs/round3_cropland_to_water_check.csv
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

SP = f"{DRIVE}/stage2_outputs/futureII_spatial"
PRESENT, BREAKS = f"{SP}/present_susceptibility_recalibrated.tif", f"{SP}/susceptibility_breaks_recalibrated.json"
ZONES, LAMCSV = f"{DRIVE}/lambda_zones_station_id.tif", f"{DRIVE}/per_station_lambda.csv"
RES = f"{DRIVE}/reservoir_zone_hydro.tif"
if not Path(RES).exists(): RES = f"{DRIVE}/reservoir_zone.tif"
POP, STACK = f"{DRIVE}/population.tif", f"{DRIVE}/LPP_paper5_predictor_stack_static.tif"
LU = {y: f"{DRIVE}/LULC/DW_lulc_{y}.tif" for y in (2016, 2020, 2024)}
LU[2045] = f"{DRIVE}/LULC/lulc_2045_camarkov.tif"
OUT = Path(f"{DRIVE}/revision_outputs"); OUT.mkdir(parents=True, exist_ok=True)

SSPS = ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]; HORIZONS = ["near_2036-2065", "far_2071-2100"]
INBASIN = [250751, 250771, 250781]
CROP, WATER, PIXEL_HA, POP_SCALE, EPS = 40, 80, 0.09, (30 / 100) ** 2, 1e-6


def aligned(path, t, crs, H, W, band=1):
    with rasterio.open(path) as s:
        if s.shape == (H, W) and s.transform == t and s.crs == crs:
            return s.read(band).astype("float64")
        d = np.zeros((H, W), "float64")
        reproject(rasterio.band(s, band), d, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=t, dst_crs=crs, resampling=Resampling.nearest)
        return d


def main():
    with rasterio.open(PRESENT) as s:
        S = s.read(1).astype("float64"); H, W = s.height, s.width; t, crs = s.transform, s.crs
    dom = np.isfinite(S)
    br = json.load(open(BREAKS))["breaks"]
    zones = aligned(ZONES, t, crs, H, W).astype("int64")
    res = aligned(RES, t, crs, H, W).astype("int64")
    keep = dom & ~np.isin(res, [3, 4])                       # pool + drawdown ring removed
    print(f"zones from {Path(RES).name}; ring+pool removed: {int((dom & ~keep).sum()) * PIXEL_HA / 100:.1f} km2 "
          f"({100 * (dom & ~keep).sum() / dom.sum():.2f}% of domain)")

    def high(p, m):
        c = np.zeros(p.shape, "int16"); v = m & np.isfinite(p)
        c[v] = np.digitize(p[v], br) + 1
        return (c >= 4) & m

    odds0 = np.clip(S, EPS, 1 - EPS) / (1 - np.clip(S, EPS, 1 - EPS))
    L = pd.read_csv(LAMCSV)
    lulc = np.round(aligned(STACK, t, crs, H, W, band=11))
    try: pop = aligned(POP, t, crs, H, W) * POP_SCALE
    except Exception: pop = np.zeros((H, W))

    def expo(mask):
        return round(int((mask & (lulc == CROP)).sum()) * PIXEL_HA, 1), round(float(np.nansum(pop[mask])), 0)

    rowsA, rowsB, rowsE = [], [], []
    hi0 = {k: high(S, m) for k, m in (("full", dom), ("ring_removed", keep))}
    c0, p0 = expo(hi0["ring_removed"])
    rowsB.append({"scenario": "present", "cropland_ha_in_high": c0, "population_in_high": p0})
    for ssp in SSPS:
        for hz in HORIZONS:
            sub = L[(L.ssp == ssp) & (L.horizon == hz)].set_index("station").lam_mean
            lam_x = np.full((H, W), np.nan)
            for st, v in sub.items(): lam_x[zones == int(st)] = v
            lam_u = float(sub.loc[INBASIN].mean())
            rec = {"ssp": ssp, "horizon": hz, "lambda_uniform_inbasin": round(lam_u, 3)}
            for lab, lam in (("spatial", lam_x), ("uniform", lam_u)):
                o = odds0 * lam; Sf = o / (1 + o)
                for mk, m in (("full", dom), ("ring_removed", keep)):
                    a0, a1 = hi0[mk].sum(), high(Sf, m).sum()
                    rec[f"dArea_pct_{lab}_{mk}"] = round((a1 / a0 - 1) * 100, 2)
                if lab == "spatial":
                    for eta in (0.5, 1.0, 2.0, 3.0):
                        oe = odds0 * np.power(lam_x, eta); Se = oe / (1 + oe)
                        rowsE.append({"ssp": ssp, "horizon": hz, "eta": eta, "pct_change":
                                      round((high(Se, keep).sum() / hi0["ring_removed"].sum() - 1) * 100, 2)})
                    c, p = expo(high(Sf, keep))
                    rowsB.append({"scenario": f"{ssp}_{hz}", "cropland_ha_in_high": c, "population_in_high": p})
            rowsA.append(rec)
    A = pd.DataFrame(rowsA); A.to_csv(OUT / "round3_area_change_uniform_vs_spatial.csv", index=False)
    B = pd.DataFrame(rowsB); B.to_csv(OUT / "round3_exposure_ring_excluded.csv", index=False)
    print("\n[A] high-class area change (%), eta = 1\n" + A.to_string(index=False))
    print("    self-check: dArea_pct_spatial_full must equal elasticity_sensitivity.csv (eta = 1)")
    E = pd.DataFrame(rowsE); E.to_csv(OUT / "round3_elasticity_ring_removed.csv", index=False)
    print("\n[A2] elasticity sweep, ring removed\n" + E.pivot_table(index=["ssp", "horizon"], columns="eta", values="pct_change").to_string())
    print("\n[B] exposure, pool + drawdown ring removed\n" + B.to_string(index=False))

    # ---------- C ----------
    lu = {y: np.round(aligned(p, t, crs, H, W)) for y, p in LU.items()}
    cw = dom & (lu[2024] == CROP) & (lu[2045] == WATER)
    n = int(cw.sum())
    was_w = cw & ((lu[2016] == WATER) | (lu[2020] == WATER))
    in_ring = cw & (res == 4)
    C = pd.DataFrame([{"cropland_to_water_px": n,
                       "pct_water_in_2016_or_2020": round(100 * was_w.sum() / max(n, 1), 1),
                       "pct_in_drawdown_ring": round(100 * in_ring.sum() / max(n, 1), 1),
                       "pct_either": round(100 * (was_w | in_ring).sum() / max(n, 1), 1)}])
    C.to_csv(OUT / "round3_cropland_to_water_check.csv", index=False)
    print("\n[C] cropland -> water (2024 -> 2045)\n" + C.to_string(index=False))
    print("    a high 'pct_water_in_2016_or_2020' = class oscillation (inundated paddy), not conversion")


if __name__ == "__main__":
    main()
