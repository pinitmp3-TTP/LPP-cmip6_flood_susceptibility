#!/usr/bin/env python3
# ============================================================
# REVISION 2b + 1kho(A,C) - spatially-varying lambda future projection
#
#   Fixes:
#     - 2b   : lambda varies by nearest-station zone -> real spatial change map
#              and a REAL robustness map (GCM agreement varies by zone)
#     - 1kho-C: prevalence recalibration of the balanced-sample odds before scaling
#              (answers Wright #1 "odds are not a real probability")
#     - 1kho-A: elasticity sweep odds' = lambda(x)^eta * odds  (eta in ETAS)
#              (answers Wright #1 "elasticity=1 is unjustified")
#     - reframe: output is a RELATIVE susceptibility, scaled by a RELATIVE
#              intensification factor; within-basin forcing variation is modest.
#
#   Inputs (GEE_LPP_CMIP6/):
#     stage2_outputs/present_susceptibility.tif
#     stage2_outputs/susceptibility_breaks.json
#     lambda_zones_station_id.tif            (from 09_GEE_station_zones.js)
#     per_station_lambda.csv                 (station x ssp x horizon lambda)
#   Optional: LULC/LPP_flood_frequency.tif   (to compute true prevalence tau)
#   Output: stage2_outputs/futureII_spatial/
# ============================================================
try:
    import google.colab  # noqa: F401
    from google.colab import drive
    drive.flush_and_unmount(); drive.mount("/content/drive", force_remount=True)
    for _p in ("rasterio",):
        try: __import__(_p)
        except ImportError:
            import subprocess, sys; subprocess.run([sys.executable,"-m","pip","install","-q",_p])
    DRIVE = "/content/drive/MyDrive/GEE_LPP_CMIP6"
except ImportError:
    DRIVE = "."

import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, rasterio
from rasterio.warp import reproject, Resampling
warnings.filterwarnings("ignore")

PRESENT = f"{DRIVE}/stage2_outputs/present_susceptibility.tif"
BREAKS  = f"{DRIVE}/stage2_outputs/susceptibility_breaks.json"
ZONES   = f"{DRIVE}/lambda_zones_station_id.tif"
LAMCSV  = f"{DRIVE}/per_station_lambda.csv"
FREQ    = f"{DRIVE}/LULC/LPP_flood_frequency.tif"   # optional, for tau
OUT     = Path(f"{DRIVE}/stage2_outputs/futureII_spatial"); OUT.mkdir(parents=True, exist_ok=True)

SSPS = ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]
HORIZONS = ["near_2036-2065", "far_2071-2100"]
ETAS = [0.5, 1.0, 2.0, 3.0]          # elasticity sweep
TAU_DEFAULT = 0.025                   # fallback true flood prevalence if FREQ missing
EPS = 1e-6
PIXEL_HA = (30*30)/10000.0


def aligned(path, t, crs, H, W, dtype="float64", rs=Resampling.nearest):
    with rasterio.open(path) as s:
        if s.shape == (H, W) and s.transform == t and s.crs == crs:
            return s.read(1).astype(dtype)
        d = np.zeros((H, W), dtype)
        reproject(rasterio.band(s, 1), d, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=t, dst_crs=crs, resampling=rs)
        return d


def main():
    with rasterio.open(PRESENT) as src:
        S = src.read(1).astype("float64"); prof = src.profile
        H, W = src.height, src.width; t, crs = src.transform, src.crs
    dom = np.isfinite(S)
    zones = aligned(ZONES, t, crs, H, W).astype("int64")
    lam = pd.read_csv(LAMCSV)

    # --- true prevalence tau for recalibration (1kho-C) ---
    try:
        freq = aligned(FREQ, t, crs, H, W)
        tau = float(((freq >= 1) & dom).sum() / dom.sum())
        print(f"true flood prevalence tau = {tau:.4f} (from inventory)")
    except Exception:
        tau = TAU_DEFAULT
        print(f"inventory not found; using tau = {tau} (edit TAU_DEFAULT)")

    # prevalence-corrected odds: balanced sample ybar=0.5 -> factor tau/(1-tau)
    s = np.clip(S, EPS, 1 - EPS)
    odds0 = (s / (1 - s)) * (tau / (1 - tau))
    S_cal = odds0 / (1 + odds0)                       # recalibrated present susceptibility

    # FIXED breakpoints = equal-area quantiles of the RECALIBRATED present surface
    # (recalibration is monotonic, so quantile classes are rank-stable; reused for all futures)
    vals = S_cal[dom]
    breaks = [float(np.quantile(vals, q)) for q in (0.2, 0.4, 0.6, 0.8)]
    json.dump({"method": "quantile_recalibrated", "tau": tau, "breaks": breaks},
              open(OUT / "susceptibility_breaks_recalibrated.json", "w"), indent=2)

    def classify(p):
        c = np.zeros(p.shape, "int16"); v = np.isfinite(p)
        c[v] = (np.digitize(p[v], breaks) + 1).astype("int16"); return c
    present_cls = classify(np.where(dom, S_cal, np.nan))
    present_high = int(((present_cls >= 4) & dom).sum()) * PIXEL_HA

    def write(a, name, dtype="float32", nod=np.nan):
        pr = prof.copy(); pr.update(count=1, dtype=dtype, nodata=nod, compress="lzw")
        with rasterio.open(OUT / name, "w", **pr) as d:
            d.write(np.where(dom, a, nod).astype(dtype) if dtype == "float32" else a, 1)
    write(S_cal, "present_susceptibility_recalibrated.tif")

    rows, elast = [], []
    for ssp in SSPS:
        for hz in HORIZONS:
            sub = lam[(lam.ssp == ssp) & (lam.horizon == hz)].set_index("station")
            lam_x = np.full((H, W), np.nan)             # lambda(x) from zone
            nup_x = np.full((H, W), np.nan)             # GCM agreement by zone -> spatial
            for st, r in sub.iterrows():
                m = zones == int(st)
                lam_x[m] = r.lam_mean; nup_x[m] = r.n_gcm_up
            lam_x = np.where(dom, lam_x, np.nan)

            for eta in ETAS:
                o = (np.clip(S_cal, EPS, 1-EPS) / (1 - np.clip(S_cal, EPS, 1-EPS))) * np.power(lam_x, eta)
                Sf = o / (1 + o)
                hi = int(((classify(np.where(dom, Sf, np.nan)) >= 4) & dom).sum()) * PIXEL_HA
                elast.append({"ssp": ssp, "horizon": hz, "eta": eta,
                              "high_area_ha": round(hi, 1),
                              "pct_change": round((hi/present_high - 1)*100, 2)})
                if eta == 1.0:                           # headline eta=1 maps
                    Sf1 = Sf; change = Sf - S_cal
            write(Sf1,    f"future_{ssp}_{hz}_prob.tif")
            write(change, f"future_{ssp}_{hz}_change.tif")
            write(np.where(dom, nup_x, np.nan), f"future_{ssp}_{hz}_agreement.tif")  # SPATIAL now
            robust = int(((nup_x >= 4) & dom).sum()) / dom.sum() * 100
            rows.append({"ssp": ssp, "horizon": hz,
                         "lambda_min": round(np.nanmin(lam_x), 3),
                         "lambda_max": round(np.nanmax(lam_x), 3),
                         "pct_domain_robust_ge4of5": round(robust, 1)})
    pd.DataFrame(rows).to_csv(OUT / "spatial_lambda_summary.csv", index=False)
    pd.DataFrame(elast).to_csv(OUT / "elasticity_sensitivity.csv", index=False)
    print("\n=== spatial lambda summary (lambda now VARIES across basin) ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n=== elasticity sensitivity (eta sweep; eta=1 is the headline) ===")
    print(pd.DataFrame(elast).pivot_table(index=["ssp","horizon"], columns="eta",
          values="pct_change").round(2).to_string())
    print(f"\nDone -> {OUT}/  (present high-susc baseline {present_high:.0f} ha)")


if __name__ == "__main__":
    main()
