#!/usr/bin/env python3
# ============================================================
# REVISION - reservoir-zone skill split + exposure summary
#   (answers Mitchell #2, Mohanty #3)
#
#   A. Reservoir split: inventory-capture and mean susceptibility for
#      upstream vs downstream of the dam, with the reservoir POOL flagged
#      separately (drawdown zone, not genuine flooding).
#   B. Exposure: cropland area (ha) and population in the high+very-high class,
#      present and (if available) future.
#
#   Inputs (GEE_LPP_CMIP6/):
#     stage2_outputs/present_susceptibility.tif (or futureII_spatial/present_susceptibility_recalibrated.tif)
#     stage2_outputs/futureII_spatial/susceptibility_breaks_recalibrated.json
#     reservoir_zone.tif, population.tif
#     LPP_paper5_predictor_stack_static.tif  (for the LULC band -> cropland)
#     LULC/LPP_flood_frequency.tif
#     optional future class: futureII_spatial/future_<ssp>_<hz>_prob.tif
#   Output: stage2_outputs/revision_reservoir_exposure/
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

import json, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd, rasterio
from rasterio.warp import reproject, Resampling
warnings.filterwarnings("ignore")

OUT = Path(f"{DRIVE}/stage2_outputs/revision_reservoir_exposure"); OUT.mkdir(parents=True, exist_ok=True)
PRESENT = f"{DRIVE}/stage2_outputs/futureII_spatial/present_susceptibility_recalibrated.tif"
PRESENT_FALLBACK = f"{DRIVE}/stage2_outputs/present_susceptibility.tif"
BREAKS = f"{DRIVE}/stage2_outputs/futureII_spatial/susceptibility_breaks_recalibrated.json"
ZONE = f"{DRIVE}/reservoir_zone.tif"; POP = f"{DRIVE}/population.tif"
# ROUND-2: prefer the flow-direction (dam catchment) zones if 22_dam_catchment_updown.py was run
if Path(f"{DRIVE}/reservoir_zone_hydro.tif").exists(): ZONE = f"{DRIVE}/reservoir_zone_hydro.tif"
STACK = f"{DRIVE}/LPP_paper5_predictor_stack_static.tif"
FREQ = f"{DRIVE}/LULC/LPP_flood_frequency.tif"
CROPLAND_CLASS = 40; PIXEL_HA = (30*30)/10000.0
# ROUND-2 FIX (Mohanty): population.tif is WorldPop 100 m COUNTS re-projected to 30 m
# with nearest-neighbour, so every 100 m count is copied into ~11.1 cells of 30 m.
# Summing those cells over-counts by (100/30)^2 = 11.1x. Scale each 30 m cell by
# (30/100)^2 = 0.09 so the total is preserved (sum-preserving aggregation).
POP_NATIVE_M = 100.0; GRID_M = 30.0
POP_SCALE = (GRID_M / POP_NATIVE_M) ** 2          # = 0.09


def ald(path, t, crs, H, W, band=1):
    with rasterio.open(path) as s:
        if s.shape==(H,W) and s.transform==t and s.crs==crs: return s.read(band).astype("float64")
        d=np.zeros((H,W),"float64")
        reproject(rasterio.band(s,band), d, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=t, dst_crs=crs, resampling=Resampling.nearest); return d


def main():
    ppath = PRESENT if Path(PRESENT).exists() else PRESENT_FALLBACK
    with rasterio.open(ppath) as src:
        S=src.read(1).astype("float64"); H,W=src.height,src.width; t,crs=src.transform,src.crs
    dom=np.isfinite(S)
    try: breaks=json.load(open(BREAKS))["breaks"]
    except Exception: breaks=[float(np.quantile(S[dom],q)) for q in (0.2,0.4,0.6,0.8)]
    cls=np.zeros(S.shape,"int16"); cls[dom]=(np.digitize(S[dom],breaks)+1)
    high = (cls>=4) & dom

    print(f"zones from: {Path(ZONE).name}")
    zone=ald(ZONE,t,crs,H,W).astype("int16")
    freq=ald(FREQ,t,crs,H,W); flood=dom&(freq>=1)
    lulc=ald(STACK,t,crs,H,W,band=11)        # band 11 = lulc in the stack
    try: pop=ald(POP,t,crs,H,W) * POP_SCALE
    except Exception: pop=np.zeros((H,W))
    tot_pop = float(np.nansum(pop[dom]))
    dom_km2 = int(dom.sum()) * PIXEL_HA / 100.0
    print(f"[sanity] domain population = {tot_pop:,.0f} people over {dom_km2:,.0f} km2 "
          f"-> {tot_pop/max(dom_km2,1e-9):,.0f} people/km2 (rural Korat ~100-200 is plausible)")

    # --- A. reservoir split ---
    znames={1:"upstream",2:"downstream",3:"reservoir_pool",4:"drawdown_ring"}
    rowsA=[]
    for z,nm in znames.items():
        m=dom&(zone==z)
        fm=flood&(zone==z)
        area_km2 = int(m.sum())*PIXEL_HA/100.0
        rowsA.append({"zone":nm,"area_ha":round(int(m.sum())*PIXEL_HA,1),
            "mean_S":round(float(np.nanmean(S[m])),4) if m.sum() else np.nan,
            "flood_px":int(fm.sum()),
            # ROUND-2 FIX (Mitchell): density = flood pixels per km2, not a raw count
            "flood_px_per_km2":round(int(fm.sum())/max(area_km2,1e-9),1),
            "flood_area_pct":round(100*int(fm.sum())/max(int(m.sum()),1),2),
            "pct_flood_captured_highclass":round(100*((flood&high&(zone==z)).sum())/max(fm.sum(),1),1)})
    dfA=pd.DataFrame(rowsA); dfA.to_csv(OUT/"reservoir_zone_split.csv",index=False)
    print("=== A. reservoir-zone split (pool = drawdown, not genuine flooding) ===")
    print(dfA.to_string(index=False))

    # --- B. exposure in high+very-high, present ---
    def expo(mask):
        crop=mask & (np.round(lulc)==CROPLAND_CLASS)
        return round(int(crop.sum())*PIXEL_HA,1), round(float(np.nansum(pop[mask])),0)
    tot_flood = max(int(flood.sum()), 1)
    dd = flood & (zone == 4)
    print(f"\n[drawdown] inventory pixels in the seasonal drawdown ring: {int(dd.sum())} "
          f"({100*int(dd.sum())/tot_flood:.2f}% of all inventory pixels)")
    crop_ha, pop_cnt = expo(high & (zone!=3))     # exclude pool
    rowsB=[{"scenario":"present","cropland_ha_in_high":crop_ha,"population_in_high":pop_cnt}]

    # future (if spatial future prob maps exist)
    for f in sorted(glob.glob(f"{DRIVE}/stage2_outputs/futureII_spatial/future_*_prob.tif")):
        tag=Path(f).stem.replace("future_","").replace("_prob","")
        Sf=ald(f,t,crs,H,W); hf=(np.digitize(np.where(dom,Sf,np.nan),breaks)+1>=4)&dom&(zone!=3)
        ch,pc=expo(hf); rowsB.append({"scenario":tag,"cropland_ha_in_high":ch,"population_in_high":pc})
    dfB=pd.DataFrame(rowsB); dfB.to_csv(OUT/"exposure_summary.csv",index=False)
    print("\n=== B. exposure in high+very-high susceptibility (pool excluded) ===")
    print(dfB.to_string(index=False))
    print(f"\nDone -> {OUT}/")


if __name__ == "__main__":
    main()
