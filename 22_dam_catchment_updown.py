#!/usr/bin/env python3
# ============================================================
# ROUND 2 - upstream / downstream of the dam by FLOW DIRECTION (dam catchment)
#
#   Why: the first split used "elevation >= dam elevation -> upstream". Reservoir
#   margins lie BELOW the crest elevation, so ~4,700 inventory pixels around the
#   reservoir were labelled "downstream". Here upstream = the dam's hydrological
#   catchment, delineated from MERIT Hydro D8 flow directions.
#
#   Inputs (GEE_LPP_CMIP6/):
#     merit_flowdir.tif   (from 21_GEE_reservoir_zones_population_flowdir.js, native grid)
#     reservoir_zone.tif  (zones 3 = pool and 4 = drawdown are kept as they are)
#   Output:
#     reservoir_zone_hydro.tif  (1 = dam catchment, 2 = downstream, 3 = pool, 4 = drawdown)
#     revision_outputs/round2_dam_catchment.csv
#   Then re-run 23_reservoir_exposure_summary.py (it uses the _hydro file if present).
# ============================================================
try:
    import google.colab  # noqa: F401
    from google.colab import drive
    drive.flush_and_unmount(); drive.mount("/content/drive", force_remount=True)
    import subprocess, sys
    for _p in ("rasterio", "pysheds"):
        try: __import__(_p)
        except ImportError: subprocess.run([sys.executable, "-m", "pip", "install", "-q", _p])
    DRIVE = "/content/drive/MyDrive/GEE_LPP_CMIP6"
except ImportError:
    DRIVE = "."

import warnings
from pathlib import Path
import numpy as np, pandas as pd, rasterio
from rasterio.warp import reproject, Resampling
if not hasattr(np, "in1d"):                     # numpy >= 2.4 removed in1d; pysheds 0.5 still calls it
    np.in1d = lambda a, b, **k: np.isin(np.ravel(a), b, **k)
from pysheds.grid import Grid
from pysheds.sview import Raster
warnings.filterwarnings("ignore")

FDIR = f"{DRIVE}/merit_flowdir.tif"
ZONE = f"{DRIVE}/reservoir_zone.tif"
ZONE_OUT = f"{DRIVE}/reservoir_zone_hydro.tif"
OUT = Path(f"{DRIVE}/revision_outputs"); OUT.mkdir(parents=True, exist_ok=True)
DAM = (101.781, 14.505)                    # lon, lat of the Lam Phra Phloeng dam
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)     # ESRI D8 = MERIT Hydro encoding (N NE E SE S SW W NW)
SNAP_ACC_CELLS = 2000                      # snap the dam to a channel draining >= ~17 km2
SNAP_MAX_M = 1500                          # ... within this distance, else stop


def main():
    grid = Grid.from_raster(FDIR)
    raw = grid.read_raster(FDIR)
    # MERIT codes 0 (outlet), 247 (inland depression) and 255 (nodata) -> 0 = no flow
    clean = np.where(np.isin(np.asarray(raw), DIRMAP), np.asarray(raw), 0).astype("int64")
    fdir = Raster(clean, viewfinder=raw.viewfinder)
    acc = grid.accumulation(fdir, dirmap=DIRMAP, nodata_out=0)

    # snap the dam point to the nearest high-accumulation cell
    a = np.asarray(acc); rows, cols = np.where(a >= SNAP_ACC_CELLS)
    if rows.size == 0: raise SystemExit("No channel cells found - lower SNAP_ACC_CELLS.")
    aff = grid.affine
    lon = aff.c + (cols + 0.5) * aff.a; lat = aff.f + (rows + 0.5) * aff.e
    dist_m = np.hypot((lon - DAM[0]) * 111320 * np.cos(np.deg2rad(DAM[1])), (lat - DAM[1]) * 110540)
    k = int(np.argmin(dist_m))
    if dist_m[k] > SNAP_MAX_M:
        raise SystemExit(f"Nearest channel is {dist_m[k]:.0f} m from the dam - check DAM coordinates.")
    # index mode (x = column, y = row); pysheds' coordinate mode is unreliable here
    catch = grid.catchment(x=int(cols[k]), y=int(rows[k]), fdir=fdir, dirmap=DIRMAP, xytype="index")
    cmask = np.asarray(catch).astype("uint8")
    if cmask.sum() < 0.9 * a[rows[k], cols[k]]:
        raise SystemExit(f"Catchment ({int(cmask.sum())} cells) smaller than accumulation "
                         f"({int(a[rows[k], cols[k]])}) - delineation failed.")

    # catchment area (km2) on the geographic grid
    lat_c = aff.f + (np.arange(cmask.shape[0]) + 0.5) * aff.e
    cell_km2 = (abs(aff.a) * 111.32 * np.cos(np.deg2rad(lat_c)))[:, None] * (abs(aff.e) * 110.54)
    area_km2 = float((cmask * cell_km2).sum())
    print(f"dam snapped {dist_m[k]:.0f} m to channel (acc = {int(a[rows[k], cols[k]])} cells)")
    print(f"dam catchment area = {area_km2:,.0f} km2  <- compare with the RID published catchment area")

    # resample the catchment mask to the 30 m zone grid and rebuild zones 1/2
    with rasterio.open(ZONE) as s:
        z = s.read(1); prof = s.profile; nod = s.nodata
        cm30 = np.zeros(z.shape, "uint8")
        reproject(cmask, cm30, src_transform=aff, src_crs="EPSG:4326",
                  dst_transform=s.transform, dst_crs=s.crs, resampling=Resampling.nearest)
    valid = (z > 0) if nod is None else ((z != nod) & (z > 0))
    zn = z.copy()
    m12 = valid & np.isin(z, [1, 2])
    zn[m12 & (cm30 == 1)] = 1
    zn[m12 & (cm30 == 0)] = 2
    with rasterio.open(ZONE_OUT, "w", **prof) as o:
        o.write(zn, 1)

    px_km2 = abs(prof["transform"].a * prof["transform"].e) / 1e6
    rec = []
    for code, name in {1: "upstream (dam catchment)", 2: "downstream", 3: "reservoir pool", 4: "drawdown ring"}.items():
        rec.append({"zone": name, "old_km2": round(int((valid & (z == code)).sum()) * px_km2, 1),
                    "new_km2": round(int((valid & (zn == code)).sum()) * px_km2, 1)})
    moved = int((m12 & (z != zn)).sum())
    R = pd.DataFrame(rec); R.to_csv(OUT / "round2_dam_catchment.csv", index=False)
    print("\nlowland-domain zone areas, elevation rule (old) vs dam catchment (new):\n" + R.to_string(index=False))
    print(f"\npixels re-assigned between upstream/downstream: {moved:,} ({moved*px_km2:,.1f} km2)")
    print(f"wrote {ZONE_OUT} -> now re-run 23_reservoir_exposure_summary.py")


if __name__ == "__main__":
    main()
