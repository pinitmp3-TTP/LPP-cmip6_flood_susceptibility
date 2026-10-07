#!/usr/bin/env python3
# ============================================================
# Figure 4 (present susceptibility) and Figure 7 (far-future SSP5-8.5 change +
# number of GCMs projecting an increase), with the reservoir pool and seasonal
# drawdown ring HATCHED (reservoir level is set by dam operation, so these pixels
# are excluded from all future statistics; see 26_round3_uniform_lambda_ring_mask.py).
#
#   Inputs (GEE_LPP_CMIP6/):
#     stage2_outputs/present_susceptibility.tif                  (balanced scale, Fig 4)
#     stage2_outputs/futureII_spatial/future_ssp5_8_5_far_2071-2100_change.tif
#     stage2_outputs/futureII_spatial/future_ssp5_8_5_far_2071-2100_agreement.tif
#     reservoir_zone_hydro.tif (or reservoir_zone.tif)           zones 3 pool, 4 drawdown
#   Output: revision_outputs/fig4_present.png, revision_outputs/fig7_robustness.png
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

from pathlib import Path
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

SP = f"{DRIVE}/stage2_outputs/futureII_spatial"
PRESENT = f"{DRIVE}/stage2_outputs/present_susceptibility.tif"
CHANGE = f"{SP}/future_ssp5_8_5_far_2071-2100_change.tif"
AGREE = f"{SP}/future_ssp5_8_5_far_2071-2100_agreement.tif"
RES = f"{DRIVE}/reservoir_zone_hydro.tif"
if not Path(RES).exists(): RES = f"{DRIVE}/reservoir_zone.tif"
OUT = Path(f"{DRIVE}/revision_outputs"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["hatch.color"] = "#333333"; plt.rcParams["hatch.linewidth"] = 0.6


def read(path, ref=None):
    with rasterio.open(path) as s:
        if ref is None or (s.shape == ref[0] and s.transform == ref[1] and s.crs == ref[2]):
            a = s.read(1).astype("float64")
            if s.nodata is not None and np.isfinite(s.nodata): a[a == s.nodata] = np.nan
            return a, (s.shape, s.transform, s.crs)
        d = np.zeros(ref[0], "float64")
        reproject(rasterio.band(s, 1), d, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=ref[1], dst_crs=ref[2], resampling=Resampling.nearest)
        return d, ref


def hatch(ax, mask):
    if mask.any():
        ax.contourf(mask.astype(float), levels=[0.5, 1.5], colors="none", hatches=["//////"])
        ax.contour(mask.astype(float), levels=[0.5], colors="#333333", linewidths=0.4)


def ring_handle():
    return Patch(facecolor="none", edgecolor="#333333", hatch="//////", label="Reservoir pool and drawdown ring (excluded)")


def main():
    # ---------- Figure 4 ----------
    S, ref = read(PRESENT)
    res, _ = read(RES, ref); ring = np.isin(np.nan_to_num(res), [3, 4]) & np.isfinite(S)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(np.where(ring, np.nan, S), cmap="RdYlBu_r", vmin=0, vmax=1)
    hatch(ax, ring); ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Flood susceptibility (balanced-sample scale)")
    ax.legend(handles=[ring_handle()], loc="lower left", fontsize=8, frameon=False)
    plt.savefig(OUT / "fig4_present.png", dpi=200, bbox_inches="tight"); plt.close()
    print(f"Fig 4 -> {OUT}/fig4_present.png  (ring+pool px: {int(ring.sum())})")

    # ---------- Figure 7 ----------
    if not (Path(CHANGE).exists() and Path(AGREE).exists()):
        print("Fig 7 skipped: change/agreement rasters not found in", SP); return
    chg, ref = read(CHANGE); agr, _ = read(AGREE, ref)
    res, _ = read(RES, ref); ring = np.isin(np.nan_to_num(res), [3, 4]) & np.isfinite(chg)
    chg = np.where(ring, np.nan, chg); agr = np.where(ring, np.nan, agr)
    v = np.nanpercentile(np.abs(chg), 99.5) or 0.01
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    im0 = ax[0].imshow(chg, cmap="RdBu_r", vmin=-v, vmax=v); hatch(ax[0], ring)
    ax[0].set_title("(a) Change in susceptibility, far-future SSP5-8.5\n(prevalence-recalibrated scale, η = 1)", fontsize=10)
    fig.colorbar(im0, ax=ax[0], shrink=0.7, label="future − present")
    cmap = ListedColormap(plt.get_cmap("YlGnBu")(np.linspace(0.08, 0.95, 6)))
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)
    im1 = ax[1].imshow(agr, cmap=cmap, norm=norm); hatch(ax[1], ring)
    ax[1].set_title("(b) Number of the five GCMs projecting an increase", fontsize=10)
    cb = fig.colorbar(im1, ax=ax[1], shrink=0.7, ticks=range(6)); cb.set_label("GCMs (0–5)")
    for a in ax: a.axis("off")
    ax[1].legend(handles=[ring_handle()], loc="lower left", fontsize=8, frameon=False)
    plt.tight_layout(); plt.savefig(OUT / "fig7_robustness.png", dpi=200, bbox_inches="tight"); plt.close()
    print(f"Fig 7 -> {OUT}/fig7_robustness.png")


if __name__ == "__main__":
    main()
