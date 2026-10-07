#!/usr/bin/env python3
# ============================================================
# Paper #5 - CMIP6 flood susceptibility, Lam Phra Phloeng
# STAGE 3b (Python)  -  quantitative validation of present map
#
#   Reads : present_susceptibility.tif, present_susceptibility_5class.tif
#           LPP_flood_frequency.tif   (export from GEE, same grid)
#   Reports (for Results):
#     (1) class distribution of flood pixels  -> % of floods in high+very-high
#     (2) success-rate curve + AUC
#     (3) frequency gradient (semi-independent: training used binary labels)
#
#   HONESTY: the inventory was used to TRAIN the model, so (1)-(2) are a
#   success-rate (apparent) check, not independent validation. Independent
#   performance = spatial-CV AUC (Stage 2). (3) is semi-independent because
#   the recurrence count (1/2/3) was NOT a training target.
#
# Paste into ONE Colab cell and Run (after Stage 3).
# ============================================================

# --- Colab bootstrap ---
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB:
    import os, subprocess, sys
    if not os.path.exists("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
    try:
        import rasterio  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio"], check=False)

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DRIVE_FOLDER = "/content/drive/MyDrive/GEE_LPP_CMIP6"   # <<< EDIT if elsewhere
OUT_DIR   = Path(DRIVE_FOLDER) / "stage2_outputs"
SUS_PROB  = OUT_DIR / "present_susceptibility.tif"
SUS_CLASS = OUT_DIR / "present_susceptibility_5class.tif"
FREQ_TIF  = f"{DRIVE_FOLDER}/LPP_flood_frequency.tif"
PIXEL_HA  = (30 * 30) / 10000.0
CLASS_NAMES = {1: "very-low", 2: "low", 3: "moderate", 4: "high", 5: "very-high"}


def load_aligned():
    """Load prob + class on their grid; reproject frequency onto the same grid if needed."""
    with rasterio.open(SUS_PROB) as ref:
        prob = ref.read(1).astype("float64")
        H, W = ref.shape
        ref_t, ref_crs = ref.transform, ref.crs
    with rasterio.open(SUS_CLASS) as c:
        cls = c.read(1)
    with rasterio.open(FREQ_TIF) as f:
        if f.shape == (H, W) and f.transform == ref_t and f.crs == ref_crs:
            freq = f.read(1).astype("float64")
        else:
            freq = np.zeros((H, W), dtype="float64")
            reproject(source=rasterio.band(f, 1), destination=freq,
                      src_transform=f.transform, src_crs=f.crs,
                      dst_transform=ref_t, dst_crs=ref_crs,
                      resampling=Resampling.nearest)
    return prob, cls, freq


def main():
    prob, cls, freq = load_aligned()
    domain = np.isfinite(prob)
    freq = np.where(np.isfinite(freq), freq, 0)
    flood = domain & (freq >= 1)

    n_dom, n_flood = int(domain.sum()), int(flood.sum())
    print(f"In-domain px: {n_dom} ({n_dom*PIXEL_HA:.0f} ha) | flood px: {n_flood} ({n_flood*PIXEL_HA:.0f} ha)")
    print("NOTE: inventory was used in training -> (1)-(2) are SUCCESS-RATE (apparent).")
    print("      Independent performance = spatial-CV AUC (Stage 2). (3) is semi-independent.\n")

    # ---------- (1) class distribution ----------
    rows = []
    for c in range(1, 6):
        in_c = domain & (cls == c)
        fl_c = flood & (cls == c)
        rows.append({
            "class": c, "name": CLASS_NAMES[c],
            "area_ha": round(int(in_c.sum()) * PIXEL_HA, 1),
            "area_pct": round(in_c.sum() / n_dom * 100, 1),
            "flood_ha": round(int(fl_c.sum()) * PIXEL_HA, 1),
            "flood_pct": round(fl_c.sum() / n_flood * 100, 1),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT_DIR / "validation_class_distribution.csv", index=False)
    high = tab[tab["class"] >= 4]["flood_pct"].sum()
    high_area = tab[tab["class"] >= 4]["area_pct"].sum()
    print("(1) Flood distribution by susceptibility class:")
    print(tab.to_string(index=False))
    print(f"    -> high+very-high capture {high:.1f}% of floods "
          f"while covering only {high_area:.1f}% of the domain\n")

    # ---------- (2) success-rate curve + AUC ----------
    p = prob[domain]
    fl = flood[domain].astype(int)
    order = np.argsort(p)[::-1]                      # high susceptibility first
    cum_flood = np.cumsum(fl[order]) / fl.sum()
    cum_area = np.arange(1, len(fl) + 1) / len(fl)
    # area under success-rate curve, anchored at origin (version-safe trapezoid)
    ca = np.concatenate([[0.0], cum_area])
    cf = np.concatenate([[0.0], cum_flood])
    sr_auc = float(np.sum(np.diff(ca) * (cf[1:] + cf[:-1]) / 2.0))   # 0.5 random, ->1 perfect
    pd.DataFrame({"cum_area": cum_area, "cum_flood": cum_flood}) \
        .iloc[:: max(1, len(fl) // 500)] \
        .to_csv(OUT_DIR / "validation_success_rate_curve.csv", index=False)
    print(f"(2) Success-rate AUC = {sr_auc:.3f}  (0.5 = random)\n")

    # ---------- (3) frequency gradient (semi-independent) ----------
    print("(3) Mean S(x) by observed flood frequency (should increase):")
    grad = []
    for fr in [0, 1, 2, 3]:
        m = domain & (freq == fr)
        if m.sum():
            grad.append({"freq": fr, "n_px": int(m.sum()), "mean_Sx": round(float(prob[m].mean()), 3)})
    gtab = pd.DataFrame(grad)
    gtab.to_csv(OUT_DIR / "validation_frequency_gradient.csv", index=False)
    print(gtab.to_string(index=False))
    mono = gtab.sort_values("freq")["mean_Sx"].is_monotonic_increasing
    print(f"    -> monotonic increase with recurrence: {mono}  "
          f"(True = model ranks recurrent floods higher, genuine skill)\n")

    # ---------- success-rate plot ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5.5, 5.5))
        plt.plot(cum_area * 100, cum_flood * 100, lw=2, label=f"Success rate (AUC={sr_auc:.3f})")
        plt.plot([0, 100], [0, 100], "--", color="grey", label="Random")
        plt.xlabel("Cumulative % of area (ranked by susceptibility)")
        plt.ylabel("Cumulative % of flood inventory captured")
        plt.title("Success-rate curve - present susceptibility")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(OUT_DIR / "validation_success_rate_curve.png", dpi=150)
        plt.close()
        print("Saved validation_success_rate_curve.png")
    except Exception as e:
        print(f"Plot skipped: {e}")

    print(f"\nDone. Validation outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
