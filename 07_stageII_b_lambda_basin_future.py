#!/usr/bin/env python3
# ============================================================
# Paper #5 - Stage II-b: climate scaling (lambda) -> future risk maps
#
#   Inputs:
#     indices_output/indices_annual_long.csv   (Stage II-a)
#     stage2_outputs/present_susceptibility.tif (Stage 3)
#     stage2_outputs/susceptibility_breaks.json (Stage 3, FIXED breakpoints)
#
#   Method (PATH A - honest uncertainty front and centre):
#     lambda_gcm = R95p_future / R95p_baseline   (3 LPP stations, per GCM)
#     future odds = present odds x lambda   (odds scaling, bounded [0,1])
#     propagate 5 GCMs SEPARATELY -> ensemble mean + inter-GCM spread + agreement
#
#   Outputs (stage2_outputs/futureII/):
#     lambda_table.csv                      (KEY result: lambda per GCM x SSP x horizon)
#     future_<ssp>_<horizon>_prob.tif       ensemble future susceptibility
#     future_<ssp>_<horizon>_5class.tif     fixed-breakpoint classes
#     future_<ssp>_<horizon>_change.tif     future - present
#     future_<ssp>_<horizon>_spread.tif     inter-GCM std (uncertainty)
#     change_summary.csv                    class-area change + GCM agreement + range
#     change_panel.png                      2x3 change-map panel
#
# Paste into ONE Colab cell and Run.
# ============================================================

# --- Colab bootstrap ---
try:
    import google.colab  # noqa: F401
    import os as _os
    if not _os.path.exists("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
    for _p in ("rasterio",):
        try:
            __import__(_p)
        except ImportError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", _p], check=False)
except ImportError:
    pass

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DRIVE   = "/content/drive/MyDrive/GEE_LPP_CMIP6"              # <<< EDIT if elsewhere
ANNUAL  = f"{DRIVE}/indices_output/indices_annual_long.csv"
PRESENT = f"{DRIVE}/stage2_outputs/present_susceptibility.tif"
BREAKS  = f"{DRIVE}/stage2_outputs/susceptibility_breaks.json"
OUT     = Path(f"{DRIVE}/stage2_outputs/futureII")

LPP_STATIONS = [250781, 250751, 250771]                      # in-basin anchor stations
SELECTED = ["MIROC6", "HadGEM3-GC31-LL", "ACCESS-CM2", "KACE-1-0-G", "MRI-ESM2-0"]
SSPS     = ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]
WINDOWS  = {"near_2036-2065": (2036, 2065), "far_2071-2100": (2071, 2100)}
BASELINE = (1985, 2014)
INDEX    = "R95p"            # lambda driver (flood-relevant extreme intensity)
PIXEL_HA = (30 * 30) / 10000.0
EPS      = 1e-6

OUT.mkdir(parents=True, exist_ok=True)


def mean_window(df, y0, y1, col):
    return df[(df.year >= y0) & (df.year <= y1)][col].mean()


def odds_scale(S, lam):
    s = np.clip(S, EPS, 1 - EPS)
    o = (s / (1 - s)) * lam
    return o / (1 + o)


def compute_lambda():
    a = pd.read_csv(ANNUAL)
    a = a[a.station.isin(LPP_STATIONS) & a.model.isin(SELECTED)]
    rows = []
    for m in SELECTED:
        g = a[a.model == m]
        base = mean_window(g[g.term == "Historical"], *BASELINE, INDEX)
        for ssp in SSPS:
            gs = g[g.scenario == ssp]
            for wname, (y0, y1) in WINDOWS.items():
                fut = mean_window(gs, y0, y1, INDEX)
                rows.append({"model": m, "ssp": ssp, "horizon": wname,
                             f"{INDEX}_base": round(base, 1), f"{INDEX}_fut": round(fut, 1),
                             "lambda": round(fut / base, 4)})
    return pd.DataFrame(rows)


def main():
    lam = compute_lambda()
    lam.to_csv(OUT / "lambda_table.csv", index=False)
    print("=== lambda (R95p ratio) per GCM x SSP x horizon ===")
    print(lam.pivot_table(index=["ssp", "horizon"], columns="model", values="lambda").round(2).to_string())

    with rasterio.open(PRESENT) as src:
        S = src.read(1).astype("float64")
        profile = src.profile
    domain = np.isfinite(S)
    breaks = json.load(open(BREAKS))["breaks"]

    def classify(p):
        c = np.full(p.shape, 0, dtype="int16")
        valid = np.isfinite(p)
        c[valid] = (np.digitize(p[valid], breaks) + 1).astype("int16")
        return c

    present_cls = classify(S)
    present_high = int(((present_cls >= 4) & domain).sum()) * PIXEL_HA

    def write(arr, path, dtype, nod):
        a = np.where(domain, arr, nod).astype(dtype)
        pr = profile.copy(); pr.update(count=1, dtype=dtype, nodata=nod, compress="lzw")
        with rasterio.open(path, "w", **pr) as d:
            d.write(a, 1)

    summary, panels = [], {}
    for ssp in SSPS:
        for hz in WINDOWS:
            lg = lam[(lam.ssp == ssp) & (lam.horizon == hz)].set_index("model")["lambda"]
            # per-GCM future surfaces
            surfaces = np.stack([odds_scale(S, lg[m]) for m in SELECTED])   # (5,H,W)
            ens = np.nanmean(surfaces, axis=0)
            spread = np.nanstd(surfaces, axis=0)
            change = ens - S
            cls = classify(ens)

            tag = f"{ssp}_{hz}"
            write(ens,    OUT / f"future_{tag}_prob.tif",   "float32", np.nan)
            write(cls,    OUT / f"future_{tag}_5class.tif",  "int16",   0)
            write(change, OUT / f"future_{tag}_change.tif",  "float32", np.nan)
            write(spread, OUT / f"future_{tag}_spread.tif",  "float32", np.nan)
            panels[tag] = np.where(domain, change, np.nan)

            # agreement + per-GCM high-class area range
            n_up = int((lg > 1).sum())
            hi_by_gcm = [int(((classify(surfaces[i]) >= 4) & domain).sum()) * PIXEL_HA
                         for i in range(len(SELECTED))]
            hi_ens = int(((cls >= 4) & domain).sum()) * PIXEL_HA
            summary.append({"ssp": ssp, "horizon": hz,
                            "lambda_mean": round(float(lg.mean()), 3),
                            "GCMs_increase": f"{n_up}/5",
                            "high_present_ha": round(present_high, 1),
                            "high_future_ha": round(hi_ens, 1),
                            "high_change_pct": round((hi_ens / present_high - 1) * 100, 1),
                            "high_range_ha": f"{round(min(hi_by_gcm),0)}-{round(max(hi_by_gcm),0)}"})

    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT / "change_summary.csv", index=False)
    print("\n=== change summary (high+very-high area, present = {:.0f} ha) ===".format(present_high))
    print(sdf.to_string(index=False))

    # change panel 2x3
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        vmax = max(np.nanmax(np.abs(p)) for p in panels.values()) or 0.1
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for j, ssp in enumerate(SSPS):
            for i, hz in enumerate(WINDOWS):
                ax = axes[i, j]
                im = ax.imshow(panels[f"{ssp}_{hz}"], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                ax.set_title(f"{ssp}  {hz.split('_')[0]}", fontsize=10)
                ax.axis("off")
        fig.colorbar(im, ax=axes, shrink=0.6, label="Change in susceptibility (future - present)")
        fig.suptitle("Climate-driven change in flood susceptibility (ensemble mean)", fontsize=13)
        plt.savefig(OUT / "change_panel.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("\nSaved change_panel.png")
    except Exception as e:
        print(f"Panel skipped: {e}")

    print(f"\nDone. Future maps + tables -> {OUT}/")
    print("Headline is the agreement/spread, not the mean: report GCMs_increase and high_range_ha.")


if __name__ == "__main__":
    main()
