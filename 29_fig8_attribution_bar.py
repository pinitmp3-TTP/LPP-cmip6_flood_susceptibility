#!/usr/bin/env python3
# ============================================================
# REVISION - Figure 8 as a BAR CHART (replaces the near-blank map version)
#
#   WHY: the factorial change is confined to the ~9.5% of LULC-changed pixels
#   and the climate signal at near-future is tiny, so a map reads as blank.
#   The numbers are sharp, so a bar chart communicates the attribution clearly.
#
#   Panel (a): mean |delta S| climate vs LULC (and LULC excl. built-up) per SSP
#              -> shows LULC is the larger driver on the changed area.
#   Panel (b): mean delta S by LULC transition type (present climate)
#              -> shows which land conversions raise vs lower susceptibility.
#
#   Inputs (GEE_LPP_CMIP6/stage2_outputs/attribution_fair/):
#       fair_attribution_summary.csv
#       lulc_effect_by_transition.csv
#   Output: GEE_LPP_CMIP6/revision_outputs/fig8_attribution_bar.png
# ============================================================
try:
    import google.colab  # noqa: F401
    from google.colab import drive
    drive.flush_and_unmount(); drive.mount("/content/drive", force_remount=True)
    DRIVE = "/content/drive/MyDrive/GEE_LPP_CMIP6"
except ImportError:
    DRIVE = "."

from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ATTR = f"{DRIVE}/stage2_outputs/attribution_fair"
OUT = Path(f"{DRIVE}/revision_outputs"); OUT.mkdir(parents=True, exist_ok=True)

# ESA WorldCover class codes (the LULC rasters use these; CROPLAND=40, BUILTUP=50)
WC = {10: "Tree", 20: "Shrub", 30: "Grass", 40: "Cropland", 50: "Built-up",
      60: "Bare", 70: "Snow/ice", 80: "Water", 90: "Wetland", 95: "Mangrove",
      100: "Moss"}
SSP_LABEL = {"ssp1_2_6": "SSP1-2.6", "ssp2_4_5": "SSP2-4.5", "ssp5_8_5": "SSP5-8.5"}


def name(code):
    try:
        return WC.get(int(round(float(code))), f"cls{int(round(float(code)))}")
    except Exception:
        return str(code)


def main():
    # ROUND 3: use the prevalence-recalibrated scale (consistent with Section 3.4) when available
    rec = Path(f"{ATTR}/fair_attribution_summary_recalibrated.csv")
    tt = pd.read_csv(f"{ATTR}/lulc_effect_by_transition.csv")
    if rec.exists() and "mean_dLULC_recal" in tt:
        summ, col, scale = pd.read_csv(rec), "mean_dLULC_recal", "prevalence-recalibrated scale"
    else:
        summ, col, scale = pd.read_csv(f"{ATTR}/fair_attribution_summary.csv"), "mean_dLULC", "balanced-sample scale"
    print(f"Fig 8 on the {scale}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.15]})

    # ---------- Panel (a): climate vs LULC per SSP ----------
    ssps = summ["ssp"].tolist(); x = np.arange(len(ssps)); w = 0.26
    bars = [("Climate", summ["clim_absdS_changed"].values, dict(color="#4C72B0"), -w),
            ("Land use (all)", summ["lulc_absdS_changed"].values, dict(color="#C44E52"), 0.0)]
    if "lulc_absdS_noBuiltup" in summ:
        bars.append(("Land use (excl. built-up)", summ["lulc_absdS_noBuiltup"].values,
                     dict(color="#C44E52", alpha=0.5, hatch="//"), w))
    for lab, vals, kw, off in bars:
        axL.bar(x + off, vals, w, label=lab, **kw)
        for xi, v in zip(x, vals):
            if np.isfinite(v): axL.text(xi + off, v, f"{v:.2g}", ha="center", va="bottom", fontsize=7.5)
    axL.set_xticks(x); axL.set_xticklabels([SSP_LABEL.get(s, s) for s in ssps])
    axL.set_ylabel(f"Mean |ΔS| on LULC-changed pixels\n({scale})")
    axL.set_title("(a) Driver magnitude, near future", loc="left")
    axL.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    axL.spines[["top", "right"]].set_visible(False); axL.margins(y=0.18)
    axL.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    # ---------- Panel (b): delta S by transition type ----------
    tt = tt.copy()
    tt["label"] = [f"{name(f)} → {name(t)}" for f, t in zip(tt["from"], tt["to"])]
    tt = tt.sort_values(col)
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in tt[col]]
    y = np.arange(len(tt)); span = float(np.nanmax(np.abs(tt[col]))) or 1.0
    axR.barh(y, tt[col], color=colors)
    axR.set_yticks(y); axR.set_yticklabels(tt["label"], fontsize=8)
    axR.axvline(0, color="k", lw=0.8)
    for yi, v, n in zip(y, tt[col], tt["n_px"]):
        axR.text(v + (0.01 if v >= 0 else -0.01) * span, yi, f"n = {int(n):,}",
                 va="center", ha="left" if v >= 0 else "right", fontsize=7, color="#444")
    axR.set_xlabel(f"Mean ΔS ({scale}); red raises, blue lowers susceptibility")
    axR.set_title("(b) Land-use effect by transition, present climate", loc="left")
    axR.spines[["top", "right"]].set_visible(False); axR.margins(x=0.22)
    axR.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))

    plt.tight_layout()
    plt.savefig(OUT / "fig8_attribution_bar.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Fig 8 (bar) saved -> {OUT}/fig8_attribution_bar.png")


if __name__ == "__main__":
    main()
