#!/usr/bin/env python3
# ============================================================
# Figure 2 (two-stage workflow diagram) and Figure 3 (SHAP a|b panel)
#
#   Fig 2: drawn from scratch (no inputs).
#   Fig 3: re-computes SHAP for the saved XGBoost model (deterministic, so the values
#          equal those of 03_stage2_ml_ensemble.py) and plots it with readable names
#          (class names instead of codes such as lulc_80 or soil_4).
#   Inputs: LPP_paper5_samples_static.csv, stage2_outputs/models/XGB.joblib, ensemble_config.joblib
#   Output: revision_outputs/fig2_workflow.png, revision_outputs/fig3_shap.png
# ============================================================
try:
    import google.colab  # noqa: F401
    from google.colab import drive
    drive.flush_and_unmount(); drive.mount("/content/drive", force_remount=True)
    DRIVE = "/content/drive/MyDrive/GEE_LPP_CMIP6"
except ImportError:
    DRIVE = "."

from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(f"{DRIVE}/revision_outputs"); OUT.mkdir(parents=True, exist_ok=True)


def fig2():
    fig, ax = plt.subplots(figsize=(11, 7.2)); ax.set_xlim(0, 11); ax.set_ylim(0, 7.2); ax.axis("off")

    def box(x, y, w, h, t, fc, fs=8.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=fc, ec="#444", lw=0.9))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.1))

    G, B, O = "#dff0e6", "#dde8f6", "#fbe5d6"
    ax.text(2.6, 6.95, "STAGE I — spatial pattern S(x)", ha="center", fontsize=10, weight="bold", color="#1d6b45")
    ax.text(8.3, 6.95, "STAGE II — climate magnitude λ(x)", ha="center", fontsize=10, weight="bold", color="#22508f")
    box(0.2, 5.75, 2.4, 0.95, "Sentinel-1 SAR inventory\n(Nangka 2020, Dianmu 2021,\nNoru 2022; freq. 1–3)", G)
    box(2.8, 5.75, 2.4, 0.95, "11 static factors (30 m)\nterrain · HAND · TWI · SPI\nhydrography · soil · LULC", G)
    box(0.8, 4.45, 3.8, 0.85, "Balanced sample (2,000 + 2,000; 150 m buffer)\nVIF < 5 · one-hot · aspect sin/cos", G)
    box(0.8, 3.15, 3.8, 0.9, "RF · SVM · ANN · XGBoost\nspatial-block CV (3, 5, 6 km) · LORO CV\nweighted soft-voting ensemble · SHAP", G)
    box(0.8, 1.95, 3.8, 0.8, "Prevalence recalibration (τ ≈ 0.023)\n→ present S(x), 5 equal-area classes", G)
    box(5.9, 5.75, 2.3, 0.95, "22 CMIP6 GCMs →\ndynamics-based screening\n→ 5 models", B)
    box(8.5, 5.75, 2.3, 0.95, "EQM bias correction\n(9 RID gauges, 1985–2014)\nR95p baseline threshold", B)
    box(6.2, 4.45, 4.3, 0.85, "λ = R95p(future) / R95p(1985–2014)\nper gauge × GCM × SSP × horizon", B)
    box(6.2, 3.15, 4.3, 0.9, "Nearest-gauge zones → λ(x)\nRQ2 checks: screened vs 22 · all C(22,5) subsets\nraw-GCM vs EQM λ diagnostic", B)
    box(6.2, 1.95, 4.3, 0.8, "odds′(x) = λ(x)^η · odds(x)\nη ∈ {0.5, 1, 2, 3} (η = 1 reference)", B)
    box(2.3, 0.55, 6.4, 0.95, "Future S′(x): 3 SSPs × 2 horizons × 5 GCMs  →  change · ≥4/5 agreement map\n"
        "Bounded 2×2 factorial (climate × LULC 2024/2045) on changed pixels, by transition\n"
        "Dam-catchment split · drawdown ring masked · exposure (cropland, population)", O, fs=8.4)
    for a in [(1.4, 5.75, 2.0, 5.3), (4.0, 5.75, 3.4, 5.3), (2.7, 4.45, 2.7, 4.05), (2.7, 3.15, 2.7, 2.75),
              (8.2, 6.22, 8.5, 6.22), (9.65, 5.75, 9.0, 5.3), (8.35, 4.45, 8.35, 4.05), (8.35, 3.15, 8.35, 2.75),
              (2.7, 1.95, 4.0, 1.5), (8.35, 1.95, 7.0, 1.5)]:
        arr(*a)
    ax.text(5.5, 0.18, "Reservoir operation held at the present regime (non-stationary boundary condition, not modelled)",
            ha="center", fontsize=8, style="italic", color="#555")
    plt.savefig(OUT / "fig2_workflow.png", dpi=220, bbox_inches="tight"); plt.close()
    print(f"Fig 2 -> {OUT}/fig2_workflow.png")


NAMES = {"elevation": "Elevation", "slope": "Slope", "curvature": "Curvature", "hand": "HAND",
         "twi": "TWI", "spi": "SPI", "dist_river": "Distance to river", "drain_density": "Drainage density",
         "aspect_sin": "Aspect (sin)", "aspect_cos": "Aspect (cos)"}
LULC = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
        60: "Bare/sparse", 80: "Permanent water", 90: "Herbaceous wetland"}
# OpenLandMap USDA texture classes (SOL_TEXTURE-CLASS_USDA-TT_M) - check against the GEE catalogue
SOIL = {1: "Clay", 2: "Silty clay", 3: "Sandy clay", 4: "Clay loam", 5: "Silty clay loam",
        6: "Sandy clay loam", 7: "Loam", 8: "Silt loam", 9: "Sandy loam", 10: "Silt",
        11: "Loamy sand", 12: "Sand"}


def nice(col):
    if col in NAMES: return NAMES[col]
    pre, _, code = col.partition("_")
    try: code = int(float(code))
    except ValueError: return col
    if pre == "lulc": return "Land cover: " + LULC.get(code, f"class {code}")
    if pre == "soil": return "Soil: " + SOIL.get(code, f"class {code}")
    return col


def fig3():
    import numpy as np, pandas as pd, joblib
    try:
        import shap
    except ImportError:
        import subprocess, sys; subprocess.run([sys.executable, "-m", "pip", "install", "-q", "shap"]); import shap
    M = Path(f"{DRIVE}/stage2_outputs/models")
    if not (M / "XGB.joblib").exists():
        print("Fig 3 skipped: stage2_outputs/models/XGB.joblib not found."); return
    cfg = joblib.load(M / "ensemble_config.joblib"); feats = cfg["features"]
    d = pd.read_csv(f"{DRIVE}/LPP_paper5_samples_static.csv")
    a = np.deg2rad(d["aspect"].values); d["aspect_sin"], d["aspect_cos"] = np.sin(a), np.cos(a)
    dm = pd.get_dummies(d[["soil", "lulc"]].astype("Int64").astype(str), prefix=["soil", "lulc"], dummy_na=False)
    X = pd.concat([d[list(NAMES)], dm.astype(float)], axis=1).reindex(columns=feats, fill_value=0.0)
    sv = shap.TreeExplainer(joblib.load(M / "XGB.joblib")).shap_values(X)
    Xn = X.rename(columns={c: nice(c) for c in X.columns})
    fig = plt.figure(figsize=(14, 6.5))
    ax1 = fig.add_subplot(1, 2, 1); plt.sca(ax1)
    shap.summary_plot(sv, Xn, plot_type="bar", show=False, plot_size=None, max_display=15)
    ax1.set_title("(a) Mean |SHAP|", loc="left"); ax1.set_xlabel("mean |SHAP value| (balanced-sample log-odds)")
    ax2 = fig.add_subplot(1, 2, 2); plt.sca(ax2)
    shap.summary_plot(sv, Xn, show=False, plot_size=None, max_display=15)
    ax2.set_title("(b) Per-sample effects", loc="left")
    plt.tight_layout(); plt.savefig(OUT / "fig3_shap.png", dpi=200, bbox_inches="tight"); plt.close()
    print(f"Fig 3 -> {OUT}/fig3_shap.png")


if __name__ == "__main__":
    fig2(); fig3()
