#!/usr/bin/env python3
# ============================================================
# Paper #5 - CMIP6 flood susceptibility, Lam Phra Phloeng
# STAGE 3 (Python)  -  present susceptibility map S(x)
#
#   Input : GEE_LPP_CMIP6/LPP_paper5_predictor_stack_static.tif  (GEE Stage 1)
#           GEE_LPP_CMIP6/stage2_outputs/models/                 (trained ensemble)
#   Output: present_susceptibility.tif        (probability 0-1)
#           present_susceptibility_5class.tif (very-low..very-high = 1..5)
#           susceptibility_breaks.json        (FIXED breakpoints -> reuse for future maps)
#           present_model_disagreement.tif    (ensemble uncertainty layer, 4.7)
#           present_susceptibility_preview.png
#
# Applies the SAME feature engineering as Stage 2, then the weighted ensemble.
# Paste into ONE Colab cell and Run.
# ============================================================

# --- Colab bootstrap (mount Drive + install rasterio; harmless outside Colab) ---
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
    for pkg in ("rasterio",):
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg} ...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import joblib

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIG  -  same ONE Drive folder as Stage 1 & 2
# ------------------------------------------------------------
DRIVE_FOLDER = "/content/drive/MyDrive/GEE_LPP_CMIP6"   # <<< EDIT if elsewhere
STACK_PATH   = f"{DRIVE_FOLDER}/LPP_paper5_predictor_stack_static.tif"
OUT_DIR      = Path(DRIVE_FOLDER) / "stage2_outputs"
MODEL_DIR    = OUT_DIR / "models"

# band order MUST match the GEE Stage 1 ee.Image.cat([...]) order
BAND_NAMES = ["elevation", "slope", "aspect", "curvature", "hand", "twi",
              "spi", "dist_river", "drain_density", "soil", "lulc"]
CATEGORICAL = ["soil", "lulc"]
QUANTILES   = [0.20, 0.40, 0.60, 0.80]   # 5 classes by equal-area; saved & reused for future maps
PIXEL_HA    = (30 * 30) / 10000.0        # 0.09 ha per 30 m pixel


def build_features(df_valid):
    """Reproduce Stage-2 feature engineering: aspect sin/cos + one-hot soil/lulc."""
    d = df_valid.copy()
    a = np.deg2rad(d["aspect"].values)
    d["aspect_sin"] = np.sin(a)
    d["aspect_cos"] = np.cos(a)
    dummies = pd.get_dummies(d[CATEGORICAL].astype("Int64").astype(str),
                             prefix=CATEGORICAL, dummy_na=False)
    cont = ["elevation", "slope", "curvature", "hand", "twi", "spi",
            "dist_river", "drain_density", "aspect_sin", "aspect_cos"]
    return pd.concat([d[cont], dummies.astype(float)], axis=1)


def main():
    # --- read stack ---
    with rasterio.open(STACK_PATH) as src:
        arr = src.read().astype("float64")          # (bands, H, W)
        profile = src.profile
        nodata = src.nodata
    n_bands, H, W = arr.shape
    assert n_bands == len(BAND_NAMES), \
        f"Stack has {n_bands} bands but expected {len(BAND_NAMES)} ({BAND_NAMES})"

    flat = arr.reshape(n_bands, -1).T               # (pixels, bands)
    df = pd.DataFrame(flat, columns=BAND_NAMES)

    valid = np.isfinite(df.values).all(axis=1)
    if nodata is not None and np.isfinite(nodata):
        valid &= (df.values != nodata).all(axis=1)
    print(f"Grid {H}x{W} = {H*W} px | valid (in-domain) px: {int(valid.sum())}")

    # --- features aligned to the trained model ---
    cfg = joblib.load(MODEL_DIR / "ensemble_config.joblib")
    features, weights = cfg["features"], cfg["weights"]
    X = build_features(df[valid]).reindex(columns=features, fill_value=0.0)

    # --- ensemble prediction (weighted soft-voting) ---
    prob = np.zeros(len(X))
    per_model = {}
    for name, wt in weights.items():
        mdl = joblib.load(MODEL_DIR / f"{name}.joblib")
        p = mdl.predict_proba(X)[:, 1]
        per_model[name] = p
        prob += wt * p
        print(f"  {name}: predicted (weight {wt:.3f})")
    disagree = np.column_stack([per_model[n] for n in weights]).std(axis=1)

    # --- FIXED 5-class breakpoints (save for future-map reuse) ---
    breaks = [float(np.quantile(prob, q)) for q in QUANTILES]
    json.dump({"method": "quantile", "quantiles": QUANTILES, "breaks": breaks,
               "note": "REUSE these exact breaks for all future maps (change-map comparability)"},
              open(OUT_DIR / "susceptibility_breaks.json", "w"), indent=2)
    cls = (np.digitize(prob, breaks) + 1).astype("int16")   # classes 1..5
    print(f"Class breakpoints (prob): {[round(b,3) for b in breaks]}")

    # --- class-area summary ---
    print("Susceptibility class areas:")
    for c in range(1, 6):
        n = int((cls == c).sum())
        print(f"  class {c}: {n:>7d} px = {n*PIXEL_HA:9.1f} ha")

    # --- write rasters ---
    def write(data1d, path, dtype, nod):
        out = np.full(H * W, nod, dtype=dtype)
        out[valid] = data1d.astype(dtype)
        prof = profile.copy()
        prof.update(count=1, dtype=dtype, nodata=nod, compress="lzw")
        with rasterio.open(path, "w", **prof) as dst:
            dst.write(out.reshape(H, W), 1)

    write(prob,     OUT_DIR / "present_susceptibility.tif",        "float32", np.nan)
    write(cls,      OUT_DIR / "present_susceptibility_5class.tif", "int16",   0)
    write(disagree, OUT_DIR / "present_model_disagreement.tif",    "float32", np.nan)

    # --- PNG preview ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pm = np.full(H * W, np.nan); pm[valid] = prob; pm = pm.reshape(H, W)
        plt.figure(figsize=(7, 6))
        im = plt.imshow(pm, cmap="RdYlBu_r", vmin=0, vmax=1)
        plt.colorbar(im, label="Flood susceptibility (present)")
        plt.title("Present flood susceptibility S(x) - Lam Phra Phloeng")
        plt.axis("off"); plt.tight_layout()
        plt.savefig(OUT_DIR / "present_susceptibility_preview.png", dpi=150)
        plt.close()
        print("Saved preview PNG.")
    except Exception as e:
        print(f"Preview skipped: {e}")

    print(f"\nDone. Present map + fixed breakpoints in {OUT_DIR}/")
    print("Stage II (climate): future risk = S(x) x lambda(R10mm,R95p), reusing THESE breakpoints.")


if __name__ == "__main__":
    main()
