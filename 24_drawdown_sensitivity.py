#!/usr/bin/env python3
# ============================================================
# ROUND 2 - Does the reservoir DRAWDOWN ring drive the Stage-I model?
#           (follows from Mitchell R2: 22% of inventory pixels lie in the
#            seasonal drawdown ring = reservoir level fluctuation, not river flooding)
#
#   1. Tags every training sample with its reservoir zone (zone 4 = drawdown ring).
#   2. 5 km spatial-block CV of the ensemble (same hyper-parameters, sklearn.clone):
#        A  trained on ALL samples        -> AUC on all test samples
#        B  trained on ALL samples        -> AUC on test samples OUTSIDE the ring
#        C  trained WITHOUT ring samples  -> AUC on test samples OUTSIDE the ring
#      B vs C (same test samples) shows whether the ring changes skill elsewhere.
#   3. Spearman correlation of B and C predictions outside the ring
#      (~1 = the susceptibility pattern outside the reservoir does not depend on the ring).
#
#   Inputs (GEE_LPP_CMIP6/):
#     LPP_paper5_samples_static.csv, reservoir_zone.tif (zone 4 = drawdown, from Round-2 GEE)
#     stage2_outputs/models/*.joblib, ensemble_config.joblib
#   Output: revision_outputs/round2_drawdown_sensitivity.csv
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

import warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib, rasterio
from rasterio.warp import transform as warp_xy
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

SAMPLES = f"{DRIVE}/LPP_paper5_samples_static.csv"
ZONE = f"{DRIVE}/reservoir_zone.tif"
MODELS = Path(f"{DRIVE}/stage2_outputs/models")
OUT = Path(f"{DRIVE}/revision_outputs"); OUT.mkdir(parents=True, exist_ok=True)
CAT = ["soil", "lulc"]
CONT = ["elevation", "slope", "curvature", "hand", "twi", "spi", "dist_river", "drain_density",
        "aspect_sin", "aspect_cos"]
BLOCK_KM, N_FOLDS, DRAWDOWN = 5, 5, 4


def features(d, feats):
    d = d.copy(); a = np.deg2rad(d["aspect"].values)
    d["aspect_sin"], d["aspect_cos"] = np.sin(a), np.cos(a)
    dm = pd.get_dummies(d[CAT].astype("Int64").astype(str), prefix=CAT, dummy_na=False)
    return pd.concat([d[CONT], dm.astype(float)], axis=1).reindex(columns=feats, fill_value=0.0)


def blocks(d, km):
    by = np.floor(d.latitude / (km / 111.0)).astype(int).astype(str)
    bx = np.floor(d.longitude / (km / (111.0 * np.cos(np.deg2rad(14.6))))).astype(int).astype(str)
    return pd.factorize(by + "_" + bx)[0]


def sample_zone(d):
    with rasterio.open(ZONE) as s:
        xs, ys = warp_xy("EPSG:4326", s.crs, d.longitude.tolist(), d.latitude.tolist())
        v = np.array([z[0] for z in s.sample(zip(xs, ys))], dtype="float64")
        if s.nodata is not None: v[v == s.nodata] = 0
    return np.nan_to_num(v, nan=0).astype(int)


def main():
    d = pd.read_csv(SAMPLES); y = d["label"].astype(int).values
    d["zone"] = sample_zone(d); ring = d["zone"].values == DRAWDOWN
    cfg = joblib.load(MODELS / "ensemble_config.joblib"); feats, w = cfg["features"], cfg["weights"]
    base = {m: joblib.load(MODELS / f"{m}.joblib") for m in w if (MODELS / f"{m}.joblib").exists()}
    X = features(d, feats); g = blocks(d, BLOCK_KM)

    tab = pd.crosstab(d["zone"], d["label"]).rename(columns={0: "absence", 1: "presence"})
    print("samples by reservoir zone (1 up, 2 down, 3 pool, 4 drawdown, 0 outside):\n" + tab.to_string())
    print(f"\npresence samples in drawdown ring: {int((ring & (y == 1)).sum())} of {int((y == 1).sum())} "
          f"({100*(ring & (y == 1)).sum()/max((y == 1).sum(), 1):.1f}%)")

    pA = np.zeros(len(y)); pC = np.full(len(y), np.nan)
    wsum = sum(w[m] for m in base)
    for tr, te in GroupKFold(N_FOLDS).split(X, y, g):
        trC = tr[~ring[tr]]
        pA[te] = sum(w[m] * clone(e).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
                     for m, e in base.items()) / wsum
        pC[te] = sum(w[m] * clone(e).fit(X.iloc[trC], y[trC]).predict_proba(X.iloc[te])[:, 1]
                     for m, e in base.items()) / wsum
    out = ~ring
    rows = [
        {"case": "A train all / test all", "n_test": len(y), "AUC": roc_auc_score(y, pA)},
        {"case": "B train all / test outside ring", "n_test": int(out.sum()), "AUC": roc_auc_score(y[out], pA[out])},
        {"case": "C train without ring / test outside ring", "n_test": int(out.sum()), "AUC": roc_auc_score(y[out], pC[out])},
    ]
    rho = spearmanr(pA[out], pC[out]).correlation
    R = pd.DataFrame(rows); R["AUC"] = R["AUC"].round(3)
    R.loc[len(R)] = {"case": "Spearman rho, B vs C predictions outside ring", "n_test": int(out.sum()), "AUC": round(rho, 3)}
    R.to_csv(OUT / "round2_drawdown_sensitivity.csv", index=False)
    print("\n" + R.to_string(index=False))
    print("\nREAD: |AUC_C - AUC_B| < 0.01 and rho > 0.95 -> the ring does not shape the model outside the"
          "\n      reservoir; keep Stage I and report this as a caveat. Larger differences -> retrain"
          "\n      Stage I without the ring and re-run the downstream steps.")


if __name__ == "__main__":
    main()
