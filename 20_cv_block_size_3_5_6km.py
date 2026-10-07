#!/usr/bin/env python3
# ============================================================
# ROUND 2 - spatial-block CV with 5 km blocks (answers Xia R2: autocorrelation
#           range 4-5 km > 3 km blocks) + metric clean-up (kappa == TSS issue)
#
#   * Re-runs spatial-block CV for EVERY model in the ensemble with the SAME
#     hyper-parameters (sklearn.clone of the saved models), at 3 km and 5 km.
#   * Reports AUC and TSS (prevalence-independent). Cohen's kappa is NOT kept:
#     on a balanced 1:1 test set it is algebraically ~= TSS, so it adds nothing.
#   * Also reports the weighted soft-voting ensemble (same weights as Stage 2).
#
#   Inputs (GEE_LPP_CMIP6/):
#     LPP_paper5_samples_static.csv
#     stage2_outputs/models/*.joblib, ensemble_config.joblib
#   Output: revision_outputs/round2_block_cv_3km_vs_5km.csv
# ============================================================
try:
    import google.colab  # noqa: F401
    from google.colab import drive
    drive.flush_and_unmount(); drive.mount("/content/drive", force_remount=True)
    DRIVE = "/content/drive/MyDrive/GEE_LPP_CMIP6"
except ImportError:
    DRIVE = "."

import warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
warnings.filterwarnings("ignore")

SAMPLES = f"{DRIVE}/LPP_paper5_samples_static.csv"
MODELS = Path(f"{DRIVE}/stage2_outputs/models")
OUT = Path(f"{DRIVE}/revision_outputs"); OUT.mkdir(parents=True, exist_ok=True)
CAT = ["soil", "lulc"]
CONT = ["elevation", "slope", "curvature", "hand", "twi", "spi", "dist_river", "drain_density",
        "aspect_sin", "aspect_cos"]
BLOCKS_KM = [3, 5, 6]
N_FOLDS = 5


def features(d, feats):
    d = d.copy(); a = np.deg2rad(d["aspect"].values)
    d["aspect_sin"], d["aspect_cos"] = np.sin(a), np.cos(a)
    dm = pd.get_dummies(d[CAT].astype("Int64").astype(str), prefix=CAT, dummy_na=False)
    return pd.concat([d[CONT], dm.astype(float)], axis=1).reindex(columns=feats, fill_value=0.0)


def blocks(d, km):
    by = np.floor(d.latitude / (km / 111.0)).astype(int).astype(str)
    bx = np.floor(d.longitude / (km / (111.0 * np.cos(np.deg2rad(14.6))))).astype(int).astype(str)
    return pd.factorize(by + "_" + bx)[0]


def tss(y, p, thr=0.5):
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1]).ravel()
    return tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1


def main():
    d = pd.read_csv(SAMPLES); y = d["label"].astype(int).values
    cfg = joblib.load(MODELS / "ensemble_config.joblib"); feats, w = cfg["features"], cfg["weights"]
    base = {m: joblib.load(MODELS / f"{m}.joblib") for m in w if (MODELS / f"{m}.joblib").exists()}
    X = features(d, feats)
    print(f"{len(d)} samples | models: {list(base)} | weights: {w}")

    rows = []
    for km in BLOCKS_KM:
        g = blocks(d, km)
        oof = {m: np.zeros(len(y)) for m in base}
        for tr, te in GroupKFold(N_FOLDS).split(X, y, g):
            for m, est in base.items():
                oof[m][te] = clone(est).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        wsum = sum(w[m] for m in base)
        oof["Ensemble"] = sum(w[m] * oof[m] for m in base) / wsum
        for m, p in oof.items():
            rows.append({"block_km": km, "n_blocks": int(g.max() + 1), "model": m,
                         "AUC": round(roc_auc_score(y, p), 3), "TSS": round(tss(y, p), 3)})
            print(f"  {km} km blocks ({g.max()+1} blocks)  {m:9s} AUC={rows[-1]['AUC']:.3f}  TSS={rows[-1]['TSS']:.3f}")
    R = pd.DataFrame(rows)
    piv = R.pivot(index="model", columns="block_km", values="AUC")
    piv.columns = [f"AUC_{c}km" for c in piv.columns]; piv["dAUC_5_minus_3"] = piv["AUC_5km"] - piv["AUC_3km"]; piv["dAUC_6_minus_3"] = piv["AUC_6km"] - piv["AUC_3km"]
    R.to_csv(OUT / "round2_block_cv_3km_vs_5km.csv", index=False)
    print("\n" + piv.round(3).to_string())
    print("\nREAD: if dAUC is small (|d| < 0.02), the 3 km result was not inflated by autocorrelation;"
          "\n      report the 5 km numbers as the main Table 2 values either way.")


if __name__ == "__main__":
    main()
