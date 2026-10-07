#!/usr/bin/env python3
# ============================================================
# Paper #5 - CMIP6 flood susceptibility, Lam Phra Phloeng
# STAGE 2 (Python)  -  PATH A, Stage I: spatial susceptibility model
#
#   Input : GEE_LPP_CMIP6/LPP_paper5_samples_static.csv  (from GEE Stage 1)
#   Output: GEE_LPP_CMIP6/stage2_outputs/  (metrics, SHAP, trained models)
#   Models: RF / SVM / ANN / XGBoost  (no CNN) -> weighted + stacking ensemble
#   Honest evaluation: spatial-block CV vs random CV (both reported)
#
# Just paste this whole file into ONE Colab cell and press Run.
# ============================================================

# ------------------------------------------------------------
# COLAB BOOTSTRAP  (mounts Drive + installs shap; harmless outside Colab)
# ------------------------------------------------------------
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
        import shap  # noqa: F401
    except ImportError:
        print("Installing shap ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "shap"], check=False)

# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, average_precision_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             roc_auc_score)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIG  -  everything reads/writes this ONE Drive folder
# ------------------------------------------------------------
DRIVE_FOLDER = "/content/drive/MyDrive/GEE_LPP_CMIP6"   # <<< EDIT if your folder is elsewhere
CSV_PATH   = f"{DRIVE_FOLDER}/LPP_paper5_samples_static.csv"
OUT_DIR    = Path(DRIVE_FOLDER) / "stage2_outputs"
MODEL_DIR  = OUT_DIR / "models"

SEED       = 42
N_OUTER    = 5          # outer CV folds (generalization estimate)
N_INNER    = 3          # inner CV folds (hyperparameter tuning)
BLOCK_KM   = 3.0        # spatial block size for spatial-block CV
LAT_REF    = 14.7       # basin latitude, for km->degree conversion
VIF_THRESH = 5.0
SHAP_ON    = True       # set False to skip SHAP

CONTINUOUS = ["elevation", "slope", "curvature", "hand", "twi", "spi",
              "dist_river", "drain_density"]
CATEGORICAL = ["soil", "lulc"]

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 1.  LOAD + PREPROCESS
# ------------------------------------------------------------
def preprocess(df):
    """aspect -> sin/cos ; one-hot soil & lulc ; return X, y, groups, feature lists."""
    df = df.copy()
    a = np.deg2rad(df["aspect"].values)          # circular transform (359 deg ~ 1 deg)
    df["aspect_sin"] = np.sin(a)
    df["aspect_cos"] = np.cos(a)

    dummies = pd.get_dummies(df[CATEGORICAL].astype("Int64").astype(str),
                             prefix=CATEGORICAL, dummy_na=False)
    cont = CONTINUOUS + ["aspect_sin", "aspect_cos"]

    X = pd.concat([df[cont], dummies.astype(float)], axis=1)
    y = df["label"].astype(int).values

    dlat = BLOCK_KM / 111.0                       # spatial block id from lon/lat (km->deg)
    dlon = BLOCK_KM / (111.0 * np.cos(np.deg2rad(LAT_REF)))
    blk = (np.floor(df["latitude"].values / dlat).astype(int).astype(str) + "_" +
           np.floor(df["longitude"].values / dlon).astype(int).astype(str))
    groups = pd.factorize(blk)[0]
    return X, y, groups, cont, list(dummies.columns)


# ------------------------------------------------------------
# 2.  VIF screening (continuous predictors only)
# ------------------------------------------------------------
def vif_prune(X, cont_cols, thresh=VIF_THRESH):
    keep, dropped = list(cont_cols), []
    while len(keep) > 1:
        vifs = [variance_inflation_factor(X[keep].values, i) for i in range(len(keep))]
        if max(vifs) <= thresh:
            break
        drop = keep[int(np.argmax(vifs))]
        dropped.append((drop, round(max(vifs), 2)))
        keep.remove(drop)
    vif_final = {keep[i]: round(variance_inflation_factor(X[keep].values, i), 2)
                 for i in range(len(keep))}
    return keep, dropped, vif_final


# ------------------------------------------------------------
# 3.  Frequency-ratio (FR) diagnostic for categorical factors
# ------------------------------------------------------------
def frequency_ratio(df, factor):
    tab = df.groupby(factor)["label"].agg(["size", "mean"])
    fr = (tab["mean"] / tab["mean"].mean()).round(3)
    return pd.DataFrame({"n": tab["size"], "flood_share": tab["mean"].round(3), "FR": fr})


# ------------------------------------------------------------
# 4.  Metrics
# ------------------------------------------------------------
def metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "TSS": sens + spec - 1,
        "F1": f1_score(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "Acc": accuracy_score(y_true, y_pred),
    }


# ------------------------------------------------------------
# 5.  Model zoo (estimator, hyperparameter grid)
# ------------------------------------------------------------
def model_zoo():
    return {
        "RF": (RandomForestClassifier(random_state=SEED, class_weight="balanced", n_jobs=-1),
               {"n_estimators": [300], "max_depth": [None, 10, 20], "max_features": ["sqrt", 0.5]}),
        "XGB": (XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=-1),
                {"n_estimators": [300], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}),
        "SVM": (Pipeline([("sc", StandardScaler()),
                          ("clf", SVC(probability=True, random_state=SEED, class_weight="balanced"))]),
                {"clf__C": [1, 10], "clf__gamma": ["scale"]}),
        "ANN": (Pipeline([("sc", StandardScaler()),
                          ("clf", MLPClassifier(random_state=SEED, max_iter=500, early_stopping=True))]),
                {"clf__hidden_layer_sizes": [(32,), (64, 32)], "clf__alpha": [1e-3, 1e-2]}),
    }


# ------------------------------------------------------------
# 6.  Out-of-fold predictions with nested tuning
# ------------------------------------------------------------
def oof_nested(est, grid, X, y, groups, mode):
    oof = np.zeros(len(y))
    if mode == "spatial":
        splits = GroupKFold(n_splits=N_OUTER).split(X, y, groups)
    else:
        splits = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED).split(X, y)
    for tr, te in splits:
        if mode == "spatial":
            gs = GridSearchCV(est, grid, cv=GroupKFold(n_splits=N_INNER),
                              scoring="roc_auc", n_jobs=-1)
            gs.fit(X.iloc[tr], y[tr], groups=groups[tr])
        else:
            gs = GridSearchCV(est, grid, cv=StratifiedKFold(N_INNER, shuffle=True, random_state=SEED),
                              scoring="roc_auc", n_jobs=-1)
            gs.fit(X.iloc[tr], y[tr])
        oof[te] = gs.predict_proba(X.iloc[te])[:, 1]
    return oof


# ------------------------------------------------------------
# 7.  MAIN
# ------------------------------------------------------------
def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples | class balance: {df['label'].value_counts().to_dict()}")

    X, y, groups, cont, dummy_cols = preprocess(df)
    print(f"Spatial blocks: {len(np.unique(groups))}")

    keep_cont, dropped, vif_final = vif_prune(X, cont)
    pd.DataFrame([{"predictor": k, "VIF": v} for k, v in vif_final.items()]) \
        .to_csv(OUT_DIR / "vif_report.csv", index=False)
    print(f"VIF kept: {keep_cont}")
    if dropped:
        print(f"VIF dropped: {dropped}")

    features = keep_cont + dummy_cols
    X = X[features]
    for f in CATEGORICAL:
        frequency_ratio(df, f).to_csv(OUT_DIR / f"fr_{f}.csv")

    zoo = model_zoo()
    rows, oof_spatial = [], {}
    for name, (est, grid) in zoo.items():
        for mode in ("random", "spatial"):
            oof = oof_nested(est, grid, X, y, groups, mode)
            m = metrics(y, oof); m.update({"model": name, "cv": mode})
            rows.append(m)
            if mode == "spatial":
                oof_spatial[name] = oof
            print(f"{name:4s} [{mode:7s}]  AUC={m['AUC']:.3f}  PR_AUC={m['PR_AUC']:.3f}  TSS={m['TSS']:.3f}")

    cv_table = pd.DataFrame(rows)[["model", "cv", "AUC", "PR_AUC", "TSS", "F1", "Kappa", "Acc"]]

    # Ensemble: weights from spatial-CV AUC above 0.5
    aucs = {n: roc_auc_score(y, p) for n, p in oof_spatial.items()}
    w = {n: max(a - 0.5, 0) for n, a in aucs.items()}
    tot = sum(w.values()) or 1.0
    w = {n: v / tot for n, v in w.items()}
    ens_oof = sum(w[n] * oof_spatial[n] for n in oof_spatial)
    ens_m = metrics(y, ens_oof); ens_m.update({"model": "ENSEMBLE_weighted", "cv": "spatial"})
    print(f"ENSEMBLE (weighted)  AUC={ens_m['AUC']:.3f}  PR_AUC={ens_m['PR_AUC']:.3f}  TSS={ens_m['TSS']:.3f}")
    print(f"  weights: { {n: round(v,3) for n,v in w.items()} }")

    # Stacking: meta-learner on OOF probabilities
    Z = np.column_stack([oof_spatial[n] for n in oof_spatial])
    stack_oof = np.zeros(len(y))
    for tr, te in GroupKFold(N_OUTER).split(Z, y, groups):
        meta = LogisticRegression(max_iter=1000).fit(Z[tr], y[tr])
        stack_oof[te] = meta.predict_proba(Z[te])[:, 1]
    stack_m = metrics(y, stack_oof); stack_m.update({"model": "ENSEMBLE_stacking", "cv": "spatial"})
    print(f"ENSEMBLE (stacking)  AUC={stack_m['AUC']:.3f}  PR_AUC={stack_m['PR_AUC']:.3f}  TSS={stack_m['TSS']:.3f}")

    pd.concat([cv_table, pd.DataFrame([ens_m, stack_m])[cv_table.columns]], ignore_index=True) \
        .to_csv(OUT_DIR / "metrics_cv.csv", index=False)
    json.dump(w, open(OUT_DIR / "ensemble_weights.json", "w"), indent=2)

    # Refit tuned models on ALL data (deployment)
    fitted = {}
    for name, (est, grid) in zoo.items():
        gs = GridSearchCV(est, grid, cv=GroupKFold(N_INNER), scoring="roc_auc", n_jobs=-1)
        gs.fit(X, y, groups=groups)
        fitted[name] = gs.best_estimator_
        joblib.dump(gs.best_estimator_, MODEL_DIR / f"{name}.joblib")
    joblib.dump({"features": features, "weights": w}, MODEL_DIR / "ensemble_config.joblib")
    print(f"Saved {len(fitted)} models + ensemble config to {MODEL_DIR}")

    # SHAP (tree model: mechanism check)
    if SHAP_ON:
        try:
            import shap, matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            sv = shap.TreeExplainer(fitted["XGB"]).shap_values(X)
            shap.summary_plot(sv, X, plot_type="bar", show=False)
            plt.tight_layout(); plt.savefig(OUT_DIR / "shap_bar.png", dpi=150); plt.close()
            shap.summary_plot(sv, X, show=False)
            plt.tight_layout(); plt.savefig(OUT_DIR / "shap_beeswarm.png", dpi=150); plt.close()
            print("SHAP: saved shap_bar.png, shap_beeswarm.png")
        except Exception as e:
            print(f"SHAP skipped: {e}")

    print(f"\nDone. Outputs in {OUT_DIR}/")
    print("Next (Stage 3): apply ensemble to the wall-to-wall static stack -> present S(x) map;")
    print("               then multiply by climate lambda (Stage II) for future risk.")


if __name__ == "__main__":
    main()
