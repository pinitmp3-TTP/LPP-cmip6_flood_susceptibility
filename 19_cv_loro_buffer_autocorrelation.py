#!/usr/bin/env python3
# ============================================================
# REVISION - validation hardening (answers Reviewer Xia #3)
#   1. Spatial autocorrelation RANGE of flood occurrence (justifies 3 km blocks)
#   2. Leave-one-region-out CV (nearest-station zones = sub-regions)
#   3. Buffer sensitivity: AUC vs absence buffer (needs sample CSVs at 0/150/500 m;
#      see 18_GEE_buffer_samples.js. If only the 150 m set is present it
#      still runs items 1-2.)
#
#   Inputs (GEE_LPP_CMIP6/):
#     LPP_paper5_samples_static.csv               (buffer 150 m, the main set)
#     LPP_paper5_samples_buffer0.csv  (optional)
#     LPP_paper5_samples_buffer500.csv (optional)
#     per_station_lambda.csv  OR station coords below (for region labels)
#   Output: revision_outputs/
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
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

OUT = Path(f"{DRIVE}/revision_outputs"); OUT.mkdir(parents=True, exist_ok=True)
MAIN = f"{DRIVE}/LPP_paper5_samples_static.csv"
BUFFERS = {"0m": f"{DRIVE}/LPP_paper5_samples_buffer0.csv",
           "150m": MAIN,
           "500m": f"{DRIVE}/LPP_paper5_samples_buffer500.csv"}
# station coords for region labels (nearest-station -> leave-one-region-out)
STATIONS = {250751:(14.492778,101.683889), 250771:(14.355833,101.725),
            250781:(14.476389,101.651111), 250541:(14.861944,101.566111),
            250651:(14.592778,101.478056), 250791:(14.942222,102.001111)}
CONT = ["elevation","slope","curvature","hand","twi","spi","dist_river","drain_density"]
CAT = ["soil","lulc"]


def prep(df):
    d = df.copy(); a = np.deg2rad(d["aspect"]); d["aspect_sin"]=np.sin(a); d["aspect_cos"]=np.cos(a)
    dm = pd.get_dummies(d[CAT].astype("Int64").astype(str), prefix=CAT, dummy_na=False)
    X = pd.concat([d[CONT+["aspect_sin","aspect_cos"]], dm.astype(float)], axis=1)
    return X, d["label"].astype(int).values


def nearest_station(lat, lon):
    best, bid = 1e9, 0
    for sid,(la,lo) in STATIONS.items():
        dd = (lat-la)**2 + (lon-lo)**2
        if dd < best: best, bid = dd, sid
    return bid


def morans_range(df, maxlag_km=8, step=0.5):
    """Empirical semivariogram-style: correlation of label vs separation distance."""
    s = df.sample(min(len(df), 1500), random_state=0)
    lat = s["latitude"].values; lon = s["longitude"].values; y = s["label"].values - s["label"].mean()
    # pairwise distances (km) on a subsample
    la = lat[:,None]; lo = lon[:,None]
    dkm = np.sqrt(((la-la.T)*111.0)**2 + ((lo-lo.T)*111.0*np.cos(np.deg2rad(14.6)))**2)
    prod = np.outer(y, y)
    bins = np.arange(0, maxlag_km+step, step); out=[]
    for i in range(len(bins)-1):
        m = (dkm>=bins[i]) & (dkm<bins[i+1]) & ~np.eye(len(s),dtype=bool)
        if m.sum()>50: out.append((round((bins[i]+bins[i+1])/2,2), round(float(prod[m].mean()),4)))
    return pd.DataFrame(out, columns=["lag_km","covariance"])


def main():
    df = pd.read_csv(MAIN)
    print(f"Main sample: {len(df)} rows")

    # --- 1. spatial autocorrelation range ---
    vg = morans_range(df); vg.to_csv(OUT/"autocorrelation_range.csv", index=False)
    # range ~ where covariance drops to ~0
    rng = vg[vg.covariance <= 0].lag_km.min() if (vg.covariance<=0).any() else np.nan
    print(f"\n[1] Spatial autocorrelation ~ decays to zero near {rng} km "
          f"(supports >= 3 km blocks if range <= 3 km).")
    print(vg.head(8).to_string(index=False))

    # --- 2. leave-one-region-out CV (nearest-station zones) ---
    df["region"] = [nearest_station(la,lo) for la,lo in zip(df.latitude, df.longitude)]
    X,y = prep(df); groups = df["region"].values
    print(f"\n[2] Leave-one-region-out CV over {df.region.nunique()} station zones:")
    aucs=[]
    for reg in sorted(df.region.unique()):
        te = groups==reg; tr=~te
        if te.sum()<20 or len(np.unique(y[tr]))<2: continue
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42).fit(X[tr],y[tr])
        if len(np.unique(y[te]))<2:
            continue
        au = roc_auc_score(y[te], m.predict_proba(X[te])[:,1]); aucs.append((reg,int(te.sum()),round(au,3)))
        print(f"    hold-out zone {reg}: n={int(te.sum())}  AUC={au:.3f}")
    pd.DataFrame(aucs,columns=["holdout_zone","n","AUC"]).to_csv(OUT/"loro_cv.csv",index=False)
    if aucs: print(f"    mean LORO AUC = {np.mean([a[2] for a in aucs]):.3f}")

    # --- 3. buffer sensitivity ---
    print("\n[3] Buffer sensitivity (spatial-block AUC vs absence buffer):")
    rows=[]
    for blab,path in BUFFERS.items():
        if not Path(path).exists():
            print(f"    {blab}: file not found, skipped"); continue
        d=pd.read_csv(path); Xb,yb=prep(d)
        # 3 km blocks
        blk=(np.floor(d.latitude/(3/111)).astype(int).astype(str)+"_"+
             np.floor(d.longitude/(3/(111*np.cos(np.deg2rad(14.6))))).astype(int).astype(str))
        g=pd.factorize(blk)[0]; oof=np.zeros(len(yb))
        for tr,te in GroupKFold(5).split(Xb,yb,g):
            mm=RandomForestClassifier(n_estimators=300,class_weight="balanced",n_jobs=-1,random_state=42).fit(Xb.iloc[tr],yb[tr])
            oof[te]=mm.predict_proba(Xb.iloc[te])[:,1]
        au=roc_auc_score(yb,oof); rows.append({"buffer":blab,"n":len(d),"spatial_AUC":round(au,3)})
        print(f"    {blab}: n={len(d)}  spatial-CV AUC={au:.3f}")
    pd.DataFrame(rows).to_csv(OUT/"buffer_sensitivity.csv",index=False)
    print(f"\nDone -> {OUT}/")


if __name__ == "__main__":
    main()
