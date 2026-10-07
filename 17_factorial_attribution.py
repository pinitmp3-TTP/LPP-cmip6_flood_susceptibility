#!/usr/bin/env python3
# ============================================================
# REVISION - fair factorial attribution (answers Reviewer Mohanty #1-2)
#
#   Fixes vs the first-submission attribution (superseded Stage 4c):
#     1. ALL 3 SSP (not only SSP5-8.5), each with ITS OWN ensemble-mean lambda
#     2. Attribution statistics restricted to pixels where LULC CHANGED
#        (removes the "climate dominates 93.5%" design artefact)
#     3. LULC effect reported BY TRANSITION TYPE, with a variant that EXCLUDES
#        built-up expansion (which carries the urban under-detection artefact)
#     4. Reports climate vs LULC magnitude only on the comparable (changed) area
#
#   Inputs (GEE_LPP_CMIP6/):
#     LPP_paper5_predictor_stack_static.tif
#     stage2_outputs/models/*.joblib, ensemble_config.joblib
#     stage2_outputs/susceptibility_breaks.json
#     LULC/DW_lulc_2024.tif, LULC/lulc_2045_camarkov.tif
#     stage2_outputs/futureII/lambda_table.csv   (per-GCM lambda; we take ens-mean)
#   Output: stage2_outputs/attribution_fair/
# ============================================================
try:
    import google.colab  # noqa: F401
    from google.colab import drive
    drive.flush_and_unmount(); drive.mount("/content/drive", force_remount=True)
    for _p in ("rasterio",):
        try: __import__(_p)
        except ImportError:
            import subprocess, sys; subprocess.run([sys.executable,"-m","pip","install","-q",_p])
    DRIVE = "/content/drive/MyDrive/GEE_LPP_CMIP6"
except ImportError:
    DRIVE = "."

import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, rasterio, joblib
from rasterio.warp import reproject, Resampling
warnings.filterwarnings("ignore")

STACK  = f"{DRIVE}/LPP_paper5_predictor_stack_static.tif"
MODELS = Path(f"{DRIVE}/stage2_outputs/models")
BREAKS = f"{DRIVE}/stage2_outputs/susceptibility_breaks.json"
LULC24 = f"{DRIVE}/LULC/DW_lulc_2024.tif"
LULC45 = f"{DRIVE}/LULC/lulc_2045_camarkov.tif"
LAMTAB = f"{DRIVE}/stage2_outputs/futureII/lambda_table.csv"
# ROUND 3: tau from the prevalence recalibration step (10) -> report on the recalibrated scale too
BREAKS_R = f"{DRIVE}/stage2_outputs/futureII_spatial/susceptibility_breaks_recalibrated.json"
OUT    = Path(f"{DRIVE}/stage2_outputs/attribution_fair"); OUT.mkdir(parents=True, exist_ok=True)

SSPS = ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]
HORIZON = "near_2036-2065"
BUILTUP = 50   # ESA WorldCover built-up class (carries the urban under-detection artefact)
BAND_NAMES = ["elevation","slope","aspect","curvature","hand","twi","spi","dist_river","drain_density","soil","lulc"]
CATEGORICAL = ["soil","lulc"]; EPS = 1e-6


def odds_scale(S, lam):
    s = np.clip(S, EPS, 1-EPS); o = s/(1-s)*lam; return o/(1+o)

def recal(S, tau):
    """Prior (prevalence) correction of balanced-sample probabilities (King & Zeng 2001)."""
    s = np.clip(S, EPS, 1-EPS); o = s/(1-s)*(tau/(1-tau)); return o/(1+o)

def build_features(dfv):
    d = dfv.copy(); a = np.deg2rad(d["aspect"].values)
    d["aspect_sin"], d["aspect_cos"] = np.sin(a), np.cos(a)
    dm = pd.get_dummies(d[CATEGORICAL].astype("Int64").astype(str), prefix=CATEGORICAL, dummy_na=False)
    cont = ["elevation","slope","curvature","hand","twi","spi","dist_river","drain_density","aspect_sin","aspect_cos"]
    return pd.concat([d[cont], dm.astype(float)], axis=1)

def aligned(path, t, crs, H, W):
    with rasterio.open(path) as s:
        if s.shape==(H,W) and s.transform==t and s.crs==crs: return s.read(1).astype("float64")
        d = np.zeros((H,W),"float64")
        reproject(rasterio.band(s,1), d, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=t, dst_crs=crs, resampling=Resampling.nearest); return d


def main():
    with rasterio.open(STACK) as src:
        arr = src.read().astype("float64"); prof = src.profile
        H,W = src.height, src.width; t, crs, nod = src.transform, src.crs, src.nodata
    df = pd.DataFrame(arr.reshape(len(BAND_NAMES),-1).T, columns=BAND_NAMES)
    valid = np.isfinite(df.values).all(axis=1)
    if nod is not None and np.isfinite(nod): valid &= (df.values != nod).all(axis=1)

    cfg = joblib.load(MODELS/"ensemble_config.joblib"); feats, w = cfg["features"], cfg["weights"]
    mdl = {m: joblib.load(MODELS/f"{m}.joblib") for m in w}
    lam_df = pd.read_csv(LAMTAB)
    l24 = aligned(LULC24, t, crs, H, W).reshape(-1)
    l45 = aligned(LULC45, t, crs, H, W).reshape(-1)
    changed = valid & (l24 != l45)                              # <-- KEY: only changed pixels
    builtup_gain = changed & (l45 == BUILTUP) & (l24 != BUILTUP)
    print(f"valid px {int(valid.sum())} | LULC-changed px {int(changed.sum())} "
          f"({100*changed.sum()/valid.sum():.1f}% of domain) | built-up gain {int(builtup_gain.sum())}")

    def predict(lulc_vec):
        d = df.copy(); d["lulc"] = lulc_vec
        X = build_features(d[valid]).reindex(columns=feats, fill_value=0.0)
        p = sum(w[m]*mdl[m].predict_proba(X)[:,1] for m in w)
        out = np.full(H*W, np.nan); out[valid] = p; return out

    S_ref  = predict(l24)              # present climate, 2024 LULC
    S_lulc = predict(l45)              # present climate, 2045 LULC
    try:
        tau = float(json.load(open(BREAKS_R))["tau"])
    except (FileNotFoundError, KeyError):
        raise SystemExit(f"tau not found in {BREAKS_R} - run 10_stageII_d_spatial_lambda_recalibration.py first.")
    R_ref, R_lulc = recal(S_ref, tau), recal(S_lulc, tau)   # recalibrated scale (as Section 3.4)
    print(f"tau = {tau:.4f} (recalibrated-scale results written alongside)")
    recs_r = []

    recs = []
    for ssp in SSPS:
        lam = float(lam_df[(lam_df.ssp==ssp)&(lam_df.horizon==HORIZON)]["lambda"].mean())
        S_clim = odds_scale(S_ref,  lam)
        S_comb = odds_scale(S_lulc, lam)
        dClim, dLULC = S_clim - S_ref, S_lulc - S_ref
        dInt = (S_comb - S_ref) - dClim - dLULC

        # write per-component rasters for SSP5-8.5 (inputs for Figure 8)
        if ssp == "ssp5_8_5":
            def _w(flat, name):
                pr = prof.copy(); pr.update(count=1, dtype="float32", nodata=np.nan, compress="lzw")
                img = np.where(valid, flat, np.nan).reshape(H, W).astype("float32")
                with rasterio.open(OUT / name, "w", **pr) as d:
                    d.write(img, 1)
            _w(dClim, "attr_climate.tif"); _w(dLULC, "attr_lulc.tif")
            _w(dInt, "attr_interaction.tif"); _w(dClim + dLULC + dInt, "attr_total.tif")

        # compare ONLY on changed pixels (fair); and a built-up-excluded variant
        ch = changed; nb = changed & ~builtup_gain
        def mabs(x, mask): return float(np.nanmean(np.abs(x[mask]))) if mask.sum() else np.nan
        recs.append({"ssp": ssp, "lambda": round(lam,3),
            "clim_absdS_changed": round(mabs(dClim, ch), 5),
            "lulc_absdS_changed": round(mabs(dLULC, ch), 5),
            "lulc_absdS_noBuiltup": round(mabs(dLULC, nb), 5),
            "ratio_lulc_over_clim_changed": round(mabs(dLULC,ch)/max(mabs(dClim,ch),1e-9), 2),
            "ratio_noBuiltup": round(mabs(dLULC,nb)/max(mabs(dClim,nb),1e-9), 2)})
        # same decomposition on the RECALIBRATED scale: recalibrate first, then scale the odds
        rC, rL = odds_scale(R_ref, lam) - R_ref, R_lulc - R_ref
        rI = (odds_scale(R_lulc, lam) - R_ref) - rC - rL
        recs_r.append({"ssp": ssp, "lambda": round(lam,3),
            "clim_absdS_changed": round(mabs(rC, ch), 6),
            "lulc_absdS_changed": round(mabs(rL, ch), 6),
            "lulc_absdS_noBuiltup": round(mabs(rL, nb), 6),
            "interaction_absdS_changed": round(mabs(rI, ch), 6),
            "ratio_lulc_over_clim_changed": round(mabs(rL,ch)/max(mabs(rC,ch),1e-12), 2),
            "ratio_noBuiltup": round(mabs(rL,nb)/max(mabs(rC,nb),1e-12), 2)})
    tab = pd.DataFrame(recs); tab.to_csv(OUT/"fair_attribution_summary.csv", index=False)
    tab_r = pd.DataFrame(recs_r); tab_r.to_csv(OUT/"fair_attribution_summary_recalibrated.csv", index=False)
    print("\n=== same, RECALIBRATED scale ===")
    print(tab_r.to_string(index=False))
    print("\n=== fair attribution (statistics on LULC-CHANGED pixels only) ===")
    print(tab.to_string(index=False))

    # LULC effect by transition type (SSP5-8.5), present climate
    rows=[]
    for frm in np.unique(l24[changed]):
        for to in np.unique(l45[changed]):
            m = changed & (l24==frm) & (l45==to)
            if m.sum() >= 20:
                rows.append({"from": int(frm), "to": int(to), "n_px": int(m.sum()),
                             "mean_dLULC": round(float(np.nanmean((S_lulc-S_ref)[m])), 4),
                             "mean_dLULC_recal": round(float(np.nanmean((R_lulc-R_ref)[m])), 5)})
    tt = pd.DataFrame(rows).sort_values("n_px", ascending=False)
    tt.to_csv(OUT/"lulc_effect_by_transition.csv", index=False)
    print("\n=== LULC effect by transition (top rows) ===")
    print(tt.head(10).to_string(index=False))
    print(f"\nNOTE: transitions INTO built-up ({BUILTUP}) carry the urban under-detection artefact; "
          "report them separately and exclude from the 'dominant driver' claim.")
    print(f"Done -> {OUT}/")


if __name__ == "__main__":
    main()
