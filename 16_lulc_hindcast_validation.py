#!/usr/bin/env python3
# ============================================================
# REVISION - CA-Markov hindcast validation (answers Reviewer Mohanty #4)
#   Train the Markov transition on 2016->2020, PREDICT 2024, compare to OBSERVED
#   2024 -> overall accuracy, Cohen's Kappa, and Figure of Merit (FoM) on the
#   pixels that actually changed 2020->2024.
#
#   Inputs : GEE_LPP_CMIP6/LULC/DW_lulc_2016.tif, _2020.tif, _2024.tif
#   Output : GEE_LPP_CMIP6/LULC/hindcast_validation.csv
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

import glob, warnings
from pathlib import Path
import numpy as np, pandas as pd, rasterio
from scipy.ndimage import uniform_filter
from sklearn.metrics import cohen_kappa_score, accuracy_score
warnings.filterwarnings("ignore")

LULC = None
for base in [f"{DRIVE}/LULC", f"{DRIVE}/lulc"]:
    if glob.glob(f"{base}/DW_lulc_2016*.tif"): LULC = Path(base); break
OUT = LULC
NEIGH = 3


def rd(y):
    f = glob.glob(f"{LULC}/DW_lulc_{y}*.tif")[0]
    with rasterio.open(f) as s: return s.read(1)


def transition(a, b, classes, valid):
    idx = {c:i for i,c in enumerate(classes)}; M=np.zeros((len(classes),)*2)
    av,bv = a[valid], b[valid]
    for ca in classes:
        m=av==ca
        if m.sum()==0: M[idx[ca],idx[ca]]=1; continue
        for cb in classes: M[idx[ca],idx[cb]]=np.sum(bv[m]==cb)/m.sum()
    return M, idx


def allocate(base, classes, idx, target_counts, valid):
    delta={c:target_counts[c]-int(np.sum(base[valid]==c)) for c in classes}
    gain=sorted([c for c in classes if delta[c]>0], key=lambda c:-delta[c])
    lose=[c for c in classes if delta[c]<0]; new=base.copy(); conv=np.zeros(base.shape,bool)
    for g in gain:
        need=delta[g]
        if need<=0: continue
        suit=uniform_filter((base==g).astype(float),size=NEIGH)
        cand=valid & np.isin(base,lose) & ~conv; ij=np.argwhere(cand)
        if len(ij)==0: continue
        order=np.argsort(suit[cand])[::-1][:need]; sel=ij[order]
        new[sel[:,0],sel[:,1]]=g; conv[sel[:,0],sel[:,1]]=True
    return new


def main():
    m16,m20,m24 = rd(2016),rd(2020),rd(2024)
    valid = (m16>0)&(m20>0)&(m24>0)
    classes=sorted(int(c) for c in np.unique(np.stack([m16,m20,m24])[:,valid]) if c>0)
    print(f"classes {classes} | valid {int(valid.sum())}")

    # train 2016->2020, predict 2024 (2 steps of 4 yr -> but 2020->2024 is 1 step)
    M,idx = transition(m16,m20,classes,valid)
    cur=np.array([np.sum(m20[valid]==c) for c in classes],float); cur/=cur.sum()
    fut=cur@M; N=int(valid.sum())
    tgt={c:int(round(fut[idx[c]]*N)) for c in classes}
    pred24=allocate(m20,classes,idx,tgt,valid)

    obs=m24[valid]; prd=pred24[valid]
    acc=accuracy_score(obs,prd); kap=cohen_kappa_score(obs,prd)
    # Figure of Merit on observed changes 2020->2024
    changed = valid & (m20!=m24)
    hit = changed & (pred24==m24)                       # predicted the right new class
    miss = changed & (pred24==m20)                      # predicted persistence
    wrong = changed & (pred24!=m24) & (pred24!=m20)     # predicted wrong change
    falsealarm = valid & (m20==m24) & (pred24!=m20)     # predicted change where none
    B=int(hit.sum()); A=int(miss.sum()); C=int(wrong.sum()); D=int(falsealarm.sum())
    FoM = B/(A+B+C+D) if (A+B+C+D) else np.nan

    res=pd.DataFrame([{"overall_accuracy":round(acc,3),"kappa":round(kap,3),
        "FoM":round(FoM,3),"hits_B":B,"misses_A":A,"wrong_hits_C":C,"false_alarms_D":D,
        "observed_changed_px":int(changed.sum())}])
    res.to_csv(OUT/"hindcast_validation.csv",index=False)
    print(res.to_string(index=False))
    print(f"\nInterpretation: Kappa>0.6 = substantial agreement; FoM for LUC models is "
          f"typically 0.1-0.3 (Pontius et al.). Report both honestly.")
    print(f"Done -> {OUT}/hindcast_validation.csv")


if __name__ == "__main__":
    main()
