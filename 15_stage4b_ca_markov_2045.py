#!/usr/bin/env python3
# ============================================================
# Paper #5 - Stage 4b: Markov-CA projection of LULC to 2045
#
#   Input : GEE_LPP_CMIP6/lulc/DW_lulc_2016.tif, _2020.tif, _2024.tif
#           (ESA WorldCover classes, same grid as predictor stack)
#   Output: GEE_LPP_CMIP6/lulc/lulc_2045_camarkov.tif  (WorldCover classes)
#           GEE_LPP_CMIP6/lulc/transition_matrix.csv
#           GEE_LPP_CMIP6/lulc/class_proportions.csv
#
#   Method:
#     1. Markov transition matrix from the two observed intervals (averaged)
#     2. Project class QUANTITIES to 2045 (matrix power, 4-yr step)
#     3. Proximity-based cellular allocation: gaining classes grow into losing
#        cells where neighbourhood agglomeration suitability is highest (the CA part)
#
#   Feeds 17_factorial_attribution.py (bounded climate-vs-LULC attribution).
# Paste into ONE Colab cell and Run.
# ============================================================

# --- Colab bootstrap (force-remount so freshly-exported files are visible) ---
try:
    import google.colab  # noqa: F401
    from google.colab import drive
    drive.flush_and_unmount()
    drive.mount("/content/drive", force_remount=True)
    for _p in ("rasterio",):
        try:
            __import__(_p)
        except ImportError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", _p], check=False)
except ImportError:
    pass

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DRIVE   = "/content/drive/MyDrive/GEE_LPP_CMIP6"      # <<< EDIT if elsewhere
YEARS   = [2016, 2020, 2024]
TARGET  = 2045
STEP_YR = 4                                           # interval length of each observed step
NEIGH   = 3                                           # CA neighbourhood size (pixels)

LULC_DIR = None                                       # resolved at runtime (case-insensitive)


def resolve_lulc_dir():
    """Find the LULC folder (any case) under DRIVE and verify the 3 input files."""
    import glob
    base = Path(DRIVE)
    if not base.exists():
        raise SystemExit(f"DRIVE not found: {DRIVE}\n"
                         f"  -> check the folder name, or Runtime > Restart session.")
    # try common case variants, then any folder whose name matches 'lulc'
    candidates = [base / "LULC", base / "lulc", base / "Lulc"]
    candidates += [Path(p) for p in glob.glob(str(base / "*"))
                   if Path(p).is_dir() and Path(p).name.lower() == "lulc"]
    for d in candidates:
        if d.exists():
            found = sorted(glob.glob(str(d / "DW_lulc_*.tif")))
            if len(found) >= 3:
                print(f"Using LULC dir: {d}")
                for f in found:
                    print(f"  found: {Path(f).name}")
                return d
    # nothing valid -> print what IS there to help debug
    print("Could not find the 3 DW_lulc_*.tif files. Contents of", base, ":")
    for p in sorted(glob.glob(str(base / "*"))):
        print("  ", p)
    for d in candidates:
        if d.exists():
            print(f"  (in {d.name}/):", [Path(x).name for x in glob.glob(str(d / "*"))])
    raise SystemExit("Fix: confirm the 3 files are in GEE_LPP_CMIP6/LULC/ and the GEE "
                     "export tasks finished, then re-run.")


def read_map(year):
    with rasterio.open(LULC_DIR / f"DW_lulc_{year}.tif") as s:
        return s.read(1), s.profile


def transition_matrix(a, b, classes, valid):
    """Row-normalised transition probabilities from map a -> map b."""
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)))
    av, bv = a[valid], b[valid]
    for ca in classes:
        m = av == ca
        if m.sum() == 0:
            M[idx[ca], idx[ca]] = 1.0
            continue
        for cb in classes:
            M[idx[ca], idx[cb]] = np.sum(bv[m] == cb) / m.sum()
    return M


def main():
    global LULC_DIR
    LULC_DIR = resolve_lulc_dir()

    maps, profs = {}, {}
    for y in YEARS:
        maps[y], profs[y] = read_map(y)
    prof = profs[YEARS[-1]]
    nod = prof.get("nodata")
    if nod is None:                       # DW export may not set a nodata value
        nod = 0

    stack = np.stack([maps[y] for y in YEARS])
    valid = np.all(stack > 0, axis=0) & np.all(np.isfinite(stack), axis=0)
    classes = sorted(int(c) for c in np.unique(stack[:, valid]) if c > 0)
    idx = {c: i for i, c in enumerate(classes)}
    print(f"Classes (WorldCover): {classes} | valid px: {int(valid.sum())}")

    # ---- 1. transition matrix (average of the two intervals) ----
    M1 = transition_matrix(maps[2016], maps[2020], classes, valid)
    M2 = transition_matrix(maps[2020], maps[2024], classes, valid)
    M = (M1 + M2) / 2
    pd.DataFrame(M, index=classes, columns=classes).round(4).to_csv(LULC_DIR / "transition_matrix.csv")
    print("Transition matrix (avg) saved. Diagonal (persistence):",
          {c: round(M[idx[c], idx[c]], 2) for c in classes})

    # ---- 2. Markov quantity projection to TARGET ----
    base = maps[YEARS[-1]]
    cur_state = np.array([np.sum(base[valid] == c) for c in classes], dtype=float)
    cur_state /= cur_state.sum()
    steps = round((TARGET - YEARS[-1]) / STEP_YR)
    Mn = np.linalg.matrix_power(M, steps)
    fut_state = cur_state @ Mn
    N = int(valid.sum())
    target_counts = {c: int(round(fut_state[idx[c]] * N)) for c in classes}
    cur_counts = {c: int(np.sum(base[valid] == c)) for c in classes}

    prop = pd.DataFrame({"class": classes,
                         "prop_2024": [cur_counts[c] / N for c in classes],
                         f"prop_{TARGET}": [target_counts[c] / N for c in classes]}).round(4)
    prop.to_csv(LULC_DIR / "class_proportions.csv", index=False)
    print(f"\nClass proportions ({steps} steps -> ~{YEARS[-1]+steps*STEP_YR}):")
    print(prop.to_string(index=False))

    # ---- 3. proximity-based cellular allocation ----
    delta = {c: target_counts[c] - cur_counts[c] for c in classes}
    gaining = sorted([c for c in classes if delta[c] > 0], key=lambda c: -delta[c])
    losing = [c for c in classes if delta[c] < 0]

    new = base.copy()
    converted = np.zeros(base.shape, dtype=bool)
    for g in gaining:
        need = delta[g]
        if need <= 0:
            continue
        suit = uniform_filter((base == g).astype(float), size=NEIGH)   # agglomeration suitability
        cand = valid & np.isin(base, losing) & (~converted)
        ij = np.argwhere(cand)
        if len(ij) == 0:
            continue
        scores = suit[cand]
        order = np.argsort(scores)[::-1][:need]                        # highest suitability first
        sel = ij[order]
        new[sel[:, 0], sel[:, 1]] = g
        converted[sel[:, 0], sel[:, 1]] = True

    new = np.where(valid, new, nod).astype("int16")
    pr = prof.copy(); pr.update(count=1, dtype="int16", nodata=nod, compress="lzw")
    with rasterio.open(LULC_DIR / "lulc_2045_camarkov.tif", "w", **pr) as d:
        d.write(new, 1)

    # verify allocation hit targets
    ach = {c: int(np.sum((new == c) & valid)) for c in classes}
    print("\nAllocation check (target vs achieved):")
    for c in classes:
        print(f"  class {c}: target {target_counts[c]:>7d}  achieved {ach[c]:>7d}")
    print(f"\nDone -> {LULC_DIR}/lulc_2045_camarkov.tif")


if __name__ == "__main__":
    main()
