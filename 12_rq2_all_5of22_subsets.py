#!/usr/bin/env python3
# ============================================================
# ROUND 2 - Is the screened-5 lambda unusually low? (answers Wright R2)
#
#   Wright: "a lower signal is not automatically a more credible one".
#   Test: compare the screened-5 ensemble-mean lambda with EVERY possible
#   5-model subset of the 22 models (C(22,5) = 26,334 subsets, enumerated
#   exactly - no random sampling needed). If the screened-5 value sits in the
#   lower tail (e.g. < 5th percentile), the screening effect is statistically
#   distinguishable from "any 5 models". If it sits mid-distribution, the
#   screened-vs-all gap is just a small-ensemble effect.
#
#   Input : indices_output/indices_annual_long.csv  (same as 11_rq2_screening_test.py)
#   Output: revision_outputs/rq2_bootstrap_5of22.csv
#           revision_outputs/rq2_bootstrap_5of22.png
# ============================================================
try:
    import google.colab  # noqa: F401
    from google.colab import drive
    drive.flush_and_unmount(); drive.mount("/content/drive", force_remount=True)
    DRIVE = "/content/drive/MyDrive/GEE_LPP_CMIP6"
except ImportError:
    DRIVE = "."

from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd

ANNUAL = f"{DRIVE}/indices_output/indices_annual_long.csv"
OUT = Path(f"{DRIVE}/revision_outputs"); OUT.mkdir(parents=True, exist_ok=True)
LPP = [250781, 250751, 250771]                 # same 3 in-basin stations as RQ2 test
SSPS = ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]
WINDOWS = {"near_2036-2065": (2036, 2065), "far_2071-2100": (2071, 2100)}
BASE = (1985, 2014)
K = 5


def lam_table(a):
    rows = []
    for m in a.model.unique():
        g = a[a.model == m]
        b = g[(g.term == "Historical") & g.year.between(*BASE)].R95p.mean()
        sel = bool(g.selected.iloc[0])
        for scen in SSPS:
            gs = g[g.scenario == scen]
            for wn, (y0, y1) in WINDOWS.items():
                f = gs[gs.year.between(y0, y1)].R95p.mean()
                if b and np.isfinite(f):
                    rows.append({"model": m, "selected": sel, "scen": scen, "horizon": wn, "lam": f / b})
    return pd.DataFrame(rows)


def main():
    a = pd.read_csv(ANNUAL)
    a = a[a.station.isin(LPP)]
    L = lam_table(a)
    if L.empty:
        raise SystemExit(f"No rows for stations {LPP} in {ANNUAL} - check the file / station IDs.")
    print(f"models available: {L.model.nunique()}  (screened: {L[L.selected].model.nunique()})")

    recs, dists = [], {}
    for (scen, hz), g in L.groupby(["scen", "horizon"]):
        lam = g.lam.values; n = len(lam)
        scr = float(g[g.selected].lam.mean())
        subs = np.array([lam[list(c)].mean() for c in combinations(range(n), K)])
        pct = 100.0 * (subs <= scr).mean()                 # percentile of screened-5 value
        dists[(scen, hz)] = (subs, scr)
        recs.append({"scenario": scen, "horizon": hz, "n_models": n, "n_subsets": len(subs),
                     "screened5_lambda": round(scr, 3), "all_mean": round(float(lam.mean()), 3),
                     "subset_p05": round(float(np.percentile(subs, 5)), 3),
                     "subset_p50": round(float(np.percentile(subs, 50)), 3),
                     "subset_p95": round(float(np.percentile(subs, 95)), 3),
                     "screened5_percentile": round(pct, 1),
                     "lower_tail_lt5pct": pct < 5.0})
    tab = pd.DataFrame(recs).sort_values(["horizon", "scenario"])
    tab.to_csv(OUT / "rq2_bootstrap_5of22.csv", index=False)
    print(tab.to_string(index=False))
    print("\nREAD: screened5_percentile < 5  -> screening effect is beyond 'any 5 models' (supports RQ2).")
    print("      percentile 5-20            -> suggestive only; report as such.")
    print("      percentile > 20            -> gap is a small-ensemble effect; drop the strong RQ2 claim.")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        keys = sorted(dists, key=lambda k: (k[1], k[0]))
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=False)
        for ax, k in zip(axes.ravel(), keys):
            subs, scr = dists[k]
            ax.hist(subs, bins=60, color="#bbbbbb")
            ax.axvline(scr, color="#1d9e75", lw=2, label=f"screened 5 = {scr:.2f}")
            ax.axvline(np.percentile(subs, 5), color="k", ls="--", lw=.8, label="5th pct")
            p = 100 * (subs <= scr).mean()
            ax.set_title(f"{k[0].replace('ssp','SSP').replace('_','-',1).replace('_','.')} {k[1].split('_')[0]}"
                         f"  (pct = {p:.1f})", fontsize=10)
            ax.set_xlabel("ensemble-mean λ of a 5-model subset"); ax.legend(fontsize=7, frameon=False)
        fig.suptitle("Screened-5 λ against all C(22,5) five-model subsets", fontsize=12)
        plt.tight_layout(); plt.savefig(OUT / "rq2_bootstrap_5of22.png", dpi=170); plt.close()
        print(f"\nSaved -> {OUT}/rq2_bootstrap_5of22.(csv|png)")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
