#!/usr/bin/env python3
# ============================================================
# REVISION - RQ2 test: does dynamics-based GCM screening change the
# climate signal vs a raw / unscreened ensemble?  (answers Reviewer Wright #2)
#
#   Input : indices_output/indices_annual_long.csv  (all 22 models, 'selected' flag)
#   Output: revision_outputs/rq2_screening_comparison.csv
#           revision_outputs/rq2_screening_comparison.png
#
#   Compares ensemble-mean lambda (R95p ratio, 3 in-basin LPP stations) for:
#     - screened 5  (passed dynamics pre-selection)
#     - unscreened 17 (failed)
#     - all 22 (naive ensemble)
#   A large gap = screening materially changes the projected signal.
# ============================================================
try:
    import google.colab  # noqa: F401
    from google.colab import drive
    drive.flush_and_unmount(); drive.mount("/content/drive", force_remount=True)
    DRIVE = "/content/drive/MyDrive/GEE_LPP_CMIP6"
except ImportError:
    DRIVE = "."

from pathlib import Path
import numpy as np, pandas as pd

ANNUAL = f"{DRIVE}/indices_output/indices_annual_long.csv"
OUT = Path(f"{DRIVE}/revision_outputs"); OUT.mkdir(parents=True, exist_ok=True)
LPP = [250781, 250751, 250771]
SSPS = ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]
WINDOWS = {"near_2036-2065": (2036, 2065), "far_2071-2100": (2071, 2100)}
BASE = (1985, 2014)


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
                    rows.append({"model": m, "selected": sel, "scen": scen,
                                 "horizon": wn, "lam": f / b})
    return pd.DataFrame(rows)


def main():
    a = pd.read_csv(ANNUAL)
    a = a[a.station.isin(LPP)]
    L = lam_table(a)

    recs = []
    for (scen, hz), g in L.groupby(["scen", "horizon"]):
        scr = g[g.selected].lam
        uns = g[~g.selected].lam
        recs.append({"scenario": scen, "horizon": hz,
                     "screened5_mean": round(scr.mean(), 3), "screened5_sd": round(scr.std(), 3),
                     "unscreened17_mean": round(uns.mean(), 3),
                     "all22_mean": round(g.lam.mean(), 3),
                     "gap_5_vs_22": round(scr.mean() - g.lam.mean(), 3)})
    tab = pd.DataFrame(recs).sort_values(["horizon", "scenario"])
    tab.to_csv(OUT / "rq2_screening_comparison.csv", index=False)
    print(tab.to_string(index=False))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, hz in zip(axes, WINDOWS):
            t = tab[tab.horizon == hz]; x = np.arange(len(t))
            ax.bar(x - 0.22, t.screened5_mean, 0.2, label="Screened 5", color="#1d9e75")
            ax.bar(x + 0.00, t.unscreened17_mean, 0.2, label="Unscreened 17", color="#d85a30")
            ax.bar(x + 0.22, t.all22_mean, 0.2, label="All 22", color="#888780")
            ax.axhline(1, ls="--", c="grey", lw=.8)
            ax.set_xticks(x); ax.set_xticklabels([{"ssp1_2_6": "SSP1-2.6", "ssp2_4_5": "SSP2-4.5", "ssp5_8_5": "SSP5-8.5"}[s] for s in t.scenario])
            ax.set_title({"near_2036-2065": "Near future (2036–2065)", "far_2071-2100": "Far future (2071–2100)"}[hz]); ax.set_ylabel("λ (R95p ratio)")
        axes[0].legend()
        
        plt.tight_layout(); plt.savefig(OUT / "rq2_screening_comparison.png", dpi=150); plt.close()
        print(f"\nSaved -> {OUT}/")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
