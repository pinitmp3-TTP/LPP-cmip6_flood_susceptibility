#!/usr/bin/env python3
# ============================================================
# Per-gauge climate factor lambda for the nearest-gauge zones (Stage II)
#
#   lambda(model, gauge, ssp, horizon) = mean annual R95p (window) / mean annual R95p (1985-2014)
#   Summarised over the 5 SCREENED models per gauge x ssp x horizon:
#     lam_mean  = ensemble mean
#     lam_sd    = ensemble standard deviation
#     n_gcm_up  = number of screened models with lambda > 1 (feeds the 4-of-5 agreement map)
#
#   Input : indices_output/indices_annual_long.csv   (from the indices step)
#   Output: per_station_lambda.csv                   (read by the spatial-lambda step)
# ============================================================
try:
    import google.colab  # noqa: F401
    from google.colab import drive
    drive.flush_and_unmount(); drive.mount("/content/drive", force_remount=True)
    DRIVE = "/content/drive/MyDrive/GEE_LPP_CMIP6"
except ImportError:
    DRIVE = "."

import numpy as np, pandas as pd

ANNUAL = f"{DRIVE}/indices_output/indices_annual_long.csv"
OUT = f"{DRIVE}/per_station_lambda.csv"
GAUGES = [250751, 250771, 250781, 250541, 250651, 250791]
SSPS = ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]
WINDOWS = {"near_2036-2065": (2036, 2065), "far_2071-2100": (2071, 2100)}
BASE = (1985, 2014)


def main():
    a = pd.read_csv(ANNUAL)
    a = a[a.station.isin(GAUGES) & a.selected.astype(bool)]
    rows = []
    for st in GAUGES:
        for scen in SSPS:
            for hz, (y0, y1) in WINDOWS.items():
                lams = []
                for m, g in a[a.station == st].groupby("model"):
                    b = g[(g.term == "Historical") & g.year.between(*BASE)].R95p.mean()
                    f = g[(g.scenario == scen) & g.year.between(y0, y1)].R95p.mean()
                    if b and np.isfinite(f): lams.append(f / b)
                lams = np.array(lams)
                rows.append({"station": st, "ssp": scen, "horizon": hz,
                             "lam_mean": round(float(lams.mean()), 4),
                             "lam_sd": round(float(lams.std(ddof=0)), 4),
                             "n_gcm_up": int((lams > 1).sum())})
    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
    print(df.to_string(index=False)); print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
