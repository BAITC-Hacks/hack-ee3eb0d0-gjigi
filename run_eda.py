"""Stage 0 EDA: data quality, seasonality, turbine agreement, time zone check.

Usage:  python scripts/run_eda.py
Writes: outputs/eda/*.png, outputs/eda/eda_summary.json
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hackalem.config import load_config
from hackalem.data.turbines import flag_quality, load_hourly, load_raw

cfg = load_config()
OUT = cfg["paths"]["outputs_dir"] / "eda"
OUT.mkdir(parents=True, exist_ok=True)
TURBINES = list(cfg["turbines"])
summary: dict = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)


# ── 1. Raw 10-min quality ────────────────────────────────────────────────────
raw = {t: flag_quality(load_raw(t)) for t in TURBINES}
for t, d in raw.items():
    full = pd.date_range(d["time"].min(), d["time"].max(), freq="10min")
    summary[f"raw_{t}"] = {
        "rows": len(d),
        "start": str(d["time"].min()), "end": str(d["time"].max()),
        "missing_10min_slots": int(len(full) - len(d)),
        "missing_pct": round(100 * (len(full) - len(d)) / len(full), 2),
        "downtime_samples": int(d["flag_downtime"].sum()),
        "below_curve_samples": int(d["flag_below_curve"].sum()),
        "stuck_samples": int(d["flag_stuck"].sum()),
        "valid_pct": round(100 * d["valid"].mean(), 2),
        "irregular_timestamps": int((d["time"].dt.minute % 10 != 0).sum()),
    }

# Power curve scatter with flags
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, (t, d) in zip(axes, raw.items()):
    s = d.sample(40000, random_state=0)
    ok = s[s["valid"]]
    ax.scatter(ok["ws"], ok["power"], s=1, alpha=0.15, c="tab:blue", label="valid")
    for flag, c in [("flag_downtime", "tab:red"), ("flag_below_curve", "tab:orange")]:
        f = d[d[flag]]
        ax.scatter(f["ws"], f["power"], s=3, alpha=0.5, c=c, label=flag.replace("flag_", ""))
    ax.set_title(f"{t}: power curve (10-min)")
    ax.set_xlabel("wind speed, m/s")
    ax.legend(markerscale=5)
axes[0].set_ylabel("normalised power")
save(fig, "01_power_curve_flags.png")

# ── 2. Hourly table ──────────────────────────────────────────────────────────
h = load_hourly(refresh=True)
summary["hourly"] = {
    "hours_total": len(h),
    **{f"power_nan_pct_{t}": round(100 * h[f"power_{t}"].isna().mean(), 2) for t in TURBINES},
    "plant_nan_pct": round(100 * h["power_plant"].isna().mean(), 2),
    "plant_capacity_factor": round(float(h["power_plant"].mean()), 3),
}

# Gaps (consecutive NaN hours of plant power)
na = h["power_plant"].isna()
gap_id = (na != na.shift()).cumsum()
gaps = h[na].groupby(gap_id[na]).apply(lambda g: pd.Series({
    "start": g.index.min(), "hours": len(g)}))
long_gaps = gaps[gaps["hours"] >= 6].sort_values("hours", ascending=False)
summary["plant_gaps_ge_6h"] = [
    {"start": str(r.start), "hours": int(r.hours)} for r in long_gaps.head(15).itertuples()]

# Monthly coverage
cov = pd.DataFrame({t: h[f"n_samples_{t}"].resample("MS").sum()
                    / (h[f"n_samples_{t}"].resample("MS").size() * 6) for t in TURBINES})
fig, ax = plt.subplots(figsize=(13, 3.5))
cov.plot.bar(ax=ax, width=0.8)
ax.set_xticklabels([d.strftime("%Y-%m") for d in cov.index], rotation=90)
ax.set_ylabel("share of 10-min samples present")
ax.set_title("Monthly data coverage")
save(fig, "02_monthly_coverage.png")

# ── 3. Seasonality ───────────────────────────────────────────────────────────
hp = h["power_plant"]
monthly = hp.groupby([hp.index.year, hp.index.month]).mean().unstack(0)
fig, ax = plt.subplots(figsize=(9, 4.5))
monthly.plot(ax=ax, marker="o")
ax.set_xlabel("month"); ax.set_ylabel("mean normalised power")
ax.set_title("Monthly capacity factor by year (plant)")
save(fig, "03_monthly_cf_by_year.png")
summary["monthly_cf"] = {f"{y}": {int(m): round(v, 3) for m, v in monthly[y].dropna().items()}
                         for y in monthly.columns}

heat = hp.groupby([hp.index.month, hp.index.hour]).mean().unstack()
fig, ax = plt.subplots(figsize=(11, 4.5))
im = ax.imshow(heat.values, aspect="auto", cmap="viridis")
ax.set_yticks(range(12)); ax.set_yticklabels(heat.index)
ax.set_xticks(range(24)); ax.set_xlabel("hour of day (local)"); ax.set_ylabel("month")
ax.set_title("Mean plant power: month x hour")
fig.colorbar(im, ax=ax)
save(fig, "04_month_hour_heatmap.png")

# ── 4. Time-zone check via diurnal temperature maximum ───────────────────────
# Kazakhstan moved from UTC+6 to UTC+5 on 2024-03-01. If SCADA timestamps are
# local civil time, the hour of the diurnal temperature max should shift ~1 h
# earlier after that date (same season compared across years).
temp = h[[f"temp_{t}" for t in TURBINES]].mean(axis=1)
diurnal = []
for (y, m), g in temp.groupby([temp.index.year, temp.index.month]):
    prof = g.groupby(g.index.hour).mean()
    if prof.notna().sum() == 24:
        # sub-hour peak via parabolic interpolation around the max
        k = int(prof.values.argmax())
        a, b, c = prof.values[(k - 1) % 24], prof.values[k], prof.values[(k + 1) % 24]
        off = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0
        diurnal.append({"year": y, "month": m, "tmax_hour": k + off,
                        "tmin_hour": int(prof.values.argmin())})
diurnal = pd.DataFrame(diurnal)
pv = diurnal.pivot(index="month", columns="year", values="tmax_hour")
summary["tmax_hour_by_year_month"] = {int(y): {int(m): round(v, 2) for m, v in pv[y].dropna().items()}
                                      for y in pv.columns}
fig, ax = plt.subplots(figsize=(9, 4.5))
pv.plot(ax=ax, marker="o")
ax.set_ylabel("hour of diurnal temperature max")
ax.set_title("Time-zone check: diurnal T-max hour by year")
save(fig, "05_tz_check_tmax_hour.png")

# Local solar noon at the site in UTC
lon = cfg["turbines"]["T1"]["lon"]
summary["solar_noon_utc_hour"] = round(12 - lon / 15, 2)

# ── 5. Turbine agreement ─────────────────────────────────────────────────────
both = h[["power_T1", "power_T2", "ws_T1", "ws_T2"]].dropna()
summary["turbine_agreement"] = {
    "hours_both": len(both),
    "corr_power": round(both["power_T1"].corr(both["power_T2"]), 4),
    "corr_ws": round(both["ws_T1"].corr(both["ws_T2"]), 4),
    "mae_power_T1_vs_T2": round(float((both["power_T1"] - both["power_T2"]).abs().mean()), 4),
    "mean_ws_T1": round(both["ws_T1"].mean(), 3), "mean_ws_T2": round(both["ws_T2"].mean(), 3),
}
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].scatter(both["power_T1"], both["power_T2"], s=1, alpha=0.1)
axes[0].plot([0, 1], [0, 1], "r--"); axes[0].set_xlabel("T1"); axes[0].set_ylabel("T2")
axes[0].set_title("Hourly power T1 vs T2")
axes[1].scatter(both["ws_T1"], both["ws_T2"], s=1, alpha=0.1)
axes[1].plot([0, 22], [0, 22], "r--"); axes[1].set_xlabel("T1"); axes[1].set_ylabel("T2")
axes[1].set_title("Hourly wind speed T1 vs T2")
save(fig, "06_turbine_agreement.png")

# ── 6. Predictability: autocorrelation & persistence skill ───────────────────
hp_full = hp.asfreq("h")
lags = range(1, 49)
acf = [hp_full.autocorr(l) for l in lags]
pers_mae = {l: float((hp_full - hp_full.shift(l)).abs().mean()) for l in (1, 6, 12, 24, 36, 48)}
clim = hp_full.groupby([hp_full.index.month, hp_full.index.hour]).transform("mean")
summary["predictability"] = {
    "acf": {l: round(a, 3) for l, a in zip(lags, acf) if l in (1, 3, 6, 12, 24, 36, 48)},
    "persistence_mae": {l: round(v, 4) for l, v in pers_mae.items()},
    "climatology_mae": round(float((hp_full - clim).abs().mean()), 4),
    "std_power": round(float(hp_full.std()), 4),
}
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(list(lags), acf, marker=".")
ax.axhline(0, c="k", lw=0.5)
ax.set_xlabel("lag, hours"); ax.set_ylabel("autocorrelation")
ax.set_title("Plant power autocorrelation")
save(fig, "07_acf.png")

# Distribution of power (U-shape matters for model choice / loss)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
hp.dropna().hist(bins=50, ax=axes[0]); axes[0].set_title("Hourly plant power distribution")
h["ws_plant"].dropna().hist(bins=50, ax=axes[1]); axes[1].set_title("Hourly wind speed distribution")
save(fig, "08_distributions.png")
summary["power_share"] = {
    "near_zero_lt_0.02": round(float((hp < 0.02).mean()), 3),
    "near_rated_gt_0.95": round(float((hp > 0.95).mean()), 3),
}

# Hourly ramps
ramp = hp_full.diff().abs()
summary["ramps"] = {"p95": round(float(ramp.quantile(0.95)), 3),
                    "p99": round(float(ramp.quantile(0.99)), 3),
                    "max": round(float(ramp.max()), 3)}

# Temperature sanity (ambient vs nacelle?)
summary["temperature"] = {
    "mean": round(float(temp.mean()), 2),
    "p01": round(float(temp.quantile(0.01)), 2), "p99": round(float(temp.quantile(0.99)), 2),
    "monthly_mean": {int(m): round(v, 1) for m, v in temp.groupby(temp.index.month).mean().items()},
}

# The last month before the test period, as a sanity look
jan = h.loc["2026-01", ["power_T1", "power_T2", "ws_plant"]]
fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
jan[["power_T1", "power_T2"]].plot(ax=ax[0]); ax[0].set_ylabel("power")
jan["ws_plant"].plot(ax=ax[1], c="tab:green"); ax[1].set_ylabel("ws, m/s")
ax[0].set_title("January 2026 (last month before test)")
save(fig, "09_jan_2026.png")

with open(OUT / "eda_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
