"""Stage 1 checks on the NWP archive.

  1. SCADA time-zone offset: cross-correlation of measured wind / temperature
     with short-lead ECMWF forecasts at candidate UTC offsets.
  2. Replay of the issue protocol over history -> data/weather/issued_forecasts.parquet
  3. Skill of each model (day D+1 vs D+2) against measured hub wind.
  4. SCADA temperature vs NWP 2 m temperature (ambient or nacelle sensor?).
  5. Audit of runs used for every test-period issue -> outputs/weather/test_issue_runs.csv

Usage: python scripts/check_weather.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hackalem.config import load_config
from hackalem.data.turbines import load_hourly
from hackalem.weather.issued import build_issued_archive
from hackalem.weather.store import load_store

cfg = load_config()
OUT = cfg["paths"]["outputs_dir"] / "weather"
OUT.mkdir(parents=True, exist_ok=True)
summary: dict = {}

store = load_store(cfg)
scada = load_hourly()
summary["store"] = {
    m: {"rows": int(len(g)), "runs": int(g["init_time"].nunique()),
        "first_valid": str(g["valid_time"].min()), "last_valid": str(g["valid_time"].max())}
    for m, g in store.groupby("model")}

# ── 1. Time-zone offset ──────────────────────────────────────────────────────
# Near-analysis series: ECMWF 9 km, lead 0..5 h of each run (covers every hour).
ec = store[(store["model"] == "ecmwf_ifs") & (store["lead_h"] < 6)]
ana = ec.sort_values("lead_h").groupby("valid_time").first()
tz = {}
for var_s, var_n in [("ws_plant", "wind_speed_100m"), ("temp_T1", "temperature_2m")]:
    corr = {}
    for off in range(0, 11):
        s = scada[var_s].copy()
        s.index = s.index - pd.Timedelta(hours=off)          # local -> UTC candidate
        j = pd.concat([s, ana[var_n]], axis=1, join="inner").dropna()
        corr[off] = round(float(j.iloc[:, 0].corr(j.iloc[:, 1])), 4)
    tz[var_s] = corr
summary["tz_offset_corr"] = tz
best = {k: max(v, key=v.get) for k, v in tz.items()}
summary["tz_best_offset_h"] = best
fig, ax = plt.subplots(figsize=(8, 4))
for k, v in tz.items():
    ax.plot(list(v), list(v.values()), marker="o", label=k)
ax.set_xlabel("assumed SCADA offset from UTC, h"); ax.set_ylabel("corr with ECMWF (lead<6h)")
ax.set_title("SCADA time-zone check"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "01_tz_offset.png", dpi=130); plt.close(fig)
if best["ws_plant"] != cfg["scada_utc_offset_h"]:
    print(f"WARNING: best wind offset {best['ws_plant']} h != config {cfg['scada_utc_offset_h']} h")

# ── 2. Issue-protocol replay ────────────────────────────────────────────────
issued = build_issued_archive()
issued = issued[issued["is_target"]].copy()   # drop neighbour-hour padding
summary["issued"] = {"issues": int(issued["issue_date"].nunique()),
                     "rows": int(len(issued)),
                     "first_issue": str(issued["issue_date"].min().date()),
                     "last_issue": str(issued["issue_date"].max().date())}
issued["day_ahead"] = (issued["time_local"].dt.normalize() - issued["issue_date"]).dt.days

# ── 3. Skill vs measured hub wind ────────────────────────────────────────────
hist = issued.merge(scada[["ws_plant", "power_plant"]], left_on="time_local",
                    right_index=True, how="inner").dropna(subset=["ws_plant", "wind_speed_100m"])
skill = []
for (m, d), g in hist.groupby(["model", "day_ahead"]):
    err = g["wind_speed_100m"] - g["ws_plant"]
    skill.append({"model": m, "day_ahead": int(d), "n": len(g),
                  "corr": round(g["wind_speed_100m"].corr(g["ws_plant"]), 3),
                  "bias_ms": round(err.mean(), 2), "mae_ms": round(err.abs().mean(), 2),
                  "lead_h_mean": round(g["lead_h"].mean(), 1)})
# multi-model mean (equal weights) as a first ensemble
ens = hist.pivot_table(index=["issue_date", "time_local"], columns="model",
                       values="wind_speed_100m").mean(axis=1).rename("ens")
ens = ens.reset_index().merge(scada[["ws_plant"]], left_on="time_local", right_index=True)
ens["day_ahead"] = (ens["time_local"].dt.normalize() - ens["issue_date"]).dt.days
for d, g in ens.dropna().groupby("day_ahead"):
    err = g["ens"] - g["ws_plant"]
    skill.append({"model": "ensemble_mean", "day_ahead": int(d), "n": len(g),
                  "corr": round(g["ens"].corr(g["ws_plant"]), 3),
                  "bias_ms": round(err.mean(), 2), "mae_ms": round(err.abs().mean(), 2),
                  "lead_h_mean": np.nan})
skill = pd.DataFrame(skill).sort_values(["day_ahead", "mae_ms"])
skill.to_csv(OUT / "nwp_skill_wind.csv", index=False)
summary["skill_wind"] = skill.to_dict("records")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, m in zip(axes, ["ecmwf_ifs", "gfs_seamless"]):
    g = hist[(hist["model"] == m) & (hist["day_ahead"] == 1)]
    ax.hexbin(g["wind_speed_100m"], g["ws_plant"], gridsize=40, mincnt=1, cmap="viridis")
    ax.plot([0, 20], [0, 20], "r--")
    ax.set_title(f"{m}, day D+1"); ax.set_xlabel("NWP wind 100 m, m/s")
axes[0].set_ylabel("measured hub wind, m/s")
fig.tight_layout(); fig.savefig(OUT / "02_nwp_vs_measured.png", dpi=130); plt.close(fig)

# Error by month for the main model (seasonal bias -> needs calibration)
g = hist[(hist["model"] == "ecmwf_ifs")]
mon = g.assign(err=g["wind_speed_100m"] - g["ws_plant"]).groupby(
    g["time_local"].dt.month)["err"].agg(["mean", lambda e: e.abs().mean()])
mon.columns = ["bias", "mae"]
summary["ecmwf_monthly_error"] = mon.round(2).to_dict()
fig, ax = plt.subplots(figsize=(8, 4))
mon.plot.bar(ax=ax); ax.set_title("ECMWF 9 km wind error by month (D+1..D+2)")
ax.set_xlabel("month"); ax.set_ylabel("m/s")
fig.tight_layout(); fig.savefig(OUT / "03_ecmwf_error_by_month.png", dpi=130); plt.close(fig)

# ── 4. Temperature sensor check ──────────────────────────────────────────────
t = issued[(issued["model"] == "ecmwf_ifs") & (issued["day_ahead"] == 1)].merge(
    scada[["temp_T1"]], left_on="time_local", right_index=True).dropna(subset=["temp_T1"])
tb = (t["temp_T1"] - t["temperature_2m"]).groupby(t["time_local"].dt.month).mean().round(1)
summary["temp_scada_minus_nwp_by_month"] = tb.to_dict()
summary["temp_corr"] = round(t["temp_T1"].corr(t["temperature_2m"]), 3)

# ── 5. Test-period audit ─────────────────────────────────────────────────────
test = issued[issued["issue_date"] >= cfg["forecast"]["first_issue"]]
audit = test.groupby(["issue_date", "model"]).agg(
    issue_time_utc=("issue_time_utc", "first"),
    runs_used=("init_time", lambda s: ", ".join(sorted({f"{x:%m-%d %H}z" for x in s}))),
    latest_available_at=("available_at", "max"),
    hours=("valid_time", "size"),
    lead_h_min=("lead_h", "min"), lead_h_max=("lead_h", "max")).reset_index()
assert (audit["latest_available_at"] <= audit["issue_time_utc"]).all()
audit.to_csv(OUT / "test_issue_runs.csv", index=False)
summary["test_audit"] = {
    "issues": int(audit["issue_date"].nunique()),
    "hours_per_model": audit.groupby("model")["hours"].min().to_dict(),
    "no_lookahead": True}

with open(OUT / "weather_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
print(json.dumps({k: v for k, v in summary.items() if k != "skill_wind"}, indent=2, default=str))
print(skill.to_string(index=False))
