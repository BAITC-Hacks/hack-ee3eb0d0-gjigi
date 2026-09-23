"""Stage 4: rolling issue-by-issue forecast for the test period (Feb 2026).

For every issue day D = 2026-01-31 ... 2026-02-27 (10:00 plant time):
    NWP published before the issue -> features -> final models -> 48 hourly values (D+1, D+2)

Also a January "rehearsal": the same pipeline with models trained only on data
before 2025-12-01, scored against actual January SCADA.

Usage: python scripts/run_backtest.py [--skip-rehearsal]
Writes: docs/forecast/ (analysis; the deliverable is written by run_agent.py)
    forecasts_all_issues.csv     every issue x unit x hour (the full rolling record)
    forecast_no_agent_feb2026.csv one value per hour of Feb 2026: freshest issue (D+1), no agent
    issue_log.csv                per issue: NWP runs used, publication times
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hackalem.config import load_config
from hackalem.evaluation import metrics
from hackalem.features.build import load_features
from hackalem.forecast import load_models, run_issue
from hackalem.models.gbm import QuantileGBM, interval_metrics
from hackalem.weather.store import load_store

cfg = load_config()
OUT = cfg["paths"]["outputs_dir"] / "forecast"
OUT.mkdir(parents=True, exist_ok=True)


def rolling(issue_dates, models, store) -> tuple[pd.DataFrame, pd.DataFrame]:
    fcs, log = [], []
    for d in issue_dates:
        r = run_issue(d, models=models, store=store)
        fcs.append(r.forecast)
        log.append({"issue_date": d.date(), "issue_time_utc": r.as_of_utc,
                    "hours": int((r.forecast["unit"] == "plant").sum()),
                    **{f"runs_{m}": ", ".join(v) for m, v in r.runs_used.items()},
                    "latest_publication_utc": r.weather.loc[r.weather["is_target"], "available_at"].max()})
    return pd.concat(fcs, ignore_index=True), pd.DataFrame(log)


def freshest(fc: pd.DataFrame) -> pd.DataFrame:
    """One forecast per unit x hour: the most recent issue (smallest horizon)."""
    return (fc.sort_values("issue_date").groupby(["unit", "time_local"], as_index=False).last()
              .sort_values(["unit", "time_local"]))


def rehearsal_january():
    """Same rolling pipeline on Jan 2026 with models that never saw Dec-Jan data."""
    feats = load_features()
    train = feats[feats["time_local"] < cfg["validation"]["holdout_start"]]
    models = {"plant": QuantileGBM(target="y_plant").fit(train)}
    store = load_store()
    issues = pd.date_range("2025-12-31", "2026-01-30")
    fc, _ = rolling(issues, models, store)
    h = feats.drop_duplicates("time_local").set_index("time_local")[["y_plant", "y_plant_raw"]]
    fc = fc.join(h, on="time_local")
    res = {}
    for d in [1, 2]:
        s = fc[fc["day_ahead"] == d]
        res[f"D+{d}"] = {"mae_p50": metrics(s["y_plant"], s["p50"])["mae"],
                         "rmse_mean": metrics(s["y_plant"], s["mean"])["rmse"],
                         **interval_metrics(s["y_plant"].to_numpy(), s["p10"].to_numpy(), s["p90"].to_numpy())}
    fc.to_csv(OUT / "rehearsal_jan2026.csv", index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rehearsal", action="store_true")
    args = ap.parse_args()

    f = cfg["forecast"]
    issues = pd.date_range(f["first_issue"], f["last_issue"])
    fc, log = rolling(issues, load_models(), load_store())
    fc.round(4).to_csv(OUT / "forecasts_all_issues.csv", index=False)
    log.to_csv(OUT / "issue_log.csv", index=False)

    sub = freshest(fc)
    sub = sub[sub["time_local"].between(cfg["periods"]["test_start"], cfg["periods"]["test_end"])]
    wide = sub.pivot(index="time_local", columns="unit", values=["p50", "p10", "p90", "mean"])
    wide.columns = [f"{u}_{v}" for v, u in wide.columns]
    wide = wide[[f"{u}_{v}" for u in ["plant", "T1", "T2"] for v in ["p50", "p10", "p90", "mean"]]]
    src = sub[sub["unit"] == "plant"].set_index("time_local")[["issue_date", "horizon_h"]]
    wide = src.join(wide).reset_index()
    wide.round(4).to_csv(OUT / "forecast_no_agent_feb2026.csv", index=False)
    assert len(wide) == 28 * 24, f"expected 672 hours, got {len(wide)}"
    assert wide.filter(like="_p50").notna().all().all()

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.fill_between(wide["time_local"], wide["plant_p10"], wide["plant_p90"], alpha=0.25, label="P10–P90")
    ax.plot(wide["time_local"], wide["plant_p50"], lw=1.2, label="P50")
    ax.set_ylim(0, 1); ax.set_ylabel("normalised power")
    ax.set_title("Plant forecast for February 2026 (freshest issue, D+1)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "01_feb2026_forecast.png", dpi=130); plt.close(fig)

    summary = {"issues": len(issues), "rows_all_issues": len(fc), "submission_hours": len(wide),
               "feb_mean_p50_plant": float(wide["plant_p50"].mean())}
    if not args.skip_rehearsal:
        summary["rehearsal_jan2026"] = rehearsal_january()
    with open(OUT / "backtest_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
