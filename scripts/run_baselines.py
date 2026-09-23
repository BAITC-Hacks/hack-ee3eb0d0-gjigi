"""Stage 2: build features, evaluate reference forecasts.

  * holdout Dec 2025 - Jan 2026 ("rehearsal of February"), by day-ahead
  * rolling-origin monthly CV Feb 2025 - Jan 2026 (robustness)
  * both targets: clean power (y_plant) and actual output incl. downtime (y_plant_raw)

Usage: python scripts/run_baselines.py [--refresh]
Writes: docs/baselines/*
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hackalem.config import load_config
from hackalem.evaluation import add_skill, evaluate, holdout_split, rolling_folds
from hackalem.features.build import feature_columns, load_features
from hackalem.models.baselines import Climatology, default_baselines, fill_with

cfg = load_config()
OUT = cfg["paths"]["outputs_dir"] / "baselines"
OUT.mkdir(parents=True, exist_ok=True)


def predict_all(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    test = test.copy()
    clim = Climatology().fit(train).predict(test)
    for b in default_baselines():
        p = b.fit(train).predict(test)
        test[b.name] = p if b.name.startswith("persistence") else fill_with(p, clim)
    return test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    feats = load_features(refresh=args.refresh)
    names = [b.name for b in default_baselines()]
    summary = {"features": {"rows": len(feats), "issues": int(feats["issue_date"].nunique()),
                            "n_features": len(feature_columns(feats))}}

    # ── Holdout ──────────────────────────────────────────────────────────────
    train, hold = holdout_split(feats)
    hp = predict_all(train, hold)
    tables = []
    for target in ["y_plant", "y_plant_raw"]:
        t = add_skill(evaluate(hp, names, target), "climatology")
        t.insert(0, "target", target)
        tables.append(t)
        tb = add_skill(evaluate(hp, names, target, by=["day_ahead"]), "climatology", ["day_ahead"])
        tb.insert(0, "target", target)
        tables.append(tb)
    hold_tab = pd.concat(tables, ignore_index=True)
    hold_tab.round(4).to_csv(OUT / "holdout_metrics.csv", index=False)
    main_tab = hold_tab[(hold_tab["target"] == "y_plant") & hold_tab["day_ahead"].isna()] \
        .sort_values("mae")
    print("HOLDOUT Dec 2025 - Jan 2026, target y_plant\n",
          main_tab[["model", "n", "mae", "rmse", "bias", "corr", "skill_vs_climatology"]]
          .round(4).to_string(index=False))

    # ── Rolling-origin CV ────────────────────────────────────────────────────
    cv = []
    for month, tr, te in rolling_folds(feats):
        p = predict_all(tr, te)
        t = evaluate(p, names, "y_plant")
        t.insert(0, "month", month)
        cv.append(t)
    cv = pd.concat(cv, ignore_index=True)
    cv.round(4).to_csv(OUT / "cv_monthly_metrics.csv", index=False)
    cv_mean = cv.groupby("model")[["mae", "rmse", "bias", "corr"]].mean().sort_values("mae")
    print("\nROLLING CV (12 monthly folds), mean over months\n", cv_mean.round(4).to_string())

    # ── Charts ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    order = cv_mean.index
    x = np.arange(len(order))
    ax.bar(x - 0.2, cv_mean["mae"], 0.4, label="rolling CV (12 months)")
    ax.bar(x + 0.2, main_tab.set_index("model").reindex(order)["mae"], 0.4, label="holdout Dec–Jan")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("MAE (share of capacity)"); ax.set_title("Baselines: MAE of hourly plant power")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "01_baseline_mae.png", dpi=130); plt.close(fig)

    piv = cv.pivot(index="month", columns="model", values="mae")[list(order[:4])]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    piv.plot(ax=ax, marker="o"); ax.set_ylabel("MAE"); ax.set_title("Monthly MAE (rolling CV)")
    fig.tight_layout(); fig.savefig(OUT / "02_cv_by_month.png", dpi=130); plt.close(fig)

    best = order[0]
    sample = hp[hp["time_local"].between("2026-01-10", "2026-01-24")]
    sample = sample[sample["day_ahead"] == 1].set_index("time_local")
    fig, ax = plt.subplots(figsize=(13, 4.5))
    sample["y_plant"].plot(ax=ax, c="k", lw=1.5, label="actual")
    sample[best].plot(ax=ax, label=best)
    sample["climatology"].plot(ax=ax, ls="--", label="climatology")
    ax.set_ylabel("normalised power"); ax.set_title("Holdout sample, day-ahead D+1")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "03_holdout_sample.png", dpi=130); plt.close(fig)

    summary["holdout"] = main_tab.round(4).to_dict("records")
    summary["cv_mean"] = cv_mean.round(4).reset_index().to_dict("records")
    with open(OUT / "baselines_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
