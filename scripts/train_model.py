"""Stage 3: train and validate the quantile GBM, then fit the final models.

  1. Holdout Dec 2025 - Jan 2026: GBM vs best baseline (MAE, RMSE, pinball, 80% coverage)
  2. Rolling monthly CV (12 folds) with ablations:
       full          all features + power-curve prior
       no_curve      without the power-curve prior
       ecmwf_only    single NWP model (ECMWF 9 km) instead of the 4-model ensemble
  3. Final models on ALL history (targets up to 2026-01-31): plant, T1, T2 -> models/

Usage:
    python scripts/train_model.py                 # everything
    python scripts/train_model.py --skip-cv       # holdout + final fit only
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hackalem.config import load_config
from hackalem.evaluation import holdout_split, metrics, rolling_folds
from hackalem.features.build import feature_columns, load_features
from hackalem.models.baselines import Climatology, MeasuredPowerCurve, fill_with
from hackalem.models.gbm import QUANTILES, QuantileGBM, interval_metrics, pinball

cfg = load_config()
OUT = cfg["paths"]["outputs_dir"] / "model"
OUT.mkdir(parents=True, exist_ok=True)
MODELS_DIR = cfg["paths"]["models_dir"]
BASELINE = "curve[ens_ws100_mean]+linear_cal"
OTHER_MODELS = ("ecmwf_ifs025", "gfs_seamless", "icon_seamless")


def ecmwf_only_features(feats: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(feats)
            if not c.startswith("ens_") and not any(c.endswith(m) for m in OTHER_MODELS)]


def configs(feats):
    return {
        "full": dict(),
        "no_curve": dict(use_curve=False),
        "ecmwf_only": dict(features=ecmwf_only_features(feats),
                           curve_inputs=["wind_speed_100m__ecmwf_ifs"]),
    }


def baseline_pred(train, test):
    clim = Climatology().fit(train).predict(test)
    return fill_with(MeasuredPowerCurve("ens_ws100_mean", calibrate=True).fit(train).predict(test), clim)


def score(y, pred: pd.DataFrame, base: np.ndarray) -> dict:
    y = np.asarray(y)
    ok = ~np.isnan(y)
    r = {"n": int(ok.sum()),
         "mae_p50": metrics(y, pred["p50"])["mae"],
         "rmse_mean": metrics(y, pred["mean"])["rmse"],
         "rmse_p50": metrics(y, pred["p50"])["rmse"],
         "bias_p50": metrics(y, pred["p50"])["bias"],
         "mae_baseline": metrics(y, base)["mae"],
         "rmse_baseline": metrics(y, base)["rmse"],
         "pinball_avg": float(np.mean([pinball(y[ok], pred[f"p{int(q*100):02d}"].to_numpy()[ok], q)
                                       for q in QUANTILES])),
         **interval_metrics(y, pred["p10"].to_numpy(), pred["p90"].to_numpy())}
    r["mae_gain_vs_baseline"] = 1 - r["mae_p50"] / r["mae_baseline"]
    return r


def run_holdout(feats):
    train, hold = holdout_split(feats)
    model = QuantileGBM().fit(train)
    pred = model.predict(hold)
    base = baseline_pred(train, hold)
    rows = []
    for target in ["y_plant", "y_plant_raw"]:
        for d in [None, 1, 2]:
            s = np.ones(len(hold), bool) if d is None else (hold["day_ahead"] == d).to_numpy()
            rows.append({"target": target, "day_ahead": d or "all",
                         **score(hold[target].to_numpy()[s], pred[s], base[s])})
    tab = pd.DataFrame(rows)
    tab.round(4).to_csv(OUT / "holdout_metrics.csv", index=False)
    print("HOLDOUT Dec 2025 - Jan 2026\n", tab.round(4).to_string(index=False))

    imp = model.feature_importance()
    imp.to_csv(OUT / "feature_importance_p50.csv", header=["gain"])
    fig, ax = plt.subplots(figsize=(8, 7))
    (imp.head(25)[::-1] / imp.sum()).plot.barh(ax=ax)
    ax.set_title("Feature importance (P50, share of gain)")
    fig.tight_layout(); fig.savefig(OUT / "02_feature_importance.png", dpi=130); plt.close(fig)

    h = hold.assign(**pred, baseline=base)
    s = h[(h["day_ahead"] == 1) & h["time_local"].between("2026-01-10", "2026-01-24")].set_index("time_local")
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.fill_between(s.index, s["p10"], s["p90"], alpha=0.25, label="P10–P90")
    s["y_plant"].plot(ax=ax, c="k", lw=1.5, label="actual")
    s["p50"].plot(ax=ax, c="tab:blue", label="GBM P50")
    s["baseline"].plot(ax=ax, c="tab:orange", ls="--", lw=1, label="best baseline")
    ax.set_ylabel("normalised power"); ax.set_title("Holdout, day-ahead D+1")
    ax.legend(loc="upper right"); fig.tight_layout()
    fig.savefig(OUT / "01_holdout_sample.png", dpi=130); plt.close(fig)

    # reliability of each quantile
    rel = {q: float((hold["y_plant"] <= pred[f"p{int(q*100):02d}"])[hold["y_plant"].notna()].mean())
           for q in QUANTILES}
    return tab, rel, model


def run_cv(feats):
    rows = []
    for month, tr, te in rolling_folds(feats):
        base = baseline_pred(tr, te)
        for name, kw in configs(feats).items():
            pred = QuantileGBM(**kw).fit(tr).predict(te)
            rows.append({"month": month, "config": name, **score(te["y_plant"].to_numpy(), pred, base)})
        print(f"  CV {month} done", flush=True)
    cv = pd.DataFrame(rows)
    cv.round(4).to_csv(OUT / "cv_metrics.csv", index=False)
    agg = cv.groupby("config")[["mae_p50", "rmse_mean", "pinball_avg", "coverage_80", "width_80",
                                "mae_baseline", "mae_gain_vs_baseline"]].mean()
    print("\nROLLING CV mean over 12 months\n", agg.round(4).to_string())

    piv = cv[cv["config"] == "full"].set_index("month")[["mae_p50", "mae_baseline"]]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    piv.plot(ax=ax, marker="o"); ax.set_ylabel("MAE")
    ax.set_title("Rolling CV: GBM P50 vs best baseline by month")
    fig.tight_layout(); fig.savefig(OUT / "03_cv_by_month.png", dpi=130); plt.close(fig)
    return agg


def fit_final(feats):
    """Final models on all history.

    Boosting rounds come from the holdout-period model (early-stopped on Oct-Nov),
    and the P10-P90 margin is calibrated on Dec-Jan predictions of that model, which
    are truly out-of-sample and seasonally closest to the February test period.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    hist = feats[feats["time_local"] <= cfg["periods"]["history_end"]]
    train, hold = holdout_split(feats)
    meta = {"trained_on_until": str(hist["time_local"].max()), "rows": int(len(hist)),
            "models": {}}
    for target in ["y_plant", "y_T1", "y_T2"]:
        ref = QuantileGBM(target=target).fit(train).calibrate(hold)
        m = QuantileGBM(target=target).fit(hist, rounds=ref.best_rounds_)
        m.cqr_margin_ = ref.cqr_margin_
        path = MODELS_DIR / f"gbm_{target}.pkl"
        m.save(path)
        meta["models"][target] = {"file": path.name, "rounds": {str(k): v for k, v in m.best_rounds_.items()},
                                  "cqr_margin": m.cqr_margin_, "n_features": len(m.feature_names_)}
        print(f"  final model {target}: saved {path.name}")
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--final-only", action="store_true")
    args = ap.parse_args()
    feats = load_features()
    tab, rel, _ = run_holdout(feats) if not args.final_only else (pd.DataFrame(), {}, None)
    summary = {"holdout": tab.round(4).to_dict("records"), "quantile_reliability": rel}
    if args.final_only:
        fit_final(feats)
        return
    if not args.skip_cv:
        summary["cv"] = run_cv(feats).round(4).reset_index().to_dict("records")
    fit_final(feats)
    with open(OUT / "model_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
