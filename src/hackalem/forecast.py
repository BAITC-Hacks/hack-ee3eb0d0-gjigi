"""Operational forecast: one issue = weather as of issue time -> features -> models -> 48 h.

This is the single entry point used by the backtest (stage 4) and by the agent (stage 5).
"""

from dataclasses import dataclass, field

import pandas as pd

from hackalem.config import load_config
from hackalem.features.build import build_features
from hackalem.models.gbm import QuantileGBM
from hackalem.timeutils import issue_time_utc, utc_to_local
from hackalem.weather.issued import issued_forecasts
from hackalem.weather.store import load_store

TARGETS = {"plant": "y_plant", "T1": "y_T1", "T2": "y_T2"}
OUTPUT_COLUMNS = ["issue_date", "issue_time_utc", "time_local", "day_ahead", "horizon_h",
                  "unit", "p10", "p50", "p90", "mean"]


@dataclass
class IssueResult:
    issue_date: pd.Timestamp
    as_of_utc: pd.Timestamp
    forecast: pd.DataFrame                 # long: one row per unit x hour
    features: pd.DataFrame                 # 48 rows, model inputs (for diagnostics)
    weather: pd.DataFrame                  # issued NWP rows used (long)
    runs_used: dict = field(default_factory=dict)


def load_models() -> dict[str, QuantileGBM]:
    d = load_config()["paths"]["models_dir"]
    return {unit: QuantileGBM.load(d / f"gbm_{t}.pkl") for unit, t in TARGETS.items()}


def run_issue(issue_date, models: dict | None = None, store: pd.DataFrame | None = None,
              as_of_utc: pd.Timestamp | None = None,
              exclude_models: list[str] | None = None) -> IssueResult:
    """Forecast local days D+1..D+2 for issue day D.

    as_of_utc: override the issue moment (e.g. a later re-run when a new NWP
               run has been published). Defaults to the protocol issue time.
    exclude_models: NWP models to drop (agent decision, e.g. a broken source).
    """
    issue_date = pd.Timestamp(issue_date).normalize()
    models = models or load_models()
    store = load_store() if store is None else store
    if exclude_models:
        store = store[~store["model"].isin(exclude_models)]
    as_of = as_of_utc if as_of_utc is not None else issue_time_utc(issue_date)

    issued = issued_forecasts([issue_date], store, as_of_utc=as_of)
    feats = build_features(issued, with_targets=False)
    as_of_local = utc_to_local(as_of)
    rows = []
    for unit, model in models.items():
        p = model.predict(feats)
        rows.append(pd.DataFrame({
            "issue_date": issue_date, "issue_time_utc": as_of,
            "time_local": feats["time_local"].to_numpy(),
            "day_ahead": feats["day_ahead"].to_numpy(),
            "horizon_h": ((feats["time_local"] - as_of_local).dt.total_seconds() / 3600).to_numpy(),
            "unit": unit, **{c: p[c].to_numpy() for c in ["p10", "p50", "p90", "mean"]}}))
    forecast = pd.concat(rows, ignore_index=True)[OUTPUT_COLUMNS]
    runs = (issued[issued["is_target"]].groupby("model")["init_time"]
            .agg(lambda s: sorted({f"{t:%Y-%m-%d %H}z" for t in s})).to_dict())
    return IssueResult(issue_date, as_of, forecast, feats, issued, runs)
