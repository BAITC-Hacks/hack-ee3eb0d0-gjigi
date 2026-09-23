"""Deterministic tools the forecasting agent can call.

Each tool returns a small JSON-serialisable dict (facts, not raw arrays), so an
LLM can reason over it; the heavy lifting (data, ML, checks) stays in code.
A ForecastSession holds state between calls for one issue.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hackalem.config import load_config
from hackalem.forecast import IssueResult, run_issue
from hackalem.timeutils import issue_time_utc, utc_to_local
from hackalem.weather.issued import issued_forecasts

MAIN_MODEL = "ecmwf_ifs"


@dataclass
class ForecastSession:
    issue_date: pd.Timestamp
    models: dict
    store: pd.DataFrame
    as_of_utc: pd.Timestamp = None
    excluded: list = field(default_factory=list)
    result: IssueResult | None = None
    history: list = field(default_factory=list)       # earlier results of this issue
    widen_factor: float = 1.0
    final: pd.DataFrame | None = None
    notes: list = field(default_factory=list)

    def __post_init__(self):
        self.issue_date = pd.Timestamp(self.issue_date).normalize()
        if self.as_of_utc is None:
            self.as_of_utc = issue_time_utc(self.issue_date)
        self.cfg = load_config()["agent"]


def _plant(fc: pd.DataFrame) -> pd.DataFrame:
    return fc[fc["unit"] == "plant"].set_index("time_local")


# ── tools ────────────────────────────────────────────────────────────────────

def check_weather_inputs(s: ForecastSession) -> dict:
    """Coverage, freshness and agreement of the NWP sources available at as_of."""
    w = issued_forecasts([s.issue_date], s.store, as_of_utc=s.as_of_utc)
    w = w[w["is_target"] & ~w["model"].isin(s.excluded)]
    ws = w.pivot_table(index="time_local", columns="model", values="wind_speed_100m")
    med = ws.median(axis=1)
    per_model = {}
    for m in ws.columns:
        g = w[w["model"] == m]
        per_model[m] = {
            "hours": int(ws[m].notna().sum()),
            "newest_run": f"{g['init_time'].max():%Y-%m-%d %H}z",
            "run_age_h": round((s.as_of_utc - g["init_time"].max()) / pd.Timedelta("1h"), 1),
            "mean_ws100": round(float(ws[m].mean()), 2),
            "mean_abs_dev_from_median": round(float((ws[m] - med).abs().mean()), 2),
        }
    spread = ws.std(axis=1)
    thr = s.cfg["outlier_model_dev_ms"]
    return {
        "issue_date": str(s.issue_date.date()),
        "as_of_local": str(utc_to_local(s.as_of_utc)),
        "models": per_model,
        "missing_models": [m for m in s.cfg["expected_models"] if m not in ws.columns and m not in s.excluded],
        "excluded_models": s.excluded,
        "ensemble_spread_ms": {"mean": round(float(spread.mean()), 2),
                               "max": round(float(spread.max()), 2),
                               "hours_above_threshold": int((spread > s.cfg["high_spread_ms"]).sum())},
        "outlier_candidates": [m for m, v in per_model.items() if v["mean_abs_dev_from_median"] > thr],
        "thresholds": {"outlier_model_dev_ms": thr, "high_spread_ms": s.cfg["high_spread_ms"]},
    }


def run_forecast(s: ForecastSession, exclude_models: list[str] | None = None) -> dict:
    """Run the ML forecast (plant, T1, T2) with the current inputs."""
    if exclude_models is not None:
        bad = [m for m in exclude_models if m == MAIN_MODEL and len(exclude_models) > 2]
        if bad:
            return {"error": "refusing to drop ECMWF together with most other sources"}
        s.excluded = list(exclude_models)
    if s.result is not None:
        s.history.append(s.result)
    s.result = run_issue(s.issue_date, models=s.models, store=s.store,
                         as_of_utc=s.as_of_utc, exclude_models=s.excluded)
    keep, s.widen_factor = s.widen_factor, 1.0
    if keep > 1.0:            # a decision to widen survives a re-run
        widen_intervals(s, keep)
    return {"status": "ok", "excluded_models": s.excluded, "widen_factor": s.widen_factor, **summarize(s)}


def summarize(s: ForecastSession) -> dict:
    p = _plant(s.result.forecast)
    days = {}
    for d, g in p.groupby("day_ahead"):
        days[f"D+{d}"] = {
            "date": str(g.index[0].date()),
            "mean_p50": round(float(g["p50"].mean()), 3),
            "max_p50": round(float(g["p50"].max()), 3),
            "hours_above_0.8": int((g["p50"] > 0.8).sum()),
            "hours_below_0.05": int((g["p50"] < 0.05).sum()),
            "mean_interval_width": round(float((g["p90"] - g["p10"]).mean()), 3),
            "expected_energy_share": round(float(g["mean"].sum() / 24), 3),
        }
    return {"plant": days, "runs_used": s.result.runs_used}


def validate_forecast(s: ForecastSession) -> dict:
    """Sanity checks on the current forecast."""
    if s.result is None:
        return {"error": "no forecast yet, call run_forecast first"}
    fc = s.result.forecast
    p = _plant(fc)
    issues = []
    if fc[["p10", "p50", "p90"]].isna().any().any():
        issues.append("missing values")
    if not ((fc["p10"] <= fc["p50"]) & (fc["p50"] <= fc["p90"])).all():
        issues.append("quantiles cross")
    if ((fc[["p10", "p50", "p90"]] < 0) | (fc[["p10", "p50", "p90"]] > 1)).any().any():
        issues.append("values outside [0, 1]")
    if len(p) != 48:
        issues.append(f"expected 48 plant hours, got {len(p)}")
    ramps = p["p50"].diff().abs()
    big_ramps = int((ramps > s.cfg["max_hourly_ramp"]).sum())
    if big_ramps:
        issues.append(f"{big_ramps} hourly ramps above {s.cfg['max_hourly_ramp']}")
    if p["p50"].std() < 0.01:
        issues.append("forecast is flat")
    # plant vs turbines consistency
    t = fc[fc["unit"].isin(["T1", "T2"])].groupby("time_local")["p50"].mean()
    gap = float((t - p["p50"]).abs().mean())
    if gap > 0.08:
        issues.append(f"plant and turbine forecasts disagree (mean gap {gap:.2f})")
    width = float((p["p90"] - p["p10"]).mean())
    return {"ok": not issues, "issues": issues, "mean_interval_width": round(width, 3),
            "max_hourly_ramp": round(float(ramps.max()), 3), "plant_vs_turbines_gap": round(gap, 3)}


def compare_with_previous(s: ForecastSession, prev_forecast: pd.DataFrame | None) -> dict:
    """Revision vs the previous issue for the overlapping day (today's D+1 = yesterday's D+2)."""
    if prev_forecast is None or s.result is None:
        return {"available": False}
    a = _plant(s.result.forecast)["p50"]
    b = _plant(prev_forecast)["p50"]
    common = a.index.intersection(b.index)
    if not len(common):
        return {"available": False}
    diff = a[common] - b[common]
    return {"available": True, "hours": len(common),
            "mean_abs_revision": round(float(diff.abs().mean()), 3),
            "mean_revision": round(float(diff.mean()), 3),
            "max_abs_revision": round(float(diff.abs().max()), 3),
            "largest_revision_at": str(diff.abs().idxmax()),
            "large_revision": bool(diff.abs().mean() > s.cfg["large_revision"])}


def check_for_new_runs(s: ForecastSession, hours_later: float = 6) -> dict:
    """Would newer NWP runs be published within `hours_later` h of the current as_of?"""
    t_new = s.as_of_utc + pd.Timedelta(hours=hours_later)
    pub = s.store[(s.store["available_at"] > s.as_of_utc) & (s.store["available_at"] <= t_new)]
    new = pub.groupby("model")["init_time"].max()
    return {"current_as_of_local": str(utc_to_local(s.as_of_utc)),
            "checked_until_local": str(utc_to_local(t_new)),
            "new_runs": {m: f"{t:%Y-%m-%d %H}z" for m, t in new.items()},
            "update_time_local": str(utc_to_local(pub["available_at"].max())) if len(pub) else None}


def rerun_with_update(s: ForecastSession, hours_later: float = 6) -> dict:
    """Move as_of to the moment the newest run is published and re-forecast."""
    info = check_for_new_runs(s, hours_later)
    if not info["new_runs"]:
        return {"status": "no new runs", **info}
    pub = s.store[(s.store["available_at"] > s.as_of_utc) &
                  (s.store["available_at"] <= s.as_of_utc + pd.Timedelta(hours=hours_later))]
    old = s.result.forecast if s.result is not None else None
    s.as_of_utc = pub["available_at"].max()
    out = run_forecast(s, s.excluded)
    rev = compare_with_previous(s, old)
    return {"status": "re-forecast with new runs", "new_as_of_local": str(utc_to_local(s.as_of_utc)),
            "new_runs": info["new_runs"], "revision_vs_before_update": rev, **out}


def widen_intervals(s: ForecastSession, factor: float) -> dict:
    """Widen P10/P90 around P50 (e.g. when NWP models disagree). factor in [1, 1.5]."""
    factor = float(np.clip(factor, 1.0, 1.5))
    fc = s.result.forecast
    for q in ("p10", "p90"):
        fc[q] = np.clip(fc["p50"] + (fc[q] - fc["p50"]) * factor / s.widen_factor, 0, 1)
    s.widen_factor = factor
    return {"status": "ok", "factor": factor,
            "mean_interval_width": round(float((_plant(fc)["p90"] - _plant(fc)["p10"]).mean()), 3)}


def finalize(s: ForecastSession, dispatcher_note: str) -> dict:
    """Accept the current forecast as the issue's final output."""
    v = validate_forecast(s)
    if not v.get("ok", False) and any("missing" in i or "cross" in i or "outside" in i for i in v["issues"]):
        return {"error": "cannot finalize: hard validation failures", **v}
    s.final = s.result.forecast.copy()
    s.final["final_issue_time_utc"] = s.as_of_utc
    s.notes.append(dispatcher_note)
    return {"status": "finalized", "issue_time_local": str(utc_to_local(s.as_of_utc))}
