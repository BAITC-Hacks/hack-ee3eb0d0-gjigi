"""Fetch archived NWP forecasts into one tidy "forecast store" format.

Store schema (one row per model x init_time x valid_time), all times naive UTC:
    model, init_time, valid_time, lead_h, <weather variables...>

Two sources:
  * Single Runs API  - the complete run exactly as issued (ECMWF IFS 9 km,
    archived since 2024-03). Every lead time is present.
  * Previous Runs API - value predicted N days before valid time for several
    models. Verified against Single Runs: init = floor_6h(valid) - N days.
"""

import numpy as np
import pandas as pd

from hackalem.weather.openmeteo import PREVIOUS_RUNS_URL, SINGLE_RUNS_URL, OpenMeteoClient

STORE_KEYS = ["model", "init_time", "valid_time", "lead_h"]


def _base_params(cfg: dict) -> dict:
    w = cfg["weather"]
    return dict(latitude=w["lat"], longitude=w["lon"], wind_speed_unit="ms", timezone="GMT")


def fetch_single_run(client: OpenMeteoClient, cfg: dict, model: str,
                     init: pd.Timestamp, offline: bool = False) -> pd.DataFrame | None:
    """One complete model run. Returns None if the run is not archived."""
    w = cfg["weather"]
    params = dict(**_base_params(cfg), models=model,
                  hourly=",".join(w["single_run_variables"]),
                  run=init.strftime("%Y-%m-%dT%H:%M"),
                  forecast_hours=w["single_run_forecast_hours"])
    data = client.get(SINGLE_RUNS_URL, params, offline=offline)
    if data.get("error"):
        return None
    h = data["hourly"]
    df = pd.DataFrame(h).rename(columns={"time": "valid_time"})
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df.insert(0, "model", model)
    df.insert(1, "init_time", init)
    df.insert(3, "lead_h", ((df["valid_time"] - init) / pd.Timedelta("1h")).astype(int))
    var_cols = w["single_run_variables"]
    df = df.dropna(subset=var_cols, how="all")
    return df[STORE_KEYS + var_cols]


def fetch_previous_runs(client: OpenMeteoClient, cfg: dict, model: str,
                        start: str, end: str, offline: bool = False) -> pd.DataFrame:
    """Values predicted 1..N days before valid time, reshaped to store format."""
    w = cfg["weather"]
    variables, days = w["prev_run_variables"], w["prev_run_days"]
    hourly = [f"{v}_previous_day{n}" for n in days for v in variables]
    params = dict(**_base_params(cfg), models=model, hourly=",".join(hourly),
                  start_date=start, end_date=end)
    data = client.get(PREVIOUS_RUNS_URL, params, offline=offline)
    if data.get("error"):
        raise RuntimeError(f"previous-runs {model} {start}..{end}: {data.get('reason')}")
    h = data["hourly"]
    valid = pd.to_datetime(h["time"])
    frames = []
    for n in days:
        cols = {v: h.get(f"{v}_previous_day{n}") for v in variables}
        df = pd.DataFrame({v: (c if c is not None else np.nan) for v, c in cols.items()})
        df.insert(0, "valid_time", valid)
        df = df.dropna(subset=variables, how="all")
        df["init_time"] = df["valid_time"].dt.floor("6h") - pd.Timedelta(days=n)
        df["lead_h"] = ((df["valid_time"] - df["init_time"]) / pd.Timedelta("1h")).astype(int)
        df["model"] = model
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out[STORE_KEYS + variables]


def month_chunks(start: str, end: str, months: int = 3) -> list[tuple[str, str]]:
    """Split [start, end] into ~`months`-long date ranges (inclusive, ISO dates)."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    while s <= e:
        nxt = min(s + pd.DateOffset(months=months) - pd.Timedelta(days=1), e)
        out.append((s.date().isoformat(), nxt.date().isoformat()))
        s = nxt + pd.Timedelta(days=1)
    return out
