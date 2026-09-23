"""Forecast store: persisted archive of NWP runs + the leak-free query `get_forecast`.

The single rule that prevents look-ahead:
    a store row may be used at time `as_of` only if
        init_time + availability_delay[model] <= as_of
Among usable rows, the most recent run wins for every (model, valid_time).
"""

from pathlib import Path

import pandas as pd

from hackalem.config import load_config
from hackalem.weather.sources import STORE_KEYS

STORE_FILE = "forecast_store.parquet"


def store_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return cfg["paths"]["weather_dir"] / STORE_FILE


def load_store(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    store = pd.read_parquet(store_path(cfg))
    return add_availability(store, cfg)


def add_availability(store: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    delays = cfg["weather"]["availability_delay_h"]
    unknown = set(store["model"].unique()) - set(delays)
    if unknown:
        raise KeyError(f"no availability delay configured for models: {unknown}")
    store = store.copy()
    store["available_at"] = store["init_time"] + pd.to_timedelta(
        store["model"].map(delays), unit="h")
    return store


def get_forecast(store: pd.DataFrame, as_of: pd.Timestamp,
                 valid_times: pd.DatetimeIndex,
                 models: list[str] | None = None) -> pd.DataFrame:
    """Latest forecast per model and valid hour that was published by `as_of`.

    All times are naive UTC. Returns a long frame:
        model, valid_time, init_time, available_at, lead_h, <variables...>
    plus `lead_from_issue_h` = hours between `as_of` and valid time.
    Missing (model, valid_time) pairs are simply absent.
    """
    rows = store[(store["available_at"] <= as_of) & store["valid_time"].isin(valid_times)]
    if models is not None:
        rows = rows[rows["model"].isin(models)]
    latest = (rows.sort_values("init_time")
                  .groupby(["model", "valid_time"], as_index=False).last())
    latest["lead_from_issue_h"] = (latest["valid_time"] - as_of) / pd.Timedelta("1h")
    assert (latest["available_at"] <= as_of).all()
    return latest.sort_values(["model", "valid_time"]).reset_index(drop=True)


def to_wide(fc: pd.DataFrame, variables: list[str] | None = None) -> pd.DataFrame:
    """Long get_forecast() output -> one row per valid_time, columns `<var>__<model>`."""
    meta = {"model", "valid_time", "init_time", "available_at", "lead_from_issue_h"}
    variables = variables or [c for c in fc.columns if c not in meta]
    wide = fc.pivot(index="valid_time", columns="model", values=variables)
    wide.columns = [f"{v}__{m}" for v, m in wide.columns]
    return wide


def merge_into_store(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    parts = [p for p in (existing, new) if p is not None and len(p)]
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["model", "init_time", "valid_time"], keep="last")
    return out.sort_values(STORE_KEYS).reset_index(drop=True)
