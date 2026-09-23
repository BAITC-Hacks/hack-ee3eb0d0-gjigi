"""Time conventions.

* NWP data and the forecast store: naive UTC.
* SCADA and forecast output: naive local plant time = UTC + `scada_utc_offset_h`
  (a fixed offset, see config).
"""

import pandas as pd

from hackalem.config import load_config


def offset() -> pd.Timedelta:
    return pd.Timedelta(hours=load_config()["scada_utc_offset_h"])


def local_to_utc(t):
    return t - offset()


def utc_to_local(t):
    return t + offset()


def issue_time_utc(issue_date: str | pd.Timestamp) -> pd.Timestamp:
    """UTC time at which the forecast for local day `issue_date` is issued."""
    cfg = load_config()
    local = pd.Timestamp(f"{pd.Timestamp(issue_date).date()} {cfg['forecast']['issue_time_local']}")
    return local_to_utc(local)


def target_hours_local(issue_date: str | pd.Timestamp) -> pd.DatetimeIndex:
    """Local hourly timestamps forecast on `issue_date` (days D+1..D+max)."""
    cfg = load_config()
    d = pd.Timestamp(issue_date).normalize()
    days = cfg["forecast"]["target_days"]
    start = d + pd.Timedelta(days=min(days))
    end = d + pd.Timedelta(days=max(days) + 1) - pd.Timedelta(hours=1)
    return pd.date_range(start, end, freq="h")
