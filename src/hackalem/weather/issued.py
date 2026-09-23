"""Replay the forecast issue protocol over history.

For every issue day D, collect what each NWP model had published by the issue
time for local days D+1..D+2. The result has the same shape the live system sees
on 2026-01-31..2026-02-27, so a model trained on it is trained exactly as it will be used.
"""

import pandas as pd

from hackalem.config import load_config
from hackalem.timeutils import issue_time_utc, offset, target_hours_local
from hackalem.weather.store import get_forecast, load_store


def issued_forecasts(issue_dates, store: pd.DataFrame | None = None,
                     pad_h: int = 3, as_of_utc: pd.Timestamp | None = None) -> pd.DataFrame:
    """Long frame: issue_date, issue_time_utc, time_local, is_target, valid_time(UTC),
    model, init_time, available_at, lead_h, lead_from_issue_h, <variables...>

    `pad_h` extra hours on each side of the 48 target hours (is_target=False)
    let features look at neighbouring NWP hours. They come from the same
    runs, published before the issue time, so padding cannot leak.
    `as_of_utc` overrides the protocol issue time (e.g. a later re-run).
    """
    store = load_store() if store is None else store
    off = offset()
    pad = pd.Timedelta(hours=pad_h)
    out = []
    for d in pd.DatetimeIndex(issue_dates):
        as_of = as_of_utc if as_of_utc is not None else issue_time_utc(d)
        target = target_hours_local(d)
        local = pd.date_range(target[0] - pad, target[-1] + pad, freq="h")
        fc = get_forecast(store, as_of, local - off)
        fc.insert(0, "issue_date", d.normalize())
        fc.insert(1, "issue_time_utc", as_of)
        fc.insert(2, "time_local", fc["valid_time"] + off)
        fc.insert(3, "is_target", fc["time_local"].isin(target))
        out.append(fc)
    return pd.concat(out, ignore_index=True)


def build_issued_archive(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Issued forecasts for every day of the archive; cached to data/weather/issued_forecasts.parquet."""
    cfg = load_config()
    w = cfg["weather"]
    start = pd.Timestamp(start or w["archive_start"]) + pd.Timedelta(days=1)
    end = pd.Timestamp(end or cfg["forecast"]["last_issue"])
    df = issued_forecasts(pd.date_range(start, end, freq="D"))
    path = cfg["paths"]["weather_dir"] / "issued_forecasts.parquet"
    df.to_parquet(path, index=False)
    return df
