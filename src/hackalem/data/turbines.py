"""Load turbine SCADA data, flag bad samples, aggregate to hourly.

Pipeline:
    load_raw()      10-min CSV  -> DataFrame[time, ws, power, temp]
    flag_quality()  adds boolean flags + `valid`
    to_hourly()     10-min -> hourly (start-of-hour label), only valid samples
    load_hourly()   both turbines + plant average, cached to parquet
"""

import numpy as np
import pandas as pd

from hackalem.config import load_config

RAW_COLUMNS = {
    "Статистическое время": "time",
    "Средняя скорость ветра(m/s)": "ws",
    "Нормализованная активная мощность": "power",
    "Средняя температура окружающей среды(°C)": "temp",
}


def load_raw(turbine: str) -> pd.DataFrame:
    cfg = load_config()
    path = cfg["paths"]["raw_dir"] / cfg["turbines"][turbine]["file"]
    df = pd.read_csv(path).rename(columns=RAW_COLUMNS)[list(RAW_COLUMNS.values())]
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def _stuck_mask(s: pd.Series, run_len: int) -> pd.Series:
    """True where the value is part of a run of >= run_len identical consecutive values."""
    run_id = (s != s.shift()).cumsum()
    run_size = s.groupby(run_id).transform("size")
    return run_size >= run_len


def _below_curve_mask(ws: pd.Series, power: pd.Series, q: dict) -> pd.Series:
    """Samples far below the empirical power curve (curtailment / derating).

    Robust per-bin threshold: median - k * 1.4826 * MAD, computed on samples
    that are not already flagged as downtime.
    """
    bins = (ws / q["curve_bin_ms"]).round().astype(int)
    grp = power.groupby(bins)
    med = grp.transform("median")
    mad = (power - med).abs().groupby(bins).transform("median")
    thr = med - q["curve_mad_k"] * 1.4826 * mad - 0.05
    return (ws >= q["cut_in_ms"] + 1.0) & (power < thr)


def flag_quality(df: pd.DataFrame) -> pd.DataFrame:
    q = load_config()["quality"]
    df = df.copy()
    df["flag_downtime"] = (df["ws"] >= q["downtime_ws_ms"]) & (df["power"] <= q["downtime_power"])
    df["flag_stuck"] = _stuck_mask(df["ws"], q["stuck_run_len"]) & (df["ws"] > 0)
    ok = ~(df["flag_downtime"] | df["flag_stuck"])
    df["flag_below_curve"] = False
    df.loc[ok, "flag_below_curve"] = _below_curve_mask(df.loc[ok, "ws"], df.loc[ok, "power"], q)
    df["valid"] = ~(df["flag_downtime"] | df["flag_stuck"] | df["flag_below_curve"])
    return df


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid 10-min samples to hourly means labelled by hour start.

    Hours with fewer than `min_samples_per_hour` valid samples get NaN power
    (kept in the index so gaps stay visible). Raw (unfiltered) power is kept
    too, because the real plant output includes downtime.
    """
    q = load_config()["quality"]
    t = df.set_index("time")
    hour = t.index.floor("h")
    valid = t[t["valid"]]
    agg = pd.DataFrame({
        "ws": valid["ws"].groupby(valid.index.floor("h")).mean(),
        "power": valid["power"].groupby(valid.index.floor("h")).mean(),
        "temp": t["temp"].groupby(hour).mean(),
        "power_raw": t["power"].groupby(hour).mean(),
        "n_samples": t["power"].groupby(hour).size(),
        "n_valid": valid["power"].groupby(valid.index.floor("h")).size(),
    })
    full = pd.date_range(agg.index.min(), agg.index.max(), freq="h", name="time")
    agg = agg.reindex(full)
    agg[["n_samples", "n_valid"]] = agg[["n_samples", "n_valid"]].fillna(0).astype(int)
    agg.loc[agg["n_valid"] < q["min_samples_per_hour"], ["ws", "power"]] = np.nan
    agg.loc[agg["n_samples"] < q["min_samples_per_hour"], ["temp", "power_raw"]] = np.nan
    return agg


def load_hourly(refresh: bool = False) -> pd.DataFrame:
    """Hourly wide table: <col>_T1, <col>_T2 and plant-level `power_plant`.

    `power_plant` = mean of available turbines' valid hourly power
    (normalised, 0..1); NaN only if both turbines are missing.
    """
    cfg = load_config()
    cache = cfg["paths"]["processed_dir"] / "hourly.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    parts = []
    for name in cfg["turbines"]:
        h = to_hourly(flag_quality(load_raw(name)))
        parts.append(h.add_suffix(f"_{name}"))
    out = pd.concat(parts, axis=1)
    out.index.name = "time"
    out["power_plant"] = out[[f"power_{n}" for n in cfg["turbines"]]].mean(axis=1)
    out["ws_plant"] = out[[f"ws_{n}" for n in cfg["turbines"]]].mean(axis=1)

    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out
