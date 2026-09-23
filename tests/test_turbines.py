import numpy as np
import pandas as pd

from hackalem.data.turbines import flag_quality, load_hourly, to_hourly


def _synthetic(n_hours=3):
    t = pd.date_range("2025-01-01", periods=6 * n_hours, freq="10min")
    ws = np.full(len(t), 8.0)
    power = np.full(len(t), 0.5)
    power[6:12] = 0.0  # hour 2: turbine stopped at 8 m/s -> downtime
    ws[12:] = ws[12:] + np.arange(len(t) - 12) * 0.01  # avoid stuck-sensor flag
    ws[:12] = ws[:12] + np.arange(12) * 0.01
    return pd.DataFrame({"time": t, "ws": ws, "power": power, "temp": 0.0})


def test_downtime_hour_becomes_nan_but_raw_kept():
    h = to_hourly(flag_quality(_synthetic()))
    assert np.isnan(h["power"].iloc[1])
    assert h["power_raw"].iloc[1] == 0.0
    assert h["power"].iloc[0] == 0.5


def test_hourly_table_is_regular_and_bounded():
    h = load_hourly()
    assert h.index.is_monotonic_increasing and h.index.is_unique
    assert (pd.Series(h.index).diff().dropna() == pd.Timedelta("1h")).all()
    p = h["power_plant"].dropna()
    assert p.between(0, 1).all()
    assert h.index.max() < pd.Timestamp("2026-02-01"), "test period must not be in training data"
