import numpy as np
import pandas as pd
import pytest

from hackalem.config import load_config
from hackalem.timeutils import issue_time_utc, target_hours_local
from hackalem.weather.store import add_availability, get_forecast, load_store, store_path

CFG = {"weather": {"availability_delay_h": {"m": 6}}}


def _synthetic_store():
    rows = []
    for init in pd.date_range("2026-01-01 00:00", "2026-01-03 18:00", freq="6h"):
        for lead in range(0, 73):
            rows.append({"model": "m", "init_time": init, "lead_h": lead,
                         "valid_time": init + pd.Timedelta(hours=lead),
                         "ws": init.value % 97 + lead})  # value encodes the run
    return add_availability(pd.DataFrame(rows), CFG)


def test_uses_latest_published_run_only():
    store = _synthetic_store()
    as_of = pd.Timestamp("2026-01-02 05:00")       # 00z run of Jan 2 not yet published (delay 6h)
    valid = pd.date_range("2026-01-02 19:00", periods=48, freq="h")
    fc = get_forecast(store, as_of, valid)
    assert (fc["available_at"] <= as_of).all()
    assert (fc["init_time"] == pd.Timestamp("2026-01-01 18:00")).all()
    assert len(fc) == 48


def test_result_independent_of_future_runs():
    """Deleting every run published after as_of must not change the forecast."""
    store = _synthetic_store()
    as_of = pd.Timestamp("2026-01-02 13:30")
    valid = pd.date_range("2026-01-03 00:00", periods=48, freq="h")
    full = get_forecast(store, as_of, valid)
    past_only = get_forecast(store[store["available_at"] <= as_of], as_of, valid)
    pd.testing.assert_frame_equal(full, past_only)


def test_issue_protocol():
    t = issue_time_utc("2026-01-31")
    assert t == pd.Timestamp("2026-01-31 10:00") - pd.Timedelta(hours=load_config()["scada_utc_offset_h"])
    hours = target_hours_local("2026-01-31")
    assert len(hours) == 48
    assert hours[0] == pd.Timestamp("2026-02-01 00:00") and hours[-1] == pd.Timestamp("2026-02-02 23:00")


@pytest.mark.skipif(not store_path().exists(), reason="forecast store not built")
def test_real_store_no_lookahead_for_test_issues():
    cfg = load_config()
    store = load_store(cfg)
    off = pd.Timedelta(hours=cfg["scada_utc_offset_h"])
    for d in pd.date_range(cfg["forecast"]["first_issue"], cfg["forecast"]["last_issue"]):
        as_of = issue_time_utc(d)
        valid = target_hours_local(d) - off
        fc = get_forecast(store, as_of, valid)
        assert (fc["init_time"] + pd.to_timedelta(fc["model"].map(
            cfg["weather"]["availability_delay_h"]), unit="h") <= as_of).all()
        assert (fc["valid_time"] > as_of).all()
        ec = fc[fc["model"] == "ecmwf_ifs"]
        assert len(ec) == 48, f"ECMWF incomplete for issue {d.date()}"
        assert not np.isnan(ec["wind_speed_100m"]).any()
