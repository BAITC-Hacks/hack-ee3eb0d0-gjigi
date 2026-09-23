"""On-demand NWP download for one forecast issue (used by the agent's fetch_weather tool).

For issue day D at time `as_of` the agent needs:
  * ECMWF 9 km: the latest runs whose publication time <= as_of (+ poll window)
  * GFS / ICON / ECMWF 0.25: previous-runs values for the target days
It queries Open-Meteo (responses cached on disk). Without network it falls back
to the forecast archive shipped in the repo, and says so in its report.
Whatever is fetched, the availability rule (init + delay <= as_of) still decides
what the model may use.
"""

import pandas as pd

from hackalem.config import load_config
from hackalem.timeutils import offset, target_hours_local
from hackalem.weather.openmeteo import OpenMeteoClient
from hackalem.weather.sources import fetch_previous_runs, fetch_single_run
from hackalem.weather.store import add_availability

PAD_H = 3


def _client(cfg) -> OpenMeteoClient:
    return OpenMeteoClient(cfg["paths"]["http_cache_dir"], cfg["weather"]["request_interval_s"])


def valid_window_utc(issue_date) -> tuple[pd.Timestamp, pd.Timestamp]:
    t = target_hours_local(issue_date)
    pad = pd.Timedelta(hours=PAD_H)
    return t[0] - pad - offset(), t[-1] + pad - offset()


def fetch_for_issue(issue_date, as_of_utc: pd.Timestamp, archive: pd.DataFrame | None = None,
                    poll_hours: float = 0, n_runs: int = 2, offline: bool = False) -> tuple[pd.DataFrame, dict]:
    """Download NWP rows for one issue. Returns (store rows, report)."""
    cfg = load_config()
    w = cfg["weather"]
    client = _client(cfg)
    vmin, vmax = valid_window_utc(issue_date)
    horizon = as_of_utc + pd.Timedelta(hours=poll_hours)
    frames, report = [], {}

    def from_archive(model):
        if archive is None:
            return None
        a = archive[(archive["model"] == model) & archive["valid_time"].between(vmin, vmax)]
        return a.drop(columns=["available_at"], errors="ignore")

    # 1. ECMWF exact runs published by `horizon`
    for m in w["single_run_models"]:
        latest = (horizon - pd.Timedelta(hours=w["availability_delay_h"][m])).floor("6h")
        inits = [latest - pd.Timedelta(hours=6 * k) for k in range(n_runs + 2)]
        got, src = [], set()
        for init in inits:
            if init.hour not in w["single_run_hours"] or len(got) >= n_runs:
                continue
            try:
                df = fetch_single_run(client, cfg, m, init, offline=offline)
                src.add("cache" if client.last_from_cache else "api")
            except Exception:  # noqa: BLE001 - network problem -> archive fallback below
                df = None
                src.add("error")
            if df is not None:
                got.append(df[df["valid_time"].between(vmin, vmax)])
        if got:
            frames += got
            report[m] = {"source": "+".join(sorted(src - {"error"})) or "api",
                         "runs": [f"{d['init_time'].iloc[0]:%Y-%m-%d %H}z" for d in got if len(d)]}
        else:
            a = from_archive(m)
            if a is not None and len(a):
                frames.append(a)
            report[m] = {"source": "local archive (offline)" if a is not None else "unavailable",
                         "runs": sorted({f"{t:%Y-%m-%d %H}z" for t in a["init_time"]})[-n_runs:] if a is not None else []}

    # 2. Multi-model previous runs for the target days
    for m in w["prev_run_models"]:
        try:
            df = fetch_previous_runs(client, cfg, m, str(vmin.date()), str(vmax.date()), offline=offline)
            frames.append(df[df["valid_time"].between(vmin, vmax)])
            report[m] = {"source": "cache" if client.last_from_cache else "api", "rows": int(len(df))}
        except Exception as e:  # noqa: BLE001
            a = from_archive(m)
            if a is not None and len(a):
                frames.append(a)
            report[m] = {"source": "local archive (offline)" if a is not None else f"unavailable: {e}"}

    rows = pd.concat([f for f in frames if f is not None and len(f)], ignore_index=True) \
        if frames else pd.DataFrame()
    if len(rows):
        rows = add_availability(rows.drop_duplicates(["model", "init_time", "valid_time"]), cfg)
    return rows, report
