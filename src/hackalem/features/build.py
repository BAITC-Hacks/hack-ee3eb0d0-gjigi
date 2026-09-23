"""Feature table: one row per (issue_date, time_local) target hour.

Inputs are ONLY what is known at issue time:
  * archived NWP forecasts published before the issue (weather/issued.py)
  * calendar / horizon information
SCADA measurements are joined solely as targets (`y_*` columns) and are never
used to build features: the organisers' SCADA ends on 2026-01-31, so any
"recent power" feature would not exist for February issues.
"""

import numpy as np
import pandas as pd

from hackalem.config import load_config
from hackalem.data.turbines import load_hourly

KEYS = ["issue_date", "time_local"]
PER_MODEL_VARS = ["wind_speed_100m", "wind_speed_10m", "wind_gusts_10m", "wd100_sin",
                  "wd100_cos", "temperature_2m", "rho", "shear", "lead_h"]
ECMWF_ONLY_VARS = ["boundary_layer_height", "cape", "relative_humidity_2m"]
MAIN_MODEL = "ecmwf_ifs"
TARGETS = ["y_plant", "y_T1", "y_T2", "y_plant_raw"]


def _derive_row_vars(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    rad = np.deg2rad(d["wind_direction_100m"])
    d["wd100_sin"], d["wd100_cos"] = np.sin(rad), np.cos(rad)
    # air density from NWP surface pressure (hPa) and 2 m temperature
    d["rho"] = d["surface_pressure"] * 100 / (287.05 * (d["temperature_2m"] + 273.15))
    # wind shear exponent between 10 m and 100 m (stability proxy)
    ratio = d["wind_speed_100m"] / d["wind_speed_10m"].clip(lower=0.3)
    d["shear"] = (np.log(ratio.clip(lower=0.1)) / np.log(10)).clip(-0.5, 1.5)
    return d


def _time_shift_features(wide: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """Neighbour-hour features of one NWP series, computed inside each issue."""
    g = wide.groupby(level="issue_date")[col]
    out = pd.DataFrame(index=wide.index)
    out[f"{prefix}_lag1"] = g.shift(1)
    out[f"{prefix}_lead1"] = g.shift(-1)
    # SCADA hour h is the mean over [h, h+1); NWP is instantaneous -> mid-interval
    out[f"{prefix}_mid"] = (wide[col] + out[f"{prefix}_lead1"]) / 2
    out[f"{prefix}_roll3"] = g.transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
    out[f"{prefix}_roll6"] = g.transform(lambda s: s.rolling(6, center=True, min_periods=1).mean())
    out[f"{prefix}_diff"] = out[f"{prefix}_lead1"] - out[f"{prefix}_lag1"]
    return out


def build_features(issued: pd.DataFrame, with_targets: bool = True) -> pd.DataFrame:
    """issued: output of weather.issued.issued_forecasts (long, with padding)."""
    d = _derive_row_vars(issued)
    wide = d.pivot_table(index=KEYS, columns="model", values=PER_MODEL_VARS, aggfunc="first")
    wide.columns = [f"{v}__{m}" for v, m in wide.columns]
    ec = d[d["model"] == MAIN_MODEL].set_index(KEYS)[ECMWF_ONLY_VARS].add_suffix(f"__{MAIN_MODEL}")
    wide = wide.join(ec)
    is_target = d.groupby(KEYS)["is_target"].first()
    wide = wide.sort_index()

    models = sorted(d["model"].unique())
    ws = wide[[f"wind_speed_100m__{m}" for m in models]]
    f = pd.DataFrame(index=wide.index)
    f["ens_ws100_mean"] = ws.mean(axis=1)
    f["ens_ws100_median"] = ws.median(axis=1)
    f["ens_ws100_std"] = ws.std(axis=1)
    f["ens_ws100_min"] = ws.min(axis=1)
    f["ens_ws100_max"] = ws.max(axis=1)
    f["ens_n_models"] = ws.notna().sum(axis=1)
    f["ens_ws10_mean"] = wide[[f"wind_speed_10m__{m}" for m in models]].mean(axis=1)
    f["ens_wd_sin"] = wide[[f"wd100_sin__{m}" for m in models]].mean(axis=1)
    f["ens_wd_cos"] = wide[[f"wd100_cos__{m}" for m in models]].mean(axis=1)
    f["ens_rho"] = wide[[f"rho__{m}" for m in models]].mean(axis=1)
    f["ens_shear"] = wide[[f"shear__{m}" for m in models]].mean(axis=1)
    # wind power density ~ rho * v^3 (normalised to standard density)
    f["ens_ws100_mean_cube_rho"] = f["ens_ws100_mean"] ** 3 * f["ens_rho"] / 1.225
    wide = wide.join(f)
    wide = wide.join(_time_shift_features(wide, "ens_ws100_mean", "ens_ws100"))
    wide = wide.join(_time_shift_features(wide, f"wind_speed_100m__{MAIN_MODEL}", "ec_ws100"))

    wide = wide[is_target.reindex(wide.index).fillna(False).astype(bool)]
    wide = wide.reset_index()

    t = wide["time_local"]
    wide["hour"] = t.dt.hour
    wide["hour_sin"], wide["hour_cos"] = np.sin(2 * np.pi * t.dt.hour / 24), np.cos(2 * np.pi * t.dt.hour / 24)
    doy = t.dt.dayofyear
    wide["doy_sin"], wide["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    wide["month"] = t.dt.month
    wide["day_ahead"] = (t.dt.normalize() - wide["issue_date"]).dt.days
    wide["lead_from_issue_h"] = (t - wide["issue_date"]).dt.total_seconds() / 3600

    if with_targets:
        wide = attach_targets(wide)
    return wide


def attach_targets(features: pd.DataFrame) -> pd.DataFrame:
    h = load_hourly()
    y = pd.DataFrame({
        "y_plant": h["power_plant"],
        "y_T1": h["power_T1"],
        "y_T2": h["power_T2"],
        # actual output incl. downtime/curtailment (what the grid really got)
        "y_plant_raw": h[["power_raw_T1", "power_raw_T2"]].mean(axis=1),
        "ws_measured": h["ws_plant"],  # diagnostics / calibration only, never a feature
    })
    return features.merge(y, left_on="time_local", right_index=True, how="left")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All model inputs: everything except keys, targets and diagnostics."""
    exclude = set(KEYS) | set(TARGETS) | {"ws_measured", "hour"}
    return [c for c in df.columns if c not in exclude]


def load_features(refresh: bool = False) -> pd.DataFrame:
    from hackalem.weather.issued import build_issued_archive

    cfg = load_config()
    path = cfg["paths"]["processed_dir"] / "features.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    issued_path = cfg["paths"]["weather_dir"] / "issued_forecasts.parquet"
    issued = pd.read_parquet(issued_path) if issued_path.exists() and not refresh else None
    if issued is None or "is_target" not in issued.columns:
        issued = build_issued_archive()
    feats = build_features(issued)
    path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(path, index=False)
    return feats
