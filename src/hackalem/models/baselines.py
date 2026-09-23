"""Reference forecasts the ML model has to beat.

Every baseline: fit(train_features) -> self ; predict(features) -> np.ndarray in [0, 1].
Fitting uses only rows whose target hours precede the evaluated period.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

from hackalem.data.turbines import load_hourly


class Climatology:
    """Mean power for (month, hour) over the training period."""
    name = "climatology"

    def fit(self, train: pd.DataFrame, target: str = "y_plant"):
        d = train.dropna(subset=[target])
        self.table = d.groupby([d["time_local"].dt.month, d["time_local"].dt.hour])[target].mean()
        self.overall = d[target].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([df["time_local"].dt.month, df["time_local"].dt.hour])
        return self.table.reindex(idx).fillna(self.overall).to_numpy()


class Persistence:
    """Power at the same hour on the issue day (D), i.e. 'tomorrow = today'.

    Reference only: it needs SCADA up to the issue time, which the organisers'
    data does not contain for February issues.
    """
    name = "persistence_D"

    def fit(self, train=None, target: str = "y_plant"):
        self.series = load_hourly()["power_plant"]
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # same clock hour on the issue day; hours after 10:00 of day D are not yet
        # known at issue time -> use day D-1 for those
        h = df["time_local"].dt.hour
        base = df["issue_date"] + pd.to_timedelta(h, unit="h")
        base = base.where(h < 10, base - pd.Timedelta(days=1))
        return self.series.reindex(base).to_numpy()


class MeasuredPowerCurve:
    """Empirical turbine curve: measured hub wind -> power (isotonic, SCADA only),
    applied to a NWP wind column. This is the classic 'NWP + power curve' method."""

    def __init__(self, wind_col: str, calibrate: bool = False):
        self.wind_col = wind_col
        self.calibrate = calibrate
        self.name = f"curve[{wind_col}]" + ("+linear_cal" if calibrate else "")

    def fit(self, train: pd.DataFrame, target: str = "y_plant"):
        h = load_hourly()
        h = h[h.index < train["time_local"].max()].dropna(subset=["ws_plant", "power_plant"])
        self.curve = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
            h["ws_plant"], h["power_plant"])
        if self.calibrate:   # measured = a * nwp + b (internship approach)
            d = train.dropna(subset=["ws_measured", self.wind_col])
            self.cal = LinearRegression().fit(d[[self.wind_col]], d["ws_measured"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        w = df[self.wind_col].to_numpy(float)
        if self.calibrate:
            ok = ~np.isnan(w)
            w = w.copy()
            if ok.any():
                w[ok] = self.cal.predict(pd.DataFrame({self.wind_col: w[ok]}))
        out = np.full(len(w), np.nan)
        ok = ~np.isnan(w)
        if ok.any():                      # a source may be entirely missing
            out[ok] = self.curve.predict(np.clip(w[ok], 0, None))
        return out


class DirectIsotonic:
    """Isotonic map fitted directly from a NWP wind column to power on
    (NWP forecast, actual power) training pairs. Absorbs NWP bias and smoothing."""

    def __init__(self, wind_col: str):
        self.wind_col = wind_col
        self.name = f"direct_iso[{wind_col}]"

    def fit(self, train: pd.DataFrame, target: str = "y_plant"):
        d = train.dropna(subset=[self.wind_col, target])
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
            d[self.wind_col], d[target])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        w = df[self.wind_col].to_numpy(float)
        out = np.full(len(w), np.nan)
        ok = ~np.isnan(w)
        if ok.any():
            out[ok] = self.iso.predict(w[ok])
        return out


def default_baselines() -> list:
    return [
        Climatology(),
        Persistence(),
        MeasuredPowerCurve("wind_speed_100m__ecmwf_ifs"),
        MeasuredPowerCurve("ens_ws100_mean"),
        MeasuredPowerCurve("ens_ws100_mean", calibrate=True),
        DirectIsotonic("wind_speed_100m__ecmwf_ifs"),
        DirectIsotonic("ens_ws100_mean"),
        DirectIsotonic("ens_ws100_mid"),
    ]


def fill_with(pred: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Replace NaN (missing NWP) with a fallback forecast so every hour is covered."""
    return np.where(np.isnan(pred), fallback, pred)
