"""Main model: LightGBM quantile regression on issued-NWP features.

* Quantiles P10 / P50 / P90 (+ an L2 "mean" model): P50 minimises MAE, mean
  minimises RMSE, P10-P90 gives an 80% interval for the agent and for bidding.
* Physics prior: the measured turbine power curve applied to the NWP ensemble
  wind (stage-2 best baseline) is added as features; the GBM learns the corrections
  (seasonal bias, direction, stability, model disagreement).
* Conformalised interval (CQR): the P10/P90 band is widened by a margin chosen on
  the held-back tail of the training period so that it really covers ~80%.
* Number of boosting rounds is chosen by early stopping on the last
  `early_stop_days` of the training period, then the model is refit on all of it.
"""

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from hackalem.config import load_config
from hackalem.features.build import feature_columns
from hackalem.models.baselines import MeasuredPowerCurve

QUANTILES = (0.1, 0.5, 0.9)
CURVE_INPUTS = ["ens_ws100_mean", "ens_ws100_mid", "ens_ws100_roll3", "wind_speed_100m__ecmwf_ifs"]

DEFAULT_PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_child_samples=40,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, verbose=-1,
)


class QuantileGBM:
    def __init__(self, target: str = "y_plant", features: list[str] | None = None,
                 use_curve: bool = True, curve_inputs: list[str] | None = None,
                 params: dict | None = None,
                 max_rounds: int = 3000, early_stop_days: int = 60, seed: int | None = None):
        self.target = target
        self.features = features
        self.use_curve = use_curve
        self.curve_inputs = curve_inputs or CURVE_INPUTS
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.params["seed"] = seed if seed is not None else load_config()["random_state"]
        self.max_rounds = max_rounds
        self.early_stop_days = early_stop_days

    # ── physics prior ────────────────────────────────────────────────────────
    def _add_curve(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.use_curve:
            return df
        df = df.copy()
        for c in self.curve_inputs:
            df[f"curve__{c}"] = self.curves[c].predict(df)
        return df

    # ── training ─────────────────────────────────────────────────────────────
    def _objective(self, q):
        if q == "mean":
            return {"objective": "regression"}
        return {"objective": "quantile", "alpha": q}

    def fit(self, train: pd.DataFrame, rounds: dict | None = None):
        """rounds: fixed boosting rounds per quantile (skips early stopping)."""
        if rounds is not None:
            return self._fit_fixed(train, rounds)
        train = train.dropna(subset=[self.target]).sort_values("time_local")
        if self.use_curve:
            self.curves = {c: MeasuredPowerCurve(c).fit(train) for c in self.curve_inputs}
        tr = self._add_curve(train)
        base = self.features or feature_columns(train)
        self.feature_names_ = base + ([f"curve__{c}" for c in self.curve_inputs] if self.use_curve else [])

        cut = tr["time_local"].max() - pd.Timedelta(days=self.early_stop_days)
        fit_part, val_part = tr[tr["time_local"] <= cut], tr[tr["time_local"] > cut]
        self.models_, self.best_rounds_ = {}, {}
        val_pred = {}
        for q in (*QUANTILES, "mean"):
            p = {**self.params, **self._objective(q)}
            dtr = lgb.Dataset(fit_part[self.feature_names_], fit_part[self.target])
            dva = lgb.Dataset(val_part[self.feature_names_], val_part[self.target])
            m = lgb.train(p, dtr, self.max_rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            n = max(int(m.best_iteration * 1.1), 50)   # a bit more data -> a bit more rounds
            self.best_rounds_[q] = n
            val_pred[q] = m.predict(val_part[self.feature_names_], num_iteration=m.best_iteration)
            full = lgb.Dataset(tr[self.feature_names_], tr[self.target])
            self.models_[q] = lgb.train(p, full, n)
        self._conformal(val_part, val_pred)
        return self

    def _fit_fixed(self, train: pd.DataFrame, rounds: dict):
        train = train.dropna(subset=[self.target]).sort_values("time_local")
        if self.use_curve:
            self.curves = {c: MeasuredPowerCurve(c).fit(train) for c in self.curve_inputs}
        tr = self._add_curve(train)
        base = self.features or feature_columns(train)
        self.feature_names_ = base + ([f"curve__{c}" for c in self.curve_inputs] if self.use_curve else [])
        self.models_, self.best_rounds_ = {}, dict(rounds)
        for q in (*QUANTILES, "mean"):
            p = {**self.params, **self._objective(q)}
            self.models_[q] = lgb.train(p, lgb.Dataset(tr[self.feature_names_], tr[self.target]), rounds[q])
        self.cqr_margin_ = {1: 0.0, 2: 0.0}
        return self

    def raw_quantiles(self, df: pd.DataFrame) -> dict:
        X = self._add_curve(df).reindex(columns=self.feature_names_)
        return {q: self.models_[q].predict(X) for q in QUANTILES}

    def calibrate(self, df: pd.DataFrame, level: float = 0.8):
        """Set the CQR margin from out-of-sample data `df` (must not be in training)."""
        d = df.dropna(subset=[self.target])
        self._conformal(d, self.raw_quantiles(d), level)
        return self

    def _conformal(self, val: pd.DataFrame, pred: dict, level: float = 0.8):
        """CQR margin per day-ahead: score = max(p10 - y, y - p90)."""
        y = val[self.target].to_numpy()
        lo = np.clip(np.minimum(pred[QUANTILES[0]], pred[QUANTILES[-1]]), 0, 1)
        hi = np.clip(np.maximum(pred[QUANTILES[0]], pred[QUANTILES[-1]]), 0, 1)
        score = np.maximum(lo - y, y - hi)
        self.cqr_margin_ = {}
        for d in sorted(val["day_ahead"].unique()):
            s = score[(val["day_ahead"] == d).to_numpy()]
            k = min(np.ceil((len(s) + 1) * level) / len(s), 1.0)
            self.cqr_margin_[int(d)] = float(np.quantile(s, k))

    # ── inference ────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # missing columns (e.g. an NWP source excluded by the agent) -> NaN, handled by LightGBM
        X = self._add_curve(df).reindex(columns=self.feature_names_)
        out = pd.DataFrame({f"p{int(q * 100):02d}": self.models_[q].predict(X) for q in QUANTILES},
                           index=df.index)
        # non-crossing quantiles, physical bounds
        out[:] = np.sort(np.clip(out.to_numpy(), 0, 1), axis=1)
        out["mean"] = np.clip(self.models_["mean"].predict(X), 0, 1)
        margin = df["day_ahead"].map(self.cqr_margin_).fillna(max(self.cqr_margin_.values()))
        out["p10"] = np.clip(out["p10"] - margin, 0, 1)
        out["p90"] = np.clip(out["p90"] + margin, 0, 1)
        return out

    def feature_importance(self, q=0.5) -> pd.Series:
        m = self.models_[q]
        return pd.Series(m.feature_importance("gain"), index=self.feature_names_) \
            .sort_values(ascending=False)

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "QuantileGBM":
        with open(path, "rb") as f:
            return pickle.load(f)


def pinball(y, q_pred, q) -> float:
    y, q_pred = np.asarray(y), np.asarray(q_pred)
    d = y - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def interval_metrics(y, p10, p90) -> dict:
    y = np.asarray(y)
    ok = ~np.isnan(y)
    return {"coverage_80": float(((y >= p10) & (y <= p90))[ok].mean()),
            "width_80": float((np.asarray(p90) - np.asarray(p10))[ok].mean())}
