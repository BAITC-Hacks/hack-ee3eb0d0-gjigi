"""Metrics and validation splits.

Power is normalised to rated capacity, so MAE is already nMAE (share of capacity).
"""

import numpy as np
import pandas as pd

from hackalem.config import load_config


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = ~(np.isnan(y) | np.isnan(p))
    y, p = y[m], p[m]
    e = p - y
    return {"n": int(m.sum()),
            "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt((e ** 2).mean())),
            "bias": float(e.mean()),
            "corr": float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 else np.nan}


def evaluate(df: pd.DataFrame, pred_cols: list[str], target: str = "y_plant",
             by: list[str] | None = None) -> pd.DataFrame:
    """Metrics per prediction column (and optional grouping), rows with a target only."""
    rows = []
    d = df[df[target].notna()]
    groups = d.groupby(by) if by else [((), d)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        for c in pred_cols:
            r = {"model": c, **dict(zip(by or [], key)), **metrics(g[target], g[c])}
            rows.append(r)
    return pd.DataFrame(rows)


def add_skill(table: pd.DataFrame, reference: str, by: list[str] | None = None) -> pd.DataFrame:
    """skill = 1 - MAE / MAE(reference) within each group."""
    by = by or []
    ref = table[table["model"] == reference].set_index(by)["mae"] if by else \
        table.loc[table["model"] == reference, "mae"].iloc[0]
    t = table.copy()
    t["skill_vs_" + reference] = 1 - t["mae"] / (t.set_index(by).index.map(ref) if by else ref)
    return t


def holdout_split(df: pd.DataFrame):
    """Train: target hours before holdout_start; holdout: Dec 2025 - Jan 2026."""
    v = load_config()["validation"]
    t = df["time_local"]
    train = df[t < v["holdout_start"]]
    hold = df[(t >= v["holdout_start"]) & (t <= v["holdout_end"])]
    return train, hold


def rolling_folds(df: pd.DataFrame):
    """Yield (month, train, test): train on all target hours before the month."""
    v = load_config()["validation"]
    for m in pd.period_range(v["cv_first_month"], v["cv_last_month"], freq="M"):
        start, end = m.start_time, m.end_time
        t = df["time_local"]
        yield str(m), df[t < start], df[(t >= start) & (t <= end)]
