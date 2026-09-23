"""Balancing-market economics: which forecast value to submit as the hourly plan.

With asymmetric imbalance prices the plan that minimises the expected cost is
not the median but the quantile q* of the forecast distribution (newsvendor):

    cost_h = c_under * max(plan - actual, 0) + c_over * max(actual - plan, 0)
    q*     = c_over / (c_over + c_under)

c_under - cost of each kWh the plant fails to deliver vs the plan (shortfall)
c_over  - value lost on each kWh delivered above the plan (surplus sold cheaper)
Prices are parameters in config.yaml (no market data is shipped).
"""

import numpy as np
import pandas as pd

from hackalem.config import load_config

QUANTILE_COLS = {0.1: "p10", 0.5: "p50", 0.9: "p90"}


def critical_quantile(c_under: float | None = None, c_over: float | None = None) -> float:
    e = load_config()["economics"]
    cu = e["shortfall_cost_tg_per_kwh"] if c_under is None else c_under
    co = e["surplus_loss_tg_per_kwh"] if c_over is None else c_over
    return co / (co + cu)


def plan_from_quantiles(fc: pd.DataFrame, q: float) -> np.ndarray:
    """Linear interpolation between P10/P50/P90 in quantile space (clamped outside)."""
    qs = np.array(sorted(QUANTILE_COLS))
    vals = fc[[QUANTILE_COLS[k] for k in qs]].to_numpy()
    return np.array([np.interp(q, qs, row) for row in vals])


def imbalance_cost_tg(plan, actual, c_under=None, c_over=None, mw: float = 1.0) -> float:
    """Total imbalance cost (tenge) for `mw` MW of installed capacity; inputs are
    normalised power (share of capacity) per hour, so 1.0 over one hour = 1000 kWh per MW."""
    e = load_config()["economics"]
    cu = e["shortfall_cost_tg_per_kwh"] if c_under is None else c_under
    co = e["surplus_loss_tg_per_kwh"] if c_over is None else c_over
    plan, actual = np.asarray(plan, float), np.asarray(actual, float)
    ok = ~np.isnan(actual)
    kwh = 1000.0 * mw
    short = np.maximum(plan[ok] - actual[ok], 0) * kwh
    surplus = np.maximum(actual[ok] - plan[ok], 0) * kwh
    return float(cu * short.sum() + co * surplus.sum())
