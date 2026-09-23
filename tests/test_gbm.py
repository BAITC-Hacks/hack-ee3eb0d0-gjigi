import numpy as np
import pytest

from hackalem.config import load_config
from hackalem.models.gbm import QuantileGBM

pytestmark = pytest.mark.skipif(
    not (load_config()["paths"]["weather_dir"] / "forecast_store.parquet").exists(),
    reason="forecast store not built")


@pytest.fixture(scope="module")
def fitted():
    from hackalem.features.build import load_features
    f = load_features()
    tr = f[f["time_local"].between("2025-03-01", "2025-08-31")]
    te = f[f["time_local"].between("2025-09-01", "2025-09-15")]
    return QuantileGBM(max_rounds=200, early_stop_days=30).fit(tr), te


def test_quantiles_ordered_and_bounded(fitted):
    model, te = fitted
    p = model.predict(te)
    assert list(p.columns) == ["p10", "p50", "p90", "mean"]
    assert ((p["p10"] <= p["p50"]) & (p["p50"] <= p["p90"])).all()
    assert ((p >= 0) & (p <= 1)).all().all()
    assert set(model.cqr_margin_) == {1, 2}


def test_model_never_sees_targets_as_features(fitted):
    model, _ = fitted
    assert not any(c.startswith("y_") or "measured" in c for c in model.feature_names_)


def test_predict_is_deterministic(fitted):
    model, te = fitted
    np.testing.assert_array_equal(model.predict(te).to_numpy(), model.predict(te).to_numpy())
