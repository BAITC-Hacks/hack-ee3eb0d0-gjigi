import pandas as pd
import pytest

from hackalem.features.build import build_features, feature_columns
from hackalem.timeutils import issue_time_utc
from hackalem.weather.issued import issued_forecasts
from hackalem.weather.store import load_store, store_path

pytestmark = pytest.mark.skipif(not store_path().exists(), reason="forecast store not built")


@pytest.fixture(scope="module")
def store():
    return load_store()


def test_no_scada_in_features(store):
    f = build_features(issued_forecasts(["2025-06-10"], store))
    cols = feature_columns(f)
    assert not any(c.startswith("y_") or "measured" in c or "power" in c for c in cols)
    assert len(f) == 48 and f["day_ahead"].isin([1, 2]).all()


def test_features_ignore_future_runs(store):
    """Features for a test issue are identical when every later run is deleted."""
    d = "2026-02-10"
    past = store[store["available_at"] <= issue_time_utc(d)]
    a = build_features(issued_forecasts([d], store), with_targets=False)
    b = build_features(issued_forecasts([d], past), with_targets=False)
    pd.testing.assert_frame_equal(a, b)
