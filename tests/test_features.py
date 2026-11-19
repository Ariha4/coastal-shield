import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader, grid_half_degree, haversine_km
from tests.conftest import make_grid_weeks


def test_haversine_one_degree_of_latitude_is_about_111_km():
    d = haversine_km(np.array([33.0]), np.array([-118.0]), 34.0, -118.0)
    assert d[0] == pytest.approx(111.2, abs=0.5)


def test_haversine_zero_distance():
    assert haversine_km(np.array([36.6]), np.array([-121.9]), 36.6, -121.9)[0] == pytest.approx(0.0)


def test_grid_snaps_to_nearest_half_degree():
    lat, lon = grid_half_degree(pd.Series([33.2, 33.3, 33.74]), pd.Series([-118.2, -118.3, -117.76]))
    assert lat.tolist() == [33.0, 33.5, 33.5]
    assert lon.tolist() == [-118.0, -118.5, -118.0]


def test_anomalies_are_computed_within_each_grid_cell():
    df = make_grid_weeks(n_weeks=6, grids=((33.0, -118.0), (34.0, -120.0)))
    df.loc[df["grid_lat"] == 33.0, "temperature"] = 10.0
    df.loc[df["grid_lat"] == 34.0, "temperature"] = 20.0
    out = DataLoader()._engineer_features(df)
    # Constant temperature inside each cell -> zero anomaly; a cross-grid leak would make it non-zero.
    assert np.allclose(out["sst_anomaly"], 0.0)
