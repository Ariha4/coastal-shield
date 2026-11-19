import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader, grid_center_lat_lon, grid_half_degree, haversine_km
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


def test_grid_center_is_the_snapped_point_itself():
    # grid_half_degree rounds to the nearest 0.5, so a grid value already sits at the
    # middle of its cell; e.g. cell 33.0 spans [32.75, 33.25).
    assert grid_center_lat_lon(33.0, -118.0) == (33.0, -118.0)


def test_bloom_label_counts_observation_inside_its_own_cell():
    # Observation 30 km south of the grid point: inside the cell and inside the 50 km
    # radius. Measuring from (lat+0.25, lon+0.25) puts it ~67 km away and misses it.
    grid = pd.DataFrame({"grid_lat": [33.0], "grid_lon": [-118.0], "week_start": [pd.Timestamp("2020-01-06")]})
    obs = pd.DataFrame({"date": [pd.Timestamp("2020-01-08")], "lat": [33.0 - 30 / 111.2], "lon": [-118.0]})
    assert DataLoader()._bloom_labels(grid, obs).tolist() == [1]


def test_bloom_label_ignores_observation_outside_radius_and_window():
    grid = pd.DataFrame({"grid_lat": [33.0], "grid_lon": [-118.0], "week_start": [pd.Timestamp("2020-01-06")]})
    obs = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-08"), pd.Timestamp("2020-02-20")],
            "lat": [34.0, 33.0],  # ~111 km away / right place but 45 days later
            "lon": [-118.0, -118.0],
        }
    )
    assert DataLoader()._bloom_labels(grid, obs).tolist() == [0]


def test_lags_use_previous_calendar_weeks():
    df = make_grid_weeks(n_weeks=4, grids=((33.0, -118.0),))
    df["nitrate"] = [1.0, 2.0, 3.0, 4.0]
    df["chlorophyll"] = [10.0, 20.0, 30.0, 40.0]
    out = DataLoader()._engineer_features(df)
    assert out["nitrate_lag7"].tolist()[1:] == [1.0, 2.0, 3.0]
    assert out["nitrate_lag14"].tolist()[2:] == [1.0, 2.0]
    assert out["chlorophyll_lag7"].tolist()[1:] == [10.0, 20.0, 30.0]
    assert np.isnan(out["nitrate_lag7"].iloc[0])


def test_lags_are_nan_across_gaps_in_sparse_cruise_data():
    # Quarterly visits: the row before 2020-04-06 is 13 weeks earlier, not 1 or 2.
    df = pd.DataFrame(
        {
            "grid_lat": 33.0,
            "grid_lon": -118.0,
            "week_start": pd.to_datetime(["2020-01-06", "2020-04-06", "2020-04-13"]),
            "nitrate": [5.0, 6.0, 7.0],
            "chlorophyll": [1.0, 2.0, 3.0],
        }
    )
    out = DataLoader()._engineer_features(df)
    assert np.isnan(out["nitrate_lag7"].iloc[1])
    assert np.isnan(out["nitrate_lag14"].iloc[1])
    assert out["nitrate_lag7"].iloc[2] == 6.0
    assert np.isnan(out["nitrate_lag14"].iloc[2])


def test_lags_never_cross_grid_cells():
    df = make_grid_weeks(n_weeks=2, grids=((33.0, -118.0), (34.0, -120.0)))
    out = DataLoader()._engineer_features(df)
    first_weeks = out[out["week_start"] == out["week_start"].min()]
    assert first_weeks["nitrate_lag7"].isna().all()
