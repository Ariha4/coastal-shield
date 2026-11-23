import pandas as pd
import pytest

from src.conformal import _chronological_three_way_split
from src.model import chronological_split_by_week, prepare_xy
from tests.conftest import make_grid_weeks


def test_time_split_puts_every_test_week_after_every_train_week(grid_weeks):
    train, test = chronological_split_by_week(grid_weeks, test_ratio=0.25)
    assert len(train) and len(test)
    assert train["week_start"].max() < test["week_start"].min()
    assert len(train) + len(test) == len(grid_weeks)


def test_time_split_keeps_all_grids_of_a_week_together(grid_weeks):
    train, test = chronological_split_by_week(grid_weeks)
    assert set(train["week_start"]).isdisjoint(set(test["week_start"]))


def test_time_split_needs_three_weeks():
    with pytest.raises(ValueError, match="3 distinct weeks"):
        chronological_split_by_week(make_grid_weeks(n_weeks=2))


def test_three_way_split_is_ordered_and_exhaustive(grid_weeks):
    train, cal, test = _chronological_three_way_split(grid_weeks)
    assert train["week_start"].max() < cal["week_start"].min()
    assert cal["week_start"].max() < test["week_start"].min()
    assert len(train) + len(cal) + len(test) == len(grid_weeks)


def test_three_way_split_needs_six_weeks():
    with pytest.raises(ValueError, match="6 distinct weeks"):
        _chronological_three_way_split(make_grid_weeks(n_weeks=5))


def test_prepare_xy_drops_unlabelled_rows_but_keeps_nan_features(grid_weeks):
    df = grid_weeks.copy()
    df.loc[0, "bloom_label"] = None
    df.loc[1, "nitrate"] = None
    X, y, cols = prepare_xy(df, ["nitrate", "temperature", "not_a_column"])
    assert cols == ["nitrate", "temperature"]
    assert len(X) == len(df) - 1
    assert X["nitrate"].isna().sum() == 1
    assert y.dtype.kind == "i"


def test_prepare_xy_rejects_frame_without_any_feature():
    df = pd.DataFrame({"bloom_label": [0, 1]})
    with pytest.raises(ValueError, match="No feature columns"):
        prepare_xy(df, ["nitrate"])
