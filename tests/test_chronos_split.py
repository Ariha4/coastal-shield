import numpy as np
import pandas as pd
import pytest

from src.model import build_chronos_sequences, chronological_split_by_week, train_chronos_hybrid
from tests.conftest import make_grid_weeks


def _with_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sst_anomaly"] = df["temperature"] - df["temperature"].mean()
    return df


def test_sequences_return_anchor_week_per_window():
    df = _with_anomaly(make_grid_weeks(n_weeks=12))
    X, y, weeks = build_chronos_sequences(df, return_weeks=True)
    assert len(X) == len(y) == len(weeks)
    # Windows come out grid by grid, so anchor weeks repeat and are not sorted overall.
    assert not np.all(np.diff(weeks.astype("int64")) >= 0)


def test_sequences_default_return_shape_is_unchanged():
    df = _with_anomaly(make_grid_weeks(n_weeks=12))
    out = build_chronos_sequences(df)
    assert len(out) == 2


def test_hybrid_test_windows_all_come_after_training_windows_plus_horizon():
    df = _with_anomaly(make_grid_weeks(n_weeks=30, grids=((33.0, -118.0), (34.0, -120.0), (35.0, -121.0))))
    out = train_chronos_hybrid(df, use_wandb=False, check_checkpoint=False)
    assert out["n_train"] > 0 and out["n_test"] > 0
    # Each training label is 2 weeks ahead of its anchor, so it must land before the test period.
    assert out["train_last_week"] + pd.Timedelta(weeks=2) < out["test_first_week"]
    assert out["chronos_checkpoint_loaded"] is False


def test_hybrid_test_set_spans_every_grid():
    # With a positional cut the test set was only the last grid cell(s) in sort order.
    grids = ((33.0, -118.0), (34.0, -120.0), (35.0, -121.0))
    df = _with_anomaly(make_grid_weeks(n_weeks=30, grids=grids))
    out = train_chronos_hybrid(df, use_wandb=False, check_checkpoint=False)
    assert out["n_test"] % len(grids) == 0
    assert out["n_test"] >= len(grids)


def test_time_split_with_tiny_test_ratio_keeps_one_test_week():
    df = make_grid_weeks(n_weeks=5)
    train, test = chronological_split_by_week(df, test_ratio=0.0)
    assert test["week_start"].nunique() == 1
    assert len(train) + len(test) == len(df)


def test_hybrid_rejects_too_little_history():
    df = _with_anomaly(make_grid_weeks(n_weeks=6))
    with pytest.raises(ValueError):
        train_chronos_hybrid(df, use_wandb=False, check_checkpoint=False)
