import numpy as np
import pytest

from src.conformal import _split_conformal_interval, recommended_action, run_conformal_pipeline
from tests.conftest import FEATURES, make_grid_weeks


@pytest.mark.parametrize(
    ("prob", "action"),
    [(0.0, "Monitor"), (0.299, "Monitor"), (0.3, "Advisory"), (0.599, "Advisory"), (0.6, "Closure Recommended")],
)
def test_recommended_action_thresholds(prob, action):
    assert recommended_action(prob) == action


def test_interval_reaches_nominal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=4000)
    y = (rng.uniform(size=4000) < p).astype(int)
    lower, upper, q_hat = _split_conformal_interval(y[:2000], p[:2000], p[2000:], alpha=0.1)
    covered = (y[2000:] >= lower) & (y[2000:] <= upper)
    assert covered.mean() >= 0.88  # 0.9 nominal, small finite-sample slack
    assert 0.0 < q_hat <= 1.0


def test_interval_is_clipped_to_probability_range():
    lower, upper, _ = _split_conformal_interval(
        np.array([0, 1, 0, 1]), np.array([0.9, 0.1, 0.8, 0.2]), np.array([0.05, 0.95])
    )
    assert (lower >= 0).all() and (upper <= 1).all()
    assert (lower <= upper).all()


def test_interval_rejects_empty_calibration_set():
    with pytest.raises(ValueError, match="empty"):
        _split_conformal_interval(np.array([]), np.array([]), np.array([0.5]))


def test_pipeline_end_to_end_on_synthetic_grid():
    df = make_grid_weeks(n_weeks=30, seed=3)
    out = run_conformal_pipeline(df, feature_cols=FEATURES, alpha=0.2)
    preds = out["predictions"]
    assert len(preds) > 0
    assert {"zone", "date", "prob", "lower", "upper", "recommended_action"} <= set(preds.columns)
    assert ((preds["lower"] <= preds["prob"]) & (preds["prob"] <= preds["upper"])).all()
    assert preds["zone"].str.match(r"^lat_-?\d+\.\d{2}_lon_-?\d+\.\d{2}$").all()
    assert 0.0 <= out["empirical_coverage"] <= 1.0
