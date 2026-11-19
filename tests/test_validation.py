import pandas as pd

from src.validation import validate_historical_events


def _write(tmp_path, events, preds):
    e, p = tmp_path / "events.csv", tmp_path / "preds.csv"
    pd.DataFrame(events).to_csv(e, index=False)
    pd.DataFrame(preds).to_csv(p, index=False)
    return e, p


def test_picks_strongest_prediction_in_lookback_window(tmp_path):
    zone = "lat_33.00_lon_-118.00"
    e, p = _write(
        tmp_path,
        [{"event_id": 1, "event_name": "A", "zone": zone, "event_date": "2020-03-20", "source": "x"}],
        [
            {"zone": zone, "date": "2020-03-01", "prob": 0.9, "lower": 0.8, "upper": 1.0, "recommended_action": "C"},
            {"zone": zone, "date": "2020-03-09", "prob": 0.4, "lower": 0.2, "upper": 0.6, "recommended_action": "A"},
            {"zone": zone, "date": "2020-03-16", "prob": 0.5, "lower": 0.3, "upper": 0.7, "recommended_action": "A"},
        ],
    )
    out = validate_historical_events(e, p, lookback_days=14)
    row = out.iloc[0]
    # 2020-03-01 is 19 days before the event, outside the 14-day window.
    assert row["prob"] == 0.5
    assert row["days_before_event"] == 4
    assert bool(row["fired_14d_prior"])


def test_event_without_predictions_is_reported_not_dropped(tmp_path):
    e, p = _write(
        tmp_path,
        [{"event_id": 7, "event_name": "B", "zone": "lat_1.00_lon_1.00", "event_date": "2020-01-10", "source": ""}],
        [{"zone": "other", "date": "2020-01-05", "prob": 0.9, "lower": 0.8, "upper": 1.0,
          "recommended_action": "C"}],
    )
    out = validate_historical_events(e, p)
    assert len(out) == 1
    assert out.iloc[0]["recommended_action"] == "No prediction"
    assert not bool(out.iloc[0]["fired_14d_prior"])
