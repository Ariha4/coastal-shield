"""Shared synthetic fixtures. Nothing here touches data/raw or the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

FEATURES = ["temperature", "chlorophyll", "nitrate", "salinity", "upwelling_proxy"]


def make_grid_weeks(
    n_weeks: int = 20,
    grids: tuple[tuple[float, float], ...] = ((33.0, -118.0), (34.5, -120.0)),
    start: str = "2020-01-06",
    seed: int = 0,
) -> pd.DataFrame:
    """Weekly rows for each grid cell with a learnable bloom signal driven by nitrate."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(start, periods=n_weeks, freq="7D")
    rows = []
    for glat, glon in grids:
        for wk in weeks:
            nitrate = rng.uniform(0, 20)
            rows.append(
                {
                    "grid_lat": glat,
                    "grid_lon": glon,
                    "week_start": wk,
                    "temperature": rng.normal(15, 2),
                    "chlorophyll": rng.uniform(0, 5),
                    "nitrate": nitrate,
                    "salinity": rng.normal(33.5, 0.3),
                    "upwelling_proxy": rng.normal(2, 1),
                    "bloom_label": int(nitrate > 12),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def grid_weeks() -> pd.DataFrame:
    return make_grid_weeks()
