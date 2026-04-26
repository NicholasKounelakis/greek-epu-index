"""Unit tests for the econometric utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.econometrics import (
    block_bootstrap_slope, diebold_mariano, newey_west_maxlags,
    stationarity_battery,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestNeweyWestMaxlags:
    def test_typical_sample(self):
        assert newey_west_maxlags(194) == 4

    def test_small_sample(self):
        assert newey_west_maxlags(50) == 3

    def test_large_sample(self):
        assert newey_west_maxlags(1000) >= 6


class TestStationarityBattery:
    def test_white_noise_is_stationary(self, rng):
        s = pd.Series(rng.standard_normal(500))
        result = stationarity_battery(s, "noise")
        assert result.adf_pvalue < 0.05
        assert result.verdict in ("stationary", "ambiguous")

    def test_random_walk_is_non_stationary(self, rng):
        s = pd.Series(rng.standard_normal(500).cumsum())
        result = stationarity_battery(s, "rw")
        assert result.adf_pvalue > 0.05


class TestBlockBootstrap:
    def test_recovers_known_slope(self, rng):
        n = 300
        x = rng.standard_normal(n)
        y = 2.0 * x + rng.standard_normal(n) * 0.5
        result = block_bootstrap_slope(y, x, n_boot=500, seed=42)
        assert abs(result["point"] - 2.0) < 0.2
        assert result["ci_lower"] < 2.0 < result["ci_upper"]

    def test_zero_slope_ci_contains_zero(self, rng):
        n = 300
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        result = block_bootstrap_slope(y, x, n_boot=500, seed=42)
        assert result["ci_lower"] < 0 < result["ci_upper"]


class TestDieboldMariano:
    def test_identical_forecasts_returns_nan(self, rng):
        # Identical forecast errors -> zero variance of loss differential
        # -> the test statistic is undefined; the implementation returns NaN.
        e = rng.standard_normal(200)
        stat, p = diebold_mariano(e, e)
        assert np.isnan(stat) and np.isnan(p)

    def test_better_model_has_lower_loss(self, rng):
        e_good = rng.standard_normal(500) * 0.5
        e_bad = rng.standard_normal(500) * 2.0
        stat, p = diebold_mariano(e_good, e_bad)
        # e_good has lower MSE, so DM is negative
        assert stat < 0
        assert p < 0.05
