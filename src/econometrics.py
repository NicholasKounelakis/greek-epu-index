"""Econometric analysis of the EPU index against the bond spread.

This module implements the full diagnostic battery needed to interpret
the relationship between the EPU index and the Greek-German 10-year
bond spread:

1. Stationarity tests (ADF, KPSS) on both series in levels and first
   differences. KPSS complements ADF because the two have opposite
   null hypotheses, and agreement between them strengthens the
   conclusion.

2. Engle-Granger cointegration test. If both series are I(1) but a
   linear combination is stationary, an OLS regression in levels is
   meaningful; otherwise the levels regression is potentially spurious
   and inference must be conducted in first differences.

3. OLS regressions estimated with Newey-West HAC standard errors. The
   maxlag is set to floor(4*(T/100)^(2/9)) following Newey & West (1994).
   - Levels (bivariate)
   - Levels with AR(1) and trend (multivariate)
   - First differences: dSpread = a + b * dEPU + e
   - Predictive regression with EPU lagged by k months

4. Granger causality tests at multiple lag orders, in first differences
   to avoid the unit root problems of testing in levels.

5. Block bootstrap confidence intervals (1000 replications) for the
   first-difference slope coefficient. This is the most defensible
   inference in the presence of non-normal residuals (Jarque-Bera
   rejects normality at 1%) and serial dependence.

6. Out-of-sample evaluation: a rolling-origin holdout that compares the
   EPU-augmented forecast against an AR(1) benchmark using the
   Diebold-Mariano test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import (
    adfuller, coint, grangercausalitytests, kpss,
)


# ---------------------------------------------------------------------------
# Stationarity & cointegration
# ---------------------------------------------------------------------------

@dataclass
class StationarityResult:
    series: str
    adf_stat: float
    adf_pvalue: float
    adf_lags: int
    kpss_stat: float
    kpss_pvalue: float
    verdict: str  # "stationary", "non-stationary", "ambiguous"


def _classify(adf_p: float, kpss_p: float) -> str:
    adf_rejects = adf_p < 0.05  # reject unit root
    kpss_rejects = kpss_p < 0.05  # reject stationarity
    if adf_rejects and not kpss_rejects:
        return "stationary"
    if not adf_rejects and kpss_rejects:
        return "non-stationary"
    return "ambiguous"


def stationarity_battery(
    series: pd.Series, name: str, regression: str = "c"
) -> StationarityResult:
    s = series.dropna()
    adf_stat, adf_p, adf_lags, *_ = adfuller(s, autolag="AIC", regression=regression)
    kpss_reg = "c" if regression == "c" else "ct"
    kpss_stat, kpss_p, *_ = kpss(s, regression=kpss_reg, nlags="auto")
    return StationarityResult(
        series=name,
        adf_stat=float(adf_stat),
        adf_pvalue=float(adf_p),
        adf_lags=int(adf_lags),
        kpss_stat=float(kpss_stat),
        kpss_pvalue=float(kpss_p),
        verdict=_classify(adf_p, kpss_p),
    )


def engle_granger_cointegration(
    y: pd.Series, x: pd.Series
) -> tuple[float, float, float]:
    """Return (statistic, p-value, 5%-critical-value) for H0: no cointegration."""
    df = pd.concat([y, x], axis=1).dropna()
    stat, p, crit = coint(df.iloc[:, 0], df.iloc[:, 1], autolag="AIC")
    return float(stat), float(p), float(crit[1])


# ---------------------------------------------------------------------------
# Newey-West maxlags rule
# ---------------------------------------------------------------------------

def newey_west_maxlags(n: int) -> int:
    """Andrews-style rule of thumb used by Newey & West (1994)."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


# ---------------------------------------------------------------------------
# OLS specifications
# ---------------------------------------------------------------------------

def fit_hac_ols(y: pd.Series, X: pd.DataFrame, maxlags: int | None = None):
    df = pd.concat([y, X], axis=1).dropna()
    y_clean = df.iloc[:, 0]
    X_clean = sm.add_constant(df.iloc[:, 1:])
    if maxlags is None:
        maxlags = newey_west_maxlags(len(df))
    return sm.OLS(y_clean, X_clean).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )


def regression_suite(
    merged: pd.DataFrame, epu_col: str, lag_for_predictive: int = 3
) -> dict[str, object]:
    """Estimate the four headline regressions and return them in a dict."""
    out: dict[str, object] = {}

    # 1. Bivariate levels
    out["bivariate_levels"] = fit_hac_ols(
        merged["spread"], merged[[epu_col]]
    )

    # 2. AR(1) + trend levels
    df_ar = merged.assign(trend=np.arange(len(merged))).copy()
    df_ar["spread_lag1"] = df_ar["spread"].shift(1)
    out["ar1_trend_levels"] = fit_hac_ols(
        df_ar["spread"], df_ar[[epu_col, "spread_lag1", "trend"]]
    )

    # 3. First differences
    df_fd = pd.DataFrame({
        "d_spread": merged["spread"].diff(),
        "d_epu": merged[epu_col].diff(),
    })
    out["first_differences"] = fit_hac_ols(
        df_fd["d_spread"], df_fd[["d_epu"]]
    )

    # 4. Predictive (lagged EPU)
    df_pred = pd.DataFrame({
        "spread": merged["spread"],
        f"{epu_col}_lag{lag_for_predictive}": merged[epu_col].shift(
            lag_for_predictive
        ),
        "trend": np.arange(len(merged)),
    })
    out["predictive_lagged"] = fit_hac_ols(
        df_pred["spread"],
        df_pred[[f"{epu_col}_lag{lag_for_predictive}", "trend"]],
    )

    return out


# ---------------------------------------------------------------------------
# Granger causality
# ---------------------------------------------------------------------------

def granger_in_differences(
    merged: pd.DataFrame, epu_col: str, max_lag: int = 6
) -> pd.DataFrame:
    """Granger causality tests on first-differenced series.

    Returns a DataFrame with one row per lag tested, in both directions
    (EPU -> spread and spread -> EPU).
    """
    df = pd.DataFrame({
        "d_spread": merged["spread"].diff(),
        "d_epu": merged[epu_col].diff(),
    }).dropna()

    rows = []
    for direction, cols in [
        ("EPU -> Spread", ["d_spread", "d_epu"]),
        ("Spread -> EPU", ["d_epu", "d_spread"]),
    ]:
        results = grangercausalitytests(df[cols], maxlag=max_lag, verbose=False)
        for lag, (tests, _) in results.items():
            f_stat, f_p = tests["ssr_ftest"][0], tests["ssr_ftest"][1]
            rows.append({
                "direction": direction,
                "lag": lag,
                "F_stat": f_stat,
                "p_value": f_p,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------

def block_bootstrap_slope(
    y: np.ndarray, x: np.ndarray, n_boot: int = 1000,
    block_size: int | None = None, seed: int = 42,
) -> dict[str, float]:
    """Block bootstrap CI for the slope of y on x.

    A moving-block bootstrap preserves serial correlation. Block size
    defaults to ceil(n^(1/3)) per the rule from Hall, Horowitz &
    Jing (1995).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    n = len(y)

    if block_size is None:
        block_size = max(1, int(np.ceil(n ** (1 / 3))))

    n_blocks = int(np.ceil(n / block_size))
    slopes = np.empty(n_boot)

    for b in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([
            np.arange(s, s + block_size) for s in starts
        ])[:n]
        y_b, x_b = y[idx], x[idx]
        # Slope via closed form
        x_centered = x_b - x_b.mean()
        denom = (x_centered ** 2).sum()
        if denom == 0:
            slopes[b] = np.nan
            continue
        slopes[b] = (x_centered * (y_b - y_b.mean())).sum() / denom

    slopes = slopes[np.isfinite(slopes)]
    point = np.mean(slopes)
    return {
        "point": float(point),
        "se": float(np.std(slopes, ddof=1)),
        "ci_lower": float(np.percentile(slopes, 2.5)),
        "ci_upper": float(np.percentile(slopes, 97.5)),
        "n_boot": int(len(slopes)),
        "block_size": int(block_size),
    }


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------

def diebold_mariano(
    e1: np.ndarray, e2: np.ndarray, h: int = 1, loss: str = "mse"
) -> tuple[float, float]:
    """Two-sided Diebold-Mariano test on forecast errors e1 vs e2.

    Returns (DM statistic, p-value). e1 corresponds to model 1, e2 to
    the benchmark; a positive DM means model 1 has higher loss.
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    if loss == "mse":
        d = e1 ** 2 - e2 ** 2
    elif loss == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")

    n = len(d)
    d_bar = d.mean()
    # Long-run variance via Newey-West
    L = h - 1
    gamma = [np.var(d, ddof=0)]
    for k in range(1, L + 1):
        gamma.append(np.cov(d[k:], d[:-k], ddof=0)[0, 1])
    long_run_var = gamma[0] + 2 * sum(
        (1 - k / (L + 1)) * gamma[k] for k in range(1, L + 1)
    )
    if long_run_var <= 0:
        return float("nan"), float("nan")

    dm = d_bar / np.sqrt(long_run_var / n)
    # Harvey-Leybourne-Newbold small-sample correction
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_adj = dm * correction
    p = 2 * (1 - stats.t.cdf(abs(dm_adj), df=n - 1))
    return float(dm_adj), float(p)


# ---------------------------------------------------------------------------
# Rolling-origin out-of-sample evaluation
# ---------------------------------------------------------------------------

def rolling_oos_forecast(
    merged: pd.DataFrame, epu_col: str, train_frac: float = 0.7,
    horizon: int = 1,
) -> dict[str, object]:
    """Rolling-origin one-step-ahead forecast: AR(1) vs AR(1)+EPU(t-h).

    Produces forecast errors for both models and the Diebold-Mariano
    test of equal forecast accuracy.
    """
    df = pd.DataFrame({
        "spread": merged["spread"],
        "spread_lag1": merged["spread"].shift(1),
        f"{epu_col}_lag{horizon}": merged[epu_col].shift(horizon),
    }).dropna().reset_index(drop=True)

    n = len(df)
    n_train = int(np.floor(train_frac * n))

    e_bench = []
    e_aug = []
    for t in range(n_train, n):
        train = df.iloc[:t]
        # Benchmark: AR(1) on spread
        m1 = sm.OLS(train["spread"], sm.add_constant(train[["spread_lag1"]])).fit()
        f1 = m1.predict(sm.add_constant(df.iloc[[t]][["spread_lag1"]]))[0]
        e_bench.append(df["spread"].iloc[t] - f1)

        # Augmented: AR(1) + lagged EPU
        X_cols = ["spread_lag1", f"{epu_col}_lag{horizon}"]
        m2 = sm.OLS(train["spread"], sm.add_constant(train[X_cols])).fit()
        f2 = m2.predict(sm.add_constant(df.iloc[[t]][X_cols]))[0]
        e_aug.append(df["spread"].iloc[t] - f2)

    e_bench = np.array(e_bench)
    e_aug = np.array(e_aug)
    rmse_bench = float(np.sqrt(np.mean(e_bench ** 2)))
    rmse_aug = float(np.sqrt(np.mean(e_aug ** 2)))
    dm_stat, dm_p = diebold_mariano(e_aug, e_bench, h=horizon)

    return {
        "rmse_benchmark_ar1": rmse_bench,
        "rmse_augmented_ar1_epu": rmse_aug,
        "rmse_ratio": rmse_aug / rmse_bench if rmse_bench > 0 else float("nan"),
        "dm_stat": dm_stat,
        "dm_pvalue": dm_p,
        "n_test": int(n - n_train),
        "interpretation": (
            "EPU improves forecasts" if rmse_aug < rmse_bench and dm_p < 0.10
            else "EPU does not significantly improve forecasts"
        ),
    }


# ---------------------------------------------------------------------------
# Method selection with holdout
# ---------------------------------------------------------------------------

def select_method_with_holdout(
    merged: pd.DataFrame, method_columns: Sequence[str],
    holdout_frac: float = 0.30, seed: int = 42,
) -> dict[str, object]:
    """Pick the best EPU variant on the in-sample fold; report on holdout.

    This mitigates the multiple-testing problem that arises when the
    method is chosen post hoc on the full sample.
    """
    rng = np.random.default_rng(seed)
    n = len(merged)
    n_train = int(np.floor((1 - holdout_frac) * n))

    train = merged.iloc[:n_train]
    test = merged.iloc[n_train:]

    in_sample = {}
    for col in method_columns:
        df = train[[col, "spread"]].dropna()
        if len(df) < 10:
            continue
        r, _ = stats.pearsonr(df[col], df["spread"])
        in_sample[col] = r

    best = max(in_sample, key=lambda k: abs(in_sample[k]))
    df_test = test[[best, "spread"]].dropna()
    r_oos, p_oos = stats.pearsonr(df_test[best], df_test["spread"])

    return {
        "best_method": best,
        "in_sample_r": float(in_sample[best]),
        "in_sample_n": int(n_train),
        "out_of_sample_r": float(r_oos),
        "out_of_sample_p": float(p_oos),
        "out_of_sample_n": int(len(df_test)),
        "all_in_sample_r": {k: float(v) for k, v in in_sample.items()},
    }
