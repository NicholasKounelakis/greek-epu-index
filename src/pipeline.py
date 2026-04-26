"""End-to-end EPU pipeline.

Usage
-----
    python -m src.pipeline --articles data/raw/news.csv \
                           --spread data/raw/spread.xlsx \
                           --output data/processed

The script reads newspaper articles from one or more CSV files, scores
each article, aggregates monthly indices, merges with the bond spread,
and runs the full diagnostic battery defined in `econometrics.py`. All
outputs are written to the directory passed via --output.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .index_builder import METHOD_COLUMNS, MonthlyAggregate, build_monthly_indices
from .econometrics import (
    block_bootstrap_slope, engle_granger_cointegration, granger_in_differences,
    newey_west_maxlags, regression_suite, rolling_oos_forecast,
    select_method_with_holdout, stationarity_battery,
)
from .lexicons import ALL_STEMS
from .scoring import score_article


CHUNK_SIZE = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greek EPU index pipeline")
    parser.add_argument(
        "--articles", type=Path, nargs="+", required=True,
        help="One or more CSV files with columns: date, content, article_title.",
    )
    parser.add_argument(
        "--spread", type=Path, required=True,
        help="Excel file with columns: Date, Spread.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed"),
        help="Output directory (created if missing).",
    )
    parser.add_argument(
        "--predictive-lag", type=int, default=3,
        help="Lag (months) used in the predictive regression and OOS test.",
    )
    parser.add_argument(
        "--holdout-frac", type=float, default=0.30,
        help="Fraction of the sample reserved for out-of-sample evaluation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for bootstrap.",
    )
    return parser.parse_args()


def process_articles(article_files: list[Path]) -> tuple[dict[str, MonthlyAggregate], Counter]:
    """Stream-process article CSVs into monthly aggregates."""
    monthly: dict[str, MonthlyAggregate] = {}
    epu_words = Counter()

    for path in article_files:
        if not path.exists():
            print(f"[warn] missing file: {path}", file=sys.stderr)
            continue

        print(f"[info] reading {path.name}")
        n_chunks = 0
        for chunk in pd.read_csv(
            path, chunksize=CHUNK_SIZE,
            usecols=["date", "content", "article_title"], dtype=str,
        ):
            n_chunks += 1
            chunk["date_parsed"] = pd.to_datetime(chunk["date"], errors="coerce")
            chunk = chunk.dropna(subset=["date_parsed"])
            chunk["year_month"] = chunk["date_parsed"].dt.to_period("M")

            for ym, group in chunk.groupby("year_month"):
                ym_str = str(ym)
                md = monthly.setdefault(ym_str, MonthlyAggregate())
                md.total += len(group)

                for _, row in group.iterrows():
                    s = score_article(row.get("content"), row.get("article_title"))
                    if s.is_epu_title:
                        md.epu_title += 1
                    if s.is_epu_content:
                        md.epu_content += 1
                        md.context_score_sum += s.context_score
                        md.context_total_sum += s.context_total
                        md.context_negated_sum += s.context_negated
                        md.context_diminished_sum += s.context_diminished
                        md.context_amplified_sum += s.context_amplified
                        md.bigram_sum += s.bigram_count
                        md.lm_sentiment_sum += s.lm_sentiment
                        md.epu_word_sum += s.word_count

                        if md.epu_content <= 50:
                            content = row.get("content")
                            if pd.notna(content):
                                words = str(content).lower().split()
                                epu_words.update(
                                    w for w in words
                                    if len(w) > 3 and any(s in w for s in ALL_STEMS)
                                )
            if n_chunks % 20 == 0:
                print(f"  ...processed {n_chunks * CHUNK_SIZE:,} rows")

    return monthly, epu_words


def merge_with_spread(epu_df: pd.DataFrame, spread_path: Path) -> pd.DataFrame:
    spread_df = pd.read_excel(spread_path)
    spread_df["date"] = pd.to_datetime(spread_df["Date"], format="mixed")
    spread_df["period"] = spread_df["date"].dt.to_period("M")
    spread_monthly = (
        spread_df.groupby("period")["Spread"].mean().reset_index()
    )

    merge_cols = [
        "period", "date", "total_articles", "epu_content", "epu_title",
        "avg_lm_sentiment",
    ]
    for col in METHOD_COLUMNS:
        merge_cols += [f"{col}_index", f"{col}_dt"]

    merged = (
        pd.merge(epu_df[merge_cols], spread_monthly, on="period", how="inner")
        .dropna(subset=["Spread"])
        .reset_index(drop=True)
        .rename(columns={"Spread": "spread"})
    )
    merged["trend"] = np.arange(len(merged))
    return merged


def run_diagnostics(
    merged: pd.DataFrame, epu_col: str, args: argparse.Namespace
) -> dict[str, object]:
    """Run the full econometric battery and return a JSON-friendly dict."""
    diag: dict[str, object] = {}

    # Stationarity
    s_spread = stationarity_battery(merged["spread"], "spread")
    s_epu = stationarity_battery(merged[epu_col], epu_col)
    s_d_spread = stationarity_battery(merged["spread"].diff(), "d_spread")
    s_d_epu = stationarity_battery(merged[epu_col].diff(), f"d_{epu_col}")
    diag["stationarity"] = {
        "levels": {
            "spread": s_spread.__dict__,
            "epu": s_epu.__dict__,
        },
        "first_differences": {
            "spread": s_d_spread.__dict__,
            "epu": s_d_epu.__dict__,
        },
    }

    # Cointegration
    eg_stat, eg_p, eg_crit5 = engle_granger_cointegration(
        merged["spread"], merged[epu_col]
    )
    diag["cointegration"] = {
        "engle_granger_stat": eg_stat,
        "p_value": eg_p,
        "critical_5pct": eg_crit5,
        "cointegrated_at_5pct": eg_p < 0.05,
    }

    # Regressions
    regs = regression_suite(merged, epu_col, lag_for_predictive=args.predictive_lag)
    diag["regressions"] = {
        name: {
            "r_squared": float(m.rsquared),
            "n_obs": int(m.nobs),
            "params": {k: float(v) for k, v in m.params.to_dict().items()},
            "pvalues": {k: float(v) for k, v in m.pvalues.to_dict().items()},
            "newey_west_maxlags": newey_west_maxlags(int(m.nobs)),
        }
        for name, m in regs.items()
    }

    # Granger (in differences)
    g_df = granger_in_differences(merged, epu_col, max_lag=6)
    diag["granger_first_differences"] = g_df.to_dict(orient="records")

    # Bootstrap on first-difference slope
    bs = block_bootstrap_slope(
        merged["spread"].diff().to_numpy(),
        merged[epu_col].diff().to_numpy(),
        seed=args.seed,
    )
    diag["bootstrap_first_difference_slope"] = bs

    # Out-of-sample
    oos = rolling_oos_forecast(merged, epu_col, horizon=args.predictive_lag)
    diag["out_of_sample"] = oos

    return diag, regs


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("[step 1/4] processing articles")
    monthly, epu_words = process_articles(args.articles)
    epu_df = build_monthly_indices(monthly)
    epu_df.to_csv(args.output / "epu_timeseries_all_methods.csv", index=False)

    print("[step 2/4] merging with bond spread")
    merged = merge_with_spread(epu_df, args.spread)
    merged.to_csv(args.output / "merged_data.csv", index=False)

    print("[step 3/4] selecting method via holdout")
    selection = select_method_with_holdout(
        merged, [f"{c}_dt" for c in METHOD_COLUMNS],
        holdout_frac=args.holdout_frac, seed=args.seed,
    )
    epu_col = selection["best_method"]
    print(
        f"  selected: {epu_col} | in-sample r = {selection['in_sample_r']:.3f} "
        f"| out-of-sample r = {selection['out_of_sample_r']:.3f}"
    )

    print("[step 4/4] running econometric battery")
    diag, regs = run_diagnostics(merged, epu_col, args)
    diag["method_selection"] = selection

    # Save the headline timeseries for downstream consumers
    headline = epu_df[["year_month", f"{epu_col}".replace("_dt", "_index"), epu_col]].copy()
    headline.columns = ["year_month", "epu_index", "epu_index_detrended"]
    headline.to_csv(args.output / "epu_timeseries.csv", index=False)

    with (args.output / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False, default=str)

    with (args.output / "regression_summaries.txt").open("w", encoding="utf-8") as f:
        for name, m in regs.items():
            f.write(f"=== {name} ===\n{m.summary().as_text()}\n\n")

    print(f"[done] outputs written to {args.output}")


if __name__ == "__main__":
    main()
