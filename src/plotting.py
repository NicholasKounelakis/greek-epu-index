"""Plotting utilities for the EPU pipeline.

Generates the three headline figures (time series, scatter, cross-
correlation) plus an OOS performance chart. All figures are written to
the directory passed via --output.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


KEY_EVENTS = {
    "2010-05": "First Bailout",
    "2012-03": "PSI/2nd Bailout",
    "2015-07": "Referendum",
    "2018-08": "End of Bailouts",
    "2020-03": "COVID-19",
    "2022-02": "Ukraine War",
}


def _annotate_events(ax, dates_min, dates_max, ypos):
    for ds, label in KEY_EVENTS.items():
        ed = pd.Timestamp(ds)
        if dates_min <= ed <= dates_max:
            ax.axvline(x=ed, color="gray", linestyle="--", alpha=0.3)
            ax.annotate(
                label, xy=(ed, ypos), fontsize=7,
                rotation=45, ha="left", va="top",
            )


def plot_epu_vs_spread(merged: pd.DataFrame, epu_col: str, output_dir: Path) -> None:
    dates = merged["date"]
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax1.plot(
        dates, merged[epu_col], color="#1f77b4", linewidth=1.3,
        label=f"EPU index ({epu_col})",
    )
    ax1.set_ylabel("EPU index", color="#1f77b4", fontsize=11)
    ax2 = ax1.twinx()
    ax2.plot(
        dates, merged["spread"], color="#d62728", linewidth=1.3,
        alpha=0.8, label="Bond spread",
    )
    ax2.set_ylabel("Spread (%)", color="#d62728", fontsize=11)

    _annotate_events(ax1, dates.min(), dates.max(), ax1.get_ylim()[1] * 0.95)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    r, _ = stats.pearsonr(
        merged[epu_col].dropna(),
        merged.loc[merged[epu_col].notna(), "spread"],
    )
    plt.title(
        f"Greek EPU vs Greece-Germany 10Y bond spread (Pearson r = {r:.3f})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "01_epu_vs_spread.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_scatter(merged: pd.DataFrame, epu_col: str, output_dir: Path) -> None:
    df = merged[[epu_col, "spread"]].dropna()
    r, _ = stats.pearsonr(df[epu_col], df["spread"])
    rho, _ = stats.spearmanr(df[epu_col], df["spread"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[epu_col], df["spread"], alpha=0.5, s=20, color="#1f77b4")
    z = np.polyfit(df[epu_col], df["spread"], 1)
    p = np.poly1d(z)
    xl = np.linspace(df[epu_col].min(), df[epu_col].max(), 100)
    ax.plot(xl, p(xl), color="red", linewidth=2)
    ax.set_xlabel("EPU index", fontsize=11)
    ax.set_ylabel("Spread (%)", fontsize=11)
    ax.set_title(
        f"EPU vs Spread | Pearson r = {r:.3f}, Spearman ρ = {rho:.3f}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "02_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_cross_correlation(
    merged: pd.DataFrame, epu_col: str, output_dir: Path,
    max_lag: int = 12,
) -> None:
    cross_corrs = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x = merged[epu_col].iloc[: len(merged) - lag].reset_index(drop=True)
            y = merged["spread"].iloc[lag:].reset_index(drop=True)
        else:
            x = merged[epu_col].iloc[-lag:].reset_index(drop=True)
            y = merged["spread"].iloc[: len(merged) + lag].reset_index(drop=True)
        n = min(len(x), len(y))
        if n > 10:
            cross_corrs[lag] = stats.pearsonr(x[:n], y[:n])[0]

    optimal = max(cross_corrs, key=cross_corrs.get)
    lags = sorted(cross_corrs)
    corrs = [cross_corrs[l] for l in lags]
    colors = ["#d62728" if l == optimal else "#1f77b4" for l in lags]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(lags, corrs, color=colors, alpha=0.7)
    ax.axhline(y=0, color="gray", linestyle="-")
    ax.set_xlabel("Lag (months, positive = EPU leads spread)", fontsize=11)
    ax.set_ylabel("Pearson r", fontsize=11)
    ax.set_title(
        f"Cross-correlation | optimal lag = {optimal} months "
        f"(r = {cross_corrs[optimal]:.3f})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "03_cross_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_first_differences(
    merged: pd.DataFrame, epu_col: str, output_dir: Path
) -> None:
    df = pd.DataFrame({
        "d_spread": merged["spread"].diff(),
        "d_epu": merged[epu_col].diff(),
    }).dropna()
    r, _ = stats.pearsonr(df["d_epu"], df["d_spread"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["d_epu"], df["d_spread"], alpha=0.5, s=20, color="#2ca02c")
    z = np.polyfit(df["d_epu"], df["d_spread"], 1)
    p = np.poly1d(z)
    xl = np.linspace(df["d_epu"].min(), df["d_epu"].max(), 100)
    ax.plot(xl, p(xl), color="red", linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("ΔEPU (month-on-month)", fontsize=11)
    ax.set_ylabel("ΔSpread (month-on-month, %)", fontsize=11)
    ax.set_title(
        f"First-difference scatter | r = {r:.3f}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "04_first_differences.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EPU figures")
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--epu-col", type=str, required=True)
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    merged = pd.read_csv(args.merged, parse_dates=["date"])

    plot_epu_vs_spread(merged, args.epu_col, args.output)
    plot_scatter(merged, args.epu_col, args.output)
    plot_cross_correlation(merged, args.epu_col, args.output)
    plot_first_differences(merged, args.epu_col, args.output)
    print(f"[done] figures written to {args.output}")


if __name__ == "__main__":
    main()
