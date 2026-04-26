"""Sensitivity analysis for the EPU index.

This script perturbs the lexicons one stem at a time and re-builds the
index, reporting how much the monthly index changes. It is intended to
be run after `pipeline.py` has produced the article-level scores; it
re-uses the cached per-article output rather than re-processing the
raw newspaper corpus.

Usage
-----
    python -m src.sensitivity --merged data/processed/merged_data.csv \
                              --output data/processed/sensitivity.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def jackknife_methods(merged: pd.DataFrame, method_cols: list[str]) -> pd.DataFrame:
    """Drop one method at a time and report range of correlations."""
    rows = []
    for col in method_cols:
        df = merged[[col, "spread"]].dropna()
        if len(df) < 10:
            continue
        r, p = stats.pearsonr(df[col], df["spread"])
        rho, _ = stats.spearmanr(df[col], df["spread"])
        rows.append({
            "method": col,
            "n": len(df),
            "pearson_r": r,
            "pearson_p": p,
            "spearman_rho": rho,
        })
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False)


def subsample_stability(
    merged: pd.DataFrame, epu_col: str, n_splits: int = 100, seed: int = 42,
) -> dict[str, float]:
    """Random 70/30 splits; report mean and std of the in-sample correlation."""
    rng = np.random.default_rng(seed)
    df = merged[[epu_col, "spread"]].dropna().reset_index(drop=True)
    n = len(df)
    n_train = int(0.7 * n)

    rs = []
    for _ in range(n_splits):
        idx = rng.permutation(n)[:n_train]
        d = df.iloc[idx]
        if len(d) > 10:
            rs.append(stats.pearsonr(d[epu_col], d["spread"])[0])
    rs = np.array(rs)
    return {
        "mean_r": float(rs.mean()),
        "std_r": float(rs.std(ddof=1)),
        "min_r": float(rs.min()),
        "max_r": float(rs.max()),
        "n_splits": int(len(rs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--epu-col", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    merged = pd.read_csv(args.merged, parse_dates=["date"])
    method_cols = [c for c in merged.columns if c.endswith("_dt")]

    jk = jackknife_methods(merged, method_cols)
    jk.to_csv(args.output, index=False)
    print(jk.to_string(index=False))

    stab = subsample_stability(merged, args.epu_col)
    print("\nSubsample stability (70/30, 100 splits):")
    for k, v in stab.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
