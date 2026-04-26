"""Construction of monthly EPU index variants from article-level scores.

Each variant is a different aggregation of the underlying article scores.
The 'basic' index is the share of EPU-relevant articles in a month, in the
spirit of Baker, Bloom & Davis (2016). Other variants weight by context,
sentiment, or bigram density.

All variants are normalised to a mean of 100 over the full sample, then
detrended by subtracting a fitted linear trend (and adding 100 back) to
remove the secular increase in article volume that would otherwise induce
a deterministic component in any subsequent regression.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# All eight variants exposed by the pipeline.
METHOD_COLUMNS: tuple[str, ...] = (
    "basic", "title_based", "context_weighted", "bigram",
    "sentiment_lm", "title_content", "ultimate", "title_bigram",
)


@dataclass
class MonthlyAggregate:
    total: int = 0
    epu_content: int = 0
    epu_title: int = 0
    context_score_sum: float = 0.0
    context_total_sum: int = 0
    context_negated_sum: int = 0
    context_diminished_sum: int = 0
    context_amplified_sum: int = 0
    bigram_sum: int = 0
    lm_sentiment_sum: float = 0.0
    epu_word_sum: int = 0


def build_monthly_indices(monthly: dict[str, MonthlyAggregate]) -> pd.DataFrame:
    """Convert a dictionary of monthly aggregates into the index DataFrame."""
    rows = []
    for ym_str, md in monthly.items():
        if md.total == 0:
            continue

        basic = md.epu_content / md.total
        title_based = md.epu_title / md.total
        context_weighted = (
            md.context_score_sum / md.total if md.epu_word_sum > 0 else 0.0
        )
        bigram = basic + (md.bigram_sum / max(md.total, 1)) * 5
        avg_lm = md.lm_sentiment_sum / md.epu_content if md.epu_content > 0 else 0.0
        lm_weight = 1 + max(0.0, -avg_lm) * 2
        sentiment_lm = basic * lm_weight
        title_content = title_based * 0.4 + basic * 0.6
        context_intensity = (
            md.context_score_sum / max(md.context_total_sum, 1)
            if md.epu_content > 0
            else 0.0
        )
        ultimate = (
            basic * (1 + context_intensity) * lm_weight
            + (md.bigram_sum / max(md.total, 1)) * 3
        )
        title_bigram = title_based + (md.bigram_sum / max(md.total, 1)) * 3

        rows.append({
            "year_month": ym_str,
            "total_articles": md.total,
            "epu_content": md.epu_content,
            "epu_title": md.epu_title,
            "basic": basic,
            "title_based": title_based,
            "context_weighted": context_weighted,
            "bigram": bigram,
            "sentiment_lm": sentiment_lm,
            "title_content": title_content,
            "ultimate": ultimate,
            "title_bigram": title_bigram,
            "avg_lm_sentiment": avg_lm,
        })

    df = pd.DataFrame(rows)
    df["period"] = pd.PeriodIndex(df["year_month"], freq="M")
    df = df.sort_values("period").reset_index(drop=True)

    for col in METHOD_COLUMNS:
        mean_val = df[col].mean()
        df[f"{col}_index"] = (df[col] / mean_val) * 100 if mean_val > 0 else 100.0
        x = np.arange(len(df))
        coeffs = np.polynomial.polynomial.polyfit(x, df[f"{col}_index"].values, 1)
        df[f"{col}_dt"] = (
            df[f"{col}_index"] - np.polynomial.polynomial.polyval(x, coeffs) + 100
        )

    df["date"] = df["period"].dt.to_timestamp()
    return df
