"""Article-level scoring for the EPU pipeline.

The scoring pipeline classifies an article as EPU-relevant if it contains
at least one keyword from each of the three categories (uncertainty,
economy, policy). For relevant articles, three context-aware adjustments
are applied:

1. Negation handling: an uncertainty keyword preceded by a negator is
   discarded.
2. Diminisher handling: an uncertainty keyword in proximity to a
   diminisher (e.g. "resolved", "improved") is discarded unless the
   diminisher itself is negated.
3. Amplifier handling: an uncertainty keyword in proximity to an
   amplifier (e.g. "deepens", "escalates") receives double weight.

The choice of weights (×0 for negated/diminished, ×1.5 for negated
diminisher, ×2 for amplified, ×1 baseline) is heuristic. Sensitivity to
these weights is examined in `analysis/sensitivity.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .lexicons import (
    AMPLIFIERS, DIMINISHERS, ECONOMY_STEMS, LM_NEGATIVE, LM_POSITIVE,
    NEGATORS, POLICY_STEMS, UNCERTAINTY_BIGRAMS, UNCERTAINTY_STEMS,
)


@dataclass
class ArticleScore:
    """Container for the scoring outputs of a single article."""
    is_epu_content: bool = False
    is_epu_title: bool = False
    context_score: float = 0.0
    context_total: int = 0
    context_negated: int = 0
    context_diminished: int = 0
    context_amplified: int = 0
    bigram_count: int = 0
    lm_sentiment: float = 0.0
    word_count: int = 0


def contains_any(text_lower: str, stems: tuple[str, ...]) -> bool:
    return any(stem.lower() in text_lower for stem in stems)


def count_matches(text_lower: str, stems: tuple[str, ...]) -> int:
    return sum(text_lower.count(stem.lower()) for stem in stems)


def count_bigrams(text_lower: str, bigrams: tuple[str, ...]) -> int:
    return sum(text_lower.count(bg.lower()) for bg in bigrams)


def context_score(
    text_lower: str,
    uncertainty_stems: tuple[str, ...] = UNCERTAINTY_STEMS,
    negators: tuple[str, ...] = NEGATORS,
    diminishers: tuple[str, ...] = DIMINISHERS,
    amplifiers: tuple[str, ...] = AMPLIFIERS,
    window: int = 5,
) -> tuple[float, int, int, int, int]:
    """Compute context-weighted uncertainty score for one article.

    Returns
    -------
    effective : float
        Sum of weights across uncertainty keywords.
    total : int
        Number of uncertainty keywords detected.
    negated : int
        Keywords suppressed by a preceding negator.
    diminished : int
        Keywords suppressed by a nearby diminisher.
    amplified : int
        Keywords boosted by a nearby amplifier.
    """
    words = text_lower.split()
    effective = 0.0
    total = negated = diminished = amplified = 0

    for i, word in enumerate(words):
        if not any(stem.lower() in word for stem in uncertainty_stems):
            continue

        total += 1
        start = max(0, i - window)
        end = min(len(words), i + window + 1)
        context = words[start:end]
        preceding = words[start:i]

        has_negator_before = any(neg in preceding for neg in negators)

        has_diminisher = False
        diminisher_negated = False
        for j, cword in enumerate(context):
            if any(dim.lower() in cword for dim in diminishers):
                has_diminisher = True
                dim_pos = start + j
                dim_preceding = words[max(0, dim_pos - 3):dim_pos]
                if any(neg in dim_preceding for neg in negators):
                    diminisher_negated = True
                break

        has_amplifier = any(
            any(amp.lower() in cword for amp in amplifiers) for cword in context
        )
        amplifier_negated = False
        if has_amplifier:
            for j, cword in enumerate(context):
                if any(amp.lower() in cword for amp in amplifiers):
                    amp_pos = start + j
                    amp_preceding = words[max(0, amp_pos - 3):amp_pos]
                    if any(neg in amp_preceding for neg in negators):
                        amplifier_negated = True
                    break

        if has_negator_before and not has_diminisher:
            negated += 1
        elif has_diminisher and not diminisher_negated:
            diminished += 1
        elif has_diminisher and diminisher_negated:
            effective += 1.5
            amplified += 1
        elif has_amplifier and not amplifier_negated:
            effective += 2.0
            amplified += 1
        elif has_amplifier and amplifier_negated:
            negated += 1
        else:
            effective += 1.0

    return effective, total, negated, diminished, amplified


def lm_sentiment_score(text_lower: str) -> float:
    """Return Loughran-McDonald polarity in [-1, 1]; 0 if no matches."""
    pos = count_matches(text_lower, LM_POSITIVE)
    neg = count_matches(text_lower, LM_NEGATIVE)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def score_article(content: object, title: object) -> ArticleScore:
    """Score one article. Returns an empty ArticleScore for unusable input."""
    score = ArticleScore()

    if pd.notna(title) and len(str(title)) > 10:
        title_lower = str(title).lower()
        has_u = contains_any(title_lower, UNCERTAINTY_STEMS)
        has_e = contains_any(title_lower, ECONOMY_STEMS)
        has_p = contains_any(title_lower, POLICY_STEMS)
        if has_u and (int(has_u) + int(has_e) + int(has_p) >= 2):
            score.is_epu_title = True

    if pd.isna(content) or len(str(content)) < 50:
        return score

    text_lower = str(content).lower()
    score.word_count = len(text_lower.split())

    if not (
        contains_any(text_lower, UNCERTAINTY_STEMS)
        and contains_any(text_lower, ECONOMY_STEMS)
        and contains_any(text_lower, POLICY_STEMS)
    ):
        return score

    eff, tot, neg, dim, amp = context_score(text_lower)
    if eff <= 0:
        return score

    score.is_epu_content = True
    score.context_score = eff
    score.context_total = tot
    score.context_negated = neg
    score.context_diminished = dim
    score.context_amplified = amp
    score.bigram_count = count_bigrams(text_lower, UNCERTAINTY_BIGRAMS)
    score.lm_sentiment = lm_sentiment_score(text_lower)

    return score
