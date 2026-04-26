# Methodology

This document describes the design choices behind the index in detail
and the reasoning for each choice.

## 1. Article filtering

Of the raw corpus, articles are excluded if any of the following hold:

- The `date` field cannot be parsed by `pd.to_datetime`.
- The `content` field is missing or shorter than 50 characters.
- The article does not contain at least one stem from each of the
  uncertainty, economy, and policy categories.

The first two are corpus-quality filters; the third is the BBD-style
classification step. The pipeline does not silently drop articles for
any other reason.

## 2. Stem matching vs lemmatisation

Greek inflectional morphology produces many surface forms per lemma
(e.g. *κρίση*, *κρίσης*, *κρίσεις*, *κρίσεων* are all forms of *κρίση*).
A full lemmatiser would be more linguistically faithful but would add a
dependency (e.g. on a Greek NLP toolkit), would introduce its own error
rate, and would be slower at the 1.5M-article scale.

Stem matching with hand-curated stems is a transparent middle path. The
trade-off is that very short stems (e.g. *χρεο-*) can over-match into
unrelated words; the curated stem list is conservative on length to
mitigate this.

## 3. Context window weighting

Each detected uncertainty stem is examined in a ±5-word window. Three
modifications can apply:

- **Negation** (within the *preceding* part of the window): the keyword
  is suppressed (weight 0).
- **Diminisher** (anywhere in the window): the keyword is suppressed
  (weight 0). If the diminisher is itself preceded by a negator (within
  3 words), the suppression is itself suppressed and the keyword is
  restored at weight 1.5 — this captures phrasings like
  *δεν ξεπεράστηκε η κρίση* ("the crisis was not overcome").
- **Amplifier** (anywhere in the window): the keyword is boosted to
  weight 2.0. If the amplifier is itself negated, it becomes a negator
  instead.

The numerical weights (0, 1, 1.5, 2) are heuristic. A grid search over
these weights is feasible but is not currently performed because the
sample size of clearly-modified mentions is small relative to the
plain-baseline mentions, which limits the statistical power to choose
between weight schemes.

## 4. The eight index variants

The pipeline emits eight candidate monthly indices:

| Variant | Definition |
| --- | --- |
| `basic` | EPU-relevant articles / total articles |
| `title_based` | EPU-relevant *titles* / total articles |
| `context_weighted` | Sum of context scores / total articles |
| `bigram` | `basic` + bigram-density bonus (×5) |
| `sentiment_lm` | `basic` reweighted by Loughran-McDonald negativity |
| `title_content` | 0.4·`title_based` + 0.6·`basic` |
| `ultimate` | Multiplicative combination of context, sentiment, bigrams |
| `title_bigram` | `title_based` + bigram-density bonus (×3) |

Each is normalised to mean 100 over the sample, then detrended by
removing the fitted linear trend and adding 100 back.

## 5. Method selection

Selecting the variant with the highest in-sample correlation against
the target inflates that correlation: with eight candidates the
expectation of the maximum is meaningfully above the expectation of any
one. The pipeline mitigates this in two ways:

1. The selection step uses only the first 70% of the sample
   (chronologically), and the held-out 30% is used to report a
   correlation that has not been optimised against.
2. All eight variants are reported, so the magnitude of the search
   penalty is visible to the reader.

## 6. Stationarity and the choice of regression specification

ADF and KPSS are run on each series in levels and in first differences.

- If both tests agree that a series is stationary in levels, regressions
  in levels are interpretable directly.
- If both tests agree that a series is non-stationary in levels, the
  Engle-Granger cointegration test is consulted. If the residual of an
  OLS levels regression is itself stationary, the levels regression
  identifies a long-run equilibrium and is interpretable.
- If neither holds, inference is conducted in first differences.

The pipeline reports all three regressions (levels bivariate, levels
with AR(1)+trend, first differences) so that the reader can see how
results depend on the specification.

## 7. HAC standard errors

All OLS regressions use Newey-West HAC standard errors. The maximum lag
is set following Newey & West (1994):

```
maxlags = floor(4 * (n / 100) ** (2 / 9))
```

For 194 monthly observations this gives 4. This is preferred to a
hand-picked maxlag because it has known asymptotic properties.

## 8. Granger causality

Granger tests are conducted on the first-differenced series, not in
levels, to avoid the well-known size distortions of Granger tests on
non-stationary series. Tests are reported at lags 1 through 6 in both
directions (EPU → spread and spread → EPU).

Granger non-causality is a statement about temporal precedence in
linear projections, not about causation. The reverse direction
(spread → EPU) is informative because if newspapers are reacting to
market stress, the index is partly endogenous.

## 9. Block bootstrap

Standard OLS standard errors and even HAC standard errors rely on
asymptotic normality, which the residuals do not satisfy in this sample
(Jarque-Bera rejects normality at the 1% level for both the levels and
first-difference equations). The pipeline therefore reports a moving-
block bootstrap confidence interval for the first-difference slope,
with block size `ceil(n^(1/3))` per Hall, Horowitz & Jing (1995). One
thousand replications are used by default.

## 10. Out-of-sample forecast comparison

The most demanding test of "the index contains information not already
in the spread" is whether augmenting an AR(1) spread forecast with
lagged EPU reduces forecast errors. The pipeline implements this as a
rolling-origin one-step-ahead forecast and compares the augmented model
against the AR(1) benchmark using a Diebold-Mariano test with
Harvey-Leybourne-Newbold small-sample correction.

If the DM p-value is below 10%, the augmented model has significantly
lower forecast loss than the benchmark; otherwise it does not. The
result is reported as-is, regardless of direction.
