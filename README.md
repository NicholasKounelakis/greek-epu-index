# Greek Economic Policy Uncertainty Index

Monthly EPU index for Greece, constructed from ~1.5 million Greek-language
newspaper articles (2010–2026), with diagnostics against the 10-year
Greece-Germany government bond spread.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

---

## What this project is

A reproducible pipeline that:

1. Streams ~1.5M Greek newspaper articles, classifies each one as
   EPU-relevant via Greek-language stem matching, and aggregates monthly
   article-share indices in the spirit of Baker, Bloom & Davis (2016).
2. Runs a complete diagnostic battery against the bond spread:
   stationarity (ADF, KPSS), Engle-Granger cointegration, Newey-West HAC
   regressions in **levels** and **first differences**, Granger causality,
   block-bootstrap confidence intervals, and a rolling-origin
   out-of-sample forecast comparison via Diebold-Mariano.
3. Selects the headline index variant on a training fold and reports its
   correlation on a held-out test fold to mitigate post-hoc method
   selection bias.

The headline series for download is in [`data/epu_timeseries.csv`](data/epu_timeseries.csv).

![EPU vs Bond Spread](figures/01_epu_vs_spread.png)

## Index methodology

### Article classification

An article is flagged as EPU-relevant if it contains at least one stem
from each of three categories simultaneously:

| Category | Example stems (translated) |
| --- | --- |
| **Uncertainty** | *αβεβαι-* (uncertainty), *κρίσ-* (crisis), *κίνδυν-* (risk), *αστάθει-* (instability) |
| **Economy** | *οικονομ-* (economy), *τραπεζ-* (bank), *ομόλογ-* (bond), *πληθωρισμ-* (inflation) |
| **Policy** | *κυβέρνησ-* (government), *μνημόνι-* (memorandum), *εκλογ-* (election), *eurogroup* |

Stem matching is used because Greek inflectional morphology produces many
surface forms per lemma; matching on stems is more permissive than exact
word-form matching and avoids the need for a full Greek lemmatiser.

### Three context-aware adjustments

1. **Negation handling** — an uncertainty keyword preceded within a small
   window by a negator (e.g. *δεν*, *χωρίς*) is discarded.
2. **Diminisher handling** — an uncertainty keyword in proximity to a
   diminisher (e.g. *βελτιώθηκε*, *λύθηκε*) is discarded; if the
   diminisher is itself negated (*δεν λύθηκε*), the keyword is restored
   with a weight of 1.5.
3. **Amplifier handling** — an uncertainty keyword in proximity to an
   amplifier (e.g. *κλιμακώνεται*, *εντείνεται*) receives weight 2.0,
   unless the amplifier itself is negated.

The choice of weights is heuristic. Sensitivity to these weights is
reported in `docs/methodology.md`.

### Index variants

The pipeline produces eight variants that differ in how article-level
scores are aggregated (basic frequency, title-only, context-weighted,
bigram-weighted, sentiment-weighted, and three combinations). A linear
trend is removed from each variant to control for the secular increase
in article volume that would otherwise contaminate any subsequent
regression.

The headline variant is selected by training-fold correlation against
the bond spread (first 70% of months); the held-out test fold (last 30%)
is used to report an out-of-sample correlation that has not been
optimised against. This is documented in `docs/methodology.md`.

## Diagnostics on the index series

Run on the 194 monthly observations covering 2010-01 to 2026-02:

| Test | Series | Statistic | p-value | Interpretation |
| --- | --- | --- | --- | --- |
| ADF | EPU (detrended), levels | −3.65 | 0.005 | Reject unit root at 1% |
| KPSS | EPU (detrended), levels | 1.58 | <0.01 | Reject stationarity at 1% |
| ADF | ΔEPU | −12.36 | <0.001 | First differences are stationary |
| KPSS | ΔEPU | 0.19 | >0.10 | First differences are stationary |

ADF and KPSS disagree on the levels: the index is near the boundary of
stationarity, which is expected given the structural breaks in Greek
political and macroeconomic conditions over the sample. **First
differences are unambiguously stationary**, so all reported regressions
are also estimated in first differences and any inference on the
levels-equation is treated with caution.

The full battery — including the bond-spread side, Engle-Granger
cointegration, Granger causality at lags 1–6, block-bootstrap CIs for
the first-difference slope, and rolling-origin Diebold-Mariano — is
produced automatically by the pipeline and written to
`data/processed/diagnostics.json`.

## What this index does *not* claim

These caveats are deliberately prominent because they are easy to elide
and easy to misread.

1. **No causal claim.** Co-movement between newspaper-derived
   uncertainty and bond spreads is consistent with several mechanisms,
   including reverse causation (markets stress -> media coverage of
   stress) and common-cause confounding (macro shock -> both move). The
   Granger tests in this repo address temporal precedence, not
   causation.
2. **No claim of market inefficiency.** The predictive regression of the
   spread on lagged EPU does not yield a coefficient that is reliably
   different from zero once a deterministic trend is included. Whether
   the index has incremental predictive value is exactly what the
   Diebold-Mariano test in the pipeline is designed to evaluate; the
   result is reported, not assumed.
3. **The lexicons are not externally validated.** The Greek adaptation
   of Loughran-McDonald sentiment terms is a translation, not a
   benchmark-validated dictionary. A meaningful next step would be a
   small annotated test set against which the classifier could be
   calibrated.
4. **Method selection inflates in-sample fit.** Trying eight index
   variants and reporting the best one gives an upward-biased estimate
   of correlation. This is why the pipeline reports both the in-sample
   and out-of-sample correlation, and why the bootstrap and DM tests
   are run on the held-out fold.
5. **~24% of articles are excluded** as malformed (HTML artifacts,
   truncated content, encoding errors). The exclusion criteria are in
   the pipeline source. Whether these criteria are correlated with
   period or outlet is not formally tested here.

## Repository layout

```
greek-epu-index/
├── src/
│   ├── lexicons.py          # Greek stem dictionaries
│   ├── scoring.py           # Article-level NLP scoring
│   ├── index_builder.py     # Monthly aggregation + detrending
│   ├── econometrics.py      # ADF/KPSS, cointegration, HAC OLS,
│   │                        # Granger, bootstrap, DM, holdout
│   ├── plotting.py          # Figure generation
│   ├── sensitivity.py       # Jackknife + subsample stability
│   └── pipeline.py          # End-to-end CLI
├── tests/                   # Unit tests (28 tests)
├── data/
│   └── epu_timeseries.csv   # Headline monthly series
├── figures/                 # Generated PNGs
├── docs/
│   ├── methodology.md       # Detailed methodology + design choices
│   └── data_schema.md       # Expected input formats
├── notebooks/
│   └── walkthrough.ipynb    # Step-by-step exploration
├── requirements.txt         # Pinned dependencies
└── LICENSE
```

## Quickstart

```bash
git clone https://github.com/NicholasKounelakis/greek-epu-index.git
cd greek-epu-index
pip install -r requirements.txt
pytest tests/
```

To reproduce the full analysis on your own data:

```bash
python -m src.pipeline \
    --articles path/to/news_part1.csv path/to/news_part2.csv \
    --spread path/to/bond_spread.xlsx \
    --output data/processed
```

The expected input formats are documented in [`docs/data_schema.md`](docs/data_schema.md).

The raw newspaper corpus is not redistributed here due to copyright on
the underlying articles; the headline monthly index series is provided.

## Tech stack

Python 3.10+, pandas, NumPy, statsmodels, SciPy, matplotlib, pytest. No
ML frameworks: the NLP pipeline is rule-based and fully transparent.

## References

- Baker, S. R., Bloom, N., & Davis, S. J. (2016). *Measuring Economic
  Policy Uncertainty.* Quarterly Journal of Economics, 131(4), 1593–1636.
- Loughran, T., & McDonald, B. (2011). *When Is a Liability Not a
  Liability? Textual Analysis, Dictionaries, and 10-Ks.* Journal of
  Finance, 66(1), 35–65.
- Newey, W. K., & West, K. D. (1994). *Automatic Lag Selection in
  Covariance Matrix Estimation.* Review of Economic Studies, 61(4),
  631–653.
- Diebold, F. X., & Mariano, R. S. (1995). *Comparing Predictive
  Accuracy.* Journal of Business & Economic Statistics, 13(3), 253–263.
- Hall, P., Horowitz, J. L., & Jing, B.-Y. (1995). *On Blocking Rules
  for the Bootstrap with Dependent Data.* Biometrika, 82(3), 561–574.

## Authors

Nicholas Kounelakis · Giorgos Papadakis

Undergraduate project · *Text as Data in Economics: The Power of AI* · 2026

## License

MIT — see [LICENSE](LICENSE).
