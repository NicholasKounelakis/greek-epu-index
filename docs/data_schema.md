# Input data schemas

The pipeline expects two inputs.

## Newspaper articles

One or more CSV files, passed via `--articles`, each with at least these
columns:

| Column | Type | Description |
| --- | --- | --- |
| `date` | string parseable by `pd.to_datetime` | Publication date |
| `content` | string | Article body |
| `article_title` | string | Article headline |

Other columns are ignored. The pipeline streams each file in chunks of
10,000 rows; it does not load the full corpus into memory.

The corpus used in the headline analysis covers the period 2010-01 to
2026-02. Articles outside this range are processed but their
contribution to the headline series depends on whether they are matched
to a corresponding bond-spread observation.

## Bond spread

A single Excel file (`.xlsx`), passed via `--spread`, with these
columns:

| Column | Type | Description |
| --- | --- | --- |
| `Date` | string parseable by `pd.to_datetime` | Observation date |
| `Spread` | float | Greece-Germany 10Y spread, in percentage points |

Daily or higher-frequency observations are accepted; the pipeline
aggregates to monthly frequency by taking the within-month mean of the
spread.

## Outputs

After a successful run, `--output` contains:

| File | Contents |
| --- | --- |
| `epu_timeseries_all_methods.csv` | All eight index variants, monthly |
| `epu_timeseries.csv` | Headline (selected) variant only |
| `merged_data.csv` | Merged EPU + spread, ready for downstream regression |
| `diagnostics.json` | Stationarity, cointegration, regressions, Granger, bootstrap, OOS results |
| `regression_summaries.txt` | Full statsmodels summary tables |
