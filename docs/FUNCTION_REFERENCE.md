# HypoForge Python function reference

## Core API (existing)

| Function / type | Description | Important parameters |
|---|---|---|
| `discover(df, target, time=None, task='auto', top_k=50, **kw)` | Generate and score candidates using in-sample information and soft numerical-risk preference | `task='regression'` or `'binary'`; `time` sorts rows, but is **not** temporal validation |
| `infer_columns(df, exclude=None)` | Infer numeric/binary/categorical/datetime columns | `exclude` set |
| `generate_candidates(df, columns, quantiles, hill_exponents, lags, windows, interactions)` | Produce `FeatureSpec` candidates | `lags` count **rows**, not time units; inputs must be sorted |
| `evaluate(spec, df)` | Evaluate one feature spec on a DataFrame | `FeatureSpec` |
| `assess_candidate(spec, df)` | Finite fraction, scale amplification and numerical-risk diagnostics | rejects unsafe candidates |
| `select_candidates(df, ranked, baseline_features=(), max_features=5, correlation_threshold=.98, max_condition_number=1e6)` | Correlation and standardized condition-number screening | `baseline_features` are raw DataFrame column names, not model design terms |
| `ColumnSpec(name, kind, availability='observed')` | Input-column metadata | availability is metadata, not enforced yet |
| `FeatureSpec(name, op, inputs, params=(), kind='numeric', lookback=0, complexity=1)` | Immutable feature definition with `sources` property | sources track raw input columns |

## Existing `evaluate` operation families

**Unary numeric:** `identity`, `log1p_abs` (signed log1p), `sqrt_abs` (signed sqrt), `square`, `cube`, `reciprocal` (zero→NA), `abs`, `hill` (`x^n/(k^n+x^n)` for nonnegative x).

**Indicators:** `is_missing`, `threshold_gt`, `threshold_lt`, `range`, `category_eq`.

**Calendar:** `weekday` (Mon=0), `month`, `dayofyear`, `is_weekend`.

**Sequential/time:** `lag`, `diff`, `pct_change`, `ewm_mean`, `rolling_mean`, `rolling_std`, `rolling_min`, `rolling_max`, `rolling_median`, `rolling_range`. Rolling statistics shift the source one row to avoid same-row exposure; generic lags still require the caller to enforce availability and grouping. The current implementation does **not** support panel-aware grouping.

**Interactions:** `add`, `subtract`, `abs_diff`, `multiply`, `ratio`, `relative_diff`, `and`.

`generate_candidates()` chooses some thresholds and Hill K from the provided DataFrame. **Supply training data only**; no fold-fit/transform object exists yet. Inference of binary columns from `nunique <= 2` can misclassify all-null or small-sample columns.

## New `hypoforge.calendar`

- `easter_sunday(year) -> datetime.date`: Gregorian Easter; 1583+.
- `event_dates(years, country='DE', extra_events=None) -> dict[str,list[date]]`: built-in named German dates, plus user-supplied dates. Not state/school-holiday complete.
- `calendar_features(dates, country='DE', extra_events=None, offsets_weeks=(-10,-4,-2,-1,0,1,2,4,10), include_daily_cycle=True) -> DataFrame`: annual Fourier sin/cos harmonics 1 and 2, weekday/month/day-of-year and daily event-offset indicators. Dates must be timezone-naive. Weekly users must aggregate a daily event grid explicitly.
- `scan_recurring_anomalies(dates, residuals, period_days=365, offsets_days=range(-70,71,7), min_cycles=3, min_hits=3) -> DataFrame`: descriptive training-residual phase scan. Multiple-comparison adjusted inference and incremental value **not implemented**.

## New `hypoforge.media`

| Function | Signature and interpretation |
|---|---|
| `geometric_adstock` | `(x, alpha=.5, l_max=8, normalize=True)`; finite causal geometric weights |
| `delayed_adstock` | `(x, alpha=.5, peak=2, l_max=12, normalize=True)`; finite peaked kernel |
| `weibull_adstock` | `(x, shape=2., scale=3., l_max=12, normalize=True)`; discrete survival-difference weights |
| `logistic_saturation` | `(x, lam=1.)`; zero-based logistic response |
| `michaelis_menten` | `(x, half_saturation=1., maximum=1.)`; hyperbolic saturation |
| `power_saturation` | `(x, exponent=.5)`; concave power for nonnegative x |
| `tanh_saturation` | `(x, scale=1.)`; bounded smooth saturation |
| `reach_frequency` | `(reach, frequency, half_saturation=2., slope=1.)`; reach times Hill-saturated frequency |

### Example: manual MMM candidate comparison

```python
import pandas as pd
from hypoforge.media import geometric_adstock, logistic_saturation

weekly_spend = pd.Series([100, 0, 40, 0, 0, 70])
x1 = geometric_adstock(weekly_spend, alpha=.4, l_max=4)
x2 = logistic_saturation(x1, lam=.02)
# Fit a baseline and challenger on TRAIN only; score on future time folds.
```

### API boundaries

- Media kernels assume a single ordered series, not grouped panels.
- NA media exposure is treated as zero; confirm this matches the application.
- Lag weights normalize over available history, so early-series behavior differs from fixed-denominator kernels.
- Saturation inputs must be nonnegative; parameters are user-specified, not fitted.
- No causal effect, confidence interval, automatic incremental lift or attribution is inferred by these functions.
