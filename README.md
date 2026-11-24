# HypoForge — Discover hypotheses, not just columns

**HypoForge** is an experimental open-source **Python** toolkit for generating and screening interpretable feature hypotheses from tabular and time-series data: nonlinear transforms, interactions, lags, calendar effects and marketing-response functions.

**Why use it?** Because a high correlation with your target is not enough. A candidate feature should be interpretable, numerically sensible and — before adoption — tested for **incremental predictive value against your existing baseline on later data**.

> **Important boundary:** Python `discover()` and `select_candidates()` support exploration and screening. The Python package does **not currently ship a native rolling-origin Ridge tournament**. The related [R package](https://github.com/sayac007/HypoForge-R) does, with a different discovery algorithm. Neither package estimates causal effects or guarantees forecast improvements.

## Install

Python 3.11+:

```bash
pip install "git+https://github.com/sayac007/HypoForge-Python.git"
```

For reproducibility, pin a Git commit SHA in your own environment. Core dependencies include NumPy, pandas and scikit-learn.

## Quick start: Find candidate relationships

```python
import pandas as pd
from hypoforge import discover, select_candidates

train = pd.DataFrame({
    "distribution": [30, 36, 40, 44, 50, 54, 60, 66, 72, 76, 82, 88],
    "price": [2.1, 2.0, 2.2, 2.1, 1.9, 2.0, 1.8, 1.9, 1.8, 1.7, 1.9, 1.8],
    "sales": [110, 130, 140, 160, 190, 205, 250, 290, 320, 345, 390, 420],
})
ranked = discover(train, "sales", task="regression", interactions=False)
selected, audit = select_candidates(
    train, ranked, baseline_features=["distribution", "price"],
    max_features=3
)
print(audit[["feature", "decision"]].head())
```

This demonstrates the API, **not** validated incremental forecast performance. Python's exploratory ranker is not mathematically identical to R's absolute Pearson ranker. A feature that ranks highly may still add no information beyond the baseline.

## Hypothetical case study: distribution, seasonality and sparse promotion signals

This section uses **only** [`examples/hypothetical_weekly_sales.csv`](examples/hypothetical_weekly_sales.csv), a deliberately constructed sequence of round, fictitious values. It contains no observations about any real product, industry, organization or time period. No model-performance findings are asserted.

### 1. Define two testable hypotheses

**Distribution and seasonality:** Distribution can rise before a hypothetical seasonal demand period begins. Its association with sales may be nonlinear or depend on seasonal activation. A squared, cubic or interaction feature is a hypothesis, not proof of a causal relationship.

**Sparse promotion extrapolation:** A signal that is almost always zero in training may take a value far outside the training range later. A Ridge coefficient penalty does not bound predictions at extreme covariate values. This is a potential failure mode, not an empirical claim that any particular fitted model failed.

### 2. Load the hypothetical sequence and reserve later observations

```python
import pandas as pd
from hypoforge import discover, select_candidates

df = pd.read_csv("examples/hypothetical_weekly_sales.csv")
df = df.sort_values("week_index").reset_index(drop=True)
assert df["week_index"].is_unique
cutoff = int(0.8 * len(df))
train = df.iloc[:cutoff].copy()
holdout = df.iloc[cutoff:].copy()  # Never use to choose hypotheses.
```

**Never randomly shuffle a forecasting time series.** Historical actual future-driver values are not necessarily available at the forecast origin. Only truly available inputs can be used in an ex-ante forecast.

### 3. Screen hypotheses on training only

```python
baseline = ["distribution", "season_index", "promotion_signal"]
ranked = discover(train[["sales", *baseline]], "sales", task="regression",
                  interactions=False, lags=(), windows=())
selected, audit = select_candidates(train[["sales", *baseline]], ranked,
                                    baseline_features=baseline, max_features=3)
print(audit[["feature", "decision"]].head())
```

The Python ranker is exploratory, not a rolling-origin tournament. Redundancy screening does not establish incremental predictive value. Construct and test nonlinear candidates only within a locked, time-ordered validation design.

### 4. Validate rather than assume an improvement

The Python package currently does not implement a native rolling-origin Ridge tournament. A separate evaluation can use `TimeSeriesSplit` and a pipeline containing fold-fitted imputation, scaling and Ridge. Compare a fixed baseline against baseline-plus-one-candidate on **identical** later folds. Keep the final holdout untouched until the entire procedure is frozen. No RMSE or improvement figures are presented here because no such benchmark is claimed.

The [R package](https://github.com/sayac007/HypoForge-R) provides a native rolling-origin tournament with a different exploratory ranker. Numerical parity requires explicitly matching folds, preprocessing and feature availability.

### 5. A reliable workflow for your own project

1. Define target, forecast horizon and true future-covariate availability.
2. Hold out the latest period **before** looking for patterns.
3. Keep an economically complete baseline.
4. Generate and screen hypotheses using **training only**.
5. Run a rolling-origin model comparison with the **same folds** for baseline and candidate.
6. Inspect consistency across windows and sensitivity to unusual campaigns.
7. Investigate extreme standardized values, sparse indicators and changes in measurement units.
8. Evaluate any new interaction or saturation idea in **new** validation runs.
9. Make strong generalization claims only after a genuinely untouched final test.

**The value of HypoForge is the disciplined path from a plausible idea to a testable, auditable feature — not an automatic declaration that a nonlinear curve is the truth.**

## Calendar hypotheses

```python
import pandas as pd
from hypoforge.calendar import calendar_features, scan_recurring_anomalies

dates = pd.date_range("2024-01-01", "2027-12-31", freq="D")
calendar = calendar_features(dates, country="DE", offsets_weeks=range(-10, 11))
# event_easter_w-10 ... event_easter_w+10, Fourier terms and weekday indicators

# Exploratory ONLY: scan TRAINING residuals, never holdout outcomes.
# candidates = scan_recurring_anomalies(train_dates, train_residuals,
#                                      period_days=364, min_cycles=3)
```

Calendar output is **daily**. Aggregate it to the actual business week when modeling weekly data. A 364-day phase is a 52-week cycle, not the same as a 365-day calendar-year phase; neither automatically replaces named moving holidays. The built-in German event set is not a full state-specific school-holiday calendar. Residual scans are **not** significance tests.

## Marketing-mix-inspired transformations

```python
from hypoforge.media import (
    geometric_adstock, delayed_adstock, weibull_adstock,
    logistic_saturation, michaelis_menten, reach_frequency
)

spend = [100, 0, 0, 80, 0]
carry = geometric_adstock(spend, alpha=0.6, l_max=4)
response = logistic_saturation(carry, lam=0.02)
```

These are **deterministic transforms**, not a fitted MMM system. They do not supply priors, causal identification, budget optimization or automatic attribution. Adstock requires a correctly ordered **single time series**; do not concatenate unrelated groups.

## Documentation and project status

- [Function reference](docs/FUNCTION_REFERENCE.md) — Python API and transform families
- [Calendar and validation](docs/CALENDAR_AND_VALIDATION.md) — moving holidays and leakage controls
- [MMM transforms and sources](docs/MMM_TRANSFORMS_AND_SOURCES.md) — framework comparisons and primary sources
- [Release notes](docs/RELEASE_NOTES.md) — implementation scope
- [CHALLENGE.md](CHALLENGE.md) — roadmap and open methodological questions

**Roadmap:** Generate → Score → Discard → Validate → Tournament. The model-aware rolling-origin tournament is still a **Python roadmap item**; it already exists in the independent R implementation.

## Contributing and license

Apache-2.0. Reproducible issues, benchmark data, tests and independent validations are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).
