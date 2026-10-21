# Calendar discovery, rare events and validation

## Three distinct questions

1. **Known-calendar hypotheses:** Could a named event (Easter, Christmas, shutdown, campaign launch) affect the target before, during or after the event? Event dates must be supplied independently of the target. These features are forecast-known.
2. **Recurring anomaly scan:** Do baseline **training residuals** exhibit unusually large signed effects at repeated 52-week (364-day) or calendar-day (365-day) phases? This is exploratory hypothesis generation, not a significance test.
3. **Incremental validation:** Does the proposed feature improve a baseline model on multiple future-origin folds, including a penalty for instability and multiple testing? **Not yet implemented** in this update.

## Why both 52 weeks and 365 days?

52 × 7 = 364 days, whereas Gregorian years have 365 or 366 days. Weekly business cycles align to weekday; fixed-date holidays align to civil calendars. Easter changes date and requires an independently computed event calendar. A single lag of 52 rows does not express all three phenomena. Also consider ISO week 53, leap days, trading-day counts and business-defined week boundaries.

## Before/after events

`calendar_features(dates, offsets_weeks=range(-10,11))` creates daily indicators at exact event-date ± 7*k days. Negative offset is **before** the event. For weekly data, create a daily date grid first and aggregate by your true week key; otherwise a Tuesday Easter-related event could be missed by Monday-only weekly timestamps.

## What rare-event evidence needs

A one-off peak is not recurrence. Require at least 3 independent cycles to explore, preferably 4–5+ to validate. Inspect effect direction, uncertainty, fold-to-fold sign stability, data coverage and overlap with trend, school holidays, advertising and production shutdowns. Hold out the final years and test candidate families together; repeated searching over offsets is a large multiple-testing problem. Never infer the event date from a future target spike.

## Leakage contracts to implement next

- `static`: known independently of time;
- `known_future`: deterministic calendar, scheduled price or preannounced event (subject to revision policy);
- `observed`: known only after observation;
- `forecast`: itself predicted and accompanied by a vintage timestamp.

For rolling-origin validation, transformations with data-fitted parameters (threshold quantiles, Hill K, scalers, imputers) must fit on training folds only and transform validation folds without refitting. Group-aware lags need sorting and isolation by group; current generic `evaluate()` does not guarantee this.

## API caveat

`scan_recurring_anomalies()` returns descriptive standardized residual differences, not p-values, adjusted p-values, model lift or causal effects. It intentionally refuses scans with fewer than `min_cycles` distinct years. A recurring phase scan is especially vulnerable to spurious discoveries when testing many candidate offsets.
