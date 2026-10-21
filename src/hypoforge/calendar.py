"""Deterministic calendar features and exploratory recurring-event diagnostics.

All features depend on timestamps or pre-declared event dates, not target values.
The anomaly scan is exploratory and MUST run on training residuals only.
"""
from __future__ import annotations

from datetime import date, timedelta
import numpy as np
import pandas as pd


def easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (Meeus/Jones/Butcher; Gregorian years)."""
    if year < 1583:
        raise ValueError("Gregorian Easter requires year >= 1583")
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def event_dates(years, *, country="DE", extra_events=None):
    """Return named events. DE includes Easter-related events; NOT state-specific holidays.

    extra_events: mapping of event name to iterable of dates (ISO strings or dates).
    """
    if country != "DE":
        raise ValueError("Only country='DE' is built in; supply extra_events for others")
    events = {"easter": [], "good_friday": [], "easter_monday": [],
              "ascension": [], "pentecost": [], "christmas": [], "new_year": []}
    for year in sorted(set(map(int, years))):
        e = easter_sunday(year)
        events["easter"].append(e)
        events["good_friday"].append(e - timedelta(days=2))
        events["easter_monday"].append(e + timedelta(days=1))
        events["ascension"].append(e + timedelta(days=39))
        events["pentecost"].append(e + timedelta(days=49))
        events["christmas"].append(date(year, 12, 25))
        events["new_year"].append(date(year, 1, 1))
    for name, dates in (extra_events or {}).items():
        if name in events:
            raise ValueError(f"Duplicate event name: {name}")
        events[name] = [pd.Timestamp(d).date() for d in dates]
    return events


def calendar_features(dates, *, country="DE", extra_events=None,
                      offsets_weeks=(-10, -4, -2, -1, 0, 1, 2, 4, 10),
                      include_daily_cycle=True):
    """Generate future-known event proximity, annual Fourier and cycle indicators.

    Offsets use signed difference (observation date - event date), in weeks.
    A weekly observation matches an event if its seven-day interval contains the
    shifted event date. Daily data matches the exact shifted event day.
    All dates are normalized; output retains original input index.
    """
    s = pd.to_datetime(pd.Series(dates), errors="raise")
    if s.isna().any():
        raise ValueError("dates contain missing values")
    if s.dt.tz is not None:
        raise ValueError("Pass timezone-naive dates")
    if len(s) == 0:
        return pd.DataFrame(index=s.index)
    normalized = s.dt.normalize()
    # Annual Fourier phase uses continuous Gregorian year fraction, including leap years.
    year_start = pd.to_datetime(s.dt.year.astype(str) + '-01-01')
    next_start = pd.to_datetime((s.dt.year + 1).astype(str) + '-01-01')
    phase = (normalized - year_start).dt.days / (next_start - year_start).dt.days
    out = pd.DataFrame(index=s.index)
    for harmonic in (1, 2):
        out[f"annual_sin_{harmonic}"] = np.sin(2 * np.pi * harmonic * phase).to_numpy()
        out[f"annual_cos_{harmonic}"] = np.cos(2 * np.pi * harmonic * phase).to_numpy()
    out["weekday"] = s.dt.weekday.to_numpy()
    out["month"] = s.dt.month.to_numpy()
    if include_daily_cycle:
        out["day_of_year"] = s.dt.dayofyear.to_numpy()
    years = range(int(s.dt.year.min()) - 1, int(s.dt.year.max()) + 2)
    events = event_dates(years, country=country, extra_events=extra_events)
    # Features are exact-date indicators. For weekly series, aggregate with max
    # over the week's days upstream; don't infer the user's week anchor.
    for name, anchors in events.items():
        anchor_days = pd.DatetimeIndex(anchors)
        for offset in offsets_weeks:
            shifted = anchor_days + pd.Timedelta(weeks=int(offset))
            out[f"event_{name}_w{int(offset):+d}"] = normalized.isin(shifted).astype('int8').to_numpy()
    return out


def scan_recurring_anomalies(dates, residuals, *, period_days=365,
                             offsets_days=range(-70, 71, 7),
                             min_cycles=3, min_hits=3):
    """Exploratory residual scan around recurring calendar phases.

    Requires at least min_cycles distinct years and min_hits observations per
    phase. Scores are descriptive only, NOT p-values or validated predictions.
    Supply TRAINING residuals from a baseline, never held-out outcomes.
    For weekly data, 364-day (52-week) cycles may be preferable to 365 days.
    """
    if period_days <= 0 or min_cycles < 2 or min_hits < 2:
        raise ValueError("Invalid scan parameters")
    dt = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    y = pd.to_numeric(pd.Series(residuals), errors='coerce').reset_index(drop=True)
    if len(dt) != len(y) or dt.isna().any():
        raise ValueError("Invalid dates/residuals")
    if dt.dt.year.nunique() < min_cycles:
        return pd.DataFrame(columns=['phase_day', 'offset_days', 'n', 'mean_residual', 'standardized_effect'])
    valid = y.notna() & np.isfinite(y)
    dt, y = dt[valid], y[valid]
    if y.std(ddof=1) <= 0 or len(y) < min_hits:
        return pd.DataFrame(columns=['phase_day', 'offset_days', 'n', 'mean_residual', 'standardized_effect'])
    origin = pd.Timestamp('2000-01-01')
    phase = ((dt.dt.normalize() - origin).dt.days % period_days).to_numpy()
    results = []
    for base in range(period_days):
        for offset in offsets_days:
            mask = phase == (base + int(offset)) % period_days
            n = int(mask.sum())
            if n >= min_hits:
                effect = float(y[mask].mean() - y.mean())
                results.append((base, int(offset), n, effect, effect / float(y.std(ddof=1))))
    return pd.DataFrame(results, columns=['phase_day', 'offset_days', 'n', 'mean_residual',
                                         'standardized_effect']).sort_values(
        'standardized_effect', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
