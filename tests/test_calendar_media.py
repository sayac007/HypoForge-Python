import numpy as np
import pandas as pd
import pytest
from hypoforge.calendar import easter_sunday, calendar_features, scan_recurring_anomalies
from hypoforge.media import (geometric_adstock, delayed_adstock, weibull_adstock,
                              logistic_saturation, michaelis_menten, reach_frequency)


def test_easter_known_dates():
    assert str(easter_sunday(2024)) == '2024-03-31'
    assert str(easter_sunday(2025)) == '2025-04-20'
    assert str(easter_sunday(2026)) == '2026-04-05'


def test_calendar_shift_and_leap():
    dates = pd.to_datetime(['2026-04-05', '2026-03-29', '2024-02-29'])
    x = calendar_features(dates, offsets_weeks=(-1, 0))
    assert x.loc[0, 'event_easter_w+0'] == 1
    assert x.loc[1, 'event_easter_w-1'] == 1
    assert np.isfinite(x.select_dtypes('number')).all().all()


def test_scan_needs_repeated_years():
    x = scan_recurring_anomalies(pd.date_range('2025-01-01', periods=90), np.arange(90))
    assert x.empty


def test_geometric_no_future_leakage():
    a = geometric_adstock([1, 0, 0, 0], alpha=.5, l_max=3, normalize=False)
    b = geometric_adstock([1, 0, 0, 999], alpha=.5, l_max=3, normalize=False)
    assert np.allclose(a.iloc[:3], b.iloc[:3])
    assert np.allclose(a.to_numpy(), [1, .5, .25, .125])


def test_delayed_and_weibull():
    assert delayed_adstock([1, 0, 0, 0], peak=2, l_max=3).iloc[2] > 0
    assert np.isfinite(weibull_adstock([1, 0, 0], shape=2, scale=3)).all()


def test_saturations():
    assert logistic_saturation([0, 1]).iloc[0] == 0
    assert michaelis_menten([0, 2], half_saturation=2).iloc[1] == .5
    assert reach_frequency([100], [2], half_saturation=2).iloc[0] == 50
    with pytest.raises(ValueError): logistic_saturation([-1])
