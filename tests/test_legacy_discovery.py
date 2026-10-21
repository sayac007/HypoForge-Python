import numpy as np
import pandas as pd
from hypoforge import discover

def test_discovers_strong_binary_driver():
    rng=np.random.default_rng(7); n=1000
    x=rng.normal(size=n); noise=rng.normal(size=n)
    y=(x>0.7).astype(int)
    df=pd.DataFrame({'x':x,'noise':noise,'y':y})
    result=discover(df,'y',top_k=10,lags=(),windows=(),interactions=False)
    assert any(result.feature.str.startswith('x__gt_'))
    assert result.iloc[0].information_score>0.1

def test_time_series_driver_is_discoverable_without_current_target_leakage():
    rng=np.random.default_rng(3); n=400
    x=rng.normal(size=n); y=np.r_[0,x[:-1]]+rng.normal(scale=.05,size=n)
    df=pd.DataFrame({'date':pd.date_range('2024-01-01',periods=n),'x':x,'y':y})
    result=discover(df,'y',time='date',task='regression',top_k=20,lags=(1,7),windows=(7,),interactions=False)
    assert 'x__lag1' in set(result.feature)
