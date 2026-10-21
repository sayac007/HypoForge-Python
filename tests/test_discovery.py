import numpy as np
import pandas as pd
from hypoforge import discover

def test_binary_driver():
    rng=np.random.default_rng(7); x=rng.normal(size=1000)
    df=pd.DataFrame({"x":x,"noise":rng.normal(size=1000),"y":(x>.7).astype(int)})
    r=discover(df,"y",top_k=50,lags=(),windows=(),interactions=False)
    assert any(r.feature.str.startswith("x__gt_"))
    assert r.iloc[0].information_score>0.1

def test_lag_driver():
    rng=np.random.default_rng(3); x=rng.normal(size=400)
    y=np.r_[0,x[:-1]]+rng.normal(scale=.05,size=400)
    r=discover(pd.DataFrame({"date":pd.date_range("2024-01-01",periods=400),"x":x,"y":y}),
               "y",time="date",task="regression",top_k=30,lags=(1,7),windows=(7,),interactions=False)
    assert "x__lag1" in set(r.feature)
