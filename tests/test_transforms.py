import numpy as np
import pandas as pd
from hypoforge import FeatureSpec,evaluate,generate_candidates,infer_columns

def test_hill():
    df=pd.DataFrame({"x":[0.,4.,8.]}); s=FeatureSpec("h","hill",("x",),(("k",4.),("n",2.)))
    o=evaluate(s,df); assert np.isclose(o.iloc[1],.5) and ((o>=0)&(o<=1)).all()

def test_hill_negative_missing():
    o=evaluate(FeatureSpec("h","hill",("x",),(("k",1.),("n",1.))),pd.DataFrame({"x":[-1.,1.]}))
    assert np.isnan(o.iloc[0])

def test_hill_generated():
    df=pd.DataFrame({"x":np.arange(1.,101.)})
    ss=generate_candidates(df,infer_columns(df),lags=(),windows=(),interactions=False)
    assert len([s for s in ss if s.op=="hill"])==12

def test_rolling_past_only():
    df=pd.DataFrame({"x":[1.,2.,3.,100.]})
    s=FeatureSpec("r","rolling_mean",("x",),(("window",3),),lookback=3)
    assert np.isclose(evaluate(s,df).iloc[3],2.)
