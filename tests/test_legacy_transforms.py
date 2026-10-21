import numpy as np
import pandas as pd
from hypoforge import FeatureSpec, evaluate, generate_candidates, infer_columns

def test_binary_does_not_get_numeric_transformations():
    df=pd.DataFrame({'flag':[0,1,0,1]})
    specs=generate_candidates(df,infer_columns(df),interactions=False,lags=(),windows=())
    ops={s.op for s in specs if s.inputs==('flag',)}
    assert 'square' not in ops and 'sqrt_abs' not in ops and 'log1p_abs' not in ops

def test_rolling_feature_uses_past_only():
    df=pd.DataFrame({'x':[1.,2.,3.,100.]})
    s=FeatureSpec('r','rolling_mean',('x',),(('window',3),),lookback=3)
    out=evaluate(s,df)
    assert np.isclose(out.iloc[3],2.0)

def test_lag_is_exact():
    df=pd.DataFrame({'x':[10,20,30]})
    s=FeatureSpec('l','lag',('x',),(('periods',1),),lookback=1)
    assert evaluate(s,df).iloc[1]==10

def test_threshold_is_binary():
    df=pd.DataFrame({'x':[1,2,3]})
    s=FeatureSpec('t','threshold_gt',('x',),(('value',2),),kind='binary')
    assert evaluate(s,df).tolist()==[0,0,1]
