import numpy as np
import pandas as pd
from hypoforge import FeatureSpec, assess_candidate, discover, select_candidates

def test_cube_gets_more_risk_than_log():
    df=pd.DataFrame({"x":np.linspace(1e5,5e5,200)})
    cube=assess_candidate(FeatureSpec("c","cube",("x",)),df)
    log=assess_candidate(FeatureSpec("l","log1p_abs",("x",)),df)
    assert cube["risk_penalty"] > log["risk_penalty"]
    assert cube["amplification_orders"] > 5

def test_preference_breaks_near_tie_toward_safer_transform():
    rng=np.random.default_rng(1); x=rng.uniform(1e5,5e5,1000)
    y=np.log1p(x)+rng.normal(scale=.03,size=1000)
    r=discover(pd.DataFrame({"x":x,"y":y}),"y",task="regression",top_k=100,
               lags=(),windows=(),interactions=False)
    cube=r.loc[r.feature=="x__cube"].iloc[0]
    log=r.loc[r.feature=="x__log1p_abs"].iloc[0]
    assert cube.risk_penalty > log.risk_penalty
    assert cube.preference_score < cube.information_score

def test_redundant_candidate_rejected_against_baseline():
    x=np.arange(1.,301.); df=pd.DataFrame({"x":x,"x_copy":x})
    spec=FeatureSpec("copy","identity",("x_copy",))
    ranked=pd.DataFrame([{"feature":"copy","spec":spec,"information_score":1.,
                          "preference_score":1.,"risk_penalty":0.}])
    selected,audit=select_candidates(df,ranked,baseline_features=("x",),max_features=1)
    assert selected.empty
    assert audit.iloc[0].decision=="redundant_correlation"

def test_interaction_keeps_all_sources():
    s=FeatureSpec("ab","multiply",("a","b"))
    assert s.sources==frozenset({"a","b"})
