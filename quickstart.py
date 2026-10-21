import numpy as np
import pandas as pd
from hypoforge import discover, select_candidates

rng=np.random.default_rng(42); x=rng.uniform(0,20,1000)
df=pd.DataFrame({"x":x,"noise":rng.normal(size=1000),
                 "target":x*x/(25+x*x)+rng.normal(scale=.02,size=1000)})
ranked=discover(df,"target",task="regression",top_k=20,lags=(),windows=(),interactions=False)
selected,audit=select_candidates(df,ranked,max_features=3)
print(selected[["feature","information_score","preference_score","risk_penalty"]])
print(audit)
