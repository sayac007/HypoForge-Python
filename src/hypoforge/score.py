import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from .transforms import evaluate
from .safety import assess_candidate

def _entropy_binary(y):
    p=y.value_counts(normalize=True)
    return float(-(p*np.log2(p+1e-15)).sum())

def score_candidate(spec,df,target,task="auto"):
    safety=assess_candidate(spec,df)
    if not safety["numerically_safe"]:
        return {"feature":spec.name,"valid":False,"spec":spec,**safety}
    x=evaluate(spec,df)
    z=pd.DataFrame({"x":x,"y":df[target]}).replace([np.inf,-np.inf],np.nan).dropna()
    if len(z)<10 or z.x.nunique()<2:
        return {"feature":spec.name,"valid":False,"spec":spec,**safety}
    if task=="auto": task="binary" if z.y.nunique()==2 else "regression"
    xv=z[["x"]]
    if not pd.api.types.is_numeric_dtype(xv.x):
        xv=pd.DataFrame({"x":pd.factorize(xv.x)[0]})
    discrete=spec.kind in ("binary","categorical")
    if task=="binary":
        mi=float(mutual_info_classif(xv,z.y,discrete_features=[discrete],random_state=0)[0])
        h=_entropy_binary(z.y); info=mi/(h*np.log(2)) if h>0 else 0.
        overall=float(z.y.mean())
        true=z.loc[z.x.astype(bool),"y"] if spec.kind=="binary" else z.loc[z.x>=z.x.median(),"y"]
        shift=float(true.mean()-overall) if len(true) else np.nan
    else:
        info=float(mutual_info_regression(xv,z.y,discrete_features=[discrete],random_state=0)[0])
        shift=float(z.loc[z.x>=z.x.median(),"y"].mean()-z.y.mean())
    # Soft preference: safer transforms win close contests; strong evidence can still win.
    preference=info/(1.0+safety["risk_penalty"])
    return {"feature":spec.name,"valid":True,"information_score":info,
            "preference_score":preference,"support":float(z.x.astype(bool).mean()) if spec.kind=="binary" else 1.,
            "target_shift":shift,"n":len(z),"spec":spec,**safety}
