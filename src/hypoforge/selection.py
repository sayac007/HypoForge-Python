"""Redundancy-aware candidate admission for downstream models."""
import numpy as np
import pandas as pd
from .transforms import evaluate

def _numeric_series(s):
    if pd.api.types.is_numeric_dtype(s): return pd.to_numeric(s,errors="coerce")
    return pd.Series(pd.factorize(s)[0],index=s.index,dtype=float)

def _standardized_matrix(cols):
    x=pd.concat(cols,axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    if x.empty: return None
    keep=x.std(ddof=0)>1e-12
    x=x.loc[:,keep]
    if x.shape[1]==0: return None
    return (x-x.mean())/x.std(ddof=0)

def _condition_number(cols):
    z=_standardized_matrix(cols)
    if z is None or z.shape[1]<2: return 1.0
    return float(np.linalg.cond(z.to_numpy(dtype=float)))

def select_candidates(df, ranked, *, baseline_features=(), max_features=5,
                      correlation_threshold=.98, max_condition_number=1e6):
    """Select candidates conservatively against baseline and already-selected columns.

    `baseline_features` are column names already used by the downstream model.
    Returns (selected_dataframe, audit_dataframe). No target is used here.
    """
    base=[]
    for c in baseline_features:
        if c in df.columns: base.append(_numeric_series(df[c]).rename(c))
    selected=[]; selected_series=[]; audit=[]
    ordered=ranked.sort_values(["preference_score","information_score"],ascending=False)
    for _,row in ordered.iterrows():
        spec=row["spec"]; s=_numeric_series(evaluate(spec,df)).rename(spec.name)
        reason="selected"; max_corr=0.0
        comparisons=base+selected_series
        for other in comparisons:
            pair=pd.concat([s,other],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
            if len(pair)>2 and pair.iloc[:,0].nunique()>1 and pair.iloc[:,1].nunique()>1:
                corr=abs(float(pair.corr().iloc[0,1])); max_corr=max(max_corr,corr)
        if max_corr>=correlation_threshold:
            reason="redundant_correlation"
        else:
            cond=_condition_number(comparisons+[s])
            if not np.isfinite(cond) or cond>max_condition_number:
                reason="condition_number"
        audit.append({"feature":spec.name,"decision":reason,"max_abs_correlation":max_corr,
                      "condition_number":_condition_number(comparisons+[s]),
                      "risk_penalty":row.get("risk_penalty",np.nan),
                      "preference_score":row.get("preference_score",np.nan),
                      "sources":"|".join(sorted(spec.sources))})
        if reason=="selected":
            selected.append(row); selected_series.append(s)
            if len(selected)>=max_features: break
    return pd.DataFrame(selected).reset_index(drop=True),pd.DataFrame(audit)
