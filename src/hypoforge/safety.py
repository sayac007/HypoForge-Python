"""Numerical safety and conservative transformation preference."""
import numpy as np
import pandas as pd
from .transforms import evaluate

# Priors are deliberately modest: they break near-ties, not override strong evidence.
# Higher values mean greater extrapolation / numerical-instability potential.
OP_RISK = {
    "identity": 0.00, "abs": 0.00, "log1p_abs": 0.00, "sqrt_abs": 0.02,
    "hill": 0.02, "threshold_gt": 0.03, "threshold_lt": 0.03, "range": 0.03,
    "weekday": 0.00, "month": 0.00, "dayofyear": 0.00, "is_weekend": 0.00,
    "lag": 0.00, "diff": 0.03, "rolling_mean": 0.00, "rolling_median": 0.00,
    "rolling_min": 0.00, "rolling_max": 0.00, "ewm_mean": 0.00,
    "rolling_std": 0.03, "rolling_range": 0.03,
    "add": 0.03, "subtract": 0.03, "abs_diff": 0.03,
    "relative_diff": 0.08, "ratio": 0.10, "reciprocal": 0.12,
    "multiply": 0.15, "pct_change": 0.12, "square": 0.20, "cube": 0.45,
}

def _robust_scale(s):
    x=pd.to_numeric(s,errors="coerce").replace([np.inf,-np.inf],np.nan).dropna().abs()
    return float(x.quantile(.95)) if len(x) else np.nan

def assess_candidate(spec, df):
    """Return diagnostics used for hard safety guards and soft preference penalties."""
    x=evaluate(spec,df)
    xn=pd.to_numeric(x,errors="coerce").replace([np.inf,-np.inf],np.nan)
    finite_fraction=float(xn.notna().mean())
    vals=xn.dropna()
    if len(vals)<2:
        return {"numerically_safe":False,"risk_penalty":1.0,"finite_fraction":finite_fraction,
                "amplification_orders":np.nan,"reason":"insufficient_finite_values"}
    if vals.nunique()<2:
        return {"numerically_safe":False,"risk_penalty":1.0,"finite_fraction":finite_fraction,
                "amplification_orders":0.0,"reason":"near_constant"}
    out_scale=_robust_scale(vals)
    input_scales=[_robust_scale(df[c]) for c in spec.inputs if pd.api.types.is_numeric_dtype(df[c])]
    ref=max([v for v in input_scales if np.isfinite(v) and v>0],default=np.nan)
    amp=0.0
    if np.isfinite(ref) and ref>0 and np.isfinite(out_scale) and out_scale>0:
        amp=max(0.0,float(np.log10(out_scale/ref)))
    # Observed scale amplification adds a data-dependent penalty. One order is tolerated;
    # each additional order adds 0.10, capped so evidence can still dominate.
    amp_penalty=min(0.40,max(0.0,amp-1.0)*0.10)
    risk=float(OP_RISK.get(spec.op,0.05)+amp_penalty)
    safe=finite_fraction>=0.80 and np.isfinite(out_scale) and out_scale<1e150
    reason="ok" if safe else ("low_finite_fraction" if finite_fraction<0.80 else "extreme_scale")
    return {"numerically_safe":safe,"risk_penalty":risk,"finite_fraction":finite_fraction,
            "amplification_orders":amp,"reason":reason}
