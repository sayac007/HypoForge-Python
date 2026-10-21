import numpy as np
import pandas as pd
from .types import FeatureSpec

def evaluate(spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
    x = df[spec.inputs[0]]
    p = dict(spec.params)
    op = spec.op
    if op == "identity": return x
    if op == "log1p_abs": return np.sign(x) * np.log1p(np.abs(x))
    if op == "sqrt_abs": return np.sign(x) * np.sqrt(np.abs(x))
    if op == "square": return x * x
    if op == "cube": return x * x * x
    if op == "reciprocal": return 1.0 / x.replace(0, np.nan)
    if op == "abs": return np.abs(x)
    if op == "hill":
        k, n = float(p["k"]), float(p["n"])
        if k <= 0 or n <= 0: raise ValueError("Hill parameters k and n must be > 0")
        xn = x.where(x >= 0).astype(float).pow(n)
        return xn / (k**n + xn)
    if op == "is_missing": return x.isna().astype("int8")
    if op == "threshold_gt": return (x > p["value"]).astype("int8")
    if op == "threshold_lt": return (x < p["value"]).astype("int8")
    if op == "range": return ((x >= p["low"]) & (x <= p["high"])).astype("int8")
    if op == "category_eq": return (x == p["value"]).astype("int8")
    if op == "weekday": return pd.to_datetime(x).dt.weekday
    if op == "month": return pd.to_datetime(x).dt.month
    if op == "dayofyear": return pd.to_datetime(x).dt.dayofyear
    if op == "is_weekend": return (pd.to_datetime(x).dt.weekday >= 5).astype("int8")
    if op == "lag": return x.shift(int(p["periods"]))
    if op == "diff": return x.diff(int(p["periods"]))
    if op == "pct_change":
        return x.pct_change(int(p["periods"]), fill_method=None).replace([np.inf, -np.inf], np.nan)
    if op == "ewm_mean": return x.shift(1).ewm(span=int(p["span"]), adjust=False).mean()
    if op.startswith("rolling_"):
        w = int(p["window"]); r = x.shift(1).rolling(w, min_periods=max(2, w // 3))
        return {"rolling_mean": r.mean, "rolling_std": r.std, "rolling_min": r.min,
                "rolling_max": r.max, "rolling_median": r.median,
                "rolling_range": lambda: r.max() - r.min()}[op]()
    y = df[spec.inputs[1]]
    if op == "add": return x + y
    if op == "subtract": return x - y
    if op == "abs_diff": return (x - y).abs()
    if op == "multiply": return x * y
    if op == "ratio": return x / y.replace(0, np.nan)
    if op == "relative_diff": return (x - y) / y.abs().replace(0, np.nan)
    if op == "and": return (x.astype(bool) & y.astype(bool)).astype("int8")
    raise ValueError(f"Unknown operation: {op}")
