from itertools import combinations
import pandas as pd
from .types import ColumnSpec, FeatureSpec

def infer_columns(df, exclude=None):
    exclude = exclude or set(); out = []
    for c in df.columns:
        if c in exclude: continue
        s = df[c]
        kind = ("datetime" if pd.api.types.is_datetime64_any_dtype(s)
                else "binary" if s.dropna().nunique() <= 2
                else "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical")
        out.append(ColumnSpec(c, kind))
    return out

def generate_candidates(df, columns, *, quantiles=(.1,.25,.5,.75,.9),
                        hill_exponents=(.5,1.,2.,3.), lags=(1,7,30),
                        windows=(7,30), interactions=True):
    out=[]; numeric=[]; binary=[]
    for col in columns:
        c=col.name
        out += [FeatureSpec(c,"identity",(c,),kind=col.kind,complexity=0),
                FeatureSpec(f"{c}__missing","is_missing",(c,),kind="binary")]
        if col.kind=="numeric":
            numeric.append(c)
            for op in ("log1p_abs","sqrt_abs","square","cube","reciprocal","abs"):
                out.append(FeatureSpec(f"{c}__{op}",op,(c,)))
            vals=df[c].dropna(); qs=[]
            for q in quantiles:
                if len(vals):
                    v=float(vals.quantile(q)); qs.append(v)
                    out += [FeatureSpec(f"{c}__gt_q{q:g}","threshold_gt",(c,),(("value",v),),kind="binary"),
                            FeatureSpec(f"{c}__lt_q{q:g}","threshold_lt",(c,),(("value",v),),kind="binary")]
            for lo,hi in zip(qs[:-1],qs[1:]):
                if lo<hi: out.append(FeatureSpec(f"{c}__range_{lo:g}_{hi:g}","range",(c,),(("low",lo),("high",hi)),kind="binary"))
            pos=vals[vals>0]
            if len(pos):
                ks=sorted({float(pos.quantile(q)) for q in (.25,.5,.75) if float(pos.quantile(q))>0})
                for k in ks:
                    for n in hill_exponents:
                        if n>0: out.append(FeatureSpec(f"{c}__hill_k{k:g}_n{n:g}","hill",(c,),(("k",k),("n",float(n))),complexity=2))
            for lag in lags:
                out += [FeatureSpec(f"{c}__lag{lag}","lag",(c,),(("periods",lag),),lookback=lag),
                        FeatureSpec(f"{c}__diff{lag}","diff",(c,),(("periods",lag),),lookback=lag),
                        FeatureSpec(f"{c}__pct{lag}","pct_change",(c,),(("periods",lag),),lookback=lag)]
            for w in windows:
                for op in ("rolling_mean","rolling_std","rolling_min","rolling_max","rolling_median","rolling_range"):
                    out.append(FeatureSpec(f"{c}__{op}{w}",op,(c,),(("window",w),),lookback=w))
                out.append(FeatureSpec(f"{c}__ewm{w}","ewm_mean",(c,),(("span",w),),lookback=1))
        elif col.kind=="binary": binary.append(c)
        elif col.kind=="categorical":
            for v in df[c].dropna().value_counts().head(20).index:
                out.append(FeatureSpec(f"{c}__eq_{v}","category_eq",(c,),(("value",v),),kind="binary"))
        elif col.kind=="datetime":
            for op,kind in (("weekday","numeric"),("month","numeric"),("dayofyear","numeric"),("is_weekend","binary")):
                out.append(FeatureSpec(f"{c}__{op}",op,(c,),kind=kind))
    if interactions:
        for a,b in combinations(numeric[:20],2):
            for op in ("add","subtract","abs_diff","multiply","ratio","relative_diff"):
                out.append(FeatureSpec(f"{a}__{op}__{b}",op,(a,b),complexity=2))
        for a,b in combinations(binary[:30],2):
            out.append(FeatureSpec(f"{a}__and__{b}","and",(a,b),kind="binary",complexity=2))
    return list({s.key:s for s in out}.values())
