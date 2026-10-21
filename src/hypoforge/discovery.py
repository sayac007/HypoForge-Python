import pandas as pd
from .generate import infer_columns, generate_candidates
from .score import score_candidate

def discover(df,target,*,time=None,task="auto",top_k=50,**kw):
    work=df.copy()
    if time is not None: work=work.sort_values(time).reset_index(drop=True)
    rows=[score_candidate(s,work,target,task) for s in generate_candidates(work,infer_columns(work,{target}),**kw)]
    rows=[r for r in rows if r.get("valid")]
    if not rows: return pd.DataFrame()
    return (pd.DataFrame(rows)
            .sort_values(["preference_score","information_score","support"],ascending=False)
            .head(top_k).reset_index(drop=True))
