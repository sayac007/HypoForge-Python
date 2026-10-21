# Prototype challenge before public release

## Addressed in v0.3

- numerical-safety diagnostics and conservative transformation preference;
- redundancy checks against baseline and already-selected candidates;
- standardized condition-number guard;
- complete source lineage for interactions;
- auditable candidate admission.

## Still required

1. Temporal availability contracts: observed, known-future, forecast and static.
2. Rolling-origin validation and stability scoring.
3. Grouped/panel-safe lags and rolling transforms.
4. Train-fold fitting of data-derived transform parameters (thresholds and Hill K).
5. Value-equivalence clustering beyond pairwise correlation.
6. Model-adapter tournament based on incremental out-of-sample value.
7. Transformation registry / extension API.
8. Batch/parallel execution for large candidate spaces.
9. Rigorous scoring semantics for binary, multiclass and continuous targets.
