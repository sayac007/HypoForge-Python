# Proposed v0.4 update — additive, experimental

Adds `hypoforge.calendar` and `hypoforge.media`, corresponding unit tests and user documentation. Existing `discover`, `evaluate`, `select_candidates` and their semantics remain unchanged. **Not** a completed temporal tournament or R parity release.

The new calendar features use independent event dates and deterministic time. The recurring residual scan is exploratory and must use training residuals. The new media functions are standalone; no automatic hyperparameter search is wired into `discover()`.

Before release: apply overlay to current repository, run `pip install -e '.[dev]'`, `python -m pytest -q`, `ruff check .`, `python -m build`, inspect wheel contents, and run at least one held-out temporal integration benchmark. Existing project lint and build artifacts may need cleanup.
