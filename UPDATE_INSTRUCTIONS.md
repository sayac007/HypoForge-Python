# Applying the Python update to the existing HypoForge repository

**This ZIP is an OVERLAY, not a standalone replacement repository.**

1. Extract the ZIP to a local folder first (do **not** drag files straight from a ZIP into GitHub; GitHub may flatten paths).
2. Copy `README.md`, `pyproject.toml`, `docs/`, `src/hypoforge/calendar.py`, `src/hypoforge/media.py` and `tests/test_calendar_media.py` into the existing Python repository, preserving relative paths. Keep all existing `src/hypoforge/*.py`, tests, LICENSE, CHALLENGE.md and CONTRIBUTING.md.
3. Review the diff. Run `python -m pip install -e '.[dev]'`, `python -m pytest -q`, `ruff check .`, `python -m build`. Fix existing repository lint/build issues separately if needed.
4. Do not call the v0.4 release fully validated until the complete original repository test suite and package build have run, and a temporal integration test is available.
5. Commit on a feature branch, then merge after CI passes.

New API is available as `from hypoforge.calendar import ...` and `from hypoforge.media import ...`. The old top-level imports remain unchanged. The `discover()` candidate generator does not yet automatically include these transforms; this is deliberate pending fold-fit/availability-safe tournament integration.
