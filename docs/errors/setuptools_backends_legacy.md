# BackendUnavailable — setuptools.backends.legacy

**Date:** 2026-04-12   
**Project:** Neural Machine Translation  
**Environment:** Windows 11, Python 3.12.9  

---

## Error message
```
pip._vendor.pyproject_hooks._impl.BackendUnavailable:
Cannot import 'setuptools.backends.legacy'
```

## Context
Running `pip install -e .` after creating `pyproject.toml` with
`build-backend = "setuptools.backends.legacy:build"`.

## Why it happened
The `setuptools.backends.legacy` module only exists in newer versions of
setuptools. The installed version was too old to support it.

## What fixed it

**Step 1 — Updated `pyproject.toml` to use the stable backend:**
```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"
```

**Step 2 — Upgraded setuptools and reinstalled:**
```bash
pip install --upgrade setuptools
pip install -e .
```

## Lesson learned
Always use `setuptools.build_meta` as the build backend — it's the stable,
well-supported option that works across all setuptools versions >= 64.
Never use `setuptools.backends.legacy` — it's version-specific and unreliable.