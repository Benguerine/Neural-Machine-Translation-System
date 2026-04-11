# ModuleNotFoundError — src/ layout not configured

**Date:** 2026-04-07  
**Project:** Neural Machine Translation  
**Environment:** Windows 11, Python 3.12.9  

---

## Error message
```
tests\test_languages.py:5: in <module>
    from ai_translator.languages import (
E   ModuleNotFoundError: No module named 'ai_translator'
```

## Context
Running pytest after confirming pytest was installed. The project uses a `src/`
layout — all source code lives inside `src/ai_translator/` instead of the root.

## Why it happened
Python searches the root folder for packages by default. With a `src/` layout,
the package is one level deeper and Python can't find it without explicit
configuration. Additionally, the package was never registered in the environment.

## Neural Machine Translation — AI Translator structure
```
Neural Machine Translation — AI Translator/
├── src/
│   └── ai_translator/      ← Python couldn't find this
│       ├── __init__.py
│       └── languages.py
├── tests/
│   └── test_languages.py
└── pyproject.toml
```

## What fixed it

**Step 1 — Created `pyproject.toml` in the root folder:**
```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "ai_translator"
version = "0.1.0"

[tool.setuptools.packages.find]
where = ["src"]
```

**Step 2 — Installed the package in editable mode:**
```bash
pip install -e .
```

## Lesson learned
The `src/` layout requires two things to work:
- `pyproject.toml` with `where = ["src"]` so setuptools knows where to look
- `pip install -e .` to register the package in the active environment

The `-e` flag means editable — Python points directly to your `src/` folder,
so code changes are reflected immediately without reinstalling.