# AGENTS.md

## Purpose

This repository contains a Python implementation of a modified Schelling segregation model with a Streamlit UI.
Agents working in this codebase should preserve a simple, readable Python style and keep changes aligned with the
current project structure.

## Working Principles

- Be explicit about uncertainty and avoid inventing APIs, files, or behavior.
- Keep responses concise, technical, and focused on the task.
- Favor deterministic, reviewable changes over broad refactors.
- Stay within the current repository scope and do not assume external services or infrastructure.
- Don't waste tokens on wording. Be short and precise in your answers
- Don't assume, always ask the operator if you don't know what to do

## Python Style

- Target Python `3.11`.
- Follow PEP 8 for formatting and naming.
- Add type hints to new or modified functions where they improve clarity.
- Prefer explicit, readable code over compact or clever code.
- Keep functions focused and small when practical.
- Use 4 spaces for indentation.
- Keep line length reasonable; wrap before lines become difficult to scan.

## Naming

- Use `snake_case` for variables, functions, and module-level helpers.
- Use `PascalCase` for classes.
- Use `UPPER_CASE` for constants.
- Prefer descriptive names such as `similarity_threshold` over short names such as `thr`.

## Imports

- Group imports in this order: standard library, third-party, local modules.
- Keep imports explicit and remove unused imports when touching a file.
- Keep one import per line unless importing closely related names from the same module.

Example:

```python
import logging
import random

import numpy as np
import streamlit as st

from Agent import Agent
from City import City
```

## Documentation

- Add short docstrings to public classes and non-trivial functions.
- Use concise comments only where intent is not obvious from the code.
- Keep comments factual and maintainable.
- Follow PEP 257 for new docstrings.

## Project Conventions

- Preserve the current module layout unless a refactor is required by the task.
- Keep simulation and domain logic in model classes such as `Agent` and `City`.
- Keep UI concerns in `main.py`.
- When changing simulation behavior, keep the Streamlit entry flow in `main.py` working.
- When adding parameters, thread them consistently through argument parsing and Streamlit controls.
- Prefer deterministic behavior where useful for simulation reproducibility.
- Avoid introducing heavyweight dependencies unless necessary.

## Testing and Validation

- Run the narrowest relevant validation available after changes.
- For logic changes, prefer small deterministic checks over manual inspection alone.
- When asked for tests, use `pytest` style unless the repository already uses something else.
- If no automated tests exist, document what was validated.

## Editing Rules

- Do not make unrelated refactors.
- Do not rename files or public interfaces without a clear need.
- Preserve user changes already present in the worktree.
- Favor minimal, reviewable diffs.
