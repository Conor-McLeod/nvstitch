"""Questions: one module each.

A question's parameters are literals at the top of its file, and its
`answer(con, ...)` function takes them as keyword defaults and returns a list of
Tables. `nvstitch ask` passes only `run`; for anything else, call answer() from
Python with the arguments you want.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType


def available() -> dict[str, ModuleType]:
    """Every question in this package, by name."""
    return {info.name: importlib.import_module(f"{__name__}.{info.name}")
            for info in pkgutil.iter_modules(__path__)
            if not info.name.startswith("_")}


def summary(question: ModuleType) -> str:
    """The first line of a question's docstring."""
    return (question.__doc__ or "").strip().splitlines()[0]
