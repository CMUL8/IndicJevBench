"""Adapter package.

Only the light abstract base is imported eagerly. Concrete adapters pull
heavy optional dependencies (torch, transformers, openai, semif) lazily in
their constructors — do NOT import them eagerly here.
"""

from indicjevbench.adapters.base import BenchAdapter, DecisionResult

__all__ = ["BenchAdapter", "DecisionResult"]
