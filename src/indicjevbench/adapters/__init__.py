"""Adapter package.

Only the light abstract base (:class:`BenchAdapter`) and the
:class:`~indicjevbench.schemas.contracts.DecisionResult` contract are
imported eagerly. Concrete adapters (``http``, ``api_llm``,
``qwen3_logprob``, ``semif``, ``laya``, ``local``) pull heavy optional
dependencies — torch, transformers, openai, semif_phase1, nirnaya —
lazily inside their constructors, so importing this package must stay
cheap and must never fail due to a missing optional extra. Do NOT add
eager imports of concrete adapters here.
"""

from indicjevbench.adapters.base import BenchAdapter, DecisionResult

__all__ = ["BenchAdapter", "DecisionResult"]
