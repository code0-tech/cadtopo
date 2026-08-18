"""Shared pytest fixtures: fake LiteLLM responses and a deterministic embedder.

Nothing in this test suite makes a real network call. Every LLM call goes
through a monkeypatched ``litellm.completion``; the embedding backend is
replaced with a small deterministic stand-in instead of loading a real
sentence-transformers model.
"""

from types import SimpleNamespace
from typing import Callable, List

import numpy as np
import pytest
from pydantic import BaseModel as BaseModelType


def make_usage(prompt_tokens: int = 10, completion_tokens: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def make_completion_response(content: str, tool_calls=None, usage=None, finish_reason="stop") -> SimpleNamespace:
    """Build a minimal object shaped like a LiteLLM/OpenAI ``ChatCompletion``.

    ``finish_reason`` is required by Instructor's response parser, so it is
    always present.
    """
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage or make_usage())


class ScriptedLLM:
    """A queue of canned responses (or callables) fed to a patched ``litellm.completion``.

    Each entry is either a ``SimpleNamespace`` response (see
    :func:`make_completion_response`) or an exception instance/class to raise
    instead — lets tests exercise the retry path deterministically.
    """

    def __init__(self, responses: List):
        self._responses = list(responses)
        self.calls: List[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item("scripted failure")
        return item


@pytest.fixture
def scripted_llm(monkeypatch) -> Callable[[List], ScriptedLLM]:
    """Patch ``litellm.completion`` (as seen by ``cadtopo.llm``) with a :class:`ScriptedLLM`."""

    def _install(responses: List) -> ScriptedLLM:
        scripted = ScriptedLLM(responses)
        monkeypatch.setattr("cadtopo.llm.litellm.completion", scripted)
        return scripted

    return _install


class ScriptedStructured:
    """A queue of canned Pydantic results for a patched ``structured_completion``.

    Each entry is a pre-built Pydantic model instance (returned as-is), a
    ``SimpleNamespace(response=…)`` standing in for a wrapped tool turn, a
    callable ``response_model -> obj`` (built lazily against the requested
    model), or an exception (raised, to exercise fail-open paths). Every call's
    kwargs are captured in ``calls`` so tests can assert what was sent.
    """

    def __init__(self, responses: List):
        self._responses = list(responses)
        self.calls: List[dict] = []

    def __call__(self, *, response_model, messages, model, api_key=None, api_base=None, label=None, max_retries=2, **extra):
        self.calls.append({
            "response_model": response_model, "messages": list(messages), "model": model,
            "label": label, "max_retries": max_retries, **extra,
        })
        if not self._responses:
            raise AssertionError("ScriptedStructured ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item("scripted failure")
        if callable(item) and not isinstance(item, BaseModelType):
            return item(response_model)
        return item


@pytest.fixture
def scripted_structured(monkeypatch) -> Callable[[List], ScriptedStructured]:
    """Patch ``cadtopo.llm.structured_completion`` with a :class:`ScriptedStructured`.

    Both ``cadtopo.agent`` and ``cadtopo.manager`` call it via the ``llm``
    module reference, so this single patch covers every structured call.
    """

    def _install(responses: List) -> ScriptedStructured:
        scripted = ScriptedStructured(responses)
        monkeypatch.setattr("cadtopo.llm.structured_completion", scripted)
        return scripted

    return _install


class FakeEmbedder:
    """Deterministic stand-in for :class:`cadtopo.embedding.EmbeddingModel`.

    Hands out a fixed vector per exact input string (looked up in
    ``vectors``, case-sensitive) so router/orchestrator tests can control
    similarity outcomes precisely without loading a real model. Unknown text
    falls back to a stable hash-based vector so cosine similarity stays
    well-defined for inputs the test didn't bother naming explicitly.
    """

    def __init__(self, vectors: dict[str, np.ndarray] | None = None, dim: int = 8):
        self.vectors = dict(vectors or {})
        self.dim = dim

    def _vector_for(self, text: str) -> np.ndarray:
        if text in self.vectors:
            return self.vectors[text]
        if not text.strip():
            return np.zeros(self.dim)
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.normal(size=self.dim)

    def get_embedding(self, text: str) -> np.ndarray:
        return self._vector_for(text)

    def get_query_embedding(self, text: str) -> np.ndarray:
        return self._vector_for(text)

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return float(np.dot(v1, v2)) / (n1 * n2)


@pytest.fixture(autouse=True)
def _reset_telemetry():
    """Every test starts with clean global COST/CALLS singletons."""
    from cadtopo.telemetry import CALLS, COST

    COST.by_agent.clear()
    COST.total = type(COST.total)()
    CALLS.calls.clear()
    yield
