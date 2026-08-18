import json

import litellm
import pytest
from pydantic import BaseModel

from cadtopo.llm import structured_completion
from cadtopo.telemetry import CALLS, COST
from tests.conftest import make_completion_response


class Simple(BaseModel):
    value: str


def test_successful_call_parses_model_and_records_in_calls_log(scripted_llm):
    scripted_llm([make_completion_response(json.dumps({"value": "hello world"}))])

    result = structured_completion(
        response_model=Simple, messages=[{"role": "user", "content": "hi"}],
        model="test-model", label="Worker",
    )

    assert isinstance(result, Simple)
    assert result.value == "hello world"
    assert len(CALLS.calls) == 1
    assert CALLS.calls[0].model == "test-model"
    assert CALLS.calls[0].label == "Worker"
    assert COST.total.calls == 1


def test_retries_on_rate_limit_then_succeeds(scripted_llm, monkeypatch):
    # Skip real sleeping — tenacity's wait_exponential would otherwise stall the test.
    monkeypatch.setattr("cadtopo.llm.structured_completion.retry.wait", lambda *a, **k: 0)

    rate_limit_error = litellm.RateLimitError("rate limited", llm_provider="test", model="test-model")
    ok = make_completion_response(json.dumps({"value": "recovered"}))
    scripted = scripted_llm([rate_limit_error, ok])

    result = structured_completion(response_model=Simple, messages=[], model="test-model")

    assert result.value == "recovered"
    assert len(scripted.calls) == 2
    # Only the successful call lands in the log — a retried failure never does.
    assert len(CALLS.calls) == 1


def test_retries_on_server_side_json_generation_failure(scripted_llm, monkeypatch):
    # Providers with server-side JSON enforcement (e.g. Groq) surface an occasional
    # sampling failure as a bare APIError, not a transport error or ValidationError.
    # A fresh sampling recovers, so it must be treated as transient rather than
    # dying after one attempt.
    monkeypatch.setattr("cadtopo.llm.structured_completion.retry.wait", lambda *a, **k: 0)

    json_fail = litellm.APIError(
        500,
        "OpenrouterException - Upstream error from Groq: Failed to generate JSON. Please adjust your prompt.",
        llm_provider="openrouter",
        model="test-model",
    )
    ok = make_completion_response(json.dumps({"value": "recovered"}))
    scripted = scripted_llm([json_fail, ok])

    result = structured_completion(response_model=Simple, messages=[], model="test-model")

    assert result.value == "recovered"
    assert len(scripted.calls) == 2


def test_non_retryable_error_propagates(scripted_llm):
    scripted_llm([ValueError("not a retryable transport error")])
    with pytest.raises(Exception):
        structured_completion(response_model=Simple, messages=[], model="test-model")
    assert len(CALLS.calls) == 0


def test_drop_params_is_forwarded_to_litellm(scripted_llm):
    scripted = scripted_llm([make_completion_response(json.dumps({"value": "x"}))])
    structured_completion(
        response_model=Simple, messages=[], model="test-model", temperature=0.4,
    )
    # Instructor forwards our kwargs to litellm.completion; drop_params lets a
    # provider silently ignore a knob it doesn't understand.
    assert scripted.calls[0]["drop_params"] is True
    assert scripted.calls[0]["temperature"] == 0.4
