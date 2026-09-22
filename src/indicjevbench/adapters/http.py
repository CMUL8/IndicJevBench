"""HTTP client adapter for a /v1/systemone-compatible decision server.

This module holds the client-side half of the "systemone" endpoint
contract used by trained Nirṇaya-style checkpoints served over HTTP.
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Self, cast

from indicjevbench.adapters.base import BenchAdapter
from indicjevbench.schemas.contracts import DecisionResult, Task

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_RETRY_DELAY_S = 2.0


class HTTPAdapter(BenchAdapter):
    """Client for a /v1/systemone-compatible HTTP decision server.

    Endpoint contract
    -----------------
    ``POST {base_url}/v1/systemone`` with a JSON body::

        {
          "model": str,
          "state": str,                      # raw customer/user message
          "questions": [
            {
              "id": str,                     # single question sent as "q0"
              "type": "choice" | "score" | "noul",
              "instructions": str,
              "options": [str, ...]          # omitted for noul
            }
          ]
        }

    Expected 200 response body::

        {
          "answers": [
            {
              "probabilities": [float, ...], # full distribution over options
              "answer": int | bool,          # argmax index, or bool for noul
              "confidence": float,           # optional, defaults to 0.0
              "expected": float              # optional; score-type mean level
            }
          ],
          "usage": {"input_tokens": int}     # optional
        }

    Transient failures (HTTP 5xx and connection-level errors) are retried
    up to two times with a fixed :data:`_RETRY_DELAY_S` second delay; the
    third failure propagates as an ``httpx.HTTPStatusError`` /
    ``httpx.TransportError``. Non-transient 4xx statuses raise on the first
    response via ``response.raise_for_status()``.

    Args:
        base_url: Base URL of the decision server (trailing slash optional).
        model: Model name sent in the ``model`` payload field.
        timeout: Per-request timeout in seconds.
        transport: Optional ``httpx`` transport (e.g. ``httpx.MockTransport``
            in tests). When ``None`` a default ``httpx.Client`` is created.

    Raises:
        ValueError: If ``base_url`` is empty or ``timeout`` is not positive.
        ImportError: If ``httpx`` is not installed. It is a core dependency;
            reinstall with ``pip install -e .`` if missing.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "my-model",
        timeout: float = 60.0,
        transport: "httpx.MockTransport | None" = None,
    ) -> None:
        if not base_url or not base_url.strip():
            raise ValueError("base_url must be a non-empty str")
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout!r}")
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "HTTPAdapter requires httpx. Reinstall the package: pip install -e ."
            ) from exc
        self._client: httpx.Client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )
        self._model = model
        logger.info("HTTPAdapter created for %s (model=%s)", base_url, model)

    def decide(self, task: Task) -> DecisionResult:
        """Request a decision for a single task from the server.

        Args:
            task: The benchmark task (state + typed question).

        Returns:
            The parsed decision result, including server-reported confidence,
            expected level, and input-token usage when present.

        Raises:
            httpx.HTTPStatusError: On a non-retryable HTTP error status.
            httpx.TransportError: If all retries against the server fail.
            ValueError: If the response body does not contain the answers.
        """
        q = task.question
        payload: dict[str, Any] = {
            "model": self._model,
            "state": task.state,
            "questions": [
                {
                    "id": "q0",
                    "type": q.type,
                    "instructions": q.instructions,
                    **({"options": list(q.options)} if q.options else {}),
                }
            ],
        }
        body, latency_ms = self._post_with_retries(payload)
        answers_obj: Any = body.get("answers")
        if not isinstance(answers_obj, list) or not answers_obj:
            raise ValueError(f"/v1/systemone response missing 'answers': {body!r}")
        answers: list[Any] = cast("list[Any]", answers_obj)
        ans: dict[str, Any] = cast("dict[str, Any]", answers[0])
        return DecisionResult(
            task_id=task.id,
            probabilities=ans["probabilities"],
            answer=ans["answer"],
            confidence=ans.get("confidence", 0.0),
            latency_ms=latency_ms,
            expected=ans.get("expected"),
            tokens_used=body.get("usage", {}).get("input_tokens"),
        )

    def _post_with_retries(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """POST the payload, retrying transient failures.

        Args:
            payload: The /v1/systemone request body.

        Returns:
            A pair of the decoded JSON response body and the request latency
            in milliseconds.

        Raises:
            httpx.HTTPStatusError: On the final attempt's HTTP error status.
            httpx.TransportError: If the final attempt fails at transport level.
        """
        t0 = time.perf_counter()
        resp: httpx.Response | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = self._client.post("/v1/systemone", json=payload)
                if resp.status_code >= 500 and attempt < _MAX_ATTEMPTS - 1:
                    logger.warning(
                        "systemone returned %d (attempt %d/%d); retrying in %.1fs",
                        resp.status_code,
                        attempt + 1,
                        _MAX_ATTEMPTS,
                        _RETRY_DELAY_S,
                    )
                    time.sleep(_RETRY_DELAY_S)
                    continue
                resp.raise_for_status()
                break
            except Exception:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
                logger.warning(
                    "systemone request failed (attempt %d/%d); retrying in %.1fs",
                    attempt + 1,
                    _MAX_ATTEMPTS,
                    _RETRY_DELAY_S,
                )
                time.sleep(_RETRY_DELAY_S)
        latency_ms = (time.perf_counter() - t0) * 1000
        if resp is None:  # unreachable: loop always assigns or raises
            raise RuntimeError("systemone request failed without a response")
        return resp.json(), latency_ms

    def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
