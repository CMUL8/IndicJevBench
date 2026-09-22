from __future__ import annotations
import time
from .base import BenchAdapter, DecisionResult

class HTTPAdapter(BenchAdapter):
    def __init__(self, base_url: str, model: str = "my-model", timeout: float = 60.0):
        try:
            import httpx
        except ImportError:
            raise ImportError("pip install httpx")
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self._model = model

    def decide(self, task) -> DecisionResult:
        import time
        q = task.question
        payload = {
            "model": self._model,
            "state": task.state,
            "questions": [{"id": "q0", "type": q["type"], "instructions": q["instructions"],
                           **({"options": q["options"]} if q.get("options") else {})}]
        }
        t0 = time.perf_counter()
        for attempt in range(3):
            try:
                resp = self._client.post("/v1/systemone", json=payload)
                if resp.status_code >= 500 and attempt < 2:
                    time.sleep(2.0); continue
                resp.raise_for_status()
                break
            except Exception:
                if attempt == 2: raise
                time.sleep(2.0)
        latency_ms = (time.perf_counter() - t0) * 1000
        ans = resp.json()["answers"][0]
        probs = ans["probabilities"]
        answer = ans["answer"]
        conf = ans.get("confidence", 0.0)
        expected = ans.get("expected")
        tokens = resp.json().get("usage", {}).get("input_tokens")
        return DecisionResult(task_id=task.id, probabilities=probs, answer=answer,
                              confidence=conf, latency_ms=latency_ms, expected=expected,
                              tokens_used=tokens)

    def close(self):
        self._client.close()
