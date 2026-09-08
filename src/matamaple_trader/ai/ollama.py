from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any
from urllib import error, request

from .reviewer import AIReview, AIReviewDecision, AIReviewInput


class OllamaError(RuntimeError):
    pass


class OllamaAnalyst:
    """Minimal local Ollama client for post-quant signal review.

    The model receives compact structured market context and must return JSON.
    It has no broker credentials and no execution methods.
    """

    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 20.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are MATAMAPLE TRADER's local risk-aware signal reviewer. "
            "You may only CONFIRM, CAUTION, or REJECT the supplied quant signal. "
            "Never create a new BUY/SELL direction, never change lot size, stop loss, "
            "take profit, or risk limits. Prefer caution when data is conflicting. "
            "Return strict JSON with keys decision, confidence, rationale. "
            "decision must be CONFIRM, CAUTION, or REJECT; confidence must be 0..1."
        )

    def _payload(self, inputs: AIReviewInput) -> dict[str, Any]:
        return {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(asdict(inputs), ensure_ascii=False, sort_keys=True),
                },
            ],
            "options": {"temperature": 0.1},
        }

    def review(self, inputs: AIReviewInput) -> AIReview:
        body = json.dumps(self._payload(inputs)).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc

        try:
            content = payload["message"]["content"]
            parsed = json.loads(content)
            decision = AIReviewDecision(str(parsed["decision"]).upper())
            confidence = float(parsed["confidence"])
            rationale = str(parsed["rationale"]).strip()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OllamaError("Ollama returned an invalid review payload") from exc

        if decision is AIReviewDecision.UNAVAILABLE:
            raise OllamaError("Model cannot return UNAVAILABLE")
        if not 0.0 <= confidence <= 1.0:
            raise OllamaError("confidence must be between 0 and 1")
        if not rationale:
            raise OllamaError("rationale must not be empty")
        return AIReview(decision, confidence, rationale, self.model)
