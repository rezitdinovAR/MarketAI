"""LLM Client — wrapper over DeepSeek API with retries, cost tracking, and guardrails."""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from openai import OpenAI

from marketmind.config import LLMConfig
from marketmind.models import LLMResponse

logger = logging.getLogger("marketmind")

# DeepSeek pricing per 1K tokens
PRICING = {
    "deepseek-chat": {"input": 0.0001, "output": 0.0002},
    "deepseek-reasoner": {"input": 0.0005, "output": 0.001},
}


class LLMBudgetExceededError(Exception):
    pass


class LLMClient:
    """Wrapper over DeepSeek API (OpenAI-compatible) with guardrails."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        # Session counters
        self._calls = 0
        self._total_tokens = 0
        self._total_cost = 0.0

    def _check_guardrails(self) -> None:
        if self._calls >= self.config.max_calls_per_request:
            raise LLMBudgetExceededError(
                f"Max LLM calls exceeded: {self._calls}/{self.config.max_calls_per_request}"
            )
        if self._total_tokens >= self.config.max_tokens_per_request:
            raise LLMBudgetExceededError(
                f"Max tokens exceeded: {self._total_tokens}/{self.config.max_tokens_per_request}"
            )
        if self._total_cost >= self.config.max_cost_per_request:
            raise LLMBudgetExceededError(
                f"Max cost exceeded: ${self._total_cost:.4f}/${self.config.max_cost_per_request}"
            )

    def _calc_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = PRICING.get(self.config.model, PRICING["deepseek-chat"])
        return (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]

    def call(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        """Make an LLM call with retries and guardrails."""
        self._check_guardrails()

        kwargs: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": self.config.timeout,
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                start = time.time()
                response = self.client.chat.completions.create(**kwargs)
                latency_ms = int((time.time() - start) * 1000)

                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                cost = self._calc_cost(input_tokens, output_tokens)

                # Update counters
                self._calls += 1
                self._total_tokens += input_tokens + output_tokens
                self._total_cost += cost

                content = response.choices[0].message.content or ""

                logger.info(
                    "LLM call completed",
                    extra={
                        "data": {
                            "model": self.config.model,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "latency_ms": latency_ms,
                            "cost_usd": round(cost, 6),
                            "attempt": attempt + 1,
                        }
                    },
                )

                return LLMResponse(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    cost_usd=cost,
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}): {e}",
                    extra={"data": {"attempt": attempt + 1, "error": str(e)}},
                )
                if attempt < self.config.max_retries:
                    time.sleep(min(2 ** attempt, 4))

        raise last_error  # type: ignore[misc]

    def call_json(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> dict:
        """Make an LLM call expecting JSON response, with retry on parse failure."""
        for attempt in range(2):
            resp = self.call(messages, temperature=temperature, max_tokens=max_tokens, response_format="json")
            try:
                return json.loads(resp.content)
            except json.JSONDecodeError:
                if attempt == 0:
                    messages = messages + [
                        {"role": "assistant", "content": resp.content},
                        {
                            "role": "user",
                            "content": "Предыдущий ответ был невалидным JSON. Ответь СТРОГО в формате JSON, без текста до/после.",
                        },
                    ]
                else:
                    raise

        return {}  # unreachable

    def get_usage_stats(self) -> dict:
        return {
            "calls": self._calls,
            "total_tokens": self._total_tokens,
            "total_cost": round(self._total_cost, 6),
        }

    def reset_session(self) -> None:
        self._calls = 0
        self._total_tokens = 0
        self._total_cost = 0.0
