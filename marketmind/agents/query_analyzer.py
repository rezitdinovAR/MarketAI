"""QueryAnalyzer — parses user query into structured QuerySpec."""

from __future__ import annotations

import logging
from pathlib import Path

from marketmind.llm_client import LLMClient
from marketmind.models import AgentState, QuerySpec, WorkflowStage

logger = logging.getLogger("marketmind")


FALLBACK_SYSTEM_PROMPT = (
    "Ты — AI-ассистент для анализа запросов покупателей. "
    "Извлеки из запроса: category, budget_min, budget_max, must_have, nice_to_have, "
    "needs_clarification, clarification_questions. Отвечай ТОЛЬКО валидным JSON."
)


def _load_prompt(prompts_dir: Path) -> str:
    path = prompts_dir / "query_analysis.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return FALLBACK_SYSTEM_PROMPT


def _sanitize_input(query: str) -> str:
    """Basic input sanitization."""
    if len(query) > 1000:
        query = query[:1000]
    return query.strip()


def run_query_analyzer(state: dict, llm: LLMClient, prompts_dir: Path) -> dict:
    """LangGraph node: parse user query into QuerySpec."""
    user_query = state.get("user_query", "")
    user_query = _sanitize_input(user_query)

    if len(user_query) < 2:
        return {
            "query_spec": QuerySpec(
                raw_query=user_query,
                needs_clarification=True,
                clarification_questions=["Пожалуйста, опишите подробнее, какой товар вы ищете."],
            ),
            "stage": WorkflowStage.QUERY_PARSED,
        }

    system_prompt = _load_prompt(prompts_dir)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    try:
        result = llm.call_json(messages, temperature=0.2, max_tokens=1024)

        query_spec = QuerySpec(
            raw_query=user_query,
            category=result.get("category"),
            budget_min=result.get("budget_min"),
            budget_max=result.get("budget_max"),
            must_have=result.get("must_have", []),
            nice_to_have=result.get("nice_to_have", []),
            needs_clarification=result.get("needs_clarification", False),
            clarification_questions=result.get("clarification_questions", []),
        )

        logger.info(
            f"Query parsed: category={query_spec.category}, budget_max={query_spec.budget_max}",
            extra={"stage": "query_analyzer"},
        )

        stats = llm.get_usage_stats()
        return {
            "query_spec": query_spec,
            "stage": WorkflowStage.QUERY_PARSED,
            "llm_calls": stats["calls"],
            "total_tokens": stats["total_tokens"],
            "total_cost": stats["total_cost"],
        }

    except Exception as e:
        logger.error(f"Query analysis failed: {e}", extra={"stage": "query_analyzer"})
        return {
            "query_spec": QuerySpec(
                raw_query=user_query,
                needs_clarification=True,
                clarification_questions=["Не удалось обработать запрос. Попробуйте переформулировать."],
            ),
            "stage": WorkflowStage.QUERY_PARSED,
            "errors": [f"QueryAnalyzer error: {e}"],
        }
