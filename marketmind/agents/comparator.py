"""Comparator — compares products using LLM + rule-based scoring."""

from __future__ import annotations

import logging
from pathlib import Path

from marketmind.llm_client import LLMClient
from marketmind.models import (
    AgentState,
    ProductAnalysis,
    QuerySpec,
    WorkflowStage,
)

logger = logging.getLogger("marketmind")


FALLBACK_SYSTEM_PROMPT = (
    "Сравни товары и оцени value_score и fit_score (0-1). "
    "Верни JSON с массивом products и comparison_summary."
)


def _load_prompt(prompts_dir: Path) -> str:
    path = prompts_dir / "comparison.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return FALLBACK_SYSTEM_PROMPT


def _build_comparison_input(analyzed: list[ProductAnalysis], query_spec: QuerySpec) -> str:
    """Build comparison prompt input."""
    lines = [
        f"Запрос пользователя: {query_spec.raw_query}",
        f"Категория: {query_spec.category or 'не определена'}",
        f"Бюджет: {query_spec.budget_min or '?'} — {query_spec.budget_max or '?'} руб.",
        f"Обязательные характеристики: {', '.join(query_spec.must_have) or 'не указаны'}",
        f"Желательные характеристики: {', '.join(query_spec.nice_to_have) or 'не указаны'}",
        "",
        "--- Товары для сравнения ---",
    ]

    for a in analyzed:
        p = a.product
        rs = a.review_summary
        lines.append(f"\nID: {p.id}")
        lines.append(f"Название: {p.name}")
        lines.append(f"Цена: {p.price} руб." + (f" (было {p.original_price} руб.)" if p.original_price else ""))
        lines.append(f"Маркетплейс: {p.marketplace}")
        lines.append(f"Рейтинг: {p.rating}/5 ({p.review_count} отзывов)")
        lines.append(f"Продавец: {p.seller_name or 'неизвестен'}")
        if rs.pros:
            lines.append(f"Плюсы: {'; '.join(rs.pros)}")
        if rs.cons:
            lines.append(f"Минусы: {'; '.join(rs.cons)}")
        lines.append(f"Резюме: {rs.summary}")
        lines.append(f"Доверие к отзывам: {rs.trust_score}")

    return "\n".join(lines)


def run_comparator(state: dict, llm: LLMClient, prompts_dir: Path) -> dict:
    """LangGraph node: compare analyzed products."""
    analyzed: list[ProductAnalysis] = state.get("analyzed_products", [])
    query_spec: QuerySpec = state.get("query_spec")

    if not analyzed or not query_spec:
        return {
            "analyzed_products": analyzed,
            "stage": WorkflowStage.COMPARED,
            "errors": ["No products to compare"],
        }

    system_prompt = _load_prompt(prompts_dir)
    user_input = _build_comparison_input(analyzed, query_spec)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    try:
        result = llm.call_json(messages, temperature=0.3, max_tokens=2048)
        llm_products = {p["product_id"]: p for p in result.get("products", [])}

        # Update analyzed products with LLM scores
        for a in analyzed:
            if a.product.id in llm_products:
                scores = llm_products[a.product.id]
                a.value_score = min(1.0, max(0.0, scores.get("value_score", 0.5)))
                a.fit_score = min(1.0, max(0.0, scores.get("fit_score", 0.5)))

        # Rule-based adjustments
        for a in analyzed:
            # Penalize products over budget
            if query_spec.budget_max and a.product.price > query_spec.budget_max:
                a.fit_score = min(a.fit_score, 0.3)
            # Boost products with discounts
            if a.product.original_price and a.product.price < a.product.original_price:
                discount = 1 - (a.product.price / a.product.original_price)
                a.value_score = min(1.0, a.value_score + discount * 0.1)

        logger.info(
            f"Compared {len(analyzed)} products",
            extra={"stage": "comparator"},
        )

    except Exception as e:
        logger.warning(f"LLM comparison failed, using heuristic: {e}")
        # Fallback: simple rule-based scoring
        max_price = max(a.product.price for a in analyzed) if analyzed else 1
        for a in analyzed:
            a.value_score = round(1 - (a.product.price / max_price) * 0.5 + a.product.rating / 10, 2)
            a.value_score = min(1.0, max(0.0, a.value_score))
            a.fit_score = round(a.product.rating / 5, 2)
            if query_spec.budget_max and a.product.price > query_spec.budget_max:
                a.fit_score = 0.2

    stats = llm.get_usage_stats()
    return {
        "analyzed_products": analyzed,
        "stage": WorkflowStage.COMPARED,
        "llm_calls": stats["calls"],
        "total_tokens": stats["total_tokens"],
        "total_cost": stats["total_cost"],
    }
