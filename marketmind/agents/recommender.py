"""Recommender — generates final top-3 recommendation with explanation."""

from __future__ import annotations

import logging
from pathlib import Path

from marketmind.llm_client import LLMClient
from marketmind.models import (
    ProductAnalysis,
    QuerySpec,
    RankedProduct,
    Recommendation,
    WorkflowStage,
)

logger = logging.getLogger("marketmind")


FALLBACK_SYSTEM_PROMPT = (
    "Сформируй рекомендацию топ-3 товаров. "
    "Верни JSON с top3, explanation и confidence."
)


def _load_prompt(prompts_dir: Path) -> str:
    path = prompts_dir / "recommendation.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return FALLBACK_SYSTEM_PROMPT


def _build_recommendation_input(analyzed: list[ProductAnalysis], query_spec: QuerySpec) -> str:
    """Build input for recommendation prompt."""
    lines = [
        f"Запрос пользователя: {query_spec.raw_query}",
        f"Категория: {query_spec.category}",
        f"Бюджет: до {query_spec.budget_max} руб." if query_spec.budget_max else "Бюджет: не указан",
        f"Обязательные: {', '.join(query_spec.must_have) or 'не указаны'}",
        "",
        "--- Товары (отсортированы по оценке) ---",
    ]

    # Sort by combined score
    sorted_products = sorted(
        analyzed,
        key=lambda a: a.fit_score * 0.6 + a.value_score * 0.4,
        reverse=True,
    )

    for a in sorted_products:
        p = a.product
        rs = a.review_summary
        lines.append(f"\nID: {p.id}")
        lines.append(f"Название: {p.name}")
        lines.append(f"Цена: {p.price} руб.")
        lines.append(f"Маркетплейс: {p.marketplace}")
        lines.append(f"Рейтинг: {p.rating}/5 ({p.review_count} отзывов)")
        lines.append(f"Оценка цена/качество: {a.value_score}")
        lines.append(f"Соответствие запросу: {a.fit_score}")
        if rs.pros:
            lines.append(f"Плюсы: {'; '.join(rs.pros)}")
        if rs.cons:
            lines.append(f"Минусы: {'; '.join(rs.cons)}")

    return "\n".join(lines)


def _validate_recommendation(result: dict, analyzed: list[ProductAnalysis]) -> bool:
    """Validate that recommended product IDs exist."""
    valid_ids = {a.product.id for a in analyzed}
    for item in result.get("top3", []):
        if item.get("product_id") not in valid_ids:
            return False
    return True


def run_recommender(state: dict, llm: LLMClient, prompts_dir: Path) -> dict:
    """LangGraph node: generate final recommendation."""
    analyzed: list[ProductAnalysis] = state.get("analyzed_products", [])
    query_spec: QuerySpec = state.get("query_spec")

    if not analyzed or not query_spec:
        return {
            "recommendation": Recommendation(
                explanation="Не удалось сформировать рекомендацию: нет данных для анализа.",
                user_query=query_spec.raw_query if query_spec else "",
                confidence=0.0,
            ),
            "stage": WorkflowStage.RECOMMENDED,
        }

    system_prompt = _load_prompt(prompts_dir)
    user_input = _build_recommendation_input(analyzed, query_spec)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    product_map = {a.product.id: a for a in analyzed}

    for attempt in range(2):
        try:
            result = llm.call_json(messages, temperature=0.4, max_tokens=2048)

            if not _validate_recommendation(result, analyzed):
                logger.warning("Recommendation validation failed, retrying")
                messages.append({"role": "assistant", "content": str(result)})
                messages.append({
                    "role": "user",
                    "content": (
                        "Некоторые product_id в ответе не совпадают с реальными. "
                        "Используй ТОЛЬКО id из списка товаров. Повтори."
                    ),
                })
                continue

            # Build RankedProduct list
            top3 = []
            for item in result.get("top3", [])[:3]:
                pid = item["product_id"]
                pa = product_map.get(pid)
                if pa:
                    top3.append(RankedProduct(
                        rank=item.get("rank", len(top3) + 1),
                        product=pa.product,
                        review_summary=pa.review_summary,
                        final_score=min(1.0, max(0.0, item.get("final_score", 0.5))),
                        fit_explanation=item.get("fit_explanation", ""),
                        main_advantage=item.get("main_advantage", ""),
                        main_caveat=item.get("main_caveat", ""),
                    ))

            recommendation = Recommendation(
                top3=top3,
                explanation=result.get("explanation", ""),
                confidence=min(1.0, max(0.0, result.get("confidence", 0.5))),
                user_query=query_spec.raw_query,
            )

            logger.info(
                f"Recommendation generated: {len(top3)} products, confidence={recommendation.confidence}",
                extra={"stage": "recommender"},
            )

            stats = llm.get_usage_stats()
            return {
                "recommendation": recommendation,
                "stage": WorkflowStage.RECOMMENDED,
                "llm_calls": stats["calls"],
                "total_tokens": stats["total_tokens"],
                "total_cost": stats["total_cost"],
            }

        except Exception as e:
            logger.error(f"Recommendation attempt {attempt + 1} failed: {e}")

    # Fallback: template-based recommendation
    sorted_products = sorted(
        analyzed,
        key=lambda a: a.fit_score * 0.6 + a.value_score * 0.4,
        reverse=True,
    )
    top3 = []
    for i, a in enumerate(sorted_products[:3]):
        top3.append(RankedProduct(
            rank=i + 1,
            product=a.product,
            review_summary=a.review_summary,
            final_score=round(a.fit_score * 0.6 + a.value_score * 0.4, 2),
            fit_explanation=f"Рейтинг {a.product.rating}/5, цена {a.product.price} руб.",
            main_advantage=a.review_summary.pros[0] if a.review_summary.pros else "Хороший рейтинг",
            main_caveat=a.review_summary.cons[0] if a.review_summary.cons else "Нет существенных минусов",
        ))

    stats = llm.get_usage_stats()
    return {
        "recommendation": Recommendation(
            top3=top3,
            explanation="Рекомендация составлена автоматически на основе рейтинга, цены и отзывов.",
            confidence=0.3,
            user_query=query_spec.raw_query,
        ),
        "stage": WorkflowStage.RECOMMENDED,
        "llm_calls": stats["calls"],
        "total_tokens": stats["total_tokens"],
        "total_cost": stats["total_cost"],
    }
