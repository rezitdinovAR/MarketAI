"""ReviewAnalyzer — summarizes product reviews using LLM."""

from __future__ import annotations

import logging
from pathlib import Path

from marketmind.llm_client import LLMClient
from marketmind.models import (
    AgentState,
    Product,
    ProductAnalysis,
    Review,
    ReviewSummary,
    WorkflowStage,
)

logger = logging.getLogger("marketmind")


FALLBACK_SYSTEM_PROMPT = (
    "Проанализируй отзывы и верни JSON: "
    '{"pros": [...], "cons": [...], "summary": "...", "trust_score": 0.0-1.0}. '
    "Отвечай ТОЛЬКО валидным JSON."
)


def _load_prompt(prompts_dir: Path) -> str:
    path = prompts_dir / "review_analysis.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return FALLBACK_SYSTEM_PROMPT


def _format_reviews(reviews: list[Review]) -> str:
    """Format reviews into text for LLM prompt."""
    lines = []
    for r in reviews:
        stars = "*" * r.rating
        verified = " [Verified]" if r.verified_purchase else ""
        lines.append(f"[{stars}]{verified} ({r.date}): {r.text}")
    return "\n\n".join(lines)


def _analyze_single_product(
    product: Product,
    reviews: list[Review],
    llm: LLMClient,
    system_prompt: str,
) -> ProductAnalysis:
    """Analyze reviews for a single product."""
    if not reviews:
        return ProductAnalysis(
            product=product,
            review_summary=ReviewSummary(
                summary="Нет отзывов для анализа",
                trust_score=0.0,
            ),
        )

    reviews_text = _format_reviews(reviews)
    user_message = (
        f"Товар: {product.name}\n"
        f"Цена: {product.price} руб.\n"
        f"Рейтинг: {product.rating}/5 ({product.review_count} отзывов)\n"
        f"Маркетплейс: {product.marketplace}\n\n"
        f"Отзывы:\n{reviews_text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    try:
        result = llm.call_json(messages, temperature=0.3, max_tokens=1024)
        review_summary = ReviewSummary(
            pros=result.get("pros", []),
            cons=result.get("cons", []),
            summary=result.get("summary", ""),
            trust_score=min(1.0, max(0.0, result.get("trust_score", 0.5))),
        )
    except Exception as e:
        logger.warning(f"Review analysis failed for {product.id}: {e}")
        # Fallback: basic summary from raw data
        review_summary = ReviewSummary(
            summary=f"Рейтинг {product.rating}/5 на основе {product.review_count} отзывов",
            trust_score=0.3,
        )

    return ProductAnalysis(product=product, review_summary=review_summary)


def run_review_analyzer(state: dict, llm: LLMClient, prompts_dir: Path) -> dict:
    """LangGraph node: analyze reviews for all found products."""
    products: list[Product] = state.get("products", [])
    product_reviews: dict[str, list[Review]] = state.get("product_reviews", {})

    if not products:
        return {
            "analyzed_products": [],
            "stage": WorkflowStage.REVIEWS_ANALYZED,
        }

    system_prompt = _load_prompt(prompts_dir)
    analyzed: list[ProductAnalysis] = []

    for product in products:
        reviews = product_reviews.get(product.id, [])
        analysis = _analyze_single_product(product, reviews, llm, system_prompt)
        analyzed.append(analysis)
        logger.info(
            f"Analyzed reviews for {product.name}: trust={analysis.review_summary.trust_score}",
            extra={"stage": "review_analyzer"},
        )

    stats = llm.get_usage_stats()
    return {
        "analyzed_products": analyzed,
        "stage": WorkflowStage.REVIEWS_ANALYZED,
        "llm_calls": stats["calls"],
        "total_tokens": stats["total_tokens"],
        "total_cost": stats["total_cost"],
    }
