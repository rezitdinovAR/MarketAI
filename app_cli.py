"""CLI entry point for MarketMind."""

from __future__ import annotations

import sys

from marketmind.config import load_settings
from marketmind.observability import setup_logger
from marketmind.orchestrator import Orchestrator

DISCLAIMER = (
    "\n--- MarketMind AI ---\n"
    "AI-ассистент по подбору товаров на маркетплейсах.\n"
    "Введите запрос на естественном языке (или 'выход' для завершения).\n"
)

WARNING = (
    "MarketMind — AI-ассистент. Рекомендации основаны на автоматическом анализе "
    "и могут содержать неточности. Перед покупкой проверьте информацию на сайте маркетплейса."
)


def format_recommendation(state: dict) -> str:
    """Format final state into readable output."""
    query_spec = state.get("query_spec")
    recommendation = state.get("recommendation")
    errors = state.get("errors", [])

    lines: list[str] = []

    # Clarification needed
    if query_spec and query_spec.needs_clarification:
        lines.append("\nНужно уточнение:")
        for q in query_spec.clarification_questions:
            lines.append(f"  - {q}")
        return "\n".join(lines)

    # No results
    products = state.get("products", [])
    if not products:
        lines.append("\nК сожалению, по вашему запросу ничего не найдено.")
        lines.append("Попробуйте расширить критерии поиска.")
        return "\n".join(lines)

    # Show recommendation
    if recommendation and recommendation.top3:
        lines.append(f"\n{'='*60}")
        lines.append(f"  РЕКОМЕНДАЦИИ ПО ЗАПРОСУ: {recommendation.user_query}")
        lines.append(f"{'='*60}")

        for rp in recommendation.top3:
            p = rp.product
            lines.append(f"\n  #{rp.rank} {p.name}")
            lines.append(f"     Цена: {p.price:,} руб.  |  Рейтинг: {p.rating}/5  |  {p.marketplace}")
            if rp.fit_explanation:
                lines.append(f"     Почему: {rp.fit_explanation}")
            if rp.main_advantage:
                lines.append(f"     + {rp.main_advantage}")
            if rp.main_caveat:
                lines.append(f"     - {rp.main_caveat}")

            # Reviews
            rs = rp.review_summary
            if rs.pros:
                lines.append(f"     Плюсы: {'; '.join(rs.pros[:3])}")
            if rs.cons:
                lines.append(f"     Минусы: {'; '.join(rs.cons[:3])}")

        if recommendation.explanation:
            lines.append(f"\n{'─'*60}")
            lines.append(f"  {recommendation.explanation}")

        lines.append(f"\n  Уверенность: {recommendation.confidence:.0%}")

    # Stats
    llm_calls = state.get("llm_calls", 0)
    total_cost = state.get("total_cost", 0)
    if llm_calls:
        lines.append(f"\n  [LLM: {llm_calls} вызовов, ${total_cost:.4f}]")

    # Errors
    if errors:
        lines.append(f"\n  Предупреждения: {'; '.join(errors)}")

    lines.append(f"\n  {WARNING}")
    return "\n".join(lines)


def main() -> None:
    settings = load_settings()
    logger = setup_logger(level=settings.app.log_level, log_dir=settings.get_logs_path())

    if not settings.llm.api_key:
        print("ОШИБКА: DEEPSEEK_API_KEY не установлен.")
        print("Создайте файл .env с содержимым: DEEPSEEK_API_KEY=sk-your-key")
        sys.exit(1)

    orchestrator = Orchestrator(settings)
    print(DISCLAIMER)

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not query:
            continue
        if query.lower() in ("выход", "exit", "quit", "q"):
            print("До свидания!")
            break

        print("\nОбработка запроса...")
        result = orchestrator.run(query)
        print(format_recommendation(result))


if __name__ == "__main__":
    main()
