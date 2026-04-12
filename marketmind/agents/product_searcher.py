"""ProductSearcher — searches products across marketplaces using tools."""

from __future__ import annotations

import logging
from typing import Optional

from marketmind.models import AgentState, Product, Review, WorkflowStage
from marketmind.tools.mock_provider import MockDataProvider

logger = logging.getLogger("marketmind")


def run_product_searcher(
    state: dict,
    mock_provider: MockDataProvider,
    enabled_sources: dict[str, bool],
    max_results_total: int = 10,
    min_rating: float = 3.5,
) -> dict:
    """LangGraph node: search products and collect reviews."""
    query_spec = state.get("query_spec")
    if not query_spec:
        return {
            "products": [],
            "stage": WorkflowStage.PRODUCTS_FOUND,
            "errors": ["No query_spec available for search"],
        }

    category = query_spec.category
    raw_query = query_spec.raw_query
    budget_max = query_spec.budget_max
    budget_min = query_spec.budget_min

    all_products: list[Product] = []
    all_reviews: dict[str, list[Review]] = {}
    errors: list[str] = []

    # Search each enabled marketplace
    for marketplace, enabled in enabled_sources.items():
        if not enabled:
            continue

        try:
            products = mock_provider.search_products(
                marketplace=marketplace,
                query=raw_query if not category else category,
                category=category,
                min_price=budget_min,
                max_price=budget_max,
            )

            # Filter by minimum rating
            products = [p for p in products if p.rating >= min_rating]

            # Collect reviews for each product
            for product in products:
                reviews = mock_provider.get_reviews(product.id, marketplace)
                if reviews:
                    all_reviews[product.id] = reviews

            all_products.extend(products)
            logger.info(
                f"Found {len(products)} products on {marketplace}",
                extra={"stage": "product_searcher"},
            )

        except Exception as e:
            logger.warning(f"Search failed for {marketplace}: {e}")
            errors.append(f"Search error on {marketplace}: {e}")

    # Sort by rating (desc) and limit
    all_products.sort(key=lambda p: (p.rating, p.review_count), reverse=True)
    all_products = all_products[:max_results_total]

    # Keep only reviews for selected products
    selected_ids = {p.id for p in all_products}
    all_reviews = {pid: revs for pid, revs in all_reviews.items() if pid in selected_ids}

    logger.info(
        f"Total: {len(all_products)} products selected from {sum(1 for e in enabled_sources.values() if e)} marketplaces",
        extra={"stage": "product_searcher"},
    )

    result = {
        "products": all_products,
        "product_reviews": all_reviews,
        "stage": WorkflowStage.PRODUCTS_FOUND,
    }
    if errors:
        result["errors"] = errors
    return result
