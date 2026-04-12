"""LangGraph Orchestrator — main pipeline for MarketMind."""

from __future__ import annotations

import logging
import time
from functools import partial
from pathlib import Path
from typing import Any, TypedDict, Optional

from langgraph.graph import END, StateGraph

from marketmind.agents.comparator import run_comparator
from marketmind.agents.product_searcher import run_product_searcher
from marketmind.agents.query_analyzer import run_query_analyzer
from marketmind.agents.recommender import run_recommender
from marketmind.agents.review_analyzer import run_review_analyzer
from marketmind.config import Settings
from marketmind.llm_client import LLMClient
from marketmind.models import (
    Product,
    ProductAnalysis,
    QuerySpec,
    Recommendation,
    Review,
    WorkflowStage,
)
from marketmind.observability import RequestTrace
from marketmind.tools.mock_provider import MockDataProvider

logger = logging.getLogger("marketmind")


# --- LangGraph State (TypedDict for LangGraph compatibility) ---

class GraphState(TypedDict, total=False):
    user_query: str
    query_spec: Optional[QuerySpec]
    products: list[Product]
    product_reviews: dict[str, list[Review]]
    analyzed_products: list[ProductAnalysis]
    recommendation: Optional[Recommendation]
    stage: WorkflowStage
    errors: list[str]
    llm_calls: int
    total_tokens: int
    total_cost: float


# --- Routing functions ---

def _route_after_parse(state: GraphState) -> str:
    """Route after query parsing: clarify or search."""
    query_spec = state.get("query_spec")
    if not query_spec:
        return "end_no_results"
    if query_spec.needs_clarification:
        return "end_clarification"
    return "search_products"


def _route_after_search(state: GraphState) -> str:
    """Route after search: analyze or no results."""
    products = state.get("products", [])
    if not products:
        return "end_no_results"
    return "analyze_reviews"


# --- Node wrappers ---

def _node_parse_query(state: GraphState, llm: LLMClient, prompts_dir: Path) -> GraphState:
    trace_start = time.time()
    result = run_query_analyzer(state, llm, prompts_dir)
    logger.info(f"parse_query took {time.time() - trace_start:.2f}s")
    return result


def _node_search_products(
    state: GraphState,
    mock_provider: MockDataProvider,
    enabled_sources: dict[str, bool],
    max_results_total: int,
    min_rating: float,
) -> GraphState:
    trace_start = time.time()
    result = run_product_searcher(state, mock_provider, enabled_sources, max_results_total, min_rating)
    logger.info(f"search_products took {time.time() - trace_start:.2f}s")
    return result


def _node_analyze_reviews(state: GraphState, llm: LLMClient, prompts_dir: Path) -> GraphState:
    trace_start = time.time()
    result = run_review_analyzer(state, llm, prompts_dir)
    logger.info(f"analyze_reviews took {time.time() - trace_start:.2f}s")
    return result


def _node_compare(state: GraphState, llm: LLMClient, prompts_dir: Path) -> GraphState:
    trace_start = time.time()
    result = run_comparator(state, llm, prompts_dir)
    logger.info(f"compare_products took {time.time() - trace_start:.2f}s")
    return result


def _node_recommend(state: GraphState, llm: LLMClient, prompts_dir: Path) -> GraphState:
    trace_start = time.time()
    result = run_recommender(state, llm, prompts_dir)
    logger.info(f"generate_recommendation took {time.time() - trace_start:.2f}s")
    return result


# --- Build graph ---

def build_graph(settings: Settings, llm: LLMClient) -> StateGraph:
    """Build the LangGraph agent pipeline."""
    prompts_dir = settings.get_prompts_path()
    mock_provider = MockDataProvider(settings.get_mock_data_path())

    # Determine enabled sources
    enabled_sources = {
        name: src.enabled
        for name, src in settings.search.sources.items()
    }

    graph = StateGraph(GraphState)

    # Add nodes with dependencies injected via partial
    graph.add_node(
        "parse_query",
        partial(_node_parse_query, llm=llm, prompts_dir=prompts_dir),
    )
    graph.add_node(
        "search_products",
        partial(
            _node_search_products,
            mock_provider=mock_provider,
            enabled_sources=enabled_sources,
            max_results_total=settings.search.max_results_total,
            min_rating=settings.search.min_rating,
        ),
    )
    graph.add_node(
        "analyze_reviews",
        partial(_node_analyze_reviews, llm=llm, prompts_dir=prompts_dir),
    )
    graph.add_node(
        "compare_products",
        partial(_node_compare, llm=llm, prompts_dir=prompts_dir),
    )
    graph.add_node(
        "generate_recommendation",
        partial(_node_recommend, llm=llm, prompts_dir=prompts_dir),
    )

    # Set entry point
    graph.set_entry_point("parse_query")

    # Conditional edges
    graph.add_conditional_edges(
        "parse_query",
        _route_after_parse,
        {
            "search_products": "search_products",
            "end_clarification": END,
            "end_no_results": END,
        },
    )
    graph.add_conditional_edges(
        "search_products",
        _route_after_search,
        {
            "analyze_reviews": "analyze_reviews",
            "end_no_results": END,
        },
    )

    # Linear edges
    graph.add_edge("analyze_reviews", "compare_products")
    graph.add_edge("compare_products", "generate_recommendation")
    graph.add_edge("generate_recommendation", END)

    return graph


class Orchestrator:
    """Main orchestrator for MarketMind pipeline."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = LLMClient(settings.llm)
        self._graph = build_graph(settings, self.llm)
        self._compiled = self._graph.compile()

    def run(self, user_query: str) -> GraphState:
        """Run the full pipeline and return final state."""
        self.llm.reset_session()

        initial_state: GraphState = {
            "user_query": user_query,
            "query_spec": None,
            "products": [],
            "product_reviews": {},
            "analyzed_products": [],
            "recommendation": None,
            "stage": WorkflowStage.INIT,
            "errors": [],
            "llm_calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }

        trace = RequestTrace()
        logger.info(
            f"Starting pipeline for query: {user_query!r}",
            extra={"request_id": trace.request_id},
        )

        try:
            final_state = self._compiled.invoke(initial_state)
            trace.finish("success")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            trace.finish("error", str(e))
            final_state = {**initial_state, "errors": [str(e)], "stage": WorkflowStage.ERROR}

        # Update trace with final stats
        stats = self.llm.get_usage_stats()
        trace.total_llm_calls = stats["calls"]
        trace.total_tokens = stats["total_tokens"]
        trace.total_cost = stats["total_cost"]

        logger.info(
            f"Pipeline finished: status={trace.final_status}, "
            f"llm_calls={stats['calls']}, tokens={stats['total_tokens']}, "
            f"cost=${stats['total_cost']:.4f}, duration={trace.duration_seconds:.1f}s",
            extra={"request_id": trace.request_id},
        )

        # Save trace
        try:
            trace.save(self.settings.get_logs_path())
        except Exception:
            pass

        return final_state

    def stream(self, user_query: str):
        """Stream pipeline execution, yielding state after each node."""
        self.llm.reset_session()

        initial_state: GraphState = {
            "user_query": user_query,
            "query_spec": None,
            "products": [],
            "product_reviews": {},
            "analyzed_products": [],
            "recommendation": None,
            "stage": WorkflowStage.INIT,
            "errors": [],
            "llm_calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }

        for event in self._compiled.stream(initial_state):
            yield event
