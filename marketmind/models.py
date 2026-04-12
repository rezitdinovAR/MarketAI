"""Pydantic models for MarketMind agent state and data structures."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, Field


# --- Enums ---

class WorkflowStage(str, Enum):
    INIT = "init"
    QUERY_PARSED = "query_parsed"
    PRODUCTS_FOUND = "products_found"
    REVIEWS_ANALYZED = "reviews_analyzed"
    COMPARED = "compared"
    RECOMMENDED = "recommended"
    ERROR = "error"
    DONE = "done"


# --- Query ---

class QuerySpec(BaseModel):
    raw_query: str
    category: Optional[str] = None
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    marketplace_priority: list[str] = Field(default_factory=lambda: ["ozon", "wildberries", "yandex"])
    needs_clarification: bool = False
    clarification_questions: list[str] = Field(default_factory=list)


# --- Product ---

class Product(BaseModel):
    id: str
    name: str
    price: int
    original_price: Optional[int] = None
    marketplace: str
    url: str
    image_url: Optional[str] = None
    rating: float
    review_count: int
    in_stock: bool = True
    seller_id: Optional[str] = None
    seller_name: Optional[str] = None


class Review(BaseModel):
    id: str
    text: str
    rating: int = Field(ge=1, le=5)
    date: str
    author: Optional[str] = None
    verified_purchase: bool = False
    helpful_count: int = 0


class SellerInfo(BaseModel):
    id: str
    name: str
    rating: float
    review_count: int
    registration_date: Optional[str] = None
    is_official: bool = False
    return_policy: Optional[str] = None
    delivery_speed: Optional[str] = None


# --- Analysis ---

class ReviewSummary(BaseModel):
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    summary: str = ""
    trust_score: float = Field(default=0.5, ge=0, le=1)


class ProductAnalysis(BaseModel):
    product: Product
    review_summary: ReviewSummary
    value_score: float = Field(default=0.0, ge=0, le=1)
    fit_score: float = Field(default=0.0, ge=0, le=1)


# --- Recommendation ---

class RankedProduct(BaseModel):
    rank: int
    product: Product
    review_summary: ReviewSummary
    final_score: float = Field(ge=0, le=1)
    fit_explanation: str = ""
    main_advantage: str = ""
    main_caveat: str = ""


class Recommendation(BaseModel):
    top3: list[RankedProduct] = Field(default_factory=list)
    explanation: str = ""
    confidence: float = Field(default=0.5, ge=0, le=1)
    user_query: str = ""


# --- LangGraph Agent State ---

def _merge_lists(left: list, right: list) -> list:
    """Reducer: replace list entirely with new value."""
    return right if right is not None else left


class AgentState(BaseModel):
    """Full agent state passed through the LangGraph pipeline."""

    # Input
    user_query: str = ""

    # Parsed
    query_spec: Optional[QuerySpec] = None

    # Search results
    products: list[Product] = Field(default_factory=list)
    product_reviews: dict[str, list[Review]] = Field(default_factory=dict)

    # Analysis
    analyzed_products: list[ProductAnalysis] = Field(default_factory=list)

    # Output
    recommendation: Optional[Recommendation] = None

    # Meta
    stage: WorkflowStage = WorkflowStage.INIT
    errors: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0


# --- LLM Response wrapper ---

class LLMResponse(BaseModel):
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0


# --- Tool result ---

class ToolResult(BaseModel):
    success: bool
    data: object = None
    fallback_used: bool = False
    error: Optional[str] = None
