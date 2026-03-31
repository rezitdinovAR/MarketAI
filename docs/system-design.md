# MarketMind — Системный дизайн
## 1. Ключевые архитектурные решения
### 1.1 Архитектурный стиль
Выбор: Event-driven Agent Graph (LangGraph)

|Решение|Обоснование|Альтернативы|
|---|----------|------|
LangGraph как оркестратор|Явный граф состояний, встроенная поддержка циклов и ветвлений, хорошая отладка|LangChain Agents (менее контроля), AutoGen (избыточно для PoC)
Единый State-объект|Упрощает debugging, все данные в одном месте|Распределённый state
Sync execution|PoC не требует async, проще отладка|Async (преждевременная оптимизация)
Mock-first data sources|Независимость от внешних API при разработке|Real parsing first (хрупко)

### 1.2 Принципы проектирования
1. Fail-fast with graceful degradation — лучше честно сказать "не смог", чем выдать мусор
2. Explicit over implicit — все решения агента логируются и объяснимы
3. Mock-friendly — любой внешний источник подменяется mock'ом через конфиг
4. Cost-aware — каждый LLM-вызов оправдан, кэширование где возможно

### 1.3 Ограничения PoC
Параметр|Значение|Комментарий
|----|---|------|
Max latency (e2e)|60 сек|Пользователь готов ждать для качественного ответа
Target latency (p50)|15-20 сек|Комфортное ожидание
Max cost per request|~$0.05|DeepSeek дешёвый, но контролируем
Max tokens per request|16K input + 4K output|Лимиты deepseek-chat
Concurrent users|1|PoC, single-threaded
Data freshness|Mock / cached|Не real-time

## 2. Модули системы
### 2.1 Обзор модулей
Module|Parts
|-|-|
UI Layer|Streamlit App / CLI
Agent Layer|Orchestrator (LangGraph): QueryAnalyzer, ProductSearcher, ReviewAnalyzer,ComparatorRecommender
Tool Layer|MarketplaceTools, ReviewTools, PriceTools
Data Layer|MockDataProvider / HTTPScraper
Infrastructure|LLMClient, StateManager, Logger, Config

### 2.2 Детализация модулей
Модуль|Роль|Input|Output
|---|-----|---|---|
UILayer|Точка входа, отображение результатов|User query (text)|Formatted recommendations
Orchestrator|Управление графом выполнения|Query + Config|Final state with recommendations
QueryAnalyzer|Парсинг и уточнение запроса|Raw query|Structured QuerySpec
ProductSearcher|Поиск товаров по источникам|QuerySpec|List[Product]
ReviewAnalyzer|Суммаризация отзывов|Product|ProductWithReviews
Comparator|Сравнительный анализ|List[ProductWithReviews]|ComparisonTable
Recommender|Финальная рекомендация|ComparisonTable + QuerySpec|Top-3 + Explanation
LLMClient|Обёртка над DeepSeek API|Prompt|Response
StateManager|Хранение и валидация state|State updates|Validated state

## 3. Workflow выполнения
### 3.1 Основной happy path
![alt text](./images/hp_paths.png)

### 3.2 Failure paths и fallbacks
![alt text](./images/fl_paths.png)

## 4. State / Memory / Context Handling
### 4.1 State Schema
```Python

from typing import TypedDict, Optional
from enum import Enum

class WorkflowStage(Enum):
    INIT = "init"
    QUERY_PARSED = "query_parsed"
    PRODUCTS_FOUND = "products_found"
    REVIEWS_ANALYZED = "reviews_analyzed"
    COMPARED = "compared"
    RECOMMENDED = "recommended"
    ERROR = "error"
    DONE = "done"

class QuerySpec(TypedDict):
    raw_query: str
    category: Optional[str]
    budget_min: Optional[int]
    budget_max: Optional[int]
    must_have: list[str]        # ["шумоподавление", "для бега"]
    nice_to_have: list[str]     # ["быстрая зарядка"]
    marketplace_priority: list[str]  # ["ozon", "wb", "yandex"]

class Product(TypedDict):
    id: str
    name: str
    price: int
    marketplace: str
    url: str
    rating: float
    review_count: int
    seller_rating: Optional[float]
    raw_reviews: list[str]      # до 20 последних

class ReviewSummary(TypedDict):
    pros: list[str]
    cons: list[str]
    summary: str
    trust_score: float          # 0-1, оценка качества отзывов

class ProductAnalysis(TypedDict):
    product: Product
    review_summary: ReviewSummary
    value_score: float          # соотношение цена/качество
    fit_score: float            # соответствие запросу

class AgentState(TypedDict):
    # Input
    user_query: str
    
    # Parsed
    query_spec: Optional[QuerySpec]
    
    # Search results
    products: list[Product]
    
    # Analysis
    analyzed_products: list[ProductAnalysis]
    
    # Output
    recommendations: list[ProductAnalysis]
    explanation: str
    
    # Meta
    stage: WorkflowStage
    errors: list[str]
    llm_calls: int
    total_tokens: int
    total_cost: float
```

### 4.2 Context Budget Management
```Python

CONTEXT_LIMITS = {
    "query_analysis": 2000,      # tokens for query parsing
    "reviews_per_product": 3000, # max tokens for reviews
    "comparison": 4000,          # tokens for comparison prompt
    "recommendation": 2000,      # tokens for final recommendation
    "max_products": 10,          # products to analyze
    "max_reviews_per_product": 15,
}

# Стратегия обрезки
# 1. Приоритет недавним отзывам
# 2. Приоритет отзывам с текстом > 50 символов
# 3. Баланс положительных/отрицательных
```

### 4.3 Memory Policy
PoC: Stateless (нет персистентности между сессиями)

В рамках сессии:
* Полный state хранится в памяти
* Каждый шаг добавляет данные, не удаляет
* При ошибке — state замораживается, добавляется в errors
* В конце — state сериализуется для логов

## 5. Retrieval-контур
### 5.1 Data Sources
Source|Type|Priority
|---|---|---|
Mock JSON|Static files|P0 (default, always works)
Ozon scraping|HTTP + parsing|P1 (if enabled)
WB scraping|HTTP + parsing|P1 (if enabled)
YaMarket|HTTP + parsing|P2 (optional)

### 5.2 Search Flow
![alt text](srch_flow.png)

### 5.3 Review Retrieval
```Python

# Для каждого продукта:
# 1. Взять до 15 отзывов
# 2. Приоритет: recent + verified + with_text
# 3. Обрезать до context_limit

def get_reviews(product_id: str, source: str) -> list[str]:
    raw = fetch_reviews(product_id, source, limit=20)
    filtered = [r for r in raw if len(r.text) > 50]
    sorted_reviews = sorted(filtered, key=lambda r: r.date, reverse=True)
    return truncate_to_tokens(sorted_reviews[:15], limit=3000)
```

## 6. Tool / API интеграции
### 6.1 Tools Registry
```Python

TOOLS = {
    "search_marketplace": {
        "description": "Поиск товаров на маркетплейсе",
        "params": {
            "marketplace": "ozon | wb | yandex",
            "query": "поисковый запрос",
            "max_price": "int | None",
            "category": "str | None"
        },
        "timeout": 10,
        "retries": 2,
        "fallback": "mock_data"
    },
    "get_product_reviews": {
        "description": "Получение отзывов о товаре",
        "params": {
            "product_id": "str",
            "marketplace": "str"
        },
        "timeout": 5,
        "retries": 1,
        "fallback": "empty_reviews"
    },
    "get_seller_info": {
        "description": "Информация о продавце",
        "params": {
            "seller_id": "str",
            "marketplace": "str"
        },
        "timeout": 5,
        "retries": 1,
        "fallback": "unknown_seller"
    }
}
```

### 6.2 LLM API Integration
```Python

class LLMClient:
    """Обёртка над DeepSeek API с защитами"""
    
    config = {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "timeout": 30,
        "max_retries": 2,
        "retry_delay": 1.0,
        "max_tokens_per_call": 4096,
    }
    
    guardrails = {
        "max_calls_per_request": 10,
        "max_total_tokens": 50000,
        "max_cost": 0.10,  # USD
    }
    
    def call(self, messages: list, **kwargs) -> str:
        # 1. Check guardrails
        # 2. Call with timeout
        # 3. Parse response
        # 4. Validate output format
        # 5. Update counters
        # 6. Log for observability
```

### 6.3 Tool Execution Wrapper
```Python

def execute_tool(tool_name: str, params: dict) -> ToolResult:
    """
    Единая обёртка для вызова любого tool с:
    - timeout handling
    - retry logic
    - fallback to mock
    - logging
    - error capture
    """
    tool = TOOLS[tool_name]
    
    for attempt in range(tool["retries"] + 1):
        try:
            with timeout(tool["timeout"]):
                result = tool["handler"](**params)
                return ToolResult(success=True, data=result)
        except TimeoutError:
            log.warning(f"{tool_name} timeout, attempt {attempt}")
        except Exception as e:
            log.error(f"{tool_name} error: {e}")
    
    # Fallback
    return ToolResult(
        success=False, 
        data=tool["fallback_handler"](**params),
        fallback_used=True
    )
```

## 7. Failure Modes, Fallbacks, Guardrails
### 7.1 LLM Guardrails
```Python

class LLMGuardrails:
    """Защита от некорректного поведения LLM"""
    
    # Input guardrails
    def sanitize_input(self, user_query: str) -> str:
        """Очистка пользовательского ввода"""
        # Удаление prompt injection паттернов
        # Ограничение длины
        # Удаление спец-символов
        
    # Output guardrails  
    def validate_query_spec(self, output: str) -> QuerySpec:
        """Валидация структуры распознанного запроса"""
        parsed = json.loads(output)
        assert "category" in parsed
        assert parsed.get("budget_max", 0) <= 10_000_000  # sanity check
        return QuerySpec(**parsed)
    
    def validate_recommendation(self, output: str, products: list) -> bool:
        """Проверка что рекомендуемые товары есть в списке"""
        recommended_ids = extract_product_ids(output)
        valid_ids = {p["id"] for p in products}
        return all(rid in valid_ids for rid in recommended_ids)
    
    # Hallucination detection
    def check_price_consistency(self, recommendation: str, products: list) -> bool:
        """Проверка что цены в рекомендации соответствуют реальным"""
        mentioned_prices = extract_prices(recommendation)
        actual_prices = {p["name"]: p["price"] for p in products}
        # Допуск 5% на округление
        return all(
            abs(mentioned - actual_prices.get(name, 0)) / actual_prices.get(name, 1) < 0.05
            for name, mentioned in mentioned_prices.items()
        )
```
### 7.2 Failure Recovery Matrix  
![alt text](images/rec_matrix.png)

### 7.3 Quality Control Checkpoints
```Python

QUALITY_CHECKPOINTS = {
    "after_query_parse": {
        "check": "query_spec has category OR budget",
        "fail_action": "clarify_with_user"
    },
    "after_search": {
        "check": "len(products) >= 1",
        "fail_action": "expand_criteria_or_exit"
    },
    "after_review_analysis": {
        "check": "all products have review_summary",
        "fail_action": "proceed_without_reviews"  # degraded
    },
    "after_comparison": {
        "check": "comparison mentions all products",
        "fail_action": "regenerate_comparison"
    },
    "after_recommendation": {
        "check": "recommended products exist in list AND prices match",
        "fail_action": "regenerate_with_constraints"
    }
}
```

## 8. Технические и операционные ограничения
### 8.1 Performance Constraints
Метрика|Target|Hard|Limit|Измерение
|---|---|---|---|---|
E2E latency (p50)|15s|60s|time.time()
E2E latency (p95)|30s|90s|time.time()
LLM calls per request|4-5|10|counter
Tokens per request|15K|50K|tiktoken
Cost per request|$0.02|$0.10|API response
Memory usage|100MB|500MB|psutil

### 8.2 Reliability
Метрика|Target|Измерение
|---|---|---|
Success rate|>90%|success/total
Partial result rate|<10%|partial/total
Hard failure rate|<5%|error/total
LLM retry rate|<20%|retries/calls

### 8.3 Качество (subjective, для future evals)
Аспект|Способ оценки
|---|---|
Relevance|Рекомендации соответствуют запросу
Factuality|Цены и характеристики корректны
Helpfulness|Объяснение понятно и полезно
Coverage|Рассмотрены разные варианты
