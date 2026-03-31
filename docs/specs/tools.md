# Tools (Marketplace Integration)

## 1. Общее описание

**Tools** — модуль интеграции с внешними источниками данных (маркетплейсы). В PoC работает с mock данными, но интерфейс готов для реального парсинга.

---

## 2. Tool Registry

| Tool | Описание | Timeout | Retries | Fallback |
|------|----------|---------|---------|----------|
| search_marketplace | Поиск товаров по запросу | 10s | 2 | mock data |
| get_product_details | Детальная информация о товаре | 5s | 1 | cached/partial |
| get_reviews | Отзывы о товаре | 5s | 1 | empty list |
| get_seller_info | Информация о продавце | 5s | 1 | unknown seller |

---

## 3. Интерфейс Tools

### 3.1 search_marketplace

Поиск товаров на маркетплейсе по запросу.

**Input:**

| Параметр | Тип | Required | Описание |
|----------|-----|----------|----------|
| marketplace | str | да | "ozon" / "wildberries" / "yandex" |
| query | str | да | Поисковый запрос |
| category | str | нет | Категория для фильтрации |
| min_price | int | нет | Минимальная цена |
| max_price | int | нет | Максимальная цена |
| limit | int | нет | Количество результатов (default 20) |

**Output:** `list[Product]`
```python
class Product:
    id: str
    name: str
    price: int
    original_price: int | None
    rating: float
    review_count: int
    url: str
    image_url: str | None
    in_stock: bool
    seller_id: str | None
```
---

### 3.2 get_product_details

Получение детальной информации о конкретном товаре.

**Input:**

| Параметр | Тип | Required | Описание |
|----------|-----|----------|----------|
| product_id | str | да | ID товара (например "ozon_12345") |
| marketplace | str | да | Маркетплейс |

**Output:** `ProductDetails`
```python
class ProductDetails:
    id: str
    name: str
    full_description: str
    price: int
    original_price: int | None
    rating: float
    review_count: int
    url: str
    image_urls: list[str]
    in_stock: bool
    attributes: dict # {"Тип": "полноразмерные", "Bluetooth": "5.0", ...}
    seller_id: str
    seller_name: str
    delivery_info: str | None
```

**Когда используется:**
- Для получения полных характеристик товара
- Когда нужна расширенная информация, которой нет в поисковой выдаче

---

### 3.3 get_reviews

Получение отзывов о товаре.

**Input:**

| Параметр | Тип | Required | Описание |
|----------|-----|----------|----------|
| product_id | str | да | ID товара |
| marketplace | str | да | Маркетплейс |
| limit | int | нет | Количество отзывов (default 20) |
| sort | str | нет | "recent" / "helpful" (default "recent") |

**Output:** `list[Review]`
```python
class Review:
    id: str
    text: str
    rating: int # 1-5
    date: str # "2024-01-15"
    author: str | None
    verified_purchase: bool
    helpful_count: int
    images: list[str]ca
```

**Особенности:**
- Приоритет отзывам с текстом > 50 символов
- Баланс положительных и отрицательных
- Приоритет verified_purchase = true

---

### 3.4 get_seller_info

Получение информации о продавце.

**Input:**

| Параметр | Тип | Required | Описание |
|----------|-----|----------|----------|
| seller_id | str | да | ID продавца |
| marketplace | str | да | Маркетплейс |

**Output:** `SellerInfo`
```python
class SellerInfo:
    id: str
    name: str
    rating: float # 0-5
    review_count: int
    registration_date: str | None
    is_official: bool # Официальный магазин бренда
    return_policy: str | None
    delivery_speed: str | None # "1-2 дня", "3-5 дней"
```

**Когда используется:**
- Для оценки надёжности продавца
- При сравнении одинаковых товаров от разных продавцов

---

## 4. Data Provider Architecture
![alt text](../images/tools.png)

---

## 5. Mock Data Provider

Загружает данные из JSON файлов:
data/mock/  
├── ozon_products.json  
├── wb_products.json  
└── yandex_products.json  

### 5.1 Mock файл структура
```json
{
  "categories": {
    "наушники": [
      {
        "id": "ozon_001",
        "name": "Sony WH-1000XM4",
        "price": 24990,
        "rating": 4.8,
        "review_count": 2341,
        "reviews": ["отзыв 1", "отзыв 2"],
        ...
      }
    ],
    "ноутбуки": [...],
    "телефоны": [...]
  }
}
```

### 5.2 Mock search логика
1. Найти категорию по query (точное или частичное совпадение)
2. Отфильтровать по max_price
3. Вернуть до limit товаров

## 6. HTTP Scraper (future)
Для будущей реализации реального парсинга:

Аспект|Подход
|---|---|
Rate limiting|1 req/sec per domain
User-Agent|Рандомизация
Retry|3 attempts с backoff
Blocking detection|Check for captcha/403
Fallback|Switch to mock on block

## 7. Error Handling
Ошибка|Действие|Fallback
|---|---|---|
Timeout	Retry|1x|Mock data
403 Forbidden|Не retry|Mock data
5xx Error|Retry с backoff|Mock data
Parse Error|Log + skip|Partial results
No results|Return empty|Expand search

## 8. Tool Execution Wrapper
Все tool вызовы проходят через единую обёртку:
1. Log tool call start
2. Check timeout
3. Execute with retry
4. Handle errors → fallback if needed
5. Log tool call end
6. Return ToolResult(success, data, fallback_used)

## 10. Метрики
Метрика|Тип|Labels
|---|---|---|
tool_calls_total|Counter|tool, marketplace, status
tool_latency_seconds|Histogram|tool, marketplace
tool_fallback_used|Counter|tool, marketplace
tool_results_count|Histogram|tool, marketplace
