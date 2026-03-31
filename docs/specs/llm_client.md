# LLM Client

## 1. Общее описание

**LLM Client** — обёртка над DeepSeek API с retry logic, cost tracking, rate limiting и guardrails. Единая точка взаимодействия с LLM для всех модулей.

---

## 2. Интерфейс модуля

### 2.1 Основные методы

| Метод | Описание |
|-------|----------|
| `call(messages, temperature, max_tokens, response_format)` | Вызов LLM |
| `call_with_json(messages, schema)` | Вызов с валидацией JSON-схемы |
| `get_usage_stats()` | Статистика текущей сессии |
| `reset_session()` | Сброс счётчиков сессии |

### 2.2 Параметры вызова

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| messages | list[dict] | required | Список сообщений |
| temperature | float | 0.7 | Креативность 0-1 |
| max_tokens | int | 2048 | Лимит токенов ответа |
| response_format | str | "text" | "text" или "json" |

### 2.3 Возвращаемые данные

| Поле | Тип | Описание |
|------|-----|----------|
| content | str | Текст ответа |
| input_tokens | int | Токены запроса |
| output_tokens | int | Токены ответа |
| latency_ms | int | Время ответа |
| cost_usd | float | Стоимость вызова |

## 3. Алгоритм работы
![alt text](../images/llm_client.png)

## 4. Retry стратегия
| Ошибка | Retry? | Backoff | Max attempts |
|--------|--------|---------|--------------|
| Timeout (30s) | Да | Нет | 2 |
| Rate Limit (429) | Да | Exponential: 1s, 2s, 4s | 3 |
| Server Error (5xx) | Да | Linear: 1s | 2 |
| Bad Request (400) | Нет | — | — |
| Auth Error (401) | Нет | — | — |
| JSON Parse Error | Да* | Нет | 2 |

*При JSON Parse Error — retry с модифицированным промптом:
```
"Предыдущий ответ был невалидным JSON.
Ответь СТРОГО в формате JSON, без текста до/после."
```

## 5. Cost Tracking

### Pricing (DeepSeek)
| Модель | Input | Output |
|--------|-------|--------|
| deepseek-chat | $0.0001 / 1K tokens | $0.0002 / 1K tokens |
| deepseek-reasoner | $0.0005 / 1K tokens | $0.001 / 1K tokens |

### Guardrails (per request)
| Лимит | Значение | При превышении |
|-------|----------|----------------|
| Max calls | 10 | BudgetExceededError |
| Max tokens | 50,000 | BudgetExceededError |
| Max cost | $0.10 | BudgetExceededError |

---

## 6. Логирование

Каждый вызов логируется в структурированном формате:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "call_id": "uuid-xxx",
  "request_id": "parent-request-uuid",
  "model": "deepseek-chat",
  "stage": "review_analysis",
  "input_tokens": 1500,
  "output_tokens": 500,
  "latency_ms": 2300,
  "cost_usd": 0.00025,
  "success": true,
  "retry_count": 0,
  "temperature": 0.5
}
```

## 7. Обработка ошибок
Исключение|Когда|Recoverable?
|---|---|---|
LLMTimeoutError|Timeout после всех retry|Да (skip step)
LLMRateLimitError|Rate limit после всех backoff|Да (wait)
LLMBudgetExceededError|Превышен лимит cost/tokens/calls|Нет
LLMResponseValidationError|Невалидный JSON после retry|Да (fallback)
LLMAuthError|Неверный API key|Нет

## 8. Метрики
Метрика|Тип|Labels
|---|---|---|
llm_calls_total|Counter|model, stage, status
llm_latency_seconds|Histogram|model, stage
llm_tokens_total|Counter|model, direction (in/out)
llm_cost_usd_total|Counter|model
llm_retries_total|Counter|model, error_type
llm_errors_total|Counter|model, error_type
