# Observability

## 1. Общее описание

**Observability** — модуль для логирования, трейсинга и сбора метрик. Обеспечивает возможность отладки, мониторинга и оценки качества системы.

---

## 2. Компоненты

### 2.1 Structured Logging

Все логи в структурированном формате (JSON) с обязательными полями:

| Поле | Описание |
|------|----------|
| timestamp | ISO8601 время |
| level | info / warning / error |
| request_id | Сквозной ID запроса |
| stage | Текущая стадия пайплайна |
| message | Человекочитаемое сообщение |
| data | Дополнительные данные (dict) |

### 2.2 Request Tracing

Каждый запрос получает trace, содержащий:

| Поле | Описание |
|------|----------|
| request_id | UUID |
| start_time / end_time | Время начала и конца |
| stages | Список StageTrace |
| total_llm_calls | Сумма LLM вызовов |
| total_tokens | Сумма токенов |
| total_cost | Сумма стоимости |
| final_status | success / partial / error |
| error_message | Текст ошибки если есть |

**StageTrace:**
| Поле | Описание |
|------|----------|
| name | Название стадии |
| start_time / end_time | Время |
| status | success / skipped / error |
| llm_calls | Количество LLM вызовов на стадии |
| metadata | Дополнительные данные |

### 2.3 Metrics

| Метрика | Тип | Labels | Описание |
|---------|-----|--------|----------|
| e2e_latency_seconds | Histogram | — | Общее время запроса |
| stage_latency_seconds | Histogram | stage | Время каждой стадии |
| llm_call_latency_seconds | Histogram | model, stage | Время LLM вызова |
| requests_total | Counter | status | Количество запросов |
| llm_calls_total | Counter | model, stage | Количество LLM вызовов |
| tokens_total | Counter | direction | Токены in/out |
| cost_usd_total | Counter | — | Общая стоимость |
| products_found | Histogram | — | Найдено товаров |
| errors_total | Counter | stage, error_type | Количество ошибок |
| retries_total | Counter | stage | Количество retry |
| fallbacks_used_total | Counter | type | Использование fallback |

---

## 3. Log Destinations

| Тип лога | Куда пишется | Формат | Retention |
|----------|--------------|--------|-----------|
| Application | stdout + logs/app.log | JSON Lines | 7 дней |
| LLM Calls | logs/llm_calls/ | JSON Lines | 30 дней |
| Request Traces | logs/traces/ | JSON per request | 30 дней |
| Errors | logs/errors.log | JSON Lines | 90 дней |
| Metrics | logs/metrics/ | JSON snapshot | Hourly |

---

## 4. Debug Mode

В debug mode дополнительно:

| Функция | Описание |
|---------|----------|
| log_full_prompts | Полные тексты промптов в логах |
| log_full_responses | Полные ответы LLM |
| save_intermediate_state | JSON dump состояния после каждой стадии |
| verbose_validation | Детальные логи валидации |

---

## 5. Quality Eval Hooks

Точки для offline оценки качества:

| Checkpoint | Input | Output | Вопрос для eval |
|------------|-------|--------|-----------------|
| query_parse | raw_query | query_spec | Корректно распознаны категория и критерии? |
| review_summary | raw_reviews | summary | Точно выделены плюсы/минусы? |
| recommendation | query + products | top3 | Релевантны рекомендации? Корректны факты? |

Данные сохраняются в `evals/{checkpoint}/{request_id}.json` для последующего анализа.

---

## 6. Health Indicators

В конце каждого запроса проверяются:

| Индикатор | Порог | Warning |
|-----------|-------|---------|
| Cost | > $0.05 | "High cost" |
| LLM calls | > 8 | "Too many LLM calls" |
| Duration | > 45s | "Slow request" |
| Status | error | "Request failed" |
