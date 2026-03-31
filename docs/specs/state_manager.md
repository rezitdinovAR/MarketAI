# State Manager

## 1. Общее описание

**State Manager** — модуль управления состоянием агента. Хранит все промежуточные данные, валидирует переходы между стадиями, создаёт checkpoints для отладки.

---

## 2. Интерфейс модуля

### 2.1 Основные методы

| Метод | Описание |
|-------|----------|
| `update(**kwargs)` | Обновить поля состояния |
| `transition(new_stage)` | Перейти на новую стадию с валидацией |
| `checkpoint()` | Сохранить текущее состояние |
| `get_state()` | Получить иммутабельную копию состояния |
| `get_summary()` | Краткая сводка для логирования |
| `add_error(error)` | Добавить ошибку в список |

### 2.2 AgentState Schema

| Поле | Тип | Описание |
|------|-----|----------|
| request_id | str | UUID запроса |
| user_query | str | Исходный запрос пользователя |
| query_spec | QuerySpec | Распарсенный запрос |
| products | list[Product] | Найденные товары |
| analyzed_products | list[ProductAnalysis] | Товары с анализом отзывов |
| comparison | ComparisonTable | Результат сравнения |
| recommendations | list[RankedProduct] | Финальные рекомендации |
| explanation | str | Текст объяснения |
| stage | WorkflowStage | Текущая стадия |
| errors | list[str] | Накопленные ошибки |
| llm_calls | int | Количество LLM вызовов |
| total_tokens | int | Суммарное количество токенов |
| total_cost | float | Суммарная стоимость |
| start_time | datetime | Время начала |

## 3. Алгоритм работы

### 3.1 Валидация переходов
![alt text](../images/state_manager.png)

### 3.2 Checkpoint механизм

При каждом checkpoint сохраняется:
- Текущая стадия
- Полный snapshot состояния (сериализованный)
- Метрики на момент checkpoint (calls, tokens, cost)
- Timestamp

В debug mode checkpoints пишутся в файлы: logs/traces/{request_id}/checkpoint_{stage}_{timestamp}.json

### 3.3 Context Budget

Отслеживание расхода токенов против лимита:

| Метод | Описание |
|-------|----------|
| `can_spend(tokens)` | Проверить, хватит ли бюджета |
| `spend(tokens)` | Списать токены с бюджета |
| `remaining()` | Сколько осталось |

При превышении бюджета → BudgetExceededError

## 4. State Summary (для логов)
```json
{
  "request_id": "abc-123",
  "stage": "REVIEWS_ANALYZED",
  "products_count": 8,
  "analyzed_count": 8,
  "recommendations_count": 0,
  "llm_calls": 4,
  "total_tokens": 12500,
  "total_cost": 0.018,
  "errors_count": 0,
  "duration_seconds": 12
}
```

## 5. Обработка ошибок
Ситуация|Действие
|---|---|
Невалидный переход|Raise InvalidTransitionError
Update несуществующего поля|Raise AttributeError
Checkpoint в error stage|Сохранить с флагом is_error=true
Budget exceeded|Raise BudgetExceededError, перейти в ERROR

