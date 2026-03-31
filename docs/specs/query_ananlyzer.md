# Query Analyzer

## 1. Общее описание

**Query Analyzer** — модуль, который преобразует свободный текстовый запрос пользователя в структурированный объект `QuerySpec`. Это первый шаг в пайплайне агента.

### Примеры работы

| Вход (raw query) | Выход (QuerySpec) |
|------------------|-------------------|
| "Хочу беспроводные наушники с шумоподавлением до 15000₽ для бега" | `category: "наушники"`, `budget_max: 15000`, `must_have: ["беспроводные", "шумоподавление", "для бега"]` |
| "ищу ноут для работы" | `category: "ноутбук"`, `must_have: ["для работы"]`, `needs_clarification: true` |
| "телефон" | `category: "телефон"`, `needs_clarification: true` |

## 2. Интерфейс модуля
### 2.1 Входные данные

```python
raw_query: str  # Текст от пользователя, 1-500 символов
```

### 2.2 Выходные данные
```Python
@dataclass
class QuerySpec:
    raw_query: str                    # Исходный запрос
    category: Optional[str]           # Категория товара: "наушники", "ноутбук", "телефон"
    budget_min: Optional[int]         # Минимальная цена в рублях
    budget_max: Optional[int]         # Максимальная цена в рублях
    must_have: list[str]              # Обязательные характеристики
    nice_to_have: list[str]           # Желательные характеристики
    marketplace_priority: list[str]   # Приоритет маркетплейсов ["ozon", "wb", "yandex"]
    needs_clarification: bool         # Нужно ли уточнение от пользователя
    clarification_questions: list[str] # Вопросы для уточнения
```

### 2.3 Исключения
```Python
class QueryTooShortError(Exception):
    """Запрос короче 2 символов"""

class QueryTooLongError(Exception):
    """Запрос длиннее 500 символов"""

class LLMParseError(Exception):
    """LLM не смог распарсить запрос после всех retry"""
```

## 3. Алгоритм работы
![alt text](../images/query_analyzer.png)
