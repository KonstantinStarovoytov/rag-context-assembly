# Engineering audit — 2026-09-13

Этот документ фиксирует только то, что проверено в коде, тестах или Langfuse.

## Рабочий путь

`ask` использует `hybrid` по умолчанию:

```text
question
  -> faithful English translation только для RU/PL
  -> Qdrant dense + BM25, RRF fusion
  -> Cohere rerank
  -> answer с citations
```

`dense` остаётся baseline-режимом. `hybrid-english` принудительно добавляет перевод
для ablation. Semantic rewrite не включён: он не улучшил champion в проверенном
наборе и ухудшил один natural-English case.

`--iterative` — отдельный экспериментальный режим. После первого rerank он просит
planner оценить именно тот evidence, который увидит генератор. При недостаточном
coverage planner может предложить до трёх follow-up queries, привязанных к уже
увиденным evidence IDs. Выполняется ровно один дополнительный retrieval round,
после которого union rerank-ится по исходному вопросу. У режима есть stop reasons
и unit tests; сравнение coverage описано ниже, преимущество iterative не подтверждено.

## Что подтверждено

### Retrieval A/B/C/D

Dataset `rag/retrieval-query-transform-v1` содержит 48 парных запросов: 8 intents
на clean/natural/short English, RU, Polish и mixed языке. Старые experiment outputs
пересчитаны одной логикой в `reports/query-transform-rescore.txt`.

| Strategy | H@1 | H@5 | H@10 | MRR |
|---|---:|---:|---:|---:|
| A original | 0.896 | 0.958 | 0.958 | 0.927 |
| B faithful English | **0.938** | **1.000** | **1.000** | **0.969** |
| C semantic | 0.896 | 1.000 | 1.000 | 0.943 |
| D English + semantic | 0.938 | 1.000 | 1.000 | 0.969 |

Faithful translation закрыла два misses для русско- и польскоязычного вопроса о
MCP tools. Semantic rewrite добавлять в runtime оснований нет: D не лучше B, а C
понизил MRR для запроса про scoped Cursor rules.

### Langfuse

В Langfuse опубликованы и помечены `production` три chat prompt:

- `doc-bot/answer` v1
- `doc-bot/translate` v1
- `doc-bot/evidence-planner` v1

Локальные шаблоны и опубликованные v1 нормализованно совпадают. Runtime выбирает
`production`, а для reproducible evaluation поддерживает pin версии через
environment variables. При недоступности Langfuse production request использует
локальный fallback; evaluation может включить `PROMPT_STRICT=true`.

Каждый `ask` создаёт root trace и observations для retrieval, rerank и generation;
generation передаёт prompt name/version в trace. Успешный end-to-end run был
проверен на вопросе об MCP server tools: root `answer-question`, `hybrid-search`,
`rerank-candidates`, `generate-answer`, prompt `doc-bot/answer` v1 и model usage
видны в Langfuse. Ошибки в traced stage получают уровень `ERROR` и остаются по тому
же `TRACE` URL.

Официальные Langfuse docs MCP и project-scoped data MCP добавлены в пользовательский
Codex config. Data endpoint настроен с read-only allowlist; оба endpoint прошли
JSON-RPC `initialize`.

## Проверки

В текущем наборе есть unit tests для routing, empty retrieval, prompt fallback/
versioning, stage tracing, error status, двухраундового control flow и offline
rescore. Они не подтверждают качество ответов, latency или стоимость на реальных
пользовательских вопросах.

### Evidence coverage pilot

Создан Langfuse dataset `rag/evidence-coverage-v1`: 4 вручную проверенных
multi-aspect English cases, где итоговый top-5 обязан покрыть каждую независимую
evidence group. Результаты зафиксированы в `reports/evidence-coverage-v1.md`.

Baseline и bounded iterative получили одинаковые значения: evidence coverage 0.833,
complete coverage 0.500. В Cursor case planner сформировал корректный follow-up о
scoped rule, но финальный rerank вытеснил этот result общими MCP/subagent chunks.
Значит, текущий bottleneck — allocation ограниченного generation context между
аспектами вопроса; включать iterative default нельзя.

Две реализации group-aware reservation проверены и отклонены. Global reservation
снизил coverage до 0.750, а topical rerank reservation — до 0.625: follow-up без
product scope мог выбрать документацию другого вендора. Обе реализации удалены из
runtime; Langfuse runs и разбор сохранены в `reports/evidence-coverage-v1.md`.

Проверена и отклонена грубая source-level cap: она снизила coverage до 0.792,
потому что отсекает разные headings одной важной страницы. В runtime оставлен
section-level cap — максимум два chunks для одинакового source + heading. Он убирает
повтор одного фрагмента без изменения measured coverage, но не считается retrieval
improvement.

## Следующие эксперименты

Контрольный vendor-scope experiment уже выполнен на 8 clean-English items:
обе ветки получили H@1/H@5/H@10/MRR = 1.0. Это нейтральный результат на небольшом
наборе с потолком метрик; он не проверяет пользу scope для multi-aspect follow-ups.
Vendor filter не включён в runtime автоматически.

Добавлен answer-quality pilot: собственный LLM judge с prompt
`doc-bot/answer-evaluator` v1 и четырьмя измерениями — faithfulness, relevance,
context utilization и citations. Он не эквивалентен реализации Ragas.
Проверка `uv pip install --dry-run ragas` предложила downgrade OpenAI 3.13 → 3.3;
зависимости приложения не менялись. Это результат конкретного dependency resolution,
а не доказательство невозможности любой совместимой установки Ragas.

1. Расширить `rag/evidence-coverage-v1` до 20+ cases: comparison, negative,
   ambiguous и multi-aspect вопросы для Claude, Cursor, Codex и MCP. Для каждого
   item оставлять independent evidence groups, canonical URLs/headings и scope.
2. Отдельно проверить product-scoped follow-up retrieval: применять Qdrant vendor
   filter только когда исходный вопрос однозначно называет один продукт. Считать
   coverage, citation precision, answer faithfulness, p50/p95 latency, tokens и
   стоимость и сравнивать с baseline.
3. Принять решение по default iterative только если gain покрывает дополнительную
   latency/cost. Не включать HyDE, GraphRAG или agent loops, пока набор не покажет
   конкретный miss, который они исправляют.
4. Сделать ingestion инкрементальным: document revision hash, stable chunk IDs,
   manifest corpus version и удаление stale chunks только после успешной загрузки.
5. Перед настройкой HNSW/embeddings измерить exact-vs-ANN recall на golden dataset.
   Это отделит проблему индекса от проблем query/document representation.
