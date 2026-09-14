# Evidence coverage pilot — 2026-09-13

## Зачем

Обычный Hit@k засчитывает один релевантный chunk. Для вопроса с несколькими
независимыми аспектами этого недостаточно: финальный top-5 должен покрыть каждый
аспект, который нужен для ответа.

Dataset `rag/evidence-coverage-v1` создан в Langfuse. В нём четыре вручную
проверенных English cases:

1. Cursor: MCP, scoped Python rule и subagent.
2. Claude Code plugin: packaging, feature composition и subagents.
3. MCP server: tools discovery/updates и authorization.
4. Codex: `AGENTS.md` и remote MCP server.

Каждый item задаёт независимые evidence groups через vendor, canonical source URL
и, когда нужно, heading. Метрика `evidence_coverage` — доля покрытых groups;
`complete_evidence_coverage` — 1 только если покрыты все группы.

## Результаты

| Run | Evidence coverage | Complete coverage | Вывод |
|---|---:|---:|---|
| [baseline v1](https://cloud.langfuse.com/project/cmtogb8i90jw1ad0i9i6g2tae/datasets/cmtzvjpmm0lllad0clueolszs/runs/0da8090b-b671-40cc-aecd-b61dcb278c43) | 0.833 | 0.500 | Single-pass hybrid + Cohere |
| [iterative v1](https://cloud.langfuse.com/project/cmtogb8i90jw1ad0i9i6g2tae/datasets/cmtzvjpmm0lllad0clueolszs/runs/e10131c3-c4b3-459b-9c1b-cdf9e9b44151) | 0.833 | 0.500 | Два раунда не дали прироста |
| source-cap v2 | 0.792 | 0.500 | Отклонён: cap по URL вытеснил нужный heading той же страницы |
| [section-cap baseline v3](https://cloud.langfuse.com/project/cmtogb8i90jw1ad0i9i6g2tae/datasets/cmtzvjpmm0lllad0clueolszs/runs/3af30ce5-3886-4c9a-a7e1-a3c196732e6e) | 0.833 | 0.500 | Повтор одного section ограничен без потери coverage |
| [section-cap iterative v3](https://cloud.langfuse.com/project/cmtogb8i90jw1ad0i9i6g2tae/datasets/cmtzvjpmm0lllad0clueolszs/runs/95f07175-a645-4671-8fba-5b474ddc43e8) | 0.833 | 0.500 | Прироста от iterative нет |
| [global group reservation v4](https://cloud.langfuse.com/project/cmtogb8i90jw1ad0i9i6g2tae/datasets/cmtzvjpmm0lllad0clueolszs/runs/938e7cb0-0232-429b-a103-c29a4df962cc) | 0.750 | 0.500 | Отклонён: global rerank выбрал общий документ вместо topic evidence |
| [topic rerank reservation v5](https://cloud.langfuse.com/project/cmtogb8i90jw1ad0i9i6g2tae/datasets/cmtzvjpmm0lllad0clueolszs/runs/9c8bb312-ece6-42ac-8779-693a4b49f7ac) | 0.625 | 0.250 | Отклонён: follow-up без product scope вернул документ другого вендора |

В section-cap runs покрыты полностью MCP и Codex cases. Cursor case стабильно
теряет evidence group `cursor-python-rule-glob`; Claude plugin case теряет
выделенный `sub-agents.md` group. В Cursor case planner сформировал хороший
follow-up о project rules и выполнил второй round, но финальный rerank вновь
выбрал более общие MCP/subagent chunks. Это локализует проблему: не query rewrite,
а allocation финальных пяти мест между независимыми аспектами.

Проверены две формы group-aware reservation. В v4 для каждого follow-up сохранялся
лучший document по global original-question rerank; это понизило coverage до 0.750,
так как общий документ мог оказаться выше документа нужной темы. В v5 follow-up
rerank-ился по своей теме, но coverage упал до 0.625: например, Cursor MCP
follow-up выбрал Claude MCP page, а MCP authorization follow-up не вернул
authorization page. В runtime обе реализации не включены.

## Решение

- Default остаётся `hybrid` с faithful translation для RU/PL.
- Semantic rewrite остаётся выключенным: A/B/C/D уже показал, что он не лучше
  translation-only champion.
- `--iterative` остаётся opt-in. Для 4-case pilot он дороже, но не лучше baseline.
- В generation context остался section-level cap: максимум два chunks с одинаковыми
  source + heading. Он предотвращает повторы одного фрагмента и не ухудшил measured
  coverage. Это presentation/context quality guard, а не доказанное retrieval gain.

Следующий осмысленный эксперимент — не ещё один rewrite. Нужно увеличить dataset
до 20+ manually reviewed multi-aspect cases. После этого можно отдельно проверить
product-scoped follow-up retrieval: применять metadata vendor filter только когда
исходный вопрос однозначно называет один продукт. Он должен сравниваться с baseline
по coverage, citation precision, latency, tokens и стоимости; без этого не включать
его в runtime.
