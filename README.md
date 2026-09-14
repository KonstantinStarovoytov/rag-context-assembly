# rag-context-assembly

A hybrid RAG system over the official Claude, Cursor, Codex and MCP documentation,
built together with the evaluation harness needed to tell whether each change actually
helped.

Runtime path: `query → (translation, only for RU/PL) → hybrid dense + BM25 with RRF →
Cohere rerank → section-level de-duplication → grounded generation with citations`.

The interesting part of the project is not the pipeline but what measuring it showed:
**for multi-aspect questions the bottleneck is context assembly, not retrieval.** Among
20 reranked candidates the required evidence is present 0.948 of the time; after
selecting the final context it is present 0.646 of the time. Full analysis, with the
experiments behind every number, is in [`docs/rag-findings.md`](docs/rag-findings.md).

## Results

| Finding | Measurement |
| --- | --- |
| Coverage is lost in selection, not retrieval | 0.948 in candidate pool → 0.646 in context |
| `generation_top_k` 5 → 8 | complete evidence coverage 0.250 → 0.750, faithfulness 0.938 → 0.956 |
| A larger candidate pool can hurt | pool 10 → 20 raises the oracle ceiling 0.812 → 0.875 but drops delivered coverage 0.750 → 0.500 |
| Bounded two-hop retrieval | no gain on 16 multi-aspect cases; the right remedy for 2 of them, so it stays opt-in |
| Aspect-aware set selection (offline) | complete coverage 0.500 → 0.688 on a frozen candidate pool |
| Hybrid RRF was non-deterministic | tied scores made the cut at `k` arbitrary; fixed with a stable tie-break key |

Caveats belong with the numbers: N = 16 with author-written ground truth, answer quality
is scored by an LLM judge, and the generator is not reproducible even at temperature 0.
Section 9 of the findings document lists the gaps in full.

## Setup

Install dependencies with `uv sync --locked` and fill `.env` from `.env.example`.
OpenAI and Cohere credentials plus a populated Qdrant collection are required.
Langfuse is only needed for `TRACING_ENABLED=true` or remote prompt fetch.

```bash
docker compose up -d
uv run python -m src.ask "How do skills work?"
uv run python -m src.ask "Как работают skills?" --strategy hybrid-english
uv run python -m src.ask "How do skills work?" --strategy dense
```

Rebuilding the hybrid collection requires an explicit flag so the index cannot be
dropped by accident:

```bash
uv run python -m src.index_hybrid --recreate
```

This is still a full rebuild, not incremental synchronisation.

## Configuration

Set in `.env`: `RETRIEVAL_STRATEGY=hybrid`, `TRANSLATE_NON_ENGLISH=true`,
`RETRIEVAL_TOP_K=10`, `PER_QUERY_TOP_K=20`, `GENERATION_TOP_K=8`.
The CLI strategy overrides `.env`.

- `hybrid` translates the query to English only when it contains Cyrillic or Polish
  characters. `hybrid-english` always adds a faithful translation, for ablations.
  Semantic paraphrase is not used at runtime: an offline A/B/C/D over 48 cases showed
  faithful English lifts Hit@5 from 0.958 to 1.000 while rewriting breaks an English case.
- Before generation, one URL plus heading may occupy at most two context slots; if no
  other sections exist, the top-k is refilled from the original rerank order.
- `GENERATION_TOP_K=8` is a measured default, not a guess — see the results table.

## Two-hop retrieval (opt-in)

```bash
uv run python -m src.ask "How do I structure a plugin for Claude and Cursor?" --iterative
```

Hybrid search and rerank run first, then an evidence planner judges the same top-k the
generator would receive. If evidence is insufficient it names structured gaps
(`category`, `target`, `slot`) and the follow-up queries are built deterministically from
them in Python, rather than written by the model. At most one extra round runs; the
second judgement cannot trigger a third. Candidates are merged by content identity and
reranked against the original question.

The loop is plain Python control flow, no LangGraph. It is off by default because it did
not improve coverage: it retrieves documents that were already candidates, and the final
rerank ranks them below the first aspect's chunks again.

## Observability and prompts

`TRACING_ENABLED=true` turns on Langfuse root and stage observations plus LangChain
callbacks; the CLI flushes before exit and prints `TRACE: <url>` even on failure.
Three chat prompts (`answer`, `translate`, `evidence-planner`) are managed in Langfuse
under the `production` label with a local fallback. For reproducible evaluation, pin
versions via `ANSWER_PROMPT_VERSION`, `TRANSLATE_PROMPT_VERSION`,
`EVIDENCE_PLANNER_PROMPT_VERSION` and set `PROMPT_STRICT=true`.

```bash
uv run python -m src.prompts.publish
```

## Layout

- `src/ingestion/` — loading, chunking, index population.
- `src/rag/` — retrieval, fusion, rerank, orchestration, generation.
- `src/prompts/` — local fallback templates and the versioned Langfuse prompt adapter.
- `src/ask.py` — the main CLI; `search*` and `compare_rerank` are diagnostics.
- `evals/` — datasets, experiment runners and offline probes.
- `tests/` — unit tests, no external API calls.
- `docs/rag-findings.md` — the experiment log and architectural conclusions.
- `docs/engineering-audit.md` — verified state, remaining gaps, planned experiments.

## Evaluation

Answer quality over multi-aspect questions, storing context, answer and scores in
Langfuse:

```bash
ANSWER_PROMPT_VERSION=1 ANSWER_EVALUATOR_PROMPT_VERSION=1 PROMPT_STRICT=true \
  uv run python -m evals.run_answer_quality_experiment
```

Only the baseline runs by default to save API calls; add `--compare-iterative` to
compare. Retrieval coverage over the multi-aspect dataset:

```bash
uv run python -m evals.run_evidence_coverage_experiment
```

Selection strategies can be compared offline against a frozen candidate pool, which
avoids paying for retrieval and reranking on every iteration — see
[`evals/FROZEN_SELECTION.md`](evals/FROZEN_SELECTION.md).

The judge is a custom LLM evaluator, not a Ragas metric; its scores need review and do
not by themselves prove an answer is complete.

```bash
uv run python -m pytest tests -q
uv run ruff check src evals tests
uv run ruff format --check src evals tests
uv run mypy
```

Ruff is the flake8 check (`E`/`W`/`F`). mypy is strict on `src/`; eval scripts stay measurement code and are not type-checked.

Unit tests cover routing, the original query reaching Cohere, the absence of semantic
variants in the selected retrieval path, empty results, Hit/RR consistency and the
deterministic RRF tie-break. They do not prove retrieval improved; dataset experiments
do that.
