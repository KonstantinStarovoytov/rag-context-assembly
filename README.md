# rag-context-assembly

A hybrid RAG system over the official Claude, Cursor, Codex and MCP documentation,
built together with the evaluation harness needed to tell whether each change actually
helped.

Runtime path: `query → (translation, only for RU/PL) → hybrid dense + BM25 with RRF → Cohere rerank → section-level de-duplication → grounded generation with citations`.

The interesting part of the project is not the pipeline but what measuring it showed:
**for multi-aspect questions the bottleneck is context assembly, not retrieval.** Among
20 reranked candidates the required evidence is present 0.948 of the time; after
selecting the final context it is present 0.646 of the time. Full analysis, with the
experiments behind every number, is in `[docs/rag-findings.md](docs/rag-findings.md)`.

## Results


| Finding                                      | Measurement                                                                                     |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Coverage is lost in selection, not retrieval | 0.948 in candidate pool → 0.646 in context                                                      |
| `generation_top_k` 5 → 8                     | complete evidence coverage 0.250 → 0.750, faithfulness 0.938 → 0.956                            |
| A larger candidate pool can hurt             | pool 10 → 20 raises the oracle ceiling 0.812 → 0.875 but drops delivered coverage 0.750 → 0.500 |
| Bounded two-hop retrieval                    | no gain on 16 multi-aspect cases; the right remedy for 2 of them, so it stays opt-in            |
| Aspect-aware set selection (offline)         | complete coverage 0.500 → 0.688 on a frozen candidate pool                                      |
| Hybrid RRF was non-deterministic             | tied scores made the cut at `k` arbitrary; fixed with a stable tie-break key                    |


Caveats belong with the numbers: N = 16 with author-written ground truth, answer quality
is scored by an LLM judge, and the generator is not reproducible even at temperature 0.
Section 9 of the findings document lists the gaps in full.

## Setup

Install dependencies with `uv sync --locked` and fill `.env` from `.env.example`.
OpenAI and Cohere credentials plus a populated Qdrant collection are required.
Langfuse is only needed for `TRACING_ENABLED=true` or remote prompt fetch, and
`API_TOKEN` only for serving over HTTP.

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

## Serving: HTTP endpoint and MCP server

The same pipeline is exposed two ways from one process, so a REST caller and an
MCP client cannot drift apart: both go through `src/service/core.py`.


| Path      | Method | Purpose                                                              |
| --------- | ------ | -------------------------------------------------------------------- |
| `/health` | GET    | Liveness. The only unauthenticated route; touches no paid model.     |
| `/ask`    | POST   | Grounded answer with citations and, when tracing is on, a trace URL. |
| `/search` | POST   | Retrieval only, for callers that want to read the passages.          |
| `/mcp`    | POST   | MCP streamable HTTP transport, tools `ask_docs` and `search_docs`.   |


`API_TOKEN` is required: the process refuses to start without it, because every
request spends OpenAI and Cohere credits. Send it as `Authorization: Bearer …`.
One shared token is a deliberate choice for a single-owner deployment; it works
with Claude Code, Cursor and the Claude API `authorization_token`, but not with
claude.ai custom connectors, which require OAuth. `API_RATE_LIMIT_PER_MINUTE`
(default 60) caps paid requests across the machine so a looping agent cannot
run up the bill; over the cap the API answers 429 with `Retry-After`.

The interactive docs at `/docs` are public and declare the bearer scheme, so
the **Authorize** button there takes the token and "Try it out" works.

**Want to try it?** An optional `API_GUEST_TOKEN` is a second token with the
same rights and the same rate limit, kept separate so it can be revoked (clear
the variable and redeploy) without rotating the owner's token. Ask me for one
and use it exactly like `API_TOKEN` below.

```bash
docker compose up -d --build     # Qdrant plus the API on :8080
curl localhost:8080/health
curl -X POST localhost:8080/ask \
  -H "Authorization: Bearer $API_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"question": "How do skills work?"}'
```



### Connecting as MCP

Remote, against the deployed URL — this is the same server the `/ask` endpoint
uses, so answers are identical:

```json
{
  "mcpServers": {
    "agent-docs": {
      "url": "https://agent-docs-mcp.onrender.com/mcp/",
      "headers": { "Authorization": "Bearer <API_TOKEN>" }
    }
  }
}
```

Local, over stdio, with no HTTP server and no token:

```json
{
  "mcpServers": {
    "agent-docs": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/rag-context-assembly", "rag-mcp"]
    }
  }
}
```

`ask_docs(question)` returns the answer with its sources.
`search_docs(query, limit=8, product=None, response_format="concise")` reranks
with Cohere (like ask_docs — it used to return raw hybrid order, whose tail
past the first few results is mostly RRF ties) and returns ranked passages
without generation, which is cheaper when the calling agent wants to reason
over the documentation itself. `concise` trims each passage to
300 characters so a first look costs little context; `detailed` returns full
chunks. `product` scopes the search to `claude-code`, `cursor`, `codex` or
`mcp`. Both results carry `index_snapshot`, the date the re-index job last
wrote to Qdrant (falling back to the `INDEX_SNAPSHOT` variable), so callers
know they are reading a snapshot rather than live docs. The iterative second retrieval round stays on the CLI and
REST surface only: it was measured as no better than one round, and the calling
agent is already its own retry loop.

Errors reach the model with their reason (`Question must not be empty`, or
"backend temporarily unavailable … retry") rather than a bare tool failure, so
it can correct the call instead of guessing.

## Keeping the index fresh

`.github/workflows/reindex.yml` runs `src/reindex.py` every morning (and on
demand from the Actions tab). It fetches every source page, fingerprints the
markdown, and re-embeds only the pages whose content changed; a day without
changes costs a few HTTP requests and no embeddings. The fingerprints live in
[data/index-manifest.json](data/index-manifest.json), so `git log` on that
file is the history of what changed when. After a change the job records the
build date in a one-point Qdrant collection (`agent_docs_meta`), which the
service reports as `index_snapshot` without a redeploy, and runs a retrieval
smoke check over the query-transform intents; a miss fails the run.

Two things it deliberately does not do. It never deletes a page that vanished
from a vendor's `llms.txt`: a rename looks identical to a removal, so the run
fails and opens a GitHub issue instead. And it does not rebuild from scratch;
`uv run python -m src.index_hybrid --recreate` stays the manual escape hatch.

Which pages are indexed is decided by `src/ingestion/sources.py`: each vendor
has an `llms.txt` and a list of URL suffixes, and every suffix must match
exactly one page or the loader refuses to run, so a vendor's rename is loud.
The MCP docs are published per spec version, so that source names a
`versioned_prefix` and the loader picks the newest dated version itself. After
you change `sources.py`, run `uv run python -m src.reindex --prune` once to
drop the pages no config selects any more.

The workflow needs the repository secrets `OPENAI_API_KEY`, `COHERE_API_KEY`
(optionally `COHERE_API_KEY_FALLBACK`, used once the primary key hits its
rate/quota limit),
`QDRANT_URL` and `QDRANT_API_KEY`. The first run indexes everything (there is
no manifest yet) and takes a few minutes.

## Deploying to Render

The container is stateless; the index lives in Qdrant Cloud, whose free 1 GB
tier holds this corpus comfortably. A free Render web service runs the same
image (measured at ~390 MB of its 512 MB after a search), so the deployment
costs nothing and a restart cannot lose the index. [render.yaml](render.yaml)
is the whole service definition.

```bash
# 1. Point the local tooling at Qdrant Cloud and build the index there.
#    QDRANT_URL=https://<cluster>.qdrant.io:6333 and QDRANT_API_KEY=… in .env
uv run python -m src.index_hybrid --recreate
```

2. In the Render dashboard: **New → Blueprint**, pick this repository. Render
   reads `render.yaml`, creates the `agent-docs-mcp` service and asks for the
   secrets marked `sync: false`: `OPENAI_API_KEY`, `COHERE_API_KEY`
   (optionally `COHERE_API_KEY_FALLBACK`),
   `QDRANT_URL`, `QDRANT_API_KEY`, `LANGFUSE_PUBLIC_KEY`,
   `LANGFUSE_SECRET_KEY` (or set `TRACING_ENABLED=false`). `API_TOKEN` is
   generated; copy it from the service's Environment tab into your MCP client
   config.

3. Verify:

```bash
curl https://agent-docs-mcp.onrender.com/health
```

Deployment details worth knowing. Render deploys `main` only after the GitHub
CI workflow passes (`autoDeployTrigger: checksPass`). The image pre-downloads
the BM25 encoder so the first question does not pay for it. A free service
spins down after 15 minutes without traffic and takes about a minute to come
back, so the first call after a pause is slow; an MCP client just sees a slow
first tool call. Free instances get 750 hours a month per workspace, which a
service that sleeps between uses never exhausts. And `API_ALLOWED_HOSTS` must
name the public host: the MCP transport blocks DNS rebinding by rejecting
unknown `Host` headers with 421, so if you rename the service or add a custom
domain, update the variable; behind a domain only that domain is accepted,
and localhost is allowed only when the variable is empty.

The actions in CI are pinned by commit SHA and the base image by digest.

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
- `src/service/` — the shared request handler, the FastAPI app and the MCP server.
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
`[evals/FROZEN_SELECTION.md](evals/FROZEN_SELECTION.md)`.

The judge is a custom LLM evaluator, not a Ragas metric; its scores need review and do
not by themselves prove an answer is complete.

### Metrics that still move

Every retrieval dataset is at hit@5 = 1.0, so hit@k and MRR can no longer tell
two configurations apart. The retrieval runners therefore also report
`precision_at_5/8/10` (how much of the window is relevant) and `ndcg_at_10`
(whether the relevant chunks come first); precision@8 is the one to watch,
since 8 chunks reach the generator.

The judge labels each answer with a `failure_mode` (`none`, `refusal`,
`empty`, `repeated_content`, `fail_follow_inst` — the taxonomy from
Databricks' long-context RAG study), so an abstention no longer hides inside a
high faithfulness score. Set `OPENAI_JUDGE_MODEL` to a model other than the
generator; every run records `evaluator_independent` so a self-graded run is
never mistaken for an independent one.

`rag/negative-v1` ([evals/negative_cases.py](evals/negative_cases.py)) holds
questions the corpus cannot answer — Copilot, Windsurf, API pricing, general
Python. The right answer is an abstention, scored by `abstention_correct`:

```bash
uv run python -m evals.seed_negative_dataset          # once
uv run python -m evals.run_answer_quality_experiment --negative
```

The recall ceiling the selector works under, per candidate pool size, without
any LLM call:

```bash
uv run python -m evals.probe_context_assembly --curve 5 10 15 20 30
```

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