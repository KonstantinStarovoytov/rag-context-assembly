# Scheduled re-index — design (2026-09-15)

## Goal
Keep the Qdrant hybrid index in sync with the vendor docs without a human
running `index_hybrid --recreate`, and tell MCP callers when the index was
last built.

## Where it runs
GitHub Actions, daily cron + `workflow_dispatch`. Free, secrets already there,
uses the repo's own ingestion code. n8n would need hosting.

## Components
- `data/index-manifest.json` (committed): `{indexed_at, documents: {url: {sha256, chunks, title}}}`.
  Git history is the audit trail of what changed when.
- `src/reindex.py`:
  1. `load_all_sources()` fetches every page (36, seconds).
  2. `fingerprint(content)` = sha256 of normalised markdown (CRLF→LF, trailing
     whitespace stripped) so cosmetic server changes do not trigger work.
  3. `diff(manifest, documents)` → added, changed, unchanged, missing.
  4. Added/changed: delete points where `metadata.source == url`, chunk, add.
     Missing (page dropped from the vendor index): **not deleted**; reported and
     the run exits non-zero so a human decides.
  5. Writes `{indexed_at, document_count}` to Qdrant collection `agent_docs_meta`
     (single point) so the running service can show `index_snapshot` without a
     redeploy.
  6. Retrieval smoke check on `INTENTS` (clean_en variant, hybrid search k=10,
     hit must match vendor + source_contains). Any miss → non-zero exit.
  7. Rewrites the manifest.
- Service: `index_snapshot` comes from `agent_docs_meta` (cached 10 min), falling
  back to `INDEX_SNAPSHOT` env.
- `.github/workflows/reindex.yml`: run script; commit manifest with `[skip ci]`
  if it changed; on failure open/refresh a GitHub Issue with the log tail.

## Non-goals
Blue/green collection swap (per-document delete+insert is fine at this size);
deleting vanished pages automatically; generation-quality evals in the cron.

## Tests
Pure logic (fingerprint, diff, smoke evaluation) with fakes; apply step with an
injected fake store/client; service snapshot fallback.
