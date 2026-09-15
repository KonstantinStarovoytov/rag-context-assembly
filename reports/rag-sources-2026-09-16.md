# RAG architecture — authoritative sources, ranked (2026-09-16)

Ranked by normativeness (primary research vs vendor blog), empirical rigour,
and applicability to this project (hybrid BM25+dense, Cohere rerank, 35-page
documentation corpus).

| # | Source | Type | Architecture guidance | Relevance here | Rank |
|---|---|---|---|---|---|
| 1 | Anthropic, Contextual Retrieval (2024) — https://www.anthropic.com/engineering/contextual-retrieval | vendor, measured | Hybrid embeddings+BM25 cuts top-20 retrieval failures 49 %; + rerank 67 % (5.7→1.9 %). Prepend document context to every chunk before embedding/BM25 | Exactly this stack; contextual chunk prefix is not implemented yet | 5 |
| 2 | Wang et al., Searching for Best Practices in RAG (2024) — https://arxiv.org/abs/2407.01219 | peer-reviewed, empirical | Component-wise search: hybrid search, rerank, "reverse" repacking (most relevant nearest the question), summarisation | Confirms hybrid+rerank; reverse repacking is a free change in context_selector | 5 |
| 3 | Microsoft, Design and develop a RAG solution (Azure Architecture Center) — https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide | vendor, methodology | End-to-end phases: chunking, enrichment (title/summary/keywords metadata), evaluation | Checklist; enrichment phase is missing here | 5 |
| 4 | Gao et al., RAG for LLMs: A Survey (2023/24) — https://arxiv.org/abs/2312.10997 | academic survey | Naive/Advanced/Modular taxonomy; map of techniques | Vocabulary and map | 4 |
| 5 | Chroma, Context Rot (2025) — https://www.trychroma.com/research/context-rot | vendor research, 18 models | Accuracy degrades well below the window; distractors hurt more than noise | Argues for generation_top_k=8 and concise search results | 4 |
| 6 | Databricks, Long Context RAG Performance (2024) — https://arxiv.org/pdf/2411.03538 | vendor, 2000+ runs | More documents help up to 16–32k tokens, then most models degrade | Upper bound for limit and context size | 4 |
| 7 | Chroma, Evaluating Chunking Strategies (2024) — https://www.trychroma.com/research/evaluating-chunking | vendor research | Token-level recall/precision/IoU; recursive splitter ~200 tokens no overlap is robust; large chunks cut precision | Chunks here are 3000 chars; test 1000–1200 on evals | 4 |
| 8 | Anthropic, Effective context engineering for AI agents (2025) — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents | vendor | Just-in-time retrieval through tools; progressive disclosure | Justifies tools over resources; concise→detailed | 4 |
| 9 | Liu et al., Lost in the Middle (2023) — https://arxiv.org/abs/2307.03172 | peer-reviewed | U-shaped use of context position | Basis for chunk ordering | 4 |
| 10 | RAGAS (2023) — https://arxiv.org/abs/2309.15217 ; https://docs.ragas.io | peer-reviewed + framework | Faithfulness, answer relevance, context precision/recall | Already used in evals/ragas | 4 |
| 11 | OpenAI, Optimizing LLM Accuracy — https://platform.openai.com/docs/guides/optimizing-llm-accuracy | vendor guide | Prompting vs RAG vs fine-tuning; retrieval-vs-generation failure diagnosis | Frame for eval failure analysis | 3 |
| 12 | Lewis et al., RAG (2020) — https://arxiv.org/abs/2005.11401 | peer-reviewed, origin | Original retriever+generator | Historical | 3 |
| 13 | Microsoft, GraphRAG (2024) — https://arxiv.org/abs/2404.16130 | vendor research | Entity graph + community summaries for global questions | Not applicable at 35 pages | 2 |
| 14 | LlamaIndex / LangChain docs (production RAG, advanced retrieval) | framework docs | Catalogue: sentence-window, parent-document, multi-query | Parent-document retrieval as a "whole section" option | 2 |
| 15 | Pinecone / Weaviate / Qdrant learning centers | vendor marketing | Chunking, hybrid, RRF explainers | Product docs, not research | 2 |

## Applicable next steps, by expected effect
1. Contextual chunk prefix (#1): include vendor/product/title/heading in the embedded and BM25 text.
2. Chunk size (#7): evaluate 1000–1200 chars vs current 3000/300 on existing evals.
3. Reverse repacking (#2, #9): most relevant chunks nearest the question.
4. Enrichment metadata (#3): generated keywords/summary per chunk for BM25; measure.
5. Keep context small (#5, #6): generation_top_k=8 and limit≤20 stay.
