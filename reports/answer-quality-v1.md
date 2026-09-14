# Answer quality pilot — 2026-09-13

[Langfuse baseline run](https://cloud.langfuse.com/project/cmtogb8i90jw1ad0i9i6g2tae/datasets/cmtzvjpmm0lllad0clueolszs/runs/e932ed57-4904-4eb8-91c6-7755317cfb8e)

Four English multi-aspect questions from `rag/evidence-coverage-v1`. Hybrid top-10,
Cohere rerank, section-diverse top-5, answer prompt v1, evaluator prompt v1.
Both answer generation and the custom judge use `gpt-5.6-luna`.
Only baseline was run to limit API usage. No runtime strategy was promoted.

| Metric | Mean |
|---|---:|
| Evidence coverage | 0.833 |
| Complete evidence coverage | 0.500 |
| Judge faithfulness | 0.9525 |
| Judge answer relevance | 0.975 |
| Judge context utilization | 0.9325 |
| Judge citation correctness | 0.965 |

The four trace outputs and score comments were retrieved after completion.
The judge flags these potential issues:

- Cursor: the answer abstains on Python-only rule syntax; it also mentions an MCP
  configuration flow that the supplied context does not explain.
- Claude plugin: task-specific examples are extrapolations from documented
  capabilities. Some may be valid recommendations rather than factual errors;
  the rubric needs to distinguish these before prompt optimization.
- MCP: selected authorization evidence describes a tutorial, not all implementation
  mechanics. The answer acknowledges missing details. The URL-based coverage
  metric nevertheless scores this case complete: source matching is too coarse
  to prove answer sufficiency.
- Codex: the judge flags repository-root placement as a stronger requirement than
  the selected documentation states; one selected context is unused.

These are diagnostic flags from an uncalibrated judge, not independently verified
factual errors. High relevance scores despite acknowledged gaps show why the
judge cannot be the sole promotion gate. Scores are custom rubric ratings, not
Ragas metrics, and temperature zero does not make model evaluation deterministic.

Next bounded experiment: add explicit required-answer facts and acceptable
abstention criteria to the dataset, review the broad MCP authorization target,
and calibrate the judge on complete, incomplete, and unsupported answers. Then
compare context selection using the same frozen candidates and prompts. Reuse
stored answers for evaluator changes to avoid paying for retrieval/generation again.
