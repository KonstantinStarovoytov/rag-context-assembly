# Vendor scope control — 2026-09-13

Dataset: `rag/retrieval-vendor-scope-clean-en-v1`, 8 clean-English questions.
Same hybrid retrieval top-10 and Cohere reranker in both arms.

| Arm | H@1 | H@5 | H@10 | MRR |
|---|---:|---:|---:|---:|
| Unscoped | 1.0 | 1.0 | 1.0 | 1.0 |
| Explicit single vendor | 1.0 | 1.0 | 1.0 | 1.0 |

[Unscoped run](https://cloud.langfuse.com/project/cmtogb8i90jw1ad0i9i6g2tae/datasets/cmtzwn4ud0m85ad0dfdgmb721/runs/6b54fb5c-07e1-4e9a-b8b9-f42e63268309)
and [scoped run](https://cloud.langfuse.com/project/cmtogb8i90jw1ad0i9i6g2tae/datasets/cmtzwn4ud0m85ad0dfdgmb721/runs/92977104-3f69-49d9-850e-2d1717faa127).

No improvement measured. The small control dataset has ceiling scores; it cannot
establish whether vendor filtering helps complex follow-up queries. Automatic
scope detection remains outside the runtime pipeline.
