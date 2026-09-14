"""Retrieval policy derived from offline A/B/C/D rescore.

Dataset: rag/retrieval-query-transform-v1, 48 paired cases.
A original-only: H@5=0.958 MRR=0.927
B original+faithful English: H@5=1.000 MRR=0.969
C original+semantic: H@5=1.000 MRR=0.943, one natural-English degradation
D English+semantic: identical to B, extra cost

Remaining misses after B at k=5: 0/48. The only original-only misses
were Russian and Polish mcp-server-tools queries, recovered by translation.
"""

ENABLE_SEMANTIC_REWRITE = False
ENABLE_ITERATIVE_BY_DEFAULT = False


def should_rewrite_after_miss() -> bool:
    return False
