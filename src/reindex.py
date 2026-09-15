"""Incremental re-index: fetch every source page, re-embed only what changed.

Runs from a daily GitHub Actions cron. The manifest under `data/` remembers a
fingerprint per page, so a run where nothing changed costs a few HTTP requests
and no embeddings. Pages that vanish from a vendor's index are reported, not
deleted: a rename looks the same as a removal, and that is a human decision.
"""

import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from qdrant_client import models

from src.config import settings
from src.ingestion.chunker import chunk_document
from src.ingestion.ingest import get_hybrid_vector_store, get_qdrant_client
from src.ingestion.loader import load_all_sources
from src.models import SourceDocument
from src.service.core import META_COLLECTION, META_POINT_ID

MANIFEST_PATH = Path("data/index-manifest.json")
SMOKE_K = 10


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    sha256: str
    chunks: int
    title: str


@dataclass(frozen=True, slots=True)
class Manifest:
    indexed_at: str | None
    documents: dict[str, ManifestEntry] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Plan:
    added: list[SourceDocument]
    changed: list[SourceDocument]
    unchanged: list[str]
    missing: list[str]

    @property
    def to_index(self) -> list[SourceDocument]:
        return [*self.added, *self.changed]


def fingerprint(content: str) -> str:
    """Hash of the markdown with cosmetic differences removed.

    CRLF and trailing spaces change between CDN nodes and deploys; they must
    not trigger a re-embed.
    """
    lines = (line.rstrip() for line in content.replace("\r\n", "\n").split("\n"))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> Manifest:
    if not path.exists():
        return Manifest(indexed_at=None, documents={})
    raw = json.loads(path.read_text())
    return Manifest(
        indexed_at=raw.get("indexed_at"),
        documents={
            url: ManifestEntry(**entry) for url, entry in raw["documents"].items()
        },
    )


def save_manifest(manifest: Manifest, path: Path = MANIFEST_PATH) -> None:
    payload = {
        "indexed_at": manifest.indexed_at,
        "documents": {
            url: asdict(entry) for url, entry in sorted(manifest.documents.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def diff(manifest: Manifest, documents: list[SourceDocument]) -> Plan:
    added: list[SourceDocument] = []
    changed: list[SourceDocument] = []
    unchanged: list[str] = []
    seen: set[str] = set()
    for document in documents:
        seen.add(document.url)
        entry = manifest.documents.get(document.url)
        if entry is None:
            added.append(document)
        elif entry.sha256 != fingerprint(document.content):
            changed.append(document)
        else:
            unchanged.append(document.url)
    missing = sorted(url for url in manifest.documents if url not in seen)
    return Plan(added=added, changed=changed, unchanged=unchanged, missing=missing)


def apply(
    plan: Plan,
    manifest: Manifest,
    *,
    delete_source: Callable[[str], None],
    add_chunks: Callable[[list[Document]], None],
    today: date,
) -> Manifest:
    """Replace the chunks of every added or changed page; leave the rest alone."""
    documents = dict(manifest.documents)
    for document in plan.to_index:
        chunks = chunk_document(document)
        delete_source(document.url)
        add_chunks(chunks)
        documents[document.url] = ManifestEntry(
            sha256=fingerprint(document.content),
            chunks=len(chunks),
            title=document.title,
        )
    indexed_at = today.isoformat() if plan.to_index else manifest.indexed_at
    return Manifest(indexed_at=indexed_at, documents=documents)


def smoke_failures(
    intents: list[dict[str, Any]],
    search: Callable[[str], list[dict[str, str]]],
) -> list[str]:
    """Intent ids whose expected page is not among the top hits."""
    failures: list[str] = []
    for intent in intents:
        hits = search(intent["variants"]["clean_en"])
        for expected in intent["relevant"]:
            if not any(
                hit["vendor"] == expected["vendor"]
                and expected["source_contains"] in hit["source"]
                for hit in hits
            ):
                failures.append(intent["id"])
                break
    return failures


# --- wiring against the real index ------------------------------------------


def _delete_source(url: str) -> None:
    client = get_qdrant_client()
    try:
        client.delete(
            collection_name=settings.qdrant_hybrid_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source", match=models.MatchValue(value=url)
                        )
                    ]
                )
            ),
        )
    finally:
        client.close()


def _add_chunks(chunks: list[Document]) -> None:
    get_hybrid_vector_store().add_documents(chunks)


def write_meta(manifest: Manifest) -> None:
    """One point the service reads to report when the index was built."""
    client = get_qdrant_client()
    try:
        if not client.collection_exists(META_COLLECTION):
            client.create_collection(
                META_COLLECTION,
                vectors_config=models.VectorParams(
                    size=1, distance=models.Distance.DOT
                ),
            )
        client.upsert(
            META_COLLECTION,
            points=[
                models.PointStruct(
                    id=META_POINT_ID,
                    vector=[0.0],
                    payload={
                        "indexed_at": manifest.indexed_at,
                        "document_count": len(manifest.documents),
                        "written_at": datetime.now(UTC).isoformat(),
                    },
                )
            ],
        )
    finally:
        client.close()


def _search(query: str) -> list[dict[str, str]]:
    from src.rag.retriever import search_hybrid

    return [
        {
            "vendor": str(r.document.metadata.get("vendor", "")),
            "source": str(r.document.metadata.get("source", "")),
        }
        for r in search_hybrid(query=query, k=SMOKE_K)
    ]


def main() -> int:
    from evals.seed_query_transform_dataset import INTENTS

    manifest = load_manifest()
    documents = load_all_sources()
    plan = diff(manifest, documents)
    print(
        f"added={len(plan.added)} changed={len(plan.changed)} "
        f"unchanged={len(plan.unchanged)} missing={len(plan.missing)}"
    )
    for document in plan.to_index:
        print(f"  reindex {document.url}")

    if plan.to_index:
        manifest = apply(
            plan,
            manifest,
            delete_source=_delete_source,
            add_chunks=_add_chunks,
            today=datetime.now(UTC).date(),
        )
        write_meta(manifest)
        save_manifest(manifest)
        failures = smoke_failures(INTENTS, _search)
        if failures:
            print(f"SMOKE FAILED: expected page missing from top hits for {failures}")
            return 1

    if plan.missing:
        print("MISSING from vendor index (not deleted; decide by hand):")
        for url in plan.missing:
            print(f"  {url}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
