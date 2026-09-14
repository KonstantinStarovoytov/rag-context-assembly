import re
from urllib.parse import urlsplit, urlunsplit

import httpx

from src.ingestion.sources import SOURCES, SourceConfig
from src.models import SourceDocument

MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)]\((https?://[^)\s]+)\)")

BARE_URL_RE = re.compile(r"https?://[^\s)>]+")


def _normalize_url(url: str) -> str:
    parts = urlsplit(url)

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            parts.query,
            "",  # remove #fragment
        )
    )


def _fallback_title(url: str) -> str:
    path = urlsplit(url).path

    name = path.rsplit("/", 1)[-1]
    name = name.removesuffix(".md")

    return name.replace("-", " ").replace("_", " ").strip().title()


def _extract_links(
    index_content: str,
) -> list[tuple[str, str]]:
    links: dict[str, str] = {}

    # Normal Markdown links:
    # [Skills](https://.../skills.md)
    for title, url in MARKDOWN_LINK_RE.findall(index_content):
        normalized = _normalize_url(url)

        links[normalized] = title.strip()

    # Cursor llms.txt also contains plain URLs,
    # so support those as well.
    for url in BARE_URL_RE.findall(index_content):
        normalized = _normalize_url(url)

        links.setdefault(
            normalized,
            _fallback_title(normalized),
        )

    return [(title, url) for url, title in links.items()]


def _should_include(
    url: str,
    config: SourceConfig,
) -> bool:
    return any(pattern in url for pattern in config.include)


def load_source(
    config: SourceConfig,
) -> list[SourceDocument]:
    documents: list[SourceDocument] = []

    with httpx.Client(
        timeout=30,
        follow_redirects=True,
    ) as client:
        index_response = client.get(config.index_url)
        index_response.raise_for_status()

        links = _extract_links(index_response.text)

        selected_links = [
            (title, url)
            for title, url in links
            if _should_include(
                url,
                config,
            )
        ]

        print(f"{config.vendor}/{config.product}: {len(selected_links)} pages")

        for title, url in selected_links:
            response = client.get(url)
            response.raise_for_status()

            documents.append(
                SourceDocument(
                    title=title,
                    content=response.text,
                    url=url,
                    vendor=config.vendor,
                    product=config.product,
                )
            )

    return documents


def load_all_sources() -> list[SourceDocument]:
    documents: list[SourceDocument] = []

    for config in SOURCES:
        source_documents = load_source(config)

        documents.extend(source_documents)

    return documents


if __name__ == "__main__":
    documents = load_all_sources()

    for document in documents:
        print(
            document.vendor,
            document.product,
            document.title,
            document.url,
        )
