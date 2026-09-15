"""Which pages a source config picks out of a vendor's llms.txt."""

import pytest

from src.ingestion import loader
from src.ingestion.sources import SourceConfig


def _cfg(**overrides: object) -> SourceConfig:
    base: dict[str, object] = {
        "vendor": "anthropic",
        "product": "claude-code",
        "index_url": "https://x/llms.txt",
        "include": ("/docs/en/skills.md",),
    }
    base.update(overrides)
    return SourceConfig(**base)  # type: ignore[arg-type]


def test_patterns_match_the_end_of_the_url_only() -> None:
    links = [
        ("Skills", "https://code.claude.com/docs/en/skills.md"),
        ("SDK skills", "https://code.claude.com/docs/en/agent-sdk/skills.md"),
    ]

    selected = loader.select_links(links, _cfg())

    assert selected == [("Skills", "https://code.claude.com/docs/en/skills.md")]


def test_missing_pattern_is_an_error_not_a_silent_gap() -> None:
    links = [("Other", "https://code.claude.com/docs/en/other.md")]

    with pytest.raises(loader.SourceSelectionError, match="skills.md"):
        loader.select_links(links, _cfg())


def test_ambiguous_pattern_is_an_error() -> None:
    links = [
        ("A", "https://x/a/docs/en/skills.md"),
        ("B", "https://x/b/docs/en/skills.md"),
    ]

    with pytest.raises(loader.SourceSelectionError, match="2 pages"):
        loader.select_links(links, _cfg())


def test_versioned_source_follows_the_newest_version() -> None:
    cfg = _cfg(
        vendor="model-context-protocol",
        product="mcp",
        include=("/learn/architecture.md",),
        versioned_prefix="/docs/",
    )
    links = [
        (
            "old",
            "https://modelcontextprotocol.io/docs/2025-11-25/learn/architecture.md",
        ),
        (
            "new",
            "https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture.md",
        ),
        ("draft", "https://modelcontextprotocol.io/docs/draft/learn/architecture.md"),
    ]

    selected = loader.select_links(links, cfg)

    assert selected == [
        ("new", "https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture.md")
    ]
    assert loader.newest_version(links, "/docs/") == "2026-07-28"
