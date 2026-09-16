"""Strip vendor boilerplate and MDX markup that survives into the index."""

from src.ingestion.clean import clean_markdown


def test_strips_leading_documentation_index_blockquote() -> None:
    raw = (
        "> ## Documentation Index\n"
        "> Fetch the complete documentation index at: https://code.claude.com/docs/llms.txt\n"
        "> Use this file to discover all available pages before exploring further.\n"
        "\n"
        "# Hooks reference\n\nReal content.\n"
    )

    cleaned = clean_markdown(raw)

    assert cleaned == "# Hooks reference\n\nReal content.\n"


def test_leaves_an_unrelated_leading_blockquote_alone() -> None:
    raw = "> A genuine documentation callout, not an index pointer.\n\n# Title\n"

    assert clean_markdown(raw) == raw


def test_strips_mdx_component_tags_but_keeps_their_text() -> None:
    raw = '<Tip>\n  Use `/init` to scaffold config.\n</Tip>\n\n<Tabs>\n<Tab title="macOS">\nrun this\n</Tab>\n</Tabs>\n'

    cleaned = clean_markdown(raw)

    assert "<Tip>" not in cleaned and "</Tip>" not in cleaned
    assert "<Tab" not in cleaned and "</Tab>" not in cleaned
    assert "Use `/init` to scaffold config." in cleaned
    assert "run this" in cleaned


def test_strips_self_closing_tags_entirely() -> None:
    raw = 'Before\n<img src="diagram.png" />\nAfter\n'

    cleaned = clean_markdown(raw)

    assert "<img" not in cleaned
    assert "Before" in cleaned and "After" in cleaned


def test_strips_code_fence_attributes_but_keeps_the_language() -> None:
    raw = '```json theme={null}\n{"a": 1}\n```\n'

    cleaned = clean_markdown(raw)

    assert cleaned.startswith("```json\n")
    assert "theme={null}" not in cleaned


def test_does_not_touch_generic_lowercase_or_typed_angle_brackets() -> None:
    """TypeScript generics and prose comparisons must survive untouched."""
    raw = "Returns `Array<string>` when `a < b` holds.\n"

    assert clean_markdown(raw) == raw
