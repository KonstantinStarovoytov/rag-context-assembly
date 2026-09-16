"""Strip vendor boilerplate and MDX markup that survives raw-markdown fetch.

`load_source` fetches the `.md` variant of each page (llms.txt convention),
which is close to plain markdown but still carries two kinds of noise: a
Mintlify-style "Documentation Index" blockquote some sites prepend to every
page, and JSX/MDX component tags (Mintlify's `<Tip>`, `<Tabs>`, ...) that
wrap real prose. Both would otherwise sit in the embedded/BM25-indexed text
and, for the blockquote, can become a standalone junk chunk of their own.
"""

import re

# Component tags are capitalised by MDX convention (`<Tab>`, `<Note>`), which
# keeps this from touching TypeScript generics (`Array<string>`) or prose
# comparisons (`a < b`) — both start with a lowercase character after `<`.
_MDX_TAG_RE = re.compile(r"</?[A-Z][A-Za-z]*(?:\s[^>\n]*)?/?>")

# Self-closing tags of any case (`<img .../>`) always end in `/>`, a marker
# specific enough that it will not match a generic (`Array<string>`) or a
# prose comparison (`a < b`), so the lowercase restriction above can be
# dropped here.
_SELF_CLOSING_TAG_RE = re.compile(r"<[A-Za-z][A-Za-z0-9]*(?:\s[^>\n]*)?/>")

# Mintlify code fences carry attributes after the language, e.g.
# ```json theme={null}```; keep the fence and language, drop the rest.
_FENCE_ATTRS_RE = re.compile(
    r"^(\s*```[A-Za-z0-9_-]*) [A-Za-z_]+=\{[^}]*\}\s*$", re.MULTILINE
)


def _strip_leading_index_blockquote(text: str) -> str:
    lines = text.split("\n")
    if not lines or not lines[0].startswith(">"):
        return text
    end = 0
    while end < len(lines) and lines[end].startswith(">"):
        end += 1
    if "llms.txt" not in "\n".join(lines[:end]):
        return text
    while end < len(lines) and lines[end].strip() == "":
        end += 1
    return "\n".join(lines[end:])


def clean_markdown(content: str) -> str:
    content = _strip_leading_index_blockquote(content)
    content = _SELF_CLOSING_TAG_RE.sub("", content)
    content = _MDX_TAG_RE.sub("", content)
    content = _FENCE_ATTRS_RE.sub(r"\1", content)
    return content
