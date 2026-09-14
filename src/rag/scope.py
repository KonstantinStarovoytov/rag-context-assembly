"""Conservative product-scope detection for retrieval experiments."""

import re

VENDOR_ALIASES = {
    "anthropic": ("anthropic", "claude"),
    "cursor": ("cursor",),
    "openai": ("openai", "codex", "chatgpt"),
}


def infer_single_vendor(query: str) -> str | None:
    """Return a vendor only when the question explicitly names exactly one."""
    normalized = query.casefold()
    matches = {
        vendor
        for vendor, aliases in VENDOR_ALIASES.items()
        if any(re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in aliases)
    }
    return next(iter(matches)) if len(matches) == 1 else None
