"""Detect the language class that failed original-only hybrid retrieval."""

POLISH_LETTERS = frozenset("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")


def needs_english_translation(query: str) -> bool:
    for char in query:
        if "\u0400" <= char <= "\u04ff" or char in POLISH_LETTERS:
            return True
    return False
