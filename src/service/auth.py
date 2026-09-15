"""Bearer token check shared by the REST endpoint and the MCP HTTP transport."""

from secrets import compare_digest

from src.config import settings


class TokenMissing(RuntimeError):
    """Refuse to serve rather than expose paid models without a token."""


def required_token() -> str:
    token = settings.api_token
    if token is None or not token.get_secret_value().strip():
        raise TokenMissing(
            "API_TOKEN must be set before serving; every request spends "
            "OpenAI and Cohere credits."
        )
    return token.get_secret_value()


def accepted_tokens() -> list[str]:
    """The owner token, plus the guest token when one is configured."""
    tokens = [required_token()]
    guest = settings.api_guest_token
    if guest is not None and guest.get_secret_value().strip():
        tokens.append(guest.get_secret_value())
    return tokens


def token_accepted(header_value: str | None) -> bool:
    """Accept `Authorization: Bearer <token>`, comparing in constant time."""
    expected = accepted_tokens()
    if not header_value:
        return False
    scheme, _, presented = header_value.partition(" ")
    if scheme.lower() != "bearer":
        return False
    presented = presented.strip()
    # Check every token rather than returning early, so timing does not reveal
    # which one matched.
    matches = [compare_digest(presented, token) for token in expected]
    return any(matches)
