from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceConfig:
    vendor: str
    product: str
    index_url: str
    # Each pattern must match the end of exactly one URL in llms.txt; the
    # loader refuses to run otherwise, so a vendor's rename is loud, not silent.
    include: tuple[str, ...]
    # For docs published per spec version under `<prefix><YYYY-MM-DD>/`: the
    # newest dated version is chosen at load time and patterns are relative to it.
    versioned_prefix: str | None = None


ANTHROPIC = SourceConfig(
    vendor="anthropic",
    product="claude-code",
    index_url="https://code.claude.com/docs/llms.txt",
    include=(
        "/docs/en/features-overview.md",
        "/docs/en/sub-agents.md",
        "/docs/en/skills.md",
        "/docs/en/plugins.md",
        "/docs/en/mcp.md",
        "/docs/en/memory.md",
        "/docs/en/hooks.md",
        "/docs/en/hooks-guide.md",
    ),
)


CURSOR = SourceConfig(
    vendor="cursor",
    product="cursor",
    index_url="https://cursor.com/llms.txt",
    include=(
        "/docs/customize-cursor.md",
        "/docs/rules.md",
        "/docs/skills.md",
        "/docs/subagents.md",
        "/docs/hooks.md",
        "/docs/mcp.md",
        "/docs/plugins.md",
        "/docs/agent/prompting.md",
    ),
)


OPENAI_CODEX = SourceConfig(
    vendor="openai",
    product="codex",
    index_url="https://developers.openai.com/codex/llms.txt",
    include=(
        "/agent-configuration/agents-md.md",
        "/agent-configuration/rules.md",
        "/agent-configuration/subagents.md",
        "/build-skills.md",
        "/customization/overview.md",
        "/extend/mcp.md",
        "/hooks.md",
        "/plugins.md",
        "/configuration.md",
    ),
)


MCP = SourceConfig(
    vendor="model-context-protocol",
    product="mcp",
    index_url="https://modelcontextprotocol.io/llms.txt",
    versioned_prefix="/docs/",
    include=(
        "/getting-started/intro.md",
        "/learn/architecture.md",
        "/learn/server-concepts.md",
        "/learn/client-concepts.md",
        "/learn/versioning.md",
        "/develop/build-server.md",
        "/develop/build-client.md",
        "/sdk.md",
        "/tutorials/security/authorization.md",
        "/tutorials/security/security_best_practices.md",
    ),
)


SOURCES = (
    ANTHROPIC,
    CURSOR,
    OPENAI_CODEX,
    MCP,
)
