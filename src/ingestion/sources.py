from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceConfig:
    vendor: str
    product: str
    index_url: str
    include: tuple[str, ...]


ANTHROPIC = SourceConfig(
    vendor="anthropic",
    product="claude-code",
    index_url="https://code.claude.com/docs/llms.txt",
    include=(
        "/features-overview.md",
        "/sub-agents.md",
        "/skills.md",
        "/plugins.md",
        "/mcp.md",
        "/memory.md",
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
    include=(
        "/docs/2026-07-28/getting-started/intro.md",
        "/docs/2026-07-28/learn/architecture.md",
        "/docs/2026-07-28/learn/server-concepts.md",
        "/docs/2026-07-28/learn/client-concepts.md",
        "/docs/2026-07-28/learn/versioning.md",
        "/docs/2026-07-28/develop/build-server.md",
        "/docs/2026-07-28/develop/build-client.md",
        "/docs/2026-07-28/sdk.md",
        "/docs/2026-07-28/tutorials/security/authorization.md",
        "/docs/2026-07-28/tutorials/security/security_best_practices.md",
    ),
)


SOURCES = (
    ANTHROPIC,
    CURSOR,
    OPENAI_CODEX,
    MCP,
)
