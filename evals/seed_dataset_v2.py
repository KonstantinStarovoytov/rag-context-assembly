from uuid import NAMESPACE_URL, uuid5

from langfuse import Langfuse

from src.config import settings

DATASET_NAME = "rag/retrieval-v2"


langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key.get_secret_value(),
    base_url=settings.langfuse_base_url,
)


CASES = [
    # ============================================================
    # SUBAGENTS
    # ============================================================
    {
        "question": "How can Claude subagents preload skills?",
        "category": "subagents",
        "difficulty": "easy",
        "relevant": [
            {
                "title": "Create custom subagents",
                "heading_contains": "Control subagent capabilities",
            }
        ],
    },
    {
        "question": "How do I restrict which tools a custom subagent can use?",
        "category": "subagents",
        "difficulty": "semantic",
        "relevant": [
            {
                "title": "Create custom subagents",
                "heading_contains": "Control subagent capabilities",
            }
        ],
    },
    {
        "question": "Can a Claude Code subagent use MCP servers?",
        "category": "subagents",
        "difficulty": "easy",
        "relevant": [
            {
                "title": "Create custom subagents",
            }
        ],
    },
    {
        "question": "Where do I configure a custom Claude Code subagent?",
        "category": "subagents",
        "difficulty": "easy",
        "relevant": [
            {
                "title": "Create custom subagents",
                "heading_contains": "Write subagent files",
            }
        ],
    },
    {
        "question": "How can a Claude subagent remember information between sessions?",
        "category": "subagents",
        "difficulty": "semantic",
        "relevant": [
            {
                "title": "Create custom subagents",
            },
            {
                "title": "How Claude remembers your project",
            },
        ],
    },
    {
        "question": "Why would I use a subagent instead of doing the work in the main conversation?",
        "category": "subagents",
        "difficulty": "hard",
        "relevant": [
            {
                "title": "Create custom subagents",
            },
            {
                "title": "Extend Claude Code",
                "heading_contains": "Compare similar features",
            },
        ],
    },
    # ============================================================
    # SKILLS
    # ============================================================
    {
        "question": "What file defines a Claude Code skill?",
        "category": "skills",
        "difficulty": "keyword",
        "relevant": [
            {
                "title": "Extend Claude with skills",
            }
        ],
    },
    {
        "question": "Where should I store reusable instructions that Claude loads only when needed?",
        "category": "skills",
        "difficulty": "semantic",
        "relevant": [
            {
                "title": "Extend Claude with skills",
            },
            {
                "title": "Extend Claude Code",
            },
            {
                "title": "How Claude remembers your project",
                "heading_contains": "When to add to CLAUDE.md",
            },
            {
                "title": "How Claude remembers your project",
                "heading_contains": "Related resources",
            },
        ],
    },
    {
        "question": "How can I invoke a Claude Code skill manually?",
        "category": "skills",
        "difficulty": "easy",
        "relevant": [
            {
                "title": "Extend Claude with skills",
            }
        ],
    },
    {
        "question": "How do I stop Claude from invoking a skill automatically?",
        "category": "skills",
        "difficulty": "hard",
        "relevant": [
            {
                "title": "Extend Claude with skills",
            }
        ],
    },
    {
        "question": "When should I move instructions from CLAUDE.md into a skill?",
        "category": "skills",
        "difficulty": "semantic",
        "relevant": [
            {
                "title": "Extend Claude with skills",
            },
            {
                "title": "Extend Claude Code",
            },
        ],
    },
    {
        "question": "Are custom slash commands now represented as skills?",
        "category": "skills",
        "difficulty": "keyword",
        "relevant": [
            {
                "title": "Extend Claude with skills",
            }
        ],
    },
    # ============================================================
    # MCP
    # ============================================================
    {
        "question": "How can Claude Code connect to external tools and services?",
        "category": "mcp",
        "difficulty": "semantic",
        "relevant": [
            {
                "title": "Connect Claude Code to tools via MCP",
            }
        ],
    },
    {
        "question": "How do I add an MCP server to Claude Code?",
        "category": "mcp",
        "difficulty": "easy",
        "relevant": [
            {
                "title": "Connect Claude Code to tools via MCP",
            }
        ],
    },
    {
        "question": "What is the difference between local, project, and user MCP server scopes?",
        "category": "mcp",
        "difficulty": "hard",
        "relevant": [
            {
                "title": "Connect Claude Code to tools via MCP",
            }
        ],
    },
    {
        "question": "Can Claude Code authenticate to a remote MCP server using OAuth?",
        "category": "mcp",
        "difficulty": "keyword",
        "relevant": [
            {
                "title": "Connect Claude Code to tools via MCP",
            }
        ],
    },
    {
        "question": "Can I make an MCP server available only to a specific subagent?",
        "category": "mcp",
        "difficulty": "hard",
        "relevant": [
            {
                "title": "Create custom subagents",
            },
            {
                "title": "Connect Claude Code to tools via MCP",
            },
        ],
    },
    # ============================================================
    # MEMORY / CLAUDE.MD
    # ============================================================
    {
        "question": "Where should I put project-wide instructions for Claude Code?",
        "category": "memory",
        "difficulty": "semantic",
        "relevant": [
            {
                "title": "How Claude remembers your project",
            }
        ],
    },
    {
        "question": "How does Claude Code load project memory?",
        "category": "memory",
        "difficulty": "easy",
        "relevant": [
            {
                "title": "How Claude remembers your project",
            }
        ],
    },
    {
        "question": "Can CLAUDE.md import instructions from another file?",
        "category": "memory",
        "difficulty": "keyword",
        "relevant": [
            {
                "title": "How Claude remembers your project",
            }
        ],
    },
    {
        "question": "What is the difference between user-level and project-level Claude instructions?",
        "category": "memory",
        "difficulty": "hard",
        "relevant": [
            {
                "title": "How Claude remembers your project",
            }
        ],
    },
    # ============================================================
    # PLUGINS / FEATURE SELECTION
    # ============================================================
    {
        "question": "How can I package Claude Code skills, hooks, and subagents together?",
        "category": "plugins",
        "difficulty": "semantic",
        "relevant": [
            {
                "title": "Create plugins",
            },
            {
                "title": "Extend Claude Code",
                "heading_contains": "Combine features",
            },
        ],
    },
    {
        "question": "When should I use a skill instead of a subagent?",
        "category": "comparison",
        "difficulty": "hard",
        "relevant": [
            {
                "title": "Extend Claude Code",
                "heading_contains": "Compare similar features",
            }
        ],
    },
    {
        "question": "What is the difference between reusable skills and isolated workers in Claude Code?",
        "category": "comparison",
        "difficulty": "semantic",
        "relevant": [
            {
                "title": "Extend Claude Code",
                "heading_contains": "Compare similar features",
            },
            {
                "title": "Create custom subagents",
            },
            {
                "title": "Extend Claude with skills",
            },
        ],
    },
    {
        "question": "How do skills affect Claude Code context usage?",
        "category": "context",
        "difficulty": "semantic",
        "relevant": [
            {
                "title": "Extend Claude Code",
                "heading_contains": "Understand how features load",
            },
            {
                "title": "Extend Claude Code",
                "heading_contains": "Understand context costs",
            },
            {
                "title": "Extend Claude with skills",
                "heading_contains": "Skill content lifecycle",
            },
        ],
    },
    {
        "question": "How do I choose between CLAUDE.md, skills, hooks, and subagents?",
        "category": "comparison",
        "difficulty": "hard",
        "relevant": [
            {
                "title": "Extend Claude Code",
            }
        ],
    },
    # ============================================================
    # AGENT SDK SKILLS
    # ============================================================
    {
        "question": "How are skills loaded when using the Claude Agent SDK?",
        "category": "agent-sdk",
        "difficulty": "easy",
        "relevant": [
            {
                "title": "Extend agents with skills",
                "heading_contains": "How skills work with the Agent SDK",
            }
        ],
    },
    {
        "question": "Where does the Claude Agent SDK discover skills from?",
        "category": "agent-sdk",
        "difficulty": "semantic",
        "relevant": [
            {
                "title": "Extend agents with skills",
                "heading_contains": "How skills work with the Agent SDK",
            }
        ],
    },
    {
        "question": "How can I restrict which skills are available in an Agent SDK session?",
        "category": "agent-sdk",
        "difficulty": "hard",
        "relevant": [
            {
                "title": "Extend agents with skills",
            }
        ],
    },
    {
        "question": "Why does the Agent SDK say a skill is not in the session allowlist?",
        "category": "agent-sdk",
        "difficulty": "keyword",
        "relevant": [
            {
                "title": "Extend agents with skills",
                "heading_contains": "Skill not being used",
            }
        ],
    },
]


def main() -> None:
    langfuse.create_dataset(
        name=DATASET_NAME,
        description=(
            "Retrieval benchmark for Claude Code documentation. "
            "Contains exact, semantic, ambiguous and keyword-heavy queries."
        ),
        metadata={
            "version": "2",
            "type": "retrieval",
            "items": len(CASES),
        },
    )

    for case in CASES:
        # Deterministic ID:
        # rerunning this script updates the same item instead of duplicating it.
        item_id = str(
            uuid5(
                NAMESPACE_URL,
                f"{DATASET_NAME}:{case['question']}",
            )
        )

        langfuse.create_dataset_item(
            id=item_id,
            dataset_name=DATASET_NAME,
            input={
                "question": case["question"],
            },
            expected_output={
                "relevant": case["relevant"],
            },
            metadata={
                "category": case["category"],
                "difficulty": case["difficulty"],
            },
        )

    print(f"Seeded {len(CASES)} cases into '{DATASET_NAME}'")


if __name__ == "__main__":
    main()
