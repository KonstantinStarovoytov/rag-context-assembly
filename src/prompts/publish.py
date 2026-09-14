"""Publish local managed prompts as production versions in Langfuse."""

from src.observability import flush
from src.prompts.managed import publish_managed_prompts


def main() -> None:
    versions = publish_managed_prompts()
    flush()
    for name, version in versions.items():
        print(f"{name}: version {version}")


if __name__ == "__main__":
    main()
