"""Unit tests use dummy credentials and never depend on the developer's .env."""

import os

for name in (
    "OPENAI_API_KEY",
    "COHERE_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
):
    os.environ[name] = "test-placeholder"

os.environ["TRACING_ENABLED"] = "false"
