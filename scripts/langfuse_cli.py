"""Run the official Langfuse CLI with credentials from the project .env."""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    env = {
        **os.environ,
        **{k: v for k, v in dotenv_values(ROOT / ".env").items() if v is not None},
    }
    env["LANGFUSE_HOST"] = env.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    return subprocess.run(
        ["npx", "-y", "@langfuse/cli", "api", *args],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


if __name__ == "__main__":
    result = run(sys.argv[1:])
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    raise SystemExit(result.returncode)
