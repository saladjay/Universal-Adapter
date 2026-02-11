"""Set provider API key environment variables.

Usage:
    python set_env.py open-router sk-xxx
    python set_env.py openai sk-xxx

Notes:
    - On Windows, this uses `setx` to persist variables.
    - New terminals must be opened to see updated values.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys

PROVIDER_ENV_MAP = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "cloudflare": "CF_API_KEY",
    "huggingface": "HF_TOKEN",
    "dashscope": "DASHSCOPE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "open-router": "OPENROUTER_API_KEY",
}


def _set_windows_env(var_name: str, value: str, system: bool) -> None:
    args = ["setx", var_name, value]
    if system:
        args.append("/M")
    subprocess.run(args, check=True, shell=True)


def _set_posix_env(var_name: str, value: str) -> None:
    print("Detected non-Windows platform.")
    print("Run the following command in your shell:")
    print(f"export {var_name}='{value}'")


def main() -> int:
    parser = argparse.ArgumentParser(description="Set provider API key environment variables.")
    parser.add_argument("provider", help="Provider name (e.g., openai, gemini, open-router)")
    parser.add_argument("api_key", help="API key value")
    parser.add_argument(
        "--system",
        action="store_true",
        help="Set system-wide env var on Windows (requires admin)",
    )
    args = parser.parse_args()

    env_var = PROVIDER_ENV_MAP.get(args.provider.lower())
    if not env_var:
        supported = ", ".join(sorted(PROVIDER_ENV_MAP.keys()))
        print(f"Unsupported provider '{args.provider}'. Supported: {supported}")
        return 1

    os.environ[env_var] = args.api_key

    if platform.system() == "Windows":
        try:
            _set_windows_env(env_var, args.api_key, args.system)
            scope = "system" if args.system else "user"
            print(f"Set {env_var} in {scope} environment variables.")
            print("Open a new terminal to pick up the change.")
        except subprocess.CalledProcessError as exc:
            print(f"Failed to set {env_var}: {exc}")
            return 1
    else:
        _set_posix_env(env_var, args.api_key)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
