from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]


def _load_dotenv_fallback() -> None:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def load_hf_token() -> str:
    """
    Load the Hugging Face token from the environment / .env file.

    - Calls load_dotenv() so HF_TOKEN from .env is available.
    - Raises RuntimeError if HF_TOKEN is missing.
    """
    if load_dotenv is not None:
        load_dotenv()
    else:
        _load_dotenv_fallback()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Make sure it is defined in your .env file."
        )
    return token

