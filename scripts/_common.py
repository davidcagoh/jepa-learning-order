"""Shared utilities for submit.py, retrieve.py, status.py. Not a user-facing script."""

import os
import pathlib
import re


def load_env() -> None:
    """Load .env from the current working directory into os.environ."""
    env = pathlib.Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def get_module_name() -> str:
    """Read the first lean_lib name from lakefile.toml in CWD.

    Returns the module name (e.g. 'AutomatedProofs') or '.' as a fallback
    that will scan all .lean files in the current directory.
    """
    lakefile = pathlib.Path("lakefile.toml")
    if not lakefile.exists():
        return "."
    text = lakefile.read_text()
    m = re.search(r'\[\[lean_lib\]\]\s*\nname\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else "."
