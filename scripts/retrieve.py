#!/usr/bin/env python3
"""Download and display a completed Aristotle result.

Run this after you get the email from Aristotle:

    python retrieve.py <project-id>
"""

import asyncio
import os
import pathlib
import subprocess
import sys

import aristotlelib
from aristotlelib import Project, ProjectStatus, AristotleAPIError


def load_env() -> None:
    env = pathlib.Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


async def main() -> None:
    load_env()

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    project_id = sys.argv[1]

    try:
        project = await Project.from_id(project_id)
    except AristotleAPIError as e:
        print(f"Could not find project: {e}")
        sys.exit(1)

    print(f"Project: {project.project_id}")
    print(f"Status:  {project.status.value}")

    if project.status not in (
        ProjectStatus.COMPLETE,
        ProjectStatus.COMPLETE_WITH_ERRORS,
        ProjectStatus.OUT_OF_BUDGET,
    ):
        print(f"\nNot ready yet — check back later.")
        sys.exit(0)

    dest = pathlib.Path("results") / f"{project.project_id}.tar.gz"
    dest.parent.mkdir(exist_ok=True)

    try:
        path = await project.get_solution(destination=str(dest))
    except AristotleAPIError as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    print(f"Saved to {path}\n")
    subprocess.run([sys.executable, "show_result.py", path])


asyncio.run(main())
