#!/usr/bin/env python3
"""Check for completed Aristotle projects not yet in results/ and download them.

Run this whenever you think a job has finished:

    python sync.py
"""

import asyncio
import os
import pathlib
import subprocess
import sys

from aristotlelib import Project, ProjectStatus, AristotleAPIError

DONE_STATUSES = {
    ProjectStatus.COMPLETE,
    ProjectStatus.COMPLETE_WITH_ERRORS,
    ProjectStatus.OUT_OF_BUDGET,
}


def load_env() -> None:
    env = pathlib.Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def already_downloaded(results_dir: pathlib.Path) -> set[str]:
    return {p.stem.split(".")[0] for p in results_dir.glob("*.tar.gz")}


async def main() -> None:
    load_env()

    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)
    have = already_downloaded(results_dir)

    print("Fetching recent projects from Aristotle…")
    try:
        projects, _ = await Project.list_projects(limit=50)
    except AristotleAPIError as e:
        print(f"Failed to list projects: {e}")
        sys.exit(1)

    new = [p for p in projects if p.status in DONE_STATUSES and p.project_id not in have]

    if not new:
        print("Nothing new to download.")
        # Show what's pending so user knows what's still in flight
        pending = [p for p in projects if p.status not in DONE_STATUSES | {ProjectStatus.FAILED, ProjectStatus.CANCELED}]
        if pending:
            print(f"\nStill in progress ({len(pending)}):")
            for p in pending:
                age = ""
                if p.input_prompt:
                    age = f'  \"{p.input_prompt[:60]}\"'
                print(f"  {p.project_id}  {p.status.value}{age}")
        return

    print(f"Found {len(new)} new completed project(s):\n")
    for project in new:
        dest = results_dir / f"{project.project_id}.tar.gz"
        label = project.input_prompt or project.project_id
        print(f"  Downloading: {label[:70]}")
        try:
            path = await project.get_solution(destination=str(dest))
            print(f"  Saved to {path}")
            subprocess.run([sys.executable, "show_result.py", path])
        except AristotleAPIError as e:
            print(f"  Download failed: {e}")


asyncio.run(main())
