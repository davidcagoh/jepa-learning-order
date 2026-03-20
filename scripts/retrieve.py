#!/usr/bin/env python3
"""Check and download completed Aristotle jobs, then annotate your papers.

Run this after submitting (or any time you think a job might be done):

    python scripts/retrieve.py

For a specific project ID:

    python scripts/retrieve.py <project-id>

For each completed job found, downloads the result and writes:
    reports/<PaperName>_annotated.md
"""

import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import load_env
import _report

from aristotlelib import Project, ProjectStatus, AristotleAPIError


DONE_STATUSES = {
    ProjectStatus.COMPLETE,
    ProjectStatus.COMPLETE_WITH_ERRORS,
    ProjectStatus.OUT_OF_BUDGET,
}


def load_meta(meta_path: pathlib.Path) -> dict:
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def annotate(tar_path: pathlib.Path, meta: dict) -> None:
    paper_str = meta.get("paper")
    paper_path = _report.find_paper(tar_path, paper_str) if not paper_str else pathlib.Path(paper_str)
    if paper_path is None or not paper_path.exists():
        paper_path = _report.find_paper(tar_path, None)
    if paper_path is None:
        print(f"             could not find paper — pass --paper to report manually")
        return
    _report.run(tar_path, paper_path)


async def process_one(project_id: str, meta: dict, results_dir: pathlib.Path) -> bool:
    """Download and annotate one project. Returns True if something new was done."""
    tar_path = results_dir / f"{project_id}.tar.gz"

    if tar_path.exists():
        print(f"  {project_id[:8]}…  already downloaded → re-annotating")
        annotate(tar_path, meta)
        return True

    try:
        project = await Project.from_id(project_id)
    except AristotleAPIError as e:
        print(f"  {project_id[:8]}…  could not fetch: {e}")
        return False

    if project.status not in DONE_STATUSES:
        pct = f" ({project.percent_complete}%)" if project.percent_complete is not None else ""
        print(f"  {project_id[:8]}…  {project.status.value}{pct}  — still in progress")
        return False

    print(f"  {project_id[:8]}…  {project.status.value}  → downloading…")
    try:
        path = await project.get_solution(destination=str(tar_path))
        print(f"             saved to {path}")
        annotate(tar_path, meta)
        return True
    except AristotleAPIError as e:
        print(f"             download failed: {e}")
        return False


async def main() -> None:
    load_env()

    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)

    # Single project ID passed explicitly
    if len(sys.argv) == 2 and sys.argv[1] not in ("-h", "--help"):
        project_id = sys.argv[1]
        meta_path = results_dir / f"{project_id}.meta.json"
        meta = load_meta(meta_path) if meta_path.exists() else {}
        await process_one(project_id, meta, results_dir)
        return

    if len(sys.argv) > 1:
        print(__doc__)
        sys.exit(0)

    # No args: scan all tracked meta.json files
    meta_files = sorted(results_dir.glob("*.meta.json"))
    if not meta_files:
        print("No tracked jobs found in results/. Submit a paper first with:")
        print('  python scripts/submit.py my_theorems/Paper.md "Fill in the sorries"')
        sys.exit(0)

    print(f"Checking {len(meta_files)} tracked job(s)…\n")
    any_done = False
    for meta_path in meta_files:
        project_id = meta_path.name.replace(".meta.json", "")
        meta = load_meta(meta_path)
        paper = meta.get("paper", "unknown")
        print(f"  [{pathlib.Path(paper).name}]  {project_id[:8]}…")
        done = await process_one(project_id, meta, results_dir)
        if done:
            any_done = True
        print()

    if not any_done:
        print("Nothing new. Check back after Aristotle emails you.")


asyncio.run(main())
