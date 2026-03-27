#!/usr/bin/env python3
"""Show current project status: sorry counts, submissions in flight, last results.

Run from the project subdirectory:
    python ../scripts/status.py
"""

import asyncio
import json
import pathlib
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import load_env, get_module_name

from aristotlelib import Project, ProjectStatus, AristotleAPIError

DONE_STATUSES = {
    ProjectStatus.COMPLETE,
    ProjectStatus.COMPLETE_WITH_ERRORS,
    ProjectStatus.OUT_OF_BUDGET,
}


def count_sorries(path: pathlib.Path) -> list[int]:
    """Return list of line numbers (1-indexed) containing a real sorry."""
    lines = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        # Skip comment lines and docstring lines
        if stripped.startswith("--") or stripped.startswith("/-") or stripped.startswith("*"):
            continue
        # Match standalone sorry (the form used in proof bodies)
        if stripped == "sorry" or stripped.endswith("by sorry") or stripped == "by sorry":
            lines.append(i)
    return lines


def show_sorry_counts() -> int:
    module_name = get_module_name()
    lean_dir = pathlib.Path(module_name) if module_name != '.' else pathlib.Path('.')
    lean_files = sorted(lean_dir.glob("**/*.lean"))
    if not lean_files:
        print(f"  (no .lean files found in {lean_dir}/)")
        return 0

    total = 0
    for f in lean_files:
        if ".lake" in f.parts:
            continue
        line_nums = count_sorries(f)
        n = len(line_nums)
        total += n
        if n == 0:
            print(f"  ✓  {f}")
        else:
            locs = ", ".join(str(l) for l in line_nums[:6])
            if len(line_nums) > 6:
                locs += ", …"
            print(f"  ✗  {f}  ({n} sorry{'s' if n != 1 else ''} at lines {locs})")
    return total


def format_age(submitted_at: str) -> str:
    try:
        dt = datetime.fromisoformat(submitted_at)
        delta = datetime.now(timezone.utc) - dt
        hours = int(delta.total_seconds() / 3600)
        if hours < 1:
            return f"{int(delta.total_seconds() / 60)}m ago"
        if hours < 48:
            return f"{hours}h ago"
        return f"{delta.days}d ago"
    except Exception:
        return "?"


async def show_submissions() -> None:
    results_dir = pathlib.Path("results")
    meta_files = sorted(
        results_dir.glob("*.meta.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not meta_files:
        print("  (no submissions found — run submit.py first)")
        return

    for meta_path in meta_files:
        meta = json.loads(meta_path.read_text())
        project_id = meta_path.name.replace(".meta.json", "")
        paper = pathlib.Path(meta.get("paper", "?")).name
        age = format_age(meta.get("submitted_at", ""))
        prompt_preview = meta.get("prompt", "?")[:72]

        tar_path = results_dir / f"{project_id}.tar.gz"
        if tar_path.exists():
            status_str = "DOWNLOADED"
        else:
            try:
                project = await Project.from_id(project_id)
                s = project.status
                pct = f" {project.percent_complete}%" if project.percent_complete is not None else ""
                status_str = f"{s.value}{pct}"
            except AristotleAPIError as e:
                status_str = f"API error: {e}"

        in_flight = not tar_path.exists() and status_str not in ("DOWNLOADED",)
        marker = "⏳" if in_flight else "✓ "
        print(f"  {marker} {project_id[:8]}…  {status_str:<28}  {age:<10}  [{paper}]")
        print(f"      prompt: {prompt_preview}")


async def main() -> None:
    load_env()

    print("\n── Sorry counts ──────────────────────────────────────────")
    total = show_sorry_counts()
    print(f"  {'─'*52}")
    if total == 0:
        print("  Total: 0  🎉 fully compiled")
    else:
        print(f"  Total: {total} remaining")

    print("\n── Submissions (most recent first) ───────────────────────")
    await show_submissions()
    print()


asyncio.run(main())
