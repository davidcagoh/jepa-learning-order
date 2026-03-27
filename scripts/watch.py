#!/usr/bin/env python3
"""Poll all in-flight Aristotle jobs, back off as they progress, auto-retrieve on completion.

Run from the project subdirectory:
    python ../scripts/watch.py

Polling schedule (adaptive — based on % complete):
    0–25%   → poll every 5 min
    25–50%  → poll every 10 min
    50–75%  → poll every 15 min
    75–99%  → poll every 8 min  (nearing the end, check more often)

ETA is estimated from (elapsed / pct_done) * pct_remaining, shown once >=5% is reported.
Auto-runs retrieve.py when any job completes.
"""

import asyncio
import json
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import load_env

from aristotlelib import Project, ProjectStatus, AristotleAPIError

TERMINAL_STATUSES = {
    ProjectStatus.COMPLETE,
    ProjectStatus.COMPLETE_WITH_ERRORS,
    ProjectStatus.OUT_OF_BUDGET,
    ProjectStatus.FAILED,
    ProjectStatus.CANCELED,
}

IN_PROGRESS_STATUSES = {
    ProjectStatus.QUEUED,
    ProjectStatus.IN_PROGRESS,
}

_RETRIEVE_PY = str(pathlib.Path(__file__).parent / "retrieve.py")


def poll_interval_seconds(pct: int | None) -> int:
    """Adaptive poll interval based on % complete."""
    if pct is None or pct < 25:
        return 5 * 60
    if pct < 50:
        return 10 * 60
    if pct < 75:
        return 15 * 60
    return 8 * 60


def fmt_seconds(s: float) -> str:
    if s < 3600:
        return f"{int(s/60)}m"
    h = int(s / 3600)
    m = int((s % 3600) / 60)
    return f"{h}h{m:02d}m"


def eta_str(started_at: datetime, pct: int | None) -> str:
    if pct is None or pct < 5:
        return ""
    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    if pct >= 100:
        return "(done)"
    remaining = elapsed / pct * (100 - pct)
    return f"ETA ~{fmt_seconds(remaining)}"


async def get_in_flight_ids() -> list[str]:
    results_dir = pathlib.Path("results")
    ids = []
    for meta_path in results_dir.glob("*.meta.json"):
        project_id = meta_path.stem.replace(".meta", "")
        tar_path = results_dir / f"{project_id}.tar.gz"
        if not tar_path.exists():
            ids.append(project_id)
    return ids


async def watch() -> None:
    load_env()

    ids = await get_in_flight_ids()
    if not ids:
        print("No in-flight jobs found. Run submit.py first.")
        return

    # Track start time and last known % per job
    start_times: dict[str, datetime] = {}
    last_pct: dict[str, int | None] = {pid: None for pid in ids}
    completed: set[str] = set()

    print(f"Watching {len(ids)} job(s): {', '.join(i[:8] for i in ids)}")
    print("Ctrl+C to stop.\n")

    while True:
        now = datetime.now(timezone.utc)
        newly_done: list[str] = []
        next_waits: list[int] = []

        for pid in ids:
            if pid in completed:
                continue
            try:
                project = await Project.from_id(pid)
            except AristotleAPIError as e:
                print(f"  [{pid[:8]}] API error: {e}")
                continue

            pct = project.percent_complete
            status = project.status

            # Record start time on first real progress
            if pid not in start_times and pct and pct > 0:
                start_times[pid] = now

            last_pct[pid] = pct
            pct_str = f"{pct}%" if pct is not None else "?"
            eta = eta_str(start_times.get(pid, now), pct)

            meta_path = pathlib.Path("results") / f"{pid}.meta.json"
            prompt_preview = ""
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                prompt_preview = meta.get("prompt", "")[:60]

            if status in TERMINAL_STATUSES:
                print(f"  [{pid[:8]}] {status.value}  ✓  — {prompt_preview}")
                completed.add(pid)
                newly_done.append(pid)
            else:
                interval = poll_interval_seconds(pct)
                next_waits.append(interval)
                print(
                    f"  [{pid[:8]}] {status.value:<14} {pct_str:>4}  {eta:<12}"
                    f"  next check in {fmt_seconds(interval)}"
                    f"  [{prompt_preview}]"
                )

        if newly_done:
            print("\nCompleted jobs detected — running retrieve.py …")
            subprocess.run([sys.executable, _RETRIEVE_PY], check=False)
            print()

        remaining = [pid for pid in ids if pid not in completed]
        if not remaining:
            print("All jobs finished.")
            break

        wait = min(next_waits) if next_waits else 5 * 60
        wake_at = datetime.now(timezone.utc).replace(microsecond=0)
        print(f"\n[{wake_at.strftime('%H:%M:%S')}] sleeping {fmt_seconds(wait)} …\n")

        try:
            await asyncio.sleep(wait)
        except asyncio.CancelledError:
            break


try:
    asyncio.run(watch())
except KeyboardInterrupt:
    print("\nStopped.")
