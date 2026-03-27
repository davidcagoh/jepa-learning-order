#!/usr/bin/env python3
"""Submit a Lean project to Aristotle, linked to a theorem paper.

Usage (run from the project subdirectory):
    python ../scripts/submit.py my_theorems/Paper.md "Fill in the sorries"
    python ../scripts/submit.py my_theorems/Paper.md "Fill in the sorries" --dry-run

The paper path is required as the first argument — it links this submission to its
source so that  python ../scripts/retrieve.py  can find and annotate the right file
when the job completes.

--dry-run  List files that would be packaged and exit without submitting.

Exits immediately after submitting. Aristotle will email when done.
"""

import asyncio
import io
import json
import os
import pathlib
import sys
import tarfile
import textwrap
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import load_env

import pathspec
from aristotlelib import Project, AristotleAPIError
from aristotlelib.local_file_utils import should_skip_input_file, get_pruned_dirnames

EXTRA_EXCLUDE_DIRS = {".claude", ".github", "results", "scripts", "my_theorems", "proofs-from-literature", "memory", "reports", "help_from_aristotle"}


def list_tar_files(tar_bytes: bytes) -> list[str]:
    buf = io.BytesIO(tar_bytes)
    with tarfile.open(fileobj=buf, mode="r:gz") as tf:
        return sorted(m.name for m in tf.getmembers() if not m.isdir())


def write_request_doc(
    project_id: str,
    paper: pathlib.Path,
    prompt: str,
    file_list: list[str],
    submitted_at: str,
) -> pathlib.Path:
    short_id = project_id[:8]
    doc_dir = pathlib.Path("help_from_aristotle")
    doc_dir.mkdir(exist_ok=True)
    existing = sorted(doc_dir.glob("[0-9][0-9]_*.md"))
    next_num = int(existing[-1].name[:2]) + 1 if existing else 1
    doc_path = doc_dir / f"{next_num:02d}_{short_id}_request.md"
    files_block = "\n".join(f"- `{f}`" for f in file_list)
    doc_path.write_text(textwrap.dedent(f"""\
        # Aristotle Submission: {short_id}

        **Job ID**: `{project_id}`
        **Submitted**: {submitted_at}
        **Paper**: `{paper}`
        **Status**: IN_PROGRESS

        ## Prompt

        ```
        {prompt}
        ```

        ## Files Sent

        {files_block}

        ## Target Lemmas

        <!-- TODO: list which lemmas this submission is targeting and why -->

        ## Justification

        <!-- TODO: explain why these lemmas need Aristotle -->

        ## Outcome

        <!-- Fill in after job completes -->
    """))
    return doc_path


def build_filtered_tar(project_dir: pathlib.Path) -> bytes:
    gitignore_path = project_dir / ".gitignore"
    spec = pathspec.PathSpec.from_lines(
        "gitwildmatch",
        gitignore_path.read_text().splitlines() if gitignore_path.exists() else [],
    )
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for dirpath, dirnames, filenames in os.walk(project_dir):
            rel_dir = pathlib.Path(dirpath).relative_to(project_dir)
            dirnames[:] = [
                d for d in get_pruned_dirnames(dirnames, rel_dir)
                if d not in EXTRA_EXCLUDE_DIRS
            ]
            for fname in sorted(filenames):
                fpath = pathlib.Path(dirpath) / fname
                rel = fpath.relative_to(project_dir)
                if spec.match_file(str(rel)):
                    continue
                if should_skip_input_file(fpath, base_path=project_dir):
                    continue
                tar.add(fpath, arcname=str(rel))
    return buf.getvalue()


async def main() -> None:
    load_env()

    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    paper = pathlib.Path(args[0])
    args = args[1:]

    if not paper.exists():
        print(f"Paper not found: {paper}")
        sys.exit(1)

    dry_run = "--dry-run" in args
    if dry_run:
        args = [a for a in args if a != "--dry-run"]

    project_dir = pathlib.Path(".")
    if "--project-dir" in args:
        idx = args.index("--project-dir")
        project_dir = pathlib.Path(args[idx + 1]).resolve()
        args = args[:idx] + args[idx + 2:]

    if not args:
        print('Usage: submit.py my_theorems/Paper.md "prompt"')
        sys.exit(1)

    prompt = " ".join(args)

    print(f"Packaging {project_dir} …")
    tar_bytes = build_filtered_tar(project_dir)

    if dry_run:
        import io as _io
        buf = _io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=buf, mode="r:gz") as tf:
            members = sorted(tf.getmembers(), key=lambda m: m.name)
            print(f"\n{'Size':>10}  File")
            print(f"{'─'*10}  {'─'*40}")
            for m in members:
                print(f"{m.size:>10,}  {m.name}")
        print(f"\n{'─'*10}")
        print(f"{len(tar_bytes):>10,}  total compressed ({len(tar_bytes)/1024:.1f} KB)")
        print(f"\nDry run — nothing submitted.")
        return

    print(f"Archive: {len(tar_bytes) / 1024:.1f} KB")

    tmp_tar = project_dir / ".lean_submission.tar.gz"
    try:
        tmp_tar.write_bytes(tar_bytes)
        project = await Project.create(
            prompt=prompt,
            tar_file_path=tmp_tar,
            public_file_path=f"{project_dir.name}.tar.gz",
        )
    except AristotleAPIError as e:
        print(f"Submission failed: {e}")
        sys.exit(1)
    finally:
        tmp_tar.unlink(missing_ok=True)

    submitted_at = datetime.now(timezone.utc).isoformat()

    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)
    meta = results_dir / f"{project.project_id}.meta.json"
    meta.write_text(json.dumps({
        "paper": str(paper),
        "prompt": prompt,
        "submitted_at": submitted_at,
    }, indent=2))

    doc_path = write_request_doc(
        project_id=project.project_id,
        paper=paper,
        prompt=prompt,
        file_list=list_tar_files(tar_bytes),
        submitted_at=submitted_at,
    )

    print(f"\nSubmitted:  {project.project_id}")
    print(f"Paper:      {paper}")
    print(f"Prompt:     {prompt}")
    print(f"\n→ Fill in justification: {doc_path}")
    print(f"\nWhen Aristotle emails, run:  python ../scripts/retrieve.py")


asyncio.run(main())
