#!/usr/bin/env python3
"""Submit a proof request to Aristotle and exit. Returns a project ID.
When Aristotle emails you that it's done, run:

    python scripts/retrieve.py <project-id>

Usage:
    python scripts/prove.py "Prove that sqrt(2) is irrational"
    python scripts/prove.py "Fill in the sorries" --project-dir .
"""

import asyncio
import io
import os
import pathlib
import sys
import tarfile

import pathspec
from aristotlelib import Project, AristotleAPIError
from aristotlelib.local_file_utils import should_skip_input_file, get_pruned_dirnames

# Directories to always exclude regardless of .gitignore
EXTRA_EXCLUDE_DIRS = {".claude", ".github", "results", "scripts", "my_theorems"}


def load_env() -> None:
    env = pathlib.Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def build_filtered_tar(project_dir: pathlib.Path) -> bytes:
    """Build an in-memory tar.gz of the project, excluding:
    - Everything in .gitignore
    - Build artifacts and .lake/packages (via aristotlelib's patterns)
    - Non-Lean directories: results/, scripts/, my_theorems/, .claude/, .github/
    """
    # Load .gitignore patterns if present
    gitignore_path = project_dir / ".gitignore"
    gitignore_spec = pathspec.PathSpec.from_lines(
        "gitwildmatch",
        gitignore_path.read_text().splitlines() if gitignore_path.exists() else [],
    )

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for dirpath, dirnames, filenames in os.walk(project_dir):
            rel_dir = pathlib.Path(dirpath).relative_to(project_dir)

            # Prune traversal: aristotlelib's build-artifact dirs + our extra excludes
            dirnames[:] = [
                d for d in get_pruned_dirnames(dirnames, rel_dir)
                if d not in EXTRA_EXCLUDE_DIRS
            ]

            for fname in sorted(filenames):
                fpath = pathlib.Path(dirpath) / fname
                rel = fpath.relative_to(project_dir)

                # Skip if matched by .gitignore
                if gitignore_spec.match_file(str(rel)):
                    continue

                # Skip build artifacts (.olean, etc.)
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

    project_dir = None
    if "--project-dir" in args:
        idx = args.index("--project-dir")
        project_dir = pathlib.Path(args[idx + 1]).resolve()
        args = args[:idx] + args[idx + 2:]

    prompt = " ".join(args)

    try:
        if project_dir:
            print(f"Packaging {project_dir} (respecting .gitignore)…")
            tar_bytes = build_filtered_tar(project_dir)
            print(f"Archive size: {len(tar_bytes) / 1024:.1f} KB")

            # Write to a temp file so the SDK can upload it
            tmp_tar = project_dir / ".lean_submission.tar.gz"
            try:
                tmp_tar.write_bytes(tar_bytes)
                project = await Project.create(
                    prompt=prompt,
                    tar_file_path=tmp_tar,
                    public_file_path=f"{project_dir.name}.tar.gz",
                )
            finally:
                tmp_tar.unlink(missing_ok=True)
        else:
            project = await Project.create(prompt=prompt)

    except AristotleAPIError as e:
        print(f"Failed to submit: {e}")
        sys.exit(1)

    print(f"Submitted. Project ID: {project.project_id}")
    print(f"\nWhen Aristotle emails you, run:")
    print(f"  python scripts/retrieve.py {project.project_id}")


asyncio.run(main())
