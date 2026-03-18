#!/usr/bin/env python3
"""Display the contents of an Aristotle result .tar.gz.

Usage:
    python show_result.py                  # opens newest file in results/
    python show_result.py results/foo.tar.gz
"""

import sys
import tarfile
import pathlib
import tempfile

def display(tar_path: pathlib.Path) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {tar_path.name}")
    print(f"{'═' * 70}")

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(tmp, filter="data")

        root = pathlib.Path(tmp)
        summary = next(root.rglob("ARISTOTLE_SUMMARY_*.md"), None)
        lean_files = sorted(root.rglob("*.lean"))

        for path in ([summary] if summary else []) + lean_files:
            rel = path.relative_to(tmp)
            print(f"\n{'─' * 70}")
            print(f"  {rel}")
            print(f"{'─' * 70}")
            print(path.read_text(errors="replace"))


def main() -> None:
    if len(sys.argv) > 1:
        targets = [pathlib.Path(sys.argv[1])]
    else:
        results_dir = pathlib.Path("results")
        targets = sorted(results_dir.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime)
        if not targets:
            print("No .tar.gz files found in results/")
            sys.exit(1)
        targets = [targets[-1]]  # newest only

    for t in targets:
        display(t)


if __name__ == "__main__":
    main()
