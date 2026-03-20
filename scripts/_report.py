"""Internal module: annotate a theorem paper with Aristotle proof results.

Imported by retrieve.py. Not a user-facing script — run  python scripts/retrieve.py  instead.
"""

import json
import pathlib
import re
import tarfile
import tempfile
from datetime import datetime, timezone


# ── Lean parsing ──────────────────────────────────────────────────────────────

DECL_RE = re.compile(
    r'^(?:noncomputable\s+)?(?:private\s+)?(?:protected\s+)?'
    r'(lemma|theorem|def|abbrev|structure|class|instance)\s+(\w+)',
    re.MULTILINE,
)

_THEOREM_NUM_RE = re.compile(
    r'\*\*\s*(Lemma|Theorem|Proposition|Corollary|Definition)\s+([\d.]+)',
    re.IGNORECASE,
)
_PART_RE = re.compile(r'Part\s+\(([A-E])\d?\)', re.IGNORECASE)


def _preceding_comment(text: str, decl_pos: int) -> str | None:
    """Return the last block comment immediately before decl_pos, if adjacent."""
    text_before = text[:decl_pos]
    matches = list(re.finditer(r'/--?[\s\S]*?-/', text_before))
    if not matches:
        return None
    last = matches[-1]
    between = text_before[last.end():].strip()
    if re.search(r'\b(?:lemma|theorem|def|structure|instance)\b', between):
        return None
    return last.group(0)


def theorem_ref(comment: str | None) -> str | None:
    """Extract 'Lemma 5.2' or 'Part (A)' from a docstring/PROBLEM block."""
    if not comment:
        return None
    m = _THEOREM_NUM_RE.search(comment)
    if m:
        return f"{m.group(1).capitalize()} {m.group(2)}"
    m = _PART_RE.search(comment)
    if m:
        return f"Part ({m.group(1).upper()})"
    return None


def decl_sorry_map(lean_text: str) -> dict[str, bool]:
    """Return {decl_name: has_sorry} for every declaration in lean_text."""
    decls = list(DECL_RE.finditer(lean_text))
    out = {}
    for i, dm in enumerate(decls):
        end = decls[i + 1].start() if i + 1 < len(decls) else len(lean_text)
        out[dm.group(2)] = bool(re.search(r'\bsorry\b', lean_text[dm.start():end]))
    return out


def local_sorry_map(lean_dir: pathlib.Path) -> dict[str, str | None]:
    """For each sorry-containing declaration in lean_dir, map name -> theorem_ref."""
    out: dict[str, str | None] = {}
    for f in sorted(lean_dir.rglob("*.lean")):
        if ".lake" in f.parts:
            continue
        text = f.read_text(errors="replace")
        for dm in DECL_RE.finditer(text):
            name = dm.group(2)
            next_dm = DECL_RE.search(text, dm.end())
            scope = text[dm.start(): next_dm.start() if next_dm else len(text)]
            if re.search(r'\bsorry\b', scope):
                out[name] = theorem_ref(_preceding_comment(text, dm.start()))
    return out


def result_remaining(result_lean_texts: list[str]) -> set[str]:
    """Return set of declaration names that still contain sorry in the result."""
    remaining: set[str] = set()
    for text in result_lean_texts:
        for name, has_sorry in decl_sorry_map(text).items():
            if has_sorry:
                remaining.add(name)
    return remaining


def result_sorry_map(result_lean_texts: list[str]) -> dict[str, str | None]:
    """Map {name: theorem_ref} for every sorry-containing decl in the result files.
    Captures declarations Aristotle added (e.g. helper lemmas) that weren't in local source."""
    out: dict[str, str | None] = {}
    for text in result_lean_texts:
        decls = list(DECL_RE.finditer(text))
        for i, dm in enumerate(decls):
            end = decls[i + 1].start() if i + 1 < len(decls) else len(text)
            if re.search(r'\bsorry\b', text[dm.start():end]):
                comment = _preceding_comment(text, dm.start())
                out[dm.group(2)] = theorem_ref(comment)
    return out


# ── Summary parsing ───────────────────────────────────────────────────────────

_ENTRY_RE = re.compile(
    r'(?:^[-\d]+[.)]\s+|^[-*]\s+)'   # list marker: "1. " or "- " or "* "
    r'\*\*`?(\w+)`?\*\*'              # **name** or **`name`**
    r'(?:\s*\([^)]*\))?'              # optional parenthetical like "(Part D)"
    r'\s*[—–-]+\s*'                   # em-dash / en-dash / hyphen separator
    r'(.+)',                           # description to end of line
    re.MULTILINE,
)


def parse_summary(text: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return (proved_map, remaining_map) each mapping lean_name -> description."""
    proved: dict[str, str] = {}
    remaining: dict[str, str] = {}
    for section in re.split(r'\n#{2,3} ', text):
        heading = section.split('\n', 1)[0].lower()
        target = remaining if 'remaining' in heading else proved
        for m in _ENTRY_RE.finditer(section):
            target.setdefault(m.group(1), m.group(2).strip())
    return proved, remaining


# ── Paper annotation ──────────────────────────────────────────────────────────

_PAPER_THREF_RE = re.compile(
    r'\*\*\s*(Lemma|Theorem|Proposition|Corollary|Definition)\s+([\d.]+)',
    re.IGNORECASE,
)


def find_paper_line(lines: list[str], ref: str) -> int | None:
    """Find the line in the paper where theorem ref is declared."""
    if ref.startswith("Part ("):
        letter = ref[6]
        for i, line in enumerate(lines):
            if re.search(r'\*\*\s*\(' + letter + r'\)\s+\w', line):
                return i
        return None
    m = re.match(r'\w+\s+([\d.]+)', ref)
    if not m:
        return None
    number = m.group(1)
    for i, line in enumerate(lines):
        pm = _PAPER_THREF_RE.search(line)
        if pm and pm.group(2) == number:
            return i
    return None


def find_insertion_point(lines: list[str], from_line: int) -> int:
    """Return the index to insert before: just before **Proof or the next ---."""
    for i in range(from_line + 1, min(from_line + 35, len(lines))):
        s = lines[i].strip()
        if re.match(r'\*\*Proof', s) or re.match(r'>\s*\*\*Proof', s):
            return i
        if s == '---':
            return i
    return from_line + 1


def build_callout(name: str, is_remaining: bool, desc: str) -> list[str]:
    if is_remaining:
        icon, label = '⚠️', 'Needs revision'
    elif re.search(r'vacuous|trivially|epsilon_0\s*=\s*0', desc, re.I):
        icon, label = '◑', 'Proved vacuously'
    else:
        icon, label = '✓', 'Proved'
    return ['', f'> {icon} **{label}** (`{name}`). {desc}', '']


def annotate_paper(
    paper_text: str,
    local_map: dict[str, str | None],
    remaining: set[str],
    proved_map: dict[str, str],
    remaining_map: dict[str, str],
) -> str:
    lines = paper_text.splitlines()
    insertions: dict[int, list[str]] = {}

    for name, ref in local_map.items():
        if not ref:
            continue
        paper_line = find_paper_line(lines, ref)
        if paper_line is None:
            continue
        insert_at = find_insertion_point(lines, paper_line)
        desc_map = remaining_map if name in remaining else proved_map
        desc = desc_map.get(name, 'Sorry still present.' if name in remaining else 'Proved.')
        callout = build_callout(name, name in remaining, desc)
        insertions.setdefault(insert_at, [])
        insertions[insert_at].extend(callout)

    for insert_at in sorted(insertions, reverse=True):
        for line in reversed(insertions[insert_at]):
            lines.insert(insert_at, line)

    return '\n'.join(lines)


def status_table(
    local_map: dict[str, str | None],
    remaining: set[str],
    proved_map: dict[str, str],
    remaining_map: dict[str, str],
) -> str:
    rows = []
    for name in sorted(local_map):
        ref = local_map[name] or '—'
        if name in remaining:
            status = '⚠️ Needs revision'
            desc = remaining_map.get(name, '')
        else:
            desc = proved_map.get(name, '')
            status = '◑ Vacuous' if re.search(r'vacuous|trivially', desc, re.I) else '✓ Proved'
        rows.append(f'| `{name}` | {ref} | {status} |')
    header = (
        '\n\n---\n\n## Formalization Status\n\n'
        '| Lean name | Paper ref | Status |\n'
        '|---|---|---|\n'
    )
    return header + '\n'.join(rows)


# ── Orchestration ─────────────────────────────────────────────────────────────

def find_paper(tar_path: pathlib.Path, paper_arg: str | None) -> pathlib.Path | None:
    if paper_arg:
        return pathlib.Path(paper_arg)
    meta = tar_path.parent / tar_path.name.replace('.tar.gz', '.meta.json')
    if meta.exists():
        data = json.loads(meta.read_text())
        if 'paper' in data:
            return pathlib.Path(data['paper'])
    candidates = list(pathlib.Path('my_theorems').glob('*.md'))
    return candidates[0] if len(candidates) == 1 else None


def run(tar_path: pathlib.Path, paper_path: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(tar_path, 'r:gz') as tf:
            tf.extractall(tmp, filter='data')
        root = pathlib.Path(tmp)
        summary_path = next(root.rglob('ARISTOTLE_SUMMARY_*.md'), None)
        summary_text = summary_path.read_text(errors='replace') if summary_path else ''
        result_lean_texts = [
            f.read_text(errors='replace')
            for f in root.rglob('*.lean') if f.stat().st_size > 0
        ]

    proved_map, remaining_map = parse_summary(summary_text)
    local_map = local_sorry_map(pathlib.Path('AutomatedProofs'))

    for name, ref in result_sorry_map(result_lean_texts).items():
        if name not in local_map or (ref is not None and local_map[name] is None):
            local_map[name] = ref

    remaining = result_remaining(result_lean_texts)

    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    header = (
        f'> *Annotated from `{tar_path.name}` — {now}. '
        f'Key: ✓ proved · ◑ vacuous · ⚠️ needs revision.*\n\n'
    )
    body = annotate_paper(paper_text=paper_path.read_text(errors='replace'),
                          local_map=local_map, remaining=remaining,
                          proved_map=proved_map, remaining_map=remaining_map)
    table = status_table(local_map, remaining, proved_map, remaining_map)

    pathlib.Path('reports').mkdir(exist_ok=True)
    out = pathlib.Path('reports') / f'{paper_path.stem}_annotated.md'
    out.write_text(header + body + table)
    print(f'Written to {out}')


if __name__ == '__main__':
    print("This is an internal module. Run:  python scripts/retrieve.py")
