# automated-proofs

A GitHub template for formalizing mathematical proofs in Lean 4 using [Aristotle](https://aristotle.harmonic.fun) — Harmonic's automated theorem proving service.

The workflow is human-assisted: a proof paper (markdown) is translated into a Lean skeleton, then submitted to Aristotle in iterative rounds until all `sorry` placeholders are filled. Tooling manages the submission/retrieval loop.

## Using this template

```bash
# 1. Click "Use this template" on GitHub, name your repo, clone it
git clone https://github.com/<you>/<your-repo> && cd <your-repo>

# 2. Initialise: rename the Lean library, seed project dirs, squash template history
python scripts/init.py <ProjectName>

# 3. Add your Aristotle API key
echo "ARISTOTLE_API_KEY=arstl_..." > .env

# 4. Scaffold a new theorem from a paper
# (uses the /new-theorem Claude Code skill)
```

## Setup

Python 3.10+ and a Lean 4 toolchain are required.

```bash
pip install aristotlelib pathspec python-dotenv
```

## Scripts

All scripts run from the project root.

```bash
# One-time init after forking the template
python scripts/init.py <ProjectName>

# See current sorry count and submission status
python scripts/status.py

# Preview what would be packaged before submitting
python scripts/submit.py my_theorems/Paper.md "Fill in the sorries" --dry-run

# Submit to Aristotle
python scripts/submit.py my_theorems/Paper.md "Fill in the sorries"

# When Aristotle emails: download results and annotate the paper
python scripts/retrieve.py
```

Annotated results are written to `reports/<PaperName>_annotated.md` with per-lemma callouts (✓ proved / ◑ proved vacuously / ⚠ needs revision).

## What's automated vs. human-assisted

| Step | Automated |
|---|---|
| Packaging and submitting to Aristotle | ✓ |
| Polling status and downloading results | ✓ |
| Annotating the paper with proof outcomes | ✓ |
| Generating the initial Lean skeleton from a paper | `/new-theorem` skill (Claude Code) |
| Diagnosing remaining sorries between rounds | Human |
| Creating helper lemmas for Mathlib gaps | Human |

## Lean environment

- **Toolchain**: `leanprover/lean4:v4.28.0` (matches Aristotle's fixed version — no porting needed)
- **Mathlib**: `v4.28.0` / commit `8f9d9cff6bd728b17a24e163c9402775d9e6a365`
- **Entry point**: `<ProjectName>.lean` — add imports here in dependency order
