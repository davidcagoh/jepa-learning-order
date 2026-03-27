# Lean 4 + Aristotle Proof Template

A GitHub template for formalising mathematical proofs in Lean 4 using [Aristotle](https://aristotle.harmonic.fun) — an automated theorem prover by Harmonic.

The workflow: write a proof paper (markdown) → translate it into a Lean skeleton with `sorry` placeholders and `PROVIDED SOLUTION` docstrings → submit to Aristotle → retrieve filled proofs. Scripts manage the submission/retrieval loop.

## Quick start

### 1. Install Python dependencies

```bash
pip install aristotlelib pathspec python-dotenv
```

### 2. Add your Aristotle API key

```bash
echo "ARISTOTLE_API_KEY=arstl_..." > .env
```

Get your key at [aristotle.harmonic.fun/dashboard](https://aristotle.harmonic.fun/dashboard).

### 3. Set up the shared Mathlib cache (saves ~7.7 GB on subsequent projects)

Clone this repo **inside a dedicated parent folder**. Lake will download Mathlib (~7.7 GB) into `.lean-packages/` in that parent folder, and any other Lean projects you clone there will reuse it automatically.

```bash
mkdir ~/lean-projects
cd ~/lean-projects
git clone <this-repo-url>
cd automated-proofs
```

```
lean-projects/
├── .lean-packages/        ← Mathlib lives here (downloaded once, ~7.7 GB)
├── automated-proofs/      ← this repo
└── another-proof/         ← reuses the same Mathlib, no re-download
```

No other setup needed — `lake build` handles the download on first run.

### 4. Build

```bash
lake build
```

### 5. Run the workflow

```bash
python scripts/status.py                                               # check sorry counts
python scripts/submit.py my_theorems/Paper.md "Fill in the sorries"   # submit to Aristotle
python scripts/retrieve.py                                             # download results
```

See `CLAUDE.md` for the full workflow, Lean type conventions, and Aristotle tips.

## Using this template

1. Click **Use this template** → **Create a new repository**
2. Clone your new repo
3. Rename the module: update `name` and `defaultTargets` in `lakefile.toml`, rename `AutomatedProofs.lean` and the `AutomatedProofs/` directory, update all `import AutomatedProofs` references
4. Follow the quick start above

## Lean environment

- **Toolchain**: `leanprover/lean4:v4.28.0`
- **Mathlib**: `v4.28.0` (commit `8f9d9cff6bd728b17a24e163c9402775d9e6a365`)
- Pinned to match Aristotle's fixed environment — returned proofs compile locally without porting.
