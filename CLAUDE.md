# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Goals (in priority order)

1. **Parse and verify** — Submit Lean files with `sorry` placeholders to Aristotle, parse the returned `.tar.gz`, verify the proofs compile locally.
2. **Optimize** — Re-submit completed proofs with prompts like "Golf all the proofs: minimize tactic count and simplify where possible."
3. **Multi-agent proof space exploration** — Run multiple parallel Aristotle submissions with varied prompts/hints, compare outputs, and select the best proof strategy.

---

## Python Setup

Install dependencies before running any scripts (Python 3.10+ required):

```bash
pip install aristotlelib pathspec python-dotenv
# or with uv:
uv pip install aristotlelib pathspec python-dotenv
```

Create `.env` in the project root:
```
ARISTOTLE_API_KEY=arstl_...
```

---

## Scripts (Workflow)

Two user-facing scripts cover the full workflow. Always run from the project root (where `.env` lives).

### `submit.py` — package and submit a paper to Aristotle

```bash
python scripts/submit.py my_theorems/Paper.md "Fill in the sorries"
python scripts/submit.py my_theorems/Paper.md "Fill in the sorries" --project-dir .
```

- Paper path is required as the first argument — it links this submission to its source.
- Packages the Lean project (respects `.gitignore`, strips `results/`, `scripts/`, `my_theorems/`, `.claude/`, `.github/`, `proofs-from-literature/`, `memory/`, `reports/`, `help_from_aristotle/`).
- Writes `results/<id>.meta.json` to track the paper→job mapping.
- Exits immediately; Aristotle emails when done.

### `retrieve.py` — check, download, and annotate

```bash
python scripts/retrieve.py                    # check all tracked jobs
python scripts/retrieve.py <project-id>       # check a specific job
```

- Scans all `results/*.meta.json` (jobs you've submitted via `submit.py`).
- For each: if already downloaded, re-annotates; if done on Aristotle, downloads and annotates; if still in progress, prints status.
- Writes `reports/<PaperName>_annotated.md` for each completed job.
- **This is the script to run when Aristotle emails you.**

### `_report.py` — internal helper (not user-facing)

Called automatically by `retrieve.py`. Do not invoke directly — it will print an error and exit. Use `retrieve.py` instead.

### `_common.py` — shared utilities (not user-facing)

Imported by `submit.py` and `retrieve.py`. Provides `load_env()` which reads `.env` from the current working directory. Do not invoke directly.

Reads the Aristotle result tar, compares against local Lean sources, and writes an annotated copy of the paper with inline callouts:

- `✓ Proved` — sorry filled
- `◑ Proved vacuously` — proved but non-substantively
- `⚠️ Needs revision` — sorry remains

### `status.py` — project dashboard

```bash
python scripts/status.py
```

- Shows sorry count per `.lean` file with line numbers.
- Shows all tracked submissions with live status (polls Aristotle API), age, and prompt preview.
- Run any time to see where things stand without reading any files manually.

### Typical session

```bash
# Check current state
python scripts/status.py

# Preview what would be packaged (always do this before submitting)
python scripts/submit.py my_theorems/JEPA.md "Fill in all the sorries" --dry-run

# Submit for real
python scripts/submit.py my_theorems/JEPA.md "Fill in all the sorries"
# → prints project ID, exits immediately

# Later: Aristotle emails you. Run:
python scripts/retrieve.py
# → downloads completed jobs, writes reports/<PaperName>_annotated.md
```

---

## Lean 4 / Aristotle Workflow

### Taking a markdown proof paper → Lean skeleton → submission

**Drop-in convention:** Place the markdown paper in `my_theorems/`. Claude will read it, create the Lean skeleton, add the import, and (after packaging confirmation) submit.

**Step 1 — Create the Lean skeleton**

Create `AutomatedProofs/<TheoremName>.lean` with this structure:

```lean
import Mathlib

/-! Module docstring summarising the paper -/

-- 1. All definitions first (structures, defs)
-- 2. Each lemma/theorem in paper order, with a /-- ... -/ docstring
--    containing the exact proof steps as a PROVIDED SOLUTION block
-- 3. Main theorem last, with full proof outline in its PROVIDED SOLUTION block
```

**Step 2 — Add the import**

```lean
-- AutomatedProofs.lean
import AutomatedProofs.Basic
import AutomatedProofs.Lemmas   -- if helper lemmas exist
import AutomatedProofs.<TheoremName>
```

**Helper lemma pattern (Lemmas.lean / OffDiagHelpers.lean):** When a theorem file needs classical results not in Mathlib, or small bridging lemmas that don't belong in the main file, create a separate helper lean file with sorry'd lemmas and detailed `PROVIDED SOLUTION` docstrings. The theorem file imports it. This lets Aristotle fill in both helpers and the main theorem in one submission. Examples:
- `Lemmas.lean` — Grönwall, Rayleigh quotient lower bounds (classical results)
- `OffDiagHelpers.lean` — ε-rpow monotonicity, Bochner integral bounds (bridging lemmas specific to one proof)

Import order in `AutomatedProofs.lean` matters: list files in dependency order (e.g. `Basic → Lemmas → OffDiagHelpers → JEPA`).

**Step 3 — Submit**

`scripts/submit.py` builds a filtered archive automatically. It respects `.gitignore` (so `.env` is excluded) and strips `results/`, `scripts/`, `my_theorems/`, `.claude/`, `.github/`, `help_from_aristotle/` from the upload. Only the Lean project files are sent:

```
.gitignore  AutomatedProofs.lean  AutomatedProofs/*.lean
lakefile.toml  lean-toolchain  lake-manifest.json  README.md
```

```bash
python scripts/submit.py my_theorems/<TheoremName>.md "Fill in the sorries"
```

Use the prompt: `"Fill in all the sorries in this project"`. For a single file, add a hint: `"Fill in the sorries in AutomatedProofs/<TheoremName>.lean"`. The paper path (first positional argument) is already recorded in the `.meta.json` so `retrieve.py` can find the paper automatically.

**Step 5 — Retrieve results**

When Aristotle emails you, say so and Claude will immediately run `python scripts/retrieve.py`. To fetch a specific ID: `python scripts/retrieve.py <project-id>`.

---

### Lean type conventions

| Mathematical object | Lean type |
|---|---|
| d×d real matrix | `Matrix (Fin d) (Fin d) ℝ` |
| Positive definite matrix | `Matrix.PosDef M` |
| Matrix inverse | `M⁻¹` (via `Matrix.nonsing_inv`) |
| Matrix trace | `Matrix.trace M` |
| Matrix transpose | `Mᵀ` (requires `open scoped Matrix` at top of file) |
| Matrix-vector product | `M.mulVec v` |
| Dot product | `dotProduct u v` (NOT `Matrix.dotProduct` — `dotProduct` is top-level, not in the `Matrix` namespace) |
| Real square root | `Real.sqrt x` |
| Real power | `x ^ (r : ℝ)` or `Real.rpow x r` for non-integer exponents |
| Norm | `‖M‖` (operator norm via `NNNorm`) |
| Inner product | `inner u v` (from `InnerProductSpace`) |
| Finite sum | `∑ i : Fin n, f i` |
| ODE derivative | `deriv f t` (for `f : ℝ → X`) |
| Set interval | `Set.Icc a b`, `Set.Ioo a b` |

When unsure of the exact Mathlib lemma name, write a best-effort attempt and add `-- TODO: check Mathlib name`.

### Common Lean proof pitfalls (confirmed in v4.29.0-rc6)

**ℕ truncation in exponents**: When building a real-valued exponent from natural number variables, **never** write `2 * (L - k) / L` where `L k : ℕ` — integer division truncates. Cast to `ℝ` first:

```lean
-- WRONG: exponent truncates (e.g. L=3, k=0 → 4/3 truncates to 1)
sigma ^ (2 * (L - k) / L)

-- RIGHT: exact rational exponent
Real.rpow sigma (2 * ((L : ℝ) - (k : ℝ)) / (L : ℝ))
```

This was the root cause of the `preconditioner` definition bug (see `help_from_aristotle/proof_decisions_log.md`).

**`Matrix.mulVec_mulVec` argument order**: The first explicit arg is `v` (the vector), **not** `M`. Passing a matrix as the first explicit arg causes silent pattern mismatch in `rw`. Use bare `← Matrix.mulVec_mulVec` without explicit args; apply multiple times to step through the chain.

```lean
-- WRONG (passes matrix as v):
rw [← Matrix.mulVec_mulVec Wbar]

-- RIGHT (bare, rewrites first occurrence of (A*B)*ᵥv):
rw [← Matrix.mulVec_mulVec]
```

**Renamed lemmas in v4.29.0-rc6**:
- `pow_le_pow_left` → `pow_le_pow_left₀`
- `div_lt_div_iff` → `div_lt_div_iff₀`

**`λ` (Unicode U+03BB) is a Lean 4 keyword** — using it in an identifier (e.g. `hλr`) causes a parse error. Use `hLr`, `hlam`, etc. instead.

**`let` bindings in theorem return types**: A `let x := expr` in the return type of a `lemma` creates a local def visible in the goal, but `x` is **not** in scope as a named term during `exact`. Inline the full expression instead of referring to `x` by name.

```lean
-- WRONG:
exact ⟨0, t_crit_leading, by simp, by simp⟩   -- "Unknown identifier 't_crit_leading'"

-- RIGHT:
exact ⟨0, (L : ℝ) / (projectedCovariance dat eb r * ...), by simp, by simp⟩
```

**`set_option maxHeartbeats N in` placement**: Must appear **before** the docstring, not between the docstring and the lemma declaration.

**Interval integral of 1**: `integral_one : ∫ _ in a..b, (1 : ℝ) = b - a` is in the **top-level** namespace (not `intervalIntegral`), declared after `end intervalIntegral` in `Mathlib.Analysis.SpecialFunctions.Integrals.Basic`. Connect Ioo set integrals to interval integrals via: `MeasureTheory.integral_Ioc_eq_integral_Ioo` (Ioo↔Ioc) then `intervalIntegral.integral_of_le` (Ioc↔interval).

---

### Verified Mathlib API patterns (v4.29.0-rc6)

These have been confirmed to compile locally — use exactly as written:

```lean
-- Scoped notations requiring explicit open:
open scoped Matrix          -- enables Mᵀ for Matrix.transpose

-- FTC (interval integral, upper limit):
intervalIntegral.integral_hasDerivAt_right
  (hf : IntervalIntegrable f volume a b)
  (hmeas : StronglyMeasurableAtFilter f (nhds b) volume)
  (hb : ContinuousAt f b) : HasDerivAt (fun u => ∫ x in a..u, f x) (f b) b

-- StronglyMeasurableAtFilter from ContinuousOn:
ContinuousOn.stronglyMeasurableAtFilter
  (hs : IsOpen s) (hf : ContinuousOn f s) :
  ∀ x ∈ s, StronglyMeasurableAtFilter f (nhds x) volume
-- Usage: ContinuousOn.stronglyMeasurableAtFilter isOpen_Ioo (hf.mono Set.Ioo_subset_Icc_self) s hs

-- Continuity of integral primitive:
intervalIntegral.continuousOn_primitive_interval' (h_int : IntervalIntegrable f μ b₁ b₂)
  (ha : a ∈ [[b₁, b₂]]) : ContinuousOn (fun b => ∫ x in a..b, f x) [[b₁, b₂]]

-- Antitone from nonpositive derivative:
antitoneOn_of_deriv_nonpos (hD : Convex ℝ D) (hf : ContinuousOn f D)
  (hf' : DifferentiableOn ℝ f (interior D))
  (hf'_nonpos : ∀ x ∈ interior D, deriv f x ≤ 0) : AntitoneOn f D

-- HasDerivAt for exp composition:
hB_da.neg.exp : HasDerivAt (fun x => Real.exp (-B x)) (Real.exp (-B s) * (-β s)) s
-- (uses HasDerivAt.exp, which is Real exp, not Complex)

-- ContinuousOn.exp for real functions:
hB_cont.neg.rexp : ContinuousOn (fun x => Real.exp (-B x)) S
-- (use .rexp not .exp for Real.exp)

-- Division/exp identities:
Real.exp_neg : Real.exp (-x) = (Real.exp x)⁻¹
div_inv_eq_mul : a / b⁻¹ = a * b
le_div_iff₀ (hc : 0 < c) : a ≤ b / c ↔ a * c ≤ b
div_le_div_of_nonneg_right (hab : a ≤ b) (hc : 0 ≤ c) : a / c ≤ b / c
```

### PROVIDED SOLUTION block pattern

```lean
/-- **Lemma N.M (Name).** Statement of the lemma.

    PROVIDED SOLUTION
    Step 1: First proof step from the paper.
    Step 2: Key substitution or identity used.
    Step 3: Conclusion. -/
lemma my_lemma ... : ... := by
  sorry
```

Aristotle reads the header docstring including `PROVIDED SOLUTION`. It does **not** read comments inside the `by` block.

---

## Lean Environment

### This project (local build)

- **Toolchain**: `leanprover/lean4:v4.28.0` (pinned in `lean-toolchain`)
- **Mathlib**: `v4.28.0` / commit `8f9d9cff6bd728b17a24e163c9402775d9e6a365` (in `lakefile.toml`)
- **Entry point**: `AutomatedProofs.lean` imports all submodules
- **Source files**: `AutomatedProofs/` — add new `.lean` files here and import them from `AutomatedProofs.lean`

### Aristotle's fixed environment (what it runs remotely)

**Critical**: Aristotle runs on fixed versions regardless of your project's toolchain:

- **Lean Toolchain**: `leanprover/lean4:v4.28.0`
- **Mathlib commit**: `8f9d9cff6bd728b17a24e163c9402775d9e6a365` (corresponds to `v4.28.0` tag)

**This project already matches Aristotle's environment.** Proofs returned by Aristotle should compile locally without porting. If a proof still fails to compile, the issue is a hallucinated lemma name or syntax error in Aristotle's output — not a version mismatch.

### Lean options (from lakefile.toml)

- `pp.unicode.fun = true` — pretty-prints lambdas as `fun a ↦ b`
- `relaxedAutoImplicit = false` — all variables must be explicitly declared
- `weak.linter.mathlibStandardSet = true` — mathlib's standard linter set is active
- `maxSynthPendingDepth = 3` — typeclass synthesis depth limit

### Build commands

```bash
lake build                          # build the whole project
lake update                         # update dependencies
lake build AutomatedProofs.Basic    # elaborate a specific file
```

CI runs `leanprover/lean-action` → `lake build` → `docgen-action`.

### Mathlib usage

```lean
import Mathlib.Tactic        -- common tactics: simp, ring, linarith, omega, norm_num, …
import Mathlib.Data.Nat.Basic -- specific modules
import Mathlib                -- full library (slow, use focused imports instead)
```

---

## Aristotle API

Aristotle is an automated theorem proving system by Harmonic that proves and formally verifies graduate and research-level problems in math, software, and more using Lean 4.

- **Dashboard / docs**: https://aristotle.harmonic.fun/dashboard/docs/
- **API key**: stored in `.env` as `ARISTOTLE_API_KEY` (file is `.gitignore`d). Load with `python-dotenv` or `source .env`.
- **Python package**: `aristotlelib` (Python 3.10+ required)

### Installation

```bash
# Run directly without install (recommended for one-offs)
uvx --from aristotlelib@latest aristotle --version

# Install into environment
uv pip install aristotlelib
# or
pip install aristotlelib
pip install --upgrade aristotlelib   # upgrade
```

### Authentication

```python
import aristotlelib

# Option 1: environment variable (preferred)
# ARISTOTLE_API_KEY=arstl_...  set in .env or shell

# Option 2: programmatic
aristotlelib.set_api_key("arstl_...")
```

---

## SDK Reference — Complete API Surface

All project methods are **async**. Import pattern:

```python
import aristotlelib
from aristotlelib import Project, ProjectStatus, AristotleAPIError
```

---

### Creating Projects

#### `Project.create(prompt, tar_file_path=None, public_file_path=None)`

Create a new project from a prompt and optional tar archive.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `prompt` | `str` | yes | Instructions telling Aristotle what to do |
| `tar_file_path` | `Path \| str` | no | Path to a `.tar.gz` file with supplementary files |
| `public_file_path` | `str` | no | Name saved for the input file for future reference (defaults to `tar_file_path`) |

**Returns**: `Project`

```python
# Prompt only
project = await Project.create(prompt="Prove that 1 + 1 = 2")

# With a tar archive
project = await Project.create(
    prompt="Fill in the sorries",
    tar_file_path="./my-project.tar.gz"
)
```

---

#### `Project.create_from_directory(prompt, project_dir)`

Create a project from a local directory. Automatically archives the directory, skipping build artifacts (`.olean`, `.lake/packages/`) and standard library packages.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `prompt` | `str` | yes | Instructions telling Aristotle what to do |
| `project_dir` | `Path \| str` | yes | Path to a directory (e.g., a Lean project) |

**Returns**: `Project`

```python
project = await Project.create_from_directory(
    prompt="Fill in the sorries",
    project_dir="./my-lean-project"
)
```

The SDK automatically:
- Detects your project root
- Validates file paths
- Resolves imports to include dependencies
- Handles file size limits (100 MB max per file)

---

### Retrieving Projects

#### `Project.from_id(project_id)`

Load an existing project by its ID.

| Parameter | Type | Description |
|---|---|---|
| `project_id` | `str` | The project UUID |

**Returns**: `Project`

```python
project = await Project.from_id("abc-123-def")
```

---

#### `Project.list_projects(pagination_key=None, limit=30, status=None)`

List projects ordered by creation date (most recent first).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pagination_key` | `str` | `None` | Key from previous response for pagination |
| `limit` | `int` | `30` | Number of results to return (1–100) |
| `status` | `ProjectStatus \| list[ProjectStatus]` | `None` | Filter by status |

**Returns**: `tuple[list[Project], str | None]` — projects list and next pagination key (or `None` if no more pages)

```python
# First page
projects, next_key = await Project.list_projects(limit=10)

# Next page
projects2, _ = await Project.list_projects(limit=10, pagination_key=next_key)

# Filter by status
active, _ = await Project.list_projects(
    status=[ProjectStatus.QUEUED, ProjectStatus.IN_PROGRESS]
)

# Only completed
done, _ = await Project.list_projects(status=ProjectStatus.COMPLETE)
```

---

### Working with Results

#### `project.wait_for_completion(destination=None, polling_interval_seconds=30)`

Poll until the project completes, then download the result.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `destination` | `Path \| str` | `None` | Where to save the solution `.tar.gz` |
| `polling_interval_seconds` | `int` | `30` | Seconds between status checks |

**Returns**: `str | None` — path to downloaded file, or `None` if canceled/failed

```python
result_path = await project.wait_for_completion(
    destination="results/output.tar.gz"
)
```

---

#### `project.get_solution(destination=None)`

Download the completed solution file immediately (does not wait).

| Parameter | Type | Description |
|---|---|---|
| `destination` | `Path \| str` | Where to save the solution |

**Returns**: Path to downloaded file

```python
path = await project.get_solution(destination="results/solution.tar.gz")
```

---

#### `project.get_input(destination=None)`

Download the original input file that was submitted.

| Parameter | Type | Description |
|---|---|---|
| `destination` | `Path \| str` | Where to save the input |

**Returns**: Path to downloaded file

---

### Project Lifecycle

#### `project.refresh()`

Refresh the project's status from the API. Call this in a polling loop.

```python
await project.refresh()
print(project.status)         # ProjectStatus.IN_PROGRESS
print(project.percent_complete)  # e.g., 42
```

#### `project.cancel()`

Cancel a QUEUED or IN_PROGRESS project.

```python
await project.cancel()
```

---

### Project Properties

| Property | Type | Description |
|---|---|---|
| `project_id` | `str` | Unique identifier (UUID) |
| `status` | `ProjectStatus` | Current status enum value |
| `created_at` | `datetime` | When the project was created |
| `last_updated_at` | `datetime` | When the status was last updated |
| `percent_complete` | `int \| None` | Completion percentage (only meaningful when IN_PROGRESS) |
| `input_prompt` | `str \| None` | The prompt text submitted |
| `file_name` | `str \| None` | Name of the input file |
| `description` | `str \| None` | Project description |

---

### ProjectStatus Enum

```python
class ProjectStatus(Enum):
    UNKNOWN = "UNKNOWN"              # unrecognized status — upgrade aristotlelib
    NOT_STARTED = "NOT_STARTED"      # DEPRECATED — legacy projects only
    QUEUED = "QUEUED"                # waiting to be processed
    IN_PROGRESS = "IN_PROGRESS"      # Aristotle is actively working
    COMPLETE = "COMPLETE"            # all tasks finished successfully
    COMPLETE_WITH_ERRORS = "COMPLETE_WITH_ERRORS"  # finished but had trouble parsing input
    OUT_OF_BUDGET = "OUT_OF_BUDGET"  # ran out of compute budget; partial results available
    FAILED = "FAILED"                # unrecoverable internal error; team notified
    CANCELED = "CANCELED"            # canceled by user
```

Terminal statuses (no further changes): `COMPLETE`, `COMPLETE_WITH_ERRORS`, `OUT_OF_BUDGET`, `FAILED`, `CANCELED`

---

### Error Handling

The SDK raises `AristotleAPIError` for API errors. It includes a `status_code` attribute when the error came from an HTTP response.

```python
from aristotlelib import AristotleAPIError

try:
    project = await Project.create(prompt="My prompt")
except AristotleAPIError as e:
    print(f"API error (HTTP {e.status_code}): {e}")
```

---

## CLI Reference

The `aristotle` CLI mirrors the SDK. All commands accept `--api-key` to override the env var.

### Submit

```bash
# Submit a prompt (returns project ID immediately)
aristotle submit "Prove that the square root of 2 is irrational"

# Submit and wait for completion with live progress
aristotle submit "Prove that sqrt(2) is irrational" --wait

# Submit a Lean project directory
aristotle submit "Fill in the sorries" --project-dir ./my-lean-project --wait

# Submit and save result to file
aristotle submit "Fill in the sorries" --project-dir ./my-lean-project --wait \
    --destination output.tar.gz
```

### Formalize a document

```bash
aristotle formalize paper.tex --wait --destination output.tar.gz
```

### List projects

```bash
aristotle list                                    # 10 most recent
aristotle list --limit 20                         # control count (1-100, default 10)
aristotle list --status COMPLETE IN_PROGRESS      # filter by status
aristotle list --pagination-key <key>             # paginate
```

### Download results

```bash
# Download completed project
aristotle result <project-id> --destination output.tar.gz

# Wait for running project, then download
aristotle result <project-id> --wait --destination output.tar.gz
```

### Cancel

```bash
aristotle cancel <project-id>
```

---

## Result File Structure

When Aristotle completes a project, the result is a `.tar.gz` file. After extraction:

```
{project_id}_aristotle/
├── ARISTOTLE_SUMMARY_{project_id}.md    # human-readable summary of what changed
├── README.md                            # citation boilerplate (tag @Aristotle-Harmonic)
├── lake-manifest.json                   # pinned deps at Aristotle's versions
├── lakefile.toml                        # configured for Aristotle's Lean/Mathlib versions
├── lean-toolchain                       # leanprover/lean4:v4.28.0
└── RequestProject/                      # the actual Lean library directory
    ├── .gitkeep
    └── {TheoremName}.lean               # proven Lean files (sorries filled in)
```

### Key files to parse

**`ARISTOTLE_SUMMARY_{project_id}.md`** — always read this first. Describes what Aristotle changed, what theorems were proven, what tactics were used, and whether the file compiles with no sorries.

**`RequestProject/*.lean`** — the proven Lean source files. These use `leanprover/lean4:v4.28.0` and `import Mathlib`. Parse these to extract the proof tactics.

**`lean-toolchain`** — always contains `leanprover/lean4:v4.28.0` (Aristotle's fixed version).

**`lakefile.toml`** — uses `rev = "v4.28.0"` for mathlib; the library is named `RequestProject`.

**`lake-manifest.json`** — fully pinned dependency tree:
- `mathlib` → `rev: "8f9d9cff6bd728b17a24e163c9402775d9e6a365"`, `inputRev: "v4.28.0"`
- Plus transitive deps: `plausible`, `LeanSearchClient`, `importGraph`, `proofwidgets`, `aesop`, `Qq`, `batteries`, `lean4-cli`

### Parsing the result in Python

```python
import tarfile, pathlib, re

def extract_result(tar_path: str, out_dir: str) -> dict:
    """Extract result tar and return key file paths."""
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)

    root = next(pathlib.Path(out_dir).iterdir())  # {project_id}_aristotle/
    project_id = root.name.removesuffix("_aristotle")

    summary = (root / f"ARISTOTLE_SUMMARY_{project_id}.md").read_text()
    lean_files = list((root / "RequestProject").glob("*.lean"))

    return {
        "project_id": project_id,
        "root": root,
        "summary": summary,
        "lean_files": lean_files,
    }
```

### Counterexample output format

When Aristotle finds a statement to be false, it leaves a block comment on the theorem:

```lean
/-
Aristotle found this block to be false.
Here is a proof of the negation:
theorem my_theorem ... := by
    negate_state;
    -- proof of negation here
-/
theorem my_theorem ... := by
  sorry
```

The original `sorry` is left in place; the negation proof is in the comment above.

---

## Submitting Lean Files to Aristotle

### Project requirements

Your Lean project must include:
- `lakefile.toml` (or `lakefile.lean`)
- `lean-toolchain`
- `.lean` source files with proper import structure

The SDK skips build artifacts (`.olean`, `.lake/packages/`) when packaging.

### Guiding Aristotle with proof sketches

Include a natural language proof sketch in the theorem's doc-comment tagged with `PROVIDED SOLUTION`. Aristotle reads the header comment but **not comments inside proof blocks**.

```lean
/--
Theorem statement here.

PROVIDED SOLUTION
Step 1: reduce to case X.
Step 2: apply lemma Y.
Step 3: conclude by monotonicity of Z.
-/
theorem my_theorem ... := by
  sorry
```

### Working with definitions

Aristotle does not modify definitions by default:

```lean
def foo : Nat := by sorry   -- Aristotle will NOT fill this
```

Aristotle will create its own data/definitions where needed, but leaves yours intact.

---

## Effective Prompts

| Goal | Prompt |
|---|---|
| Fill sorries | `"Fill in all the sorries in this project"` |
| Tactic constraints | `"Prove this using only ring and omega, avoiding heavy automation"` |
| Proof golfing | `"Golf all the proofs in this project: minimize tactic count and simplify where possible"` |
| Auxiliary lemmas | `"Build auxiliary lemmas that would help prove the main sorry'd goal in this file"` |
| API development | `"Develop API lemmas for the main structure: coercions, simp lemmas, and basic properties"` |
| Proof repair | `"Fix all compilation errors and linter warnings in this project"` |
| Formal skeleton | `"Build a formal sorry'd skeleton closely following my paper, with theorem statements matching each result"` |
| Code quality | `"Formalize this paper and make sure the code quality closely follows Mathlib standards"` |
| Docstrings | `"Fill in the sorries and add detailed docstrings explaining each definition, theorem, and proof step for Lean beginners"` |
| Modularize | `"Refactor this file into a modular structure: extract helper lemmas, group related definitions, and minimize imports"` |

---

## Resuming Out-of-Budget Projects

If a project reaches `OUT_OF_BUDGET`, partial results are available. Resume:

```bash
aristotle result <project-id> --destination partial.tar.gz
mkdir partial-output && tar -xzf partial.tar.gz -C partial-output
aristotle submit "Fill in the sorries" --project-dir ./partial-output --wait
```

Or in Python:

```python
partial_path = await project.get_solution(destination="partial.tar.gz")
# extract, then create new project from the extracted directory
new_project = await Project.create_from_directory(
    prompt="Fill in the remaining sorries",
    project_dir="partial-output/..."
)
```

---

## Gotchas and Notes

### Version mismatch between local project and Aristotle

This project is pinned to `leanprover/lean4:v4.28.0` + mathlib `8f9d9cff...` — the same as Aristotle. No porting step is needed. If an Aristotle proof fails to compile, the issue is a hallucinated lemma name or broken syntax, not a version difference. Fix by finding the correct Mathlib v4.28.0 lemma name via `exact?` / `apply?` or by searching Mathlib source.

### `aesop` non-terminal warning

```
aesop: failed to prove the goal after exhaustive search
```

This is **not an error**. It's expected when `aesop` is used as a non-terminal tactic (other tactics follow it). Suppress with:

```lean
aesop (config := { warnOnNonterminal := false })
```

### `--wait` vs polling manually

- `--wait` blocks the CLI/SDK call with a live progress display.
- Without `--wait`, the command returns immediately with a project ID.
- Use `--wait` for scripts; use manual polling (via `project.refresh()`) for fine-grained control or parallel submissions.

### `negate_state` tactic (Aristotle-internal)

When Aristotle proves a negation/counterexample, it uses a custom tactic `negate_state` that is automatically added to the file header. You do not need to define this yourself; Aristotle injects it. The definition for reference:

```lean
import Mathlib
open Lean Meta Elab Tactic in
elab "revert_all" : tactic => do
  let goals ← getGoals
  let mut newGoals : List MVarId := []
  for mvarId in goals do
    newGoals := newGoals.append [(← mvarId.revertAll)]
  setGoals newGoals

open Lean.Elab.Tactic in
macro "negate_state" : tactic => `(tactic|
  (
    guard_goal_nums 1
    revert_all
    refine @(((by admit) : ∀ {p : Prop}, ¬p → p) ?_)
    try push_neg
  )
)
```

### `COMPLETE_WITH_ERRORS` status

Aristotle finished but had trouble parsing the input. The result tar will still be available — download it and inspect `ARISTOTLE_SUMMARY_*.md` to understand what happened.

### Context files are not modified

When submitting with `--project-dir`, Lean files used as context (definitions, structures you want Aristotle to be aware of) will **not** be modified. Only files with `sorry` placeholders are targeted.

### File size limit

100 MB max per file when uploading.

---

## Python Workflow Pattern

```python
import asyncio
import aristotlelib
from aristotlelib import Project, ProjectStatus, AristotleAPIError
from dotenv import load_dotenv

load_dotenv()  # loads ARISTOTLE_API_KEY from .env

async def submit_and_retrieve(prompt: str, project_dir: str, dest: str) -> str | None:
    """Submit a Lean project, wait for completion, return path to result."""
    try:
        project = await Project.create_from_directory(
            prompt=prompt,
            project_dir=project_dir
        )
        print(f"Submitted: {project.project_id}")

        result_path = await project.wait_for_completion(
            destination=dest,
            polling_interval_seconds=30
        )

        if result_path is None:
            print(f"Project {project.project_id} did not complete: {project.status}")
            return None

        print(f"Done: {result_path}")
        return result_path
    except AristotleAPIError as e:
        print(f"API error (HTTP {e.status_code}): {e}")
        return None

asyncio.run(submit_and_retrieve(
    prompt="Fill in all the sorries",
    project_dir="./AutomatedProofs",
    dest="results/output.tar.gz"
))
```

### Parallel multi-agent exploration

```python
async def explore_proof_space(project_dir: str, prompts: list[str]) -> list[str]:
    """Submit multiple Aristotle jobs in parallel with different prompt strategies."""
    tasks = [
        Project.create_from_directory(prompt=p, project_dir=project_dir)
        for p in prompts
    ]
    projects = await asyncio.gather(*tasks)

    # Wait for all to complete
    results = await asyncio.gather(*[
        p.wait_for_completion(destination=f"results/{p.project_id}.tar.gz")
        for p in projects
    ])

    return [r for r in results if r is not None]
```

---

## Real Example: sqrt(2) is Irrational

A completed Aristotle result is in `results/sqrt2.tar.gz`. After extraction:

```
307e1afd-52cf-46be-bcda-fdab6267b932_aristotle/
├── ARISTOTLE_SUMMARY_307e1afd-52cf-46be-bcda-fdab6267b932.md
├── README.md
├── lake-manifest.json
├── lakefile.toml                     # name = "RequestProject", mathlib rev = "v4.28.0"
├── lean-toolchain                    # leanprover/lean4:v4.28.0
└── RequestProject/
    ├── .gitkeep
    └── Sqrt2Irrational.lean
```

**`Sqrt2Irrational.lean`**:

```lean
import Mathlib

theorem sqrt2_irrational : Irrational (Real.sqrt 2) := by
  -- The irrationality of √2 is a well-known result, so we can use the existing theorem.
  apply irrational_sqrt_two
```

**Summary**: "I formalized and proved that √2 is irrational in `RequestProject/Sqrt2Irrational.lean`. The theorem is stated as `theorem sqrt2_irrational : Irrational (Real.sqrt 2)`. The proof uses Mathlib's existing `irrational_sqrt_two` lemma. The file compiles successfully with no sorries."

---

## File Layout of This Repo

This repo is a GitHub template. After `python scripts/init.py <ProjectName>` the layout is:

```
<repo>/
├── .env                          # ARISTOTLE_API_KEY=arstl_... (gitignored — add your key)
├── .gitignore
├── .github/workflows/
│   ├── lean_action_ci.yml        # CI: lake build + docgen
│   ├── update.yml
│   └── create-release.yml
├── .claude/
│   └── commands/
│       └── new-theorem.md        # /new-theorem skill: scaffold + submit workflow
├── <ProjectName>.lean            # entry point: add imports here in dependency order
├── <ProjectName>/
│   └── Basic.lean                # placeholder; add theorem files here
├── my_theorems/                  # gitignored — markdown papers (source for Lean skeletons)
├── help_from_aristotle/          # gitignored — submission docs + proof_decisions_log.md
├── results/                      # gitignored — downloaded .tar.gz outputs and meta.json files
├── reports/                      # gitignored — annotated markdown reports from retrieve.py
├── scripts/
│   ├── init.py                   # one-time setup: rename library, squash history
│   ├── submit.py                 # package + submit to Aristotle
│   ├── retrieve.py               # check/download completed jobs
│   ├── status.py                 # dashboard: sorry counts + job statuses
│   ├── _report.py                # internal helper (called by retrieve.py)
│   └── _common.py                # shared utilities
├── lakefile.toml                 # project config (Lean v4.28.0, Mathlib v4.28.0, packagesDir = "../.lean-packages")
├── lean-toolchain                # leanprover/lean4:v4.28.0
├── lake-manifest.json            # pinned dependency tree
├── aristotle-docs.md             # Aristotle API docs (gitignored)
└── CLAUDE.md                     # this file
```

**Gitignored directories** (`my_theorems/`, `results/`, `reports/`, `help_from_aristotle/`) are created by `init.py` and live only on disk — they are never committed. Project-specific theorem `.lean` files in `<ProjectName>/` are also gitignored by name once they exist, keeping the committed template clean.

### Starting a new project from this template

```bash
# On GitHub: "Use this template" → name your new repo → clone it
git clone github.com/<you>/<new-repo> && cd <new-repo>
python scripts/init.py <ProjectName>   # rename, seed dirs, squash history
# Add your ARISTOTLE_API_KEY to .env
# Then: /new-theorem my_theorems/<paper>.md
```

### Shared Mathlib cache (saves ~7 GB per project)

`lakefile.toml` is pre-configured with `packagesDir = "../.lean-packages"`. This means Mathlib and all other Lake dependencies are downloaded **one level up** from the project, into a shared `.lean-packages/` directory that all sibling projects can reuse.

**Recommended layout** — clone all your Lean projects into the same parent folder:

```
lean-projects/
├── .lean-packages/          ← Mathlib + deps live here (shared, ~7.7 GB once)
├── automated-proofs/        ← this template
├── my-next-proof/           ← another project from this template
└── yet-another-proof/       ← shares the same Mathlib, no extra download
```

On first `lake build` in a new project, Lake downloads Mathlib into `../.lean-packages/` if it isn't already there. Subsequent projects in the same folder reuse it — no re-download, no duplicate GBs.

If you keep projects in different directories rather than siblings, update `packagesDir` in `lakefile.toml` and `lake-manifest.json` to point to wherever you want the shared cache to live.
