# New Theorem Skill

Scaffold a new Lean theorem from a markdown paper and prepare it for Aristotle submission.

**Usage:** `/new-theorem my_theorems/Paper.md`

This skill works in any Lean project in this workspace. It reads `lakefile.toml` to determine the module name automatically.

---

## Steps

### 1. Read the paper

Read the markdown paper at `$ARGUMENTS`. If no argument is given, ask the user for the paper path.

### 2. Determine the module name

Read `lakefile.toml` to find the `[[lean_lib]] name = "..."` field — this is the Lean module name (e.g. `AutomatedProofs`, `SimplicialLatentGeometry`). Use this throughout.

### 3. Infer the theorem name

Derive a `CamelCase` Lean module name from the paper title or filename (e.g. `JEPA_Proof_Lecture.md` → `JEPA`, a paper titled "Convergence of SGD" → `SGDConvergence`). Show the inferred name and ask the user to confirm or correct it before proceeding.

### 4. Identify the proof structure

Read the paper carefully and identify:
- All **definitions** (structures, defs)
- All **lemmas** in the order they appear
- The **main theorem(s)**
- Any lemmas that are classical results *not* directly available in Mathlib (e.g. variable-coefficient Grönwall, Rayleigh quotient bounds) — these are candidates for a separate `Lemmas.lean` or a new `<Name>Helpers.lean`

Before writing any files, briefly summarise to the user:
- How many lemmas/theorems you found
- Which (if any) you propose to factor out into a helper file and why
- The proposed file structure (e.g. `JEPA.lean` + `OffDiagHelpers.lean`)

Ask the user to confirm this structure before writing files.

### 5. Write the Lean skeleton

Create `<Module>/<TheoremName>.lean` following this structure exactly:

```lean
import Mathlib
-- import <Module>.Lemmas         -- uncomment if helper lemmas exist
-- import <Module>.<Name>Helpers  -- uncomment if bridging helpers exist

set_option linter.style.longLine false
set_option linter.style.whitespace false

/-!
# <Paper Title>

<One-paragraph summary of what is proved.>
-/

-- open scoped Matrix  -- add if matrices are used

/-! ## Definitions -/

-- All structures and defs first

/-! ## Lemmas -/

-- Each lemma with a PROVIDED SOLUTION docstring

/-! ## Main Theorem -/

-- Main theorem last
```

For each sorry'd lemma or theorem, write the docstring in this form:

```lean
/-- **Lemma N.M (Name).** <Statement in plain English>.

    PROVIDED SOLUTION
    Step 1: <first proof step from the paper>
    Step 2: <key substitution or identity>
    Step 3: <conclusion> -/
lemma my_lemma ... : ... := by
  sorry
```

**Lean type conventions to follow** (see parent CLAUDE.md):
- `Matrix (Fin d) (Fin d) ℝ` for d×d real matrices
- `Real.rpow x r` for non-integer real exponents — **never** `x ^ (n / m)` where `n m : ℕ` (integer division truncates)
- `open scoped Matrix` at top if transpose notation `Mᵀ` is needed
- `dotProduct u v` (top-level, not `Matrix.dotProduct`)
- `‖M‖` for operator norm, `inner u v` for inner product

If a helper file is needed, create it at `<Module>/<Name>Helpers.lean` with the same sorry+PROVIDED SOLUTION pattern.

### 6. Update the entry file

Add the new import(s) to `<Module>.lean` in dependency order. Read the current file first, then insert — never clobber existing imports. Helper files must appear before the file that imports them.

### 7. Run the dry-run

Run:
```bash
python scripts/submit.py $ARGUMENTS "Fill in all the sorries" --dry-run
```

Show the full output to the user (file list with sizes).

### 8. Review scope with the user

After the dry-run, run:
```bash
python scripts/status.py
```

Then ask the user these three questions explicitly. **Wait for answers before proceeding.**

1. **File list** — Does the package look right? Any unexpected files included or missing?

2. **Scope** — Is the number of sorries appropriate for a single Aristotle job? Aristotle performs best on focused asks. Guidelines:
   - ≤5 sorries in one file: usually fine as one job
   - >5 sorries, or sorries spread across multiple files: consider splitting by lemma cluster
   - If splitting, which lemmas should go in the first submission?

3. **Proof difficulty** — For each sorry, is it:
   - Mechanical/algebraic (Aristotle handles well — `ring`, `linarith`, `omega`, `norm_num`)
   - ODE/integral argument (needs a detailed PROVIDED SOLUTION with explicit steps)
   - Genuinely novel (may need multiple Aristotle rounds)

   If any sorries have thin PROVIDED SOLUTION blocks, offer to flesh them out now.

### 9. Offer to submit

Once the user has answered and you've made any adjustments (splitting files, enriching PROVIDED SOLUTIONs), ask:

> "Scope looks good. What prompt should I use? Or I can suggest one based on the sorries."

Suggest a focused prompt if the user wants one, e.g.:
- `"Fill in the sorries in <Module>/<TheoremName>.lean"` (single file)
- `"Fill in the sorries in <Module>/<TheoremName>.lean. Focus on <LemmaA> and <LemmaB>. Each has a detailed PROVIDED SOLUTION."` (targeted)

Then, with the user's confirmed prompt, run:
```bash
python scripts/submit.py $ARGUMENTS "<confirmed prompt>"
```

After submission, remind the user:
```
When Aristotle emails, run:  python scripts/retrieve.py
Check status any time:        python scripts/status.py
```
