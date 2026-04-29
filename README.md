# jepa-learning-order

Lean 4 formalization of the feature learning order of a depth-L linear JEPA model under small random initialization. Part of the [Stochastic Proofs](../stochastic-proofs-handbook) workspace.

## Repository structure

| Path | Role |
|---|---|
| `JepaLearningOrder/JEPA.lean` | Main theorem — `JEPA_rho_ordering` |
| `JepaLearningOrder/Lemmas.lean` | Supporting lemmas (Grönwall, PD bounds) |
| `JepaLearningOrder/BootstrapLemmas.lean` | Sub-lemmas decomposing `bootstrap_consistency` |
| `JepaLearningOrder/OffDiagHelpers.lean` | Off-diagonal bridging lemmas |
| `JepaLearningOrder/GronwallIntegral.lean` | Grönwall integral machinery |
| `my_theorems/paper.tex` | LaTeX paper (14pp) |
| `my_theorems/paper_draft.md` | Theorem spec submitted to Aristotle |
| `literature/` | Reference PDFs |
| `requests/` | Aristotle submission prompts |
| `results/` | Aristotle result tarballs |

## Commands

```bash
lake build
lake build JepaLearningOrder.JEPA

python ../stochastic-proofs-handbook/scripts/status.py
python ../stochastic-proofs-handbook/scripts/submit.py my_theorems/paper_draft.md "Fill in the sorries"
python ../stochastic-proofs-handbook/scripts/retrieve.py [project-id]
```

## Setup

```bash
pip install aristotlelib pathspec python-dotenv
# API key in lean-workspace/.env — no per-project .env needed
lake build
```

Lean toolchain: `leanprover/lean4:v4.28.0` · Mathlib: `v4.28.0` · Shared cache: `../.lean-packages/`
