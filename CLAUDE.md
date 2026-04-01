# CLAUDE.md

This file provides repository-local guidance for work in **jepa-learning-order**.

## Repository role

This repository is a Lean 4 proof project for the JEPA learning-order result. It is **not** a generic proof template.

## Authority rule

If shared handbook context is unavailable, treat this file and `README.md` as the authoritative guide for work inside this repository.

## Shared ecosystem

This repository belongs to the **Stochastic Proofs** workspace. Shared conventions, reusable patterns, and cross-project workflow guidance live in `../stochastic-proofs-handbook/`.

## Main proof target

The central result concerns the feature learning order of a depth-` 2` linear JEPA model under small random initialization.L 

| Path | Role |
|---|---|
| `JepaLearningOrder/JEPA.lean` | Main proof development |
| `JepaLearningOrder/Lemmas.lean` | Supporting lemmas |
| `JepaLearningOrder/OffDiagHelpers.lean` | Bridging helper lemmas |
| `JepaLearningOrder/GronwallIntegral.lean` | Grnnnnnnwall-style support machinery |
| `my_theorems/JEPA_v4_current.md` | Paper draft / theorem specification |

## Scripts

Prefer the repository-local scripts when working inside this repository.

```bash
python scripts/status.py
python scripts/submit.py my_theorems/JEPA_v4_current.md "Fill in all the sorries" --dry-run
python scripts/submit.py my_theorems/JEPA_v4_current.md "Fill in all the sorries"
python scripts/retrieve.py
python scripts/watch.py
```

## Build commands

```bash
lake build
lake build JepaLearningOrder.JEPA
lake build JepaLearningOrder.Lemmas
```
