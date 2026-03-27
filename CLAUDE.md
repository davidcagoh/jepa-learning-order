# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The parent `../CLAUDE.md` covers all shared workflow, Lean conventions, Mathlib API patterns, and Aristotle submission guidelines. This file covers only what is specific to this project.

---

## Project: JEPA Learning Dynamics Proof

**Theorem**: A depth-L ≥ 2 linear JEPA model, trained from small random initialisation, learns features in decreasing order of their generalised regression coefficient ρ\*, even when the input and cross-covariance matrices share no common eigenbasis.

**Paper draft**: `my_theorems/JEPA_v4_current.md`

---

## Lean Source Files

Import order (dependency chain):

```
AutomatedProofs/Lemmas.lean         ← classical results not in Mathlib
                                        (PD Frobenius bound, variable-coefficient Grönwall)
AutomatedProofs/OffDiagHelpers.lean ← bridging lemmas (imports Lemmas)
AutomatedProofs/JEPA.lean           ← main proof (imports Lemmas + OffDiagHelpers)
AutomatedProofs.lean                ← entry point (add imports here in dependency order)
```

`AutomatedProofs/Basic.lean` is a placeholder stub — it is not part of the JEPA proof.

---

## Scripts

Run from this directory using the parent scripts path:

```bash
python ../scripts/status.py
python ../scripts/submit.py my_theorems/JEPA_v4_current.md "Fill in all the sorries" --dry-run
python ../scripts/submit.py my_theorems/JEPA_v4_current.md "Fill in all the sorries"
python ../scripts/retrieve.py
python ../scripts/watch.py     # adaptive background poller
```

---

## Building

```bash
lake build                              # build whole project
lake build AutomatedProofs.JEPA        # elaborate JEPA.lean only
lake build AutomatedProofs.Lemmas      # elaborate Lemmas.lean only
```
