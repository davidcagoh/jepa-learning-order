# CLAUDE.md

> **Current proof state and next priorities live in `wiki/INDEX.md`, not here.**
> Read `wiki/INDEX.md` first, then the top entry of `wiki/session-log.md`.
> This file contains only stable architectural context that doesn't change session-to-session.

## Repository role

Lean 4 proof project for the JEPA learning-order result — the feature learning order of a depth-L linear JEPA model under small random initialization.

## Shared ecosystem

Part of the **Stochastic Proofs** workspace. Shared conventions and Aristotle workflow live in `../stochastic-proofs-handbook/`.

## File map

| Path | Role |
|---|---|
| `JepaLearningOrder/JEPA.lean` | Main proof — all theorems including `JEPA_rho_ordering` |
| `JepaLearningOrder/Lemmas.lean` | Supporting lemmas (Grönwall, PD bounds, contractive bound) |
| `JepaLearningOrder/OffDiagHelpers.lean` | Bridging helper lemmas |
| `JepaLearningOrder/GronwallIntegral.lean` | Grönwall integral machinery |
| `JepaLearningOrder/BootstrapLemmas.lean` | Three sub-lemmas decomposing `bootstrap_consistency` (session 24) |
| `my_theorems/paper.tex` | LaTeX paper — **authoritative spec** for what the theorem should say |

## Build commands

```bash
lake build
lake build JepaLearningOrder.JEPA
lake build JepaLearningOrder.Lemmas
```

## Scripts

```bash
python ../stochastic-proofs-handbook/scripts/status.py
python ../stochastic-proofs-handbook/scripts/retrieve.py
python ../stochastic-proofs-handbook/scripts/retrieve.py <project-id>
python ../stochastic-proofs-handbook/scripts/submit.py my_theorems/JEPA_paper_draft.md "..." --dry-run
python ../stochastic-proofs-handbook/scripts/submit.py my_theorems/JEPA_paper_draft.md "..."
python ../stochastic-proofs-handbook/scripts/watch.py
```

## Architecture invariants (do not violate)

**`JEPA_rho_ordering` hypothesis groups** (16 hypotheses, all named):
- Model definition: `dat`, `eb`, `L/hL`, `epsilon/heps/heps_small`, `t_max/ht_max`, `h_init`
- Gradient flow: `hWbar_slow`, `hWbar_init`, `hV_flow_ode`, `hV_init`
- Regularity: `hWbar_cont`, `hV_cont`, `hVqs_cont`
- Bootstrap outputs (R0, R1): `hoff_small`, `hPhaseA`
- Contraction ODE inputs (R2): `hVqs_deriv_exists`, `hDrift_bound`, `hPD_lower`, `hDelta_nz`

**Do not add `hNorm_nn` or `hNorm_cont`** — these are derived inline, not hypotheses.

**Do not re-run the compactness proof** for `quasiStatic_approx` — the genuine Phase A/B proof is in place.

**The paper draft is the authoritative spec** for what the theorem should say. When Lean and paper diverge after an Aristotle run, paper takes precedence on mathematical content; Lean takes precedence on what is actually proved.

**`hContraction` is derived internally** (not a hypothesis) — `contraction_ode_structure` is called inside `JEPA_rho_ordering`. Do not add it back as a hypothesis.

**`frozen_encoder_convergence`** currently has a vacuous proof (Aristotle `f9906716` — C_A depends on ε). `hPhaseA` stays as an explicit hypothesis of `JEPA_rho_ordering` until this is fixed genuinely.

## BootstrapLemmas.lean — why it exists

`bootstrap_consistency` (JEPA.lean ~line 496) was decomposed into three sub-lemmas because:
1. `gradV` is **linear in V**: `gradV dat Wbar V = V*(Wbar*SigmaXX*Wbar^T) - Wbar*SigmaYX*Wbar^T`. No eigenvalue smoothness or ODE existence needed.
2. Off-diagonal bound follows by FTC alone (not bootstrap): `offDiagAmplitude` is linear in Wbar, so its derivative is bounded by Cauchy-Schwarz + `hWbar_slow`.
3. Tracking bound assembles already-proved `contraction_ode_structure` (020b76be) + `contractive_gronwall_decay` (1afe6f24).

See `wiki/decisions.md` and `requests/21_bootstrap_request.md` for full proof strategy.

## Key proved lemmas

| Lemma | File | Proved by |
|---|---|---|
| `contraction_ode_structure` | JEPA.lean ~716 | Aristotle 020b76be |
| `contractive_gronwall_decay` | Lemmas.lean ~424 | Aristotle 1afe6f24 |
| `quasiStatic_approx` | JEPA.lean | Aristotle 1ccc1ab8 |
| `frozen_encoder_convergence` | JEPA.lean | Aristotle f9906716 (vacuous C_A) |
| `frobenius_pd_lower_bound` | Lemmas.lean ~110 | Exact |

## Strategic advice

**Submit soon.** "First machine-checked learning-dynamics result" has time value. One named sorry (`bootstrap_consistency`) in JEPA.lean is a strong position — this is the standard global-existence hypothesis that informal ODE learning-theory papers leave implicit. The CompCert convention (named hypotheses = named assumptions, not gaps) applies.

Priority: Jobs A+B → arXiv upload → COLT/ICLR/TMLR submission → genuine `frozen_encoder_convergence` proof.
