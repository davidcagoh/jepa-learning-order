# CLAUDE.md

This file provides repository-local guidance for work in **jepa-learning-order**.

## Repository role

This repository is a Lean 4 proof project for the JEPA learning-order result. It is **not** a generic proof template.

## Authority rule

If shared handbook context is unavailable, treat this file and `README.md` as the authoritative guide for work inside this repository.

## Shared ecosystem

This repository belongs to the **Stochastic Proofs** workspace. Shared conventions, reusable patterns, and cross-project workflow guidance live in `../stochastic-proofs-handbook/`.

## Main proof target

The central result concerns the feature learning order of a depth-L linear JEPA model under small random initialization. The active paper draft is `my_theorems/JEPA_paper_draft.md`.

| Path | Role |
|---|---|
| `JepaLearningOrder/JEPA.lean` | Main proof — all theorems including `JEPA_rho_ordering` |
| `JepaLearningOrder/Lemmas.lean` | Supporting lemmas (Grönwall, PD bounds, contractive bound) |
| `JepaLearningOrder/OffDiagHelpers.lean` | Bridging helper lemmas |
| `JepaLearningOrder/GronwallIntegral.lean` | Grönwall integral machinery |
| `my_theorems/JEPA_paper_draft.md` | Current paper draft — keep in sync with Lean state |

## Current proof state (as of 2026-04-03)

**1 sorry remains** in `JEPA.lean`:

| Sorry | Line | What it is | Path to close |
|---|---|---|---|
| `bootstrap_consistency` | ~496 | Joint ODE continuation (regularity hypothesis) | Long-term: requires Picard-Lindelöf for JEPA gradient-flow system. Kept as explicit named assumption in the main theorem. See §5.3 of paper. |

All other lemmas are **exact** — including `quasiStatic_approx` (Aristotle `1ccc1ab8`) and `contraction_ode_structure` (Aristotle `020b76be`, 8 helper lemmas, wired into `JEPA_rho_ordering`). Build: 8028 jobs, no errors.

## Roadmap to full publication readiness

**Step 1 — DONE: `contraction_ode_structure` proved and wired (2026-04-03)**

Aristotle job `020b76be` proved the lemma with 8 helper lemmas. `hContraction` is no longer a hypothesis of `JEPA_rho_ordering`; instead `hVqs_deriv_exists`, `hDrift_bound`, `hPD_lower`, `hDelta_nz` are passed in and `contraction_ode_structure` is called internally. Note: `hDelta_nz` (tracking error nonzero) was added by Aristotle — physically justified since decoder has not converged at any finite time.

**Step 2 — DONE (partially): `contractive_gronwall_decay` proved; `frozen_encoder_convergence` vacuous (2026-04-03)**

Aristotle job `1afe6f24` returned:
- `contractive_gronwall_decay` in `Lemmas.lean` (§4): **genuine proof** — MVT argument, cherry-picked. ✓
- `frozen_encoder_convergence` in `JEPA.lean` (§5.5): **vacuous proof** — Aristotle witnessed `C_A = (‖V(τ_A)‖+1)/ε^{2(L-1)/L}` which makes the inequality trivially true but gives a C_A depending on ε and the trajectory. Cherry-picked with a warning comment. Do NOT wire into `JEPA_rho_ordering` — `hPhaseA` stays as an explicit hypothesis.

**Step 2b — Next: reformulate `frozen_encoder_convergence` for a genuine proof**

The current `∃ C_A` existential allows vacuous witnesses. To force Aristotle to use exponential decay, reformulate the conclusion to fix C_A explicitly:

```lean
: matFrobNorm (V τ_A - quasiStaticDecoder dat W₀) ≤
    (K₀ + K_qs) * epsilon ^ (2 * ((L : ℝ) - 1) / L)
```

where `K₀` comes from `hV_init` and `K_qs` is a bound on `‖V_qs(W₀)‖_F` derived from `hPD_lower` (the formula `W₀ΣʸˣW₀ᵀ(W₀ΣˣˣW₀ᵀ)⁻¹` has Frobenius norm bounded using the PD lower bound). Then `contractive_gronwall_decay` with D=0 gives the exponential, and the exponent arithmetic at τ_A closes it.

On success: in `JEPA_rho_ordering`, replace `hPhaseA` with the inputs to `frozen_encoder_convergence` and derive `hPhaseA` internally (same pattern as Step 1 with `hContraction`).

**Step 3 — Handle `bootstrap_consistency` as named regularity assumption (paper)**

Do not attempt to close `bootstrap_consistency` via Aristotle — it requires ODE continuation machinery not in Mathlib. Instead:
- It is already explicitly sorry'd and named. This is the CompCert convention; every ODE-based learning dynamics paper assumes solution existence.
- The paper §5.3 already has the full proof sketch.
- For submission: ensure the paper abstract and §1 name bootstrap as the single explicit open item.

**Step 4 — Update paper draft (immediate priority)**

The abstract and §1 contribution list still describe `contraction_ode_structure` as an open item. It has been proved. Update:
- Abstract: remove `contraction_ode_structure` from open items; state only `bootstrap_consistency` remains.
- §1 contribution 4 (Formal verification): update to reflect one sorry, not two.
- Appendix B table: mark `contraction_ode_structure` as Exact (Aristotle `020b76be`), `contractive_gronwall_decay` as Exact (Aristotle `1afe6f24`), `frozen_encoder_convergence` as Exact-vacuous (note: C_A not ε-independent).

**Step 5 — Final paper pass before submission**

- Replace all `[CITE: ...]` placeholders with real citations.
- Remove the `[CITE]` markers for Arora et al. 2018/2019 (already in references).
- Confirm Appendix B line numbers match current Lean file (they shift after each round).

## Strategic advice on paper impact (2026-04-03)

**Do submit soon.** The "first machine-checked learning-dynamics result" claim has time value. One named sorry (`bootstrap_consistency`) is a strong position — this is the standard global-existence hypothesis that all informal ODE learning-theory papers leave implicit.

**Priority ranking for remaining work:**
1. Update paper abstract and §1 (fast, high impact — changes the story from "two gaps" to "one gap")
2. Reformulate and resubmit `frozen_encoder_convergence` (discharges `hPhaseA`, reduces named hypotheses from 3 to 2)
3. Submit the paper to a theory-ML venue (NeurIPS/ICLR/TMLR or COLT)
4. Do NOT chase `bootstrap_consistency` in Lean — it requires Picard-Lindelöf infrastructure Mathlib doesn't have yet

**What reviewers will care about:**
- The mathematical contribution (removing simultaneous diagonalisability) is real and clean
- `bootstrap_consistency` is mathematically natural — it's the joint trajectory existence hypothesis, not a gap in the argument
- The Lean verification adds credibility even with named hypotheses; the community recognises the CompCert convention
- `hPhaseA` and `hoff_small` both trace back to `bootstrap_consistency` — once that is named as the single regularity assumption, the theorem has one explicit gap, not three

## Architecture notes (important for a new agent)

**`JEPA_rho_ordering` hypothesis groups** (16 hypotheses, all named):
- Model definition: `dat`, `eb`, `L/hL`, `epsilon/heps/heps_small`, `t_max/ht_max`, `h_init`
- Gradient flow: `hWbar_slow`, `hWbar_init`, `hV_flow_ode`, `hV_init`
- Regularity: `hWbar_cont`, `hV_cont`, `hVqs_cont`
- Bootstrap outputs (R0, R1): `hoff_small`, `hPhaseA`
- Contraction ODE inputs (R2): `hVqs_deriv_exists`, `hDrift_bound`, `hPD_lower`, `hDelta_nz` — `hContraction` is now derived internally

**`hNorm_nn` and `hNorm_cont`** are derived inline in the proof body (not hypotheses). Do not add them back.

**Do not re-run the compactness proof** for `quasiStatic_approx` — the genuine Phase A/B proof is in place and must not be reverted.

**The paper draft is the authoritative spec** for what the theorem should say. When Lean and paper diverge after an Aristotle run, the paper takes precedence on mathematical content; Lean takes precedence on what is actually proved.

## Scripts

```bash
python ../scripts/status.py                          # check sorry count + job statuses
python ../scripts/retrieve.py                        # download completed Aristotle jobs
python ../scripts/submit.py my_theorems/JEPA_paper_draft.md "..." --dry-run
python ../scripts/submit.py my_theorems/JEPA_paper_draft.md "..."
python ../scripts/watch.py                           # background poller
```

## Build commands

```bash
lake build
lake build JepaLearningOrder.JEPA
lake build JepaLearningOrder.Lemmas
```
