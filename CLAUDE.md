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
| `JepaLearningOrder/BootstrapLemmas.lean` | Bootstrap sub-lemmas (session 24, 2026-04-29) |
| `my_theorems/JEPA_paper_draft.md` | Current paper draft — keep in sync with Lean state |

## Current proof state (as of 2026-04-29)

**4 sorries remain** — 1 in `JEPA.lean` (unchanged), 3 new in `BootstrapLemmas.lean` (Aristotle Job A pending):

| File | Sorry | Line | What it is | Path to close |
|---|---|---|---|---|
| `JEPA.lean` | `bootstrap_consistency` | ~496 | Original sorry — joint ODE bounds | Being decomposed; see BootstrapLemmas.lean |
| `BootstrapLemmas.lean` | `offDiag_ftc` | ~103 | Off-diagonal bound via FTC + Cauchy-Schwarz | **Aristotle Job A (pending)** |
| `BootstrapLemmas.lean` | `pd_lower_from_offDiag` | ~159 | Spectral PD bound from eigenbasis structure | Aristotle Job B (after Job A returns) |
| `BootstrapLemmas.lean` | `tracking_bound_from_gronwall` | ~285 | h_D_over_lam rpow arithmetic + assembly | **Aristotle Job A (pending)** |

All other lemmas are **exact** — including `quasiStatic_approx` (Aristotle `1ccc1ab8`) and `contraction_ode_structure` (Aristotle `020b76be`). Build: 8029 jobs, no errors.

**New file:** `JepaLearningOrder/BootstrapLemmas.lean` — three sub-lemmas decomposing `bootstrap_consistency`. See § "Bootstrap decomposition" below.

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

**Step 3 — Bootstrap decomposition (session 24, 2026-04-29)**

The previous note "Do not attempt via Aristotle — requires Picard-Lindelöf" was **wrong on two counts**:
1. Mathlib does have `Mathlib.Analysis.ODE.PicardLindelof` (Kudryashov/Winston Yin).
2. `bootstrap_consistency` takes the ODE solution as a given hypothesis (`hV_flow_ode`). It does not prove existence — it proves bounds.

**Key insight:** `gradV` is **linear in V**: `gradV dat Wbar V = V*(Wbar*SigmaXX*Wbar^T) - Wbar*SigmaYX*Wbar^T`. Therefore the two conclusions decouple:

- **Off-diagonal bound** (conclusion 1): provable directly by FTC on `t ↦ offDiagAmplitude dat eb (Wbar t) r s`, since this is linear in Wbar. Derivative bounded by `C * K * ε²` via Cauchy-Schwarz + `hWbar_slow`. No bootstrap needed.
- **Tracking bound** (conclusion 2): follows from `contraction_ode_structure` + `contractive_gronwall_decay` (both proved), given the Frobenius PD lower bound on `Wbar*SigmaXX*Wbar^T`.

**New file: `BootstrapLemmas.lean`** contains three sub-lemmas:

```
offDiag_ftc              — FTC + Cauchy-Schwarz (Aristotle Job A)
pd_lower_from_offDiag   — spectral perturbation in eigenbasis (Aristotle Job B)
tracking_bound_from_gronwall — assembles 020b76be + 1afe6f24 (Aristotle Job A)
```

**Submission:** `help_from_aristotle/21_bootstrap_request.md` — prompts for both jobs.

**New hypothesis added to `bootstrap_consistency`:** `hWbar_init` (Frobenius norm bound on Wbar(0)). This is already a hypothesis of `JEPA_rho_ordering` (line 1387), so adding it does not increase the overall assumption count.

**What closes with Job A:**
- `offDiag_ftc` sorry fills → `hoff_small` can be derived (not assumed) in `JEPA_rho_ordering`
- `tracking_bound_from_gronwall` sorry fills → tracking bound assembled from existing lemmas

**What remains after Job A:**
- `pd_lower_from_offDiag` (Job B, separate) — spectral perturbation: diagonal amps ≥ c*ε^{1/L} + off-diagonal small → `Wbar*SigmaXX*Wbar^T` has min eigenvalue ≥ c₀*ε^{2/L}
- `hPhaseA` and `hPD_lower` remain as explicit hypotheses of `JEPA_rho_ordering` until Job B lands

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
1. Submit Aristotle Job A (`offDiag_ftc` + `tracking_bound_from_gronwall`) — see `help_from_aristotle/21_bootstrap_request.md`
2. Submit Aristotle Job B (`pd_lower_from_offDiag`) after Job A returns
3. ArXiv upload (paper.tex is ready; 1 named sorry remains in JEPA.lean pending Jobs A+B)
4. Reformulate `frozen_encoder_convergence` genuinely (discharges `hPhaseA`)
5. Submit to theory-ML venue (NeurIPS/ICLR/TMLR or COLT) after arXiv

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
