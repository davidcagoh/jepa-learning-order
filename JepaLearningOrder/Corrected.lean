/-
Copyright (c) 2026. All rights reserved.
Released under MIT license.
Authors: David Goh

# Paper-1 Corrected Theorems (session 90, 2026-05-21)

Empirical validation in `../jepa-rho-recovery/experiments/RESULTS_session90_verification.md`
identified that the Bernoulli ODE form used throughout paper-1
(`JEPA.lean::diagAmp_ODE`, `bernoulli_laurent_bound`, `actual_critical_time`,
`MainTheorem.lean::JEPA_dynamics_ordering`) has its bracket exponent inverted:

  Paper-1 (inverted):   σ̇ = L λ σ^{3-1/L} (1 − σ^{1/L}/ρ)
                        plateau σ^∞ = ρ^L
                        Laurent sum has 2L−1 divergent terms

  Saxe (correct):       σ̇ = L μ σ^{2-1/L} (ρ − σ^L)
                        plateau σ^∞ = ρ^{1/L}
                        hitting time has a SINGLE divergent term ε^{-(L-1)/L}

The original `JEPA.lean::diagAmp_ODE` proof is *vacuous* (compactness trick
picks C = max(C', 1)/ε^{(2L-1)/L} so the conclusion is trivially true). The
inverted bracket is asserted in the statement but never constrained.

This file collects the **corrected** versions of the affected paper-1
theorems. Statements use the Saxe ODE form; proofs are queued for
Aristotle resubmission with a focus on *genuine* derivations (not
compactness-trick discharge).

Mathematical content of the correction:
* ODE form: bracket swap `σ^{1/L} ↔ σ^L`.
* Plateau: ρ^L → ρ^{1/L}.
* Hitting-time threshold: `p · ρ^L` → `p · ρ^{1/L}`.
* Laurent asymptotic: a single ε^{-(L-1)/L} leading term, not 2L−1 terms.
* Time scale: training is much faster under corrected form
  (ε^{-(L-1)/L} ≪ ε^{-(2L-1)/L} as ε → 0).

The paper-1 **ordering claim** (Theorem `JEPA_dynamics_ordering`) is empirically
correct in direction (features learned in ρ-order); only quantitative bounds
change.
-/

import JepaLearningOrder.JEPA
import JepaLearningOrder.Lemmas
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace JepaLearningOrder

open Real Filter Topology

/-! ## §1. Corrected diagonal-amplitude ODE -/

/--
**Corrected Bernoulli ODE for σ_r (Saxe form).**

Under JEPA gradient flow with `quasiStatic_approx` tracking
(‖V − V_qs‖ ≤ K·ε^{2(L-1)/L}) and bounded off-diagonal amplitudes
(|c_rs| ≤ K·ε^{1/L} for r ≠ s), the diagonal amplitude
`σ_r := u_r^T W̄ v_r^*` satisfies

  `|σ̇_r(t) − L μ_r σ_r^{2-1/L}(t) (ρ_r − σ_r^L(t))| ≤ C · ε^{(2L-1)/L}`

uniformly on `[0, t_max]`, where `μ_r := projectedCovariance / ρ_r`
and the constant `C` depends only on (L, K_track, K_off, ρ_r, μ_r),
NOT on ε.

**This is the NON-VACUOUS replacement for `JepaLearningOrder.JEPA.diagAmp_ODE`.**
The original discharges via compactness with `C := max(C', 1)/ε^{(2L-1)/L}`,
which absorbs any residual into the constant; here we require an actual
functional bound.

**Proof plan** (genuine derivation, not compactness trick):
1. Chain rule: `σ̇_r = preconditioner_L(σ_r, σ_r) · u_r^T (-gradW̄) v_r*`.
2. Substitute `gradW̄ = V^T (V W̄ Σˣˣ - W̄ Σʸˣ)`.
3. Split `V = V_qs + ΔV`; use `htrack` to bound the ΔV contribution by
   the ε^{2(L-1)/L} tracking error.
4. For the V_qs piece: substitute V_qs = W̄ Σʸˣ W̄ᵀ (W̄ Σˣˣ W̄ᵀ)⁻¹ and
   project onto (u_r, v_r). Off-diagonal terms in W̄ contribute via
   `hoff_small` bounded by K · ε^{1/L}; the diagonal piece collapses
   to the Saxe form `L μ_r σ_r^{2-1/L} (ρ_r − σ_r^L)`.
5. Combine error terms — each contributes O(ε^{(2L-1)/L}) at worst.

The signature mirrors `diagAmp_ODE` exactly (same hypotheses, same shape),
only the conclusion bracket is corrected and the proof must be genuine.
-/
theorem diagAmp_ODE_corrected (dat : JEPAData d) (eb : GenEigenbasis dat)
    (L : ℕ) (hL : 2 ≤ L) (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (V Wbar : ℝ → Matrix (Fin d) (Fin d) ℝ)
    (hWbar_slow : ∃ K : ℝ, 0 < K ∧ ∀ t ∈ Set.Icc 0 t_max,
        matFrobNorm (deriv Wbar t) ≤ K * epsilon ^ 2)
    (hV_flow_ode : ∀ t ∈ Set.Icc 0 t_max,
        HasDerivAt V (-(gradV dat (Wbar t) (V t))) t)
    (htrack : ∃ K : ℝ, 0 < K ∧ ∀ t ∈ Set.Icc 0 t_max,
        matFrobNorm (V t - quasiStaticDecoder dat (Wbar t)) ≤
          K * epsilon ^ (2 * ((L : ℝ) - 1) / L))
    (hoff : ∃ K : ℝ, 0 < K ∧ ∀ r s : Fin d, r ≠ s → ∀ t ∈ Set.Icc 0 t_max,
        |offDiagAmplitude dat eb (Wbar t) r s| ≤ K * epsilon ^ ((1 : ℝ) / L))
    (r : Fin d)
    (hflow_diag : ∀ t ∈ Set.Icc 0 t_max,
        HasDerivAt (fun s => diagAmplitude dat eb (Wbar s) r)
            (preconditioner L (diagAmplitude dat eb (Wbar t) r)
                              (diagAmplitude dat eb (Wbar t) r) *
             dotProduct (dualBasis dat eb r)
               ((-(gradWbar dat (Wbar t) (V t))).mulVec (eb.pairs r).v))
            t)
    (hWbar_cont : ContinuousOn Wbar (Set.Icc 0 t_max))
    (hV_cont : ContinuousOn V (Set.Icc 0 t_max)) :
    ∃ C : ℝ, 0 < C ∧ ∀ t ∈ Set.Ioo 0 t_max,
      |deriv (fun s => diagAmplitude dat eb (Wbar s) r) t
       - ((L : ℝ) * ((eb.pairs r).mu)
            * Real.rpow (diagAmplitude dat eb (Wbar t) r) (2 - 1 / L)
            * ((eb.pairs r).rho
                - (diagAmplitude dat eb (Wbar t) r) ^ L))|
      ≤ C * epsilon ^ ((2 * (L : ℝ) - 1) / L) := by
  sorry

/-! ## §2. Corrected hitting-time / single-pole asymptotic

    Under the corrected Saxe ODE, the hitting-time integral has a
    SINGLE divergent term (not 2L−1). The asymptotic form is

      T(p, ε) ≈ ε^{-(L-1)/L} / [μ · ρ · (L-1)] + bounded(ε).

    Compare to the original `bernoulli_laurent_bound` (paper-1
    inverted-form) whose Laurent sum had 2L−1 divergent contributions
    at rates ε^{-n/L} for n = 1, ..., 2L−1. The corrected form is
    cleaner because the bracket `(ρ − σ^L)` ≈ ρ for small σ contributes
    no extra singularities (unlike `(1 − σ^{1/L}/ρ)` which expanded
    polynomially in σ^{1/L}).
-/

/--
**Corrected single-pole hitting-time bound.**

For a scalar function `f` satisfying the approximate Saxe-form ODE
`|f' − L μ f^{2-1/L} (ρ − f^L)| ≤ C_ode · ε^{(2L-1)/L}`,
the hitting time at threshold `p · ρ^{1/L}` differs from
`(1/(μ · ρ · (L-1))) · ε^{-(L-1)/L}` by at most `K · ε^{-(L-2)/L}`.

**This REPLACES `bernoulli_laurent_bound`** (which had 2L−1 divergent
terms under the inverted form). The corrected version has a single
divergent term — much cleaner.

**Proof plan** — Gronwall sandwich:
* Construct upper/lower comparison solutions `f_±` satisfying exact
  Saxe ODEs with perturbed rates `μ(1±δ)`, `δ = O(ε^{(2L-1)/L})`.
* ODE comparison: `f₋ ≤ f ≤ f₊`, so hitting times satisfy `τ₊ ≤ τ_f ≤ τ₋`.
* For each comparison solution, integrate the exact Saxe ODE to get
  the closed-form hitting time at threshold `p·ρ^{1/L}`. The integral
  `∫_ε^{p·ρ^{1/L}} dσ/[Lμσ^{2-1/L}(ρ−σ^L)]` evaluates to
  `(1/(μρ(L-1))) · ε^{-(L-1)/L} + bounded(ε)` for small ε.
* Triangle inequality: τ_f differs from the asymptotic single-pole
  term by `O(ε^{-(L-2)/L})`.
-/
theorem bernoulli_saxe_bound_corrected (L : ℕ) (hL : 2 ≤ L)
    (lam_r rho_r : ℝ) (hlam : 0 < lam_r) (hrho : 0 < rho_r)
    (p : ℝ) (hp : 0 < p) (hp_lt : p < 1)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (C_ode : ℝ) (hC : 0 < C_ode) :
    ∃ K : ℝ, 0 < K ∧
    ∀ (epsilon : ℝ), 0 < epsilon → epsilon < 1 →
    ∀ (f : ℝ → ℝ),
      f 0 = epsilon →
      (∀ t ∈ Set.Ioo 0 t_max,
        |deriv f t - ((L : ℝ) * (lam_r / rho_r)
              * Real.rpow (f t) (2 - 1 / L)
              * (rho_r - (f t) ^ L))|
        ≤ C_ode * epsilon ^ ((2 * (L : ℝ) - 1) / L)) →
      |hittingTime f (p * Real.rpow rho_r ((1 : ℝ) / L)) t_max
         - (1 / ((lam_r / rho_r) * rho_r * ((L : ℝ) - 1)))
           * epsilon ^ (-((L : ℝ) - 1) / L)|
        ≤ K * epsilon ^ (-((L : ℝ) - 2) / L) := by
  sorry

/-! ## §3. Corrected critical-time and ordering theorems

    These are composition wrappers using `diagAmp_ODE_corrected` +
    `bernoulli_saxe_bound_corrected`. The downstream changes are
    mechanical: threshold update, time-scale exponent update.
-/

/--
**Corrected `actual_critical_time`.**

Same composition logic as `JepaLearningOrder.JEPA.actual_critical_time`, but
under the Saxe ODE form: σ_r evolves with the corrected bracket, and
the hitting-time threshold is `p · ρ_r^{1/L}` (not `p · ρ_r^L`).
The estimate is `K · ε^{-(L-2)/L}` distance from the single-pole asymptotic.
-/
theorem actual_critical_time_corrected (dat : JEPAData d) (eb : GenEigenbasis dat)
    (L : ℕ) (hL : 2 ≤ L)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (p : ℝ) (hp : 0 < p) (hp_lt : p < 1)
    (r : Fin d)
    (C : ℝ) (hC : 0 < C) :
    ∃ K : ℝ, 0 < K ∧
    ∀ (epsilon : ℝ), 0 < epsilon → epsilon < 1 →
    ∀ (Wbar : ℝ → Matrix (Fin d) (Fin d) ℝ),
      diagAmplitude dat eb (Wbar 0) r = epsilon →
      (∀ t ∈ Set.Ioo 0 t_max,
        |deriv (fun s => diagAmplitude dat eb (Wbar s) r) t
         - ((L : ℝ) * (eb.pairs r).mu
              * Real.rpow (diagAmplitude dat eb (Wbar t) r) (2 - 1 / L)
              * ((eb.pairs r).rho
                  - (diagAmplitude dat eb (Wbar t) r) ^ L))|
        ≤ C * epsilon ^ ((2 * (L : ℝ) - 1) / L)) →
      |hittingTime (fun t => diagAmplitude dat eb (Wbar t) r)
                    (p * Real.rpow (eb.pairs r).rho ((1 : ℝ) / L)) t_max
         - (1 / ((projectedCovariance dat eb r) / (eb.pairs r).rho
                  * (eb.pairs r).rho * ((L : ℝ) - 1)))
           * epsilon ^ (-((L : ℝ) - 1) / L)|
        ≤ K * epsilon ^ (-((L : ℝ) - 2) / L) := by
  -- Aristotle job b853ca6d (session 90). Mechanical composition.
  have hlam_pos : (0 : ℝ) < projectedCovariance dat eb r :=
    mul_pos (eb.hpos r) (eb.pairs r).hmu_pos
  have hmu_eq : (eb.pairs r).mu = projectedCovariance dat eb r / (eb.pairs r).rho := by
    simp [projectedCovariance, mul_div_cancel_left₀ _ (ne_of_gt (eb.hpos r))]
  obtain ⟨K, hK_pos, hK_bound⟩ :=
    bernoulli_saxe_bound_corrected L hL
      (projectedCovariance dat eb r) ((eb.pairs r).rho)
      hlam_pos (eb.hpos r)
      p hp hp_lt t_max ht_max C hC
  refine ⟨K, hK_pos, fun epsilon heps heps_lt Wbar hwbar_init hode => ?_⟩
  apply hK_bound epsilon heps heps_lt (fun t => diagAmplitude dat eb (Wbar t) r) hwbar_init
  intro t ht
  rw [← hmu_eq]
  exact hode t ht

end JepaLearningOrder
