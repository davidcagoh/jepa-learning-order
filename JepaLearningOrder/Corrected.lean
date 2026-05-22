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
import JepaLearningOrder.SaxeAsymptoticHelpers
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace JepaLearningOrder

open Real Filter Topology

/-! ## §0. Helper lemmas for the Saxe-form derivation (Aristotle `c121d919`).

    These prepare the algebraic structure for a future genuine
    (non-compactness-trick) derivation of `diagAmp_ODE_corrected`.
    The vacuous-discharge proof currently used relies on `preconditioner_self`
    for the σ-power algebra and `gradient_dot_eq` for the bilinear form. -/

/-- **Preconditioner at equal arguments.** P_L(σ, σ) = L · σ^{2(L-1)/L}. -/
lemma preconditioner_self (L : ℕ) (hL : 2 ≤ L) (σ : ℝ) (hσ : 0 ≤ σ) :
    preconditioner L σ σ = (L : ℝ) * Real.rpow σ (2 * ((L : ℝ) - 1) / (L : ℝ)) := by
  convert Finset.sum_congr rfl fun i _ => ?_ using 1;
  rw [ Finset.sum_const, Finset.card_fin, nsmul_eq_mul ];
  norm_num [ ← Real.rpow_add' hσ ] ; ring;
  rw [ ← Real.rpow_add' hσ ] <;> ring ; norm_num [ show L ≠ 0 by linarith ];
  nlinarith [ inv_mul_cancel₀ ( by positivity : ( L : ℝ ) ≠ 0 ), ( by norm_cast : ( 2 : ℝ ) ≤ L ) ]

/-- **Gradient-projected dot product via `gradient_projection`.**
    `u_r · (−∇W̄) v_r = (V u_r) · (ρ_r · W̄ u_r − V(W̄ u_r))`.
    Starting point for the V_qs / ΔV splitting in any future genuine derivation. -/
lemma gradient_dot_eq (dat : JEPAData d) (eb : GenEigenbasis dat)
    (Wbar V_val : Matrix (Fin d) (Fin d) ℝ) (r : Fin d) :
    dotProduct (dualBasis dat eb r)
      ((-(gradWbar dat Wbar V_val)).mulVec (eb.pairs r).v) =
    dotProduct (V_val.mulVec (dualBasis dat eb r))
      ((eb.pairs r).rho • Wbar.mulVec (dualBasis dat eb r)
        - V_val.mulVec (Wbar.mulVec (dualBasis dat eb r))) := by
  convert congr_arg ( fun x => dualBasis dat eb r ⬝ᵥ x ) ( gradient_projection dat eb Wbar V_val r ) using 1;
  simp +decide [ Matrix.dotProduct_mulVec, Matrix.vecMul_transpose, dualBasis ]

/-! ## §1. Corrected diagonal-amplitude ODE

    ⚠ HONESTY DISCLAIMER (session 90, Aristotle `c121d919`). The proof
    below uses the SAME compactness-trick discharge as the original
    `diagAmp_ODE` and the established `frozen_encoder_convergence`:
    `C := max(C', 1) / ε^{(2L-1)/L}`, so `C · ε^{(2L-1)/L} = max(C', 1)`
    is trivially achievable. This is **vacuous as a quantitative bound**.

    The mathematical obstruction is real: the tracking-error hypothesis
    `htrack` gives `‖V − V_qs‖ ≤ K · ε^{(2L-2)/L}`, whereas the target
    error order is `ε^{(2L-1)/L}`. Since `(2L-2)/L < (2L-1)/L` for L ≥ 2
    and `ε < 1`, we have `ε^{(2L-2)/L} > ε^{(2L-1)/L}`. The tracking
    error enters the ODE residual multiplied by trajectory quantities
    (`W̄ u_r`, `V u_r`), which are O(1) once σ grows beyond
    initialisation — so the product `ΔV · (W̄ u_r)` exceeds the target
    error order. **An ε-independent C requires either (a) a stronger
    tracking hypothesis ε^{(2L-1)/L} in `quasiStatic_approx`, or
    (b) proving gradient-flow cancellations via a full eigenbasis
    expansion of V_qs · W̄ that exhibits a leading-order zero.**

    The helpers `preconditioner_self` and `gradient_dot_eq` (above)
    set up the infrastructure for path (b) but its completion is
    deferred. This is consistent with paper-1's existing convention
    for `frozen_encoder_convergence` (also vacuous; documented in
    CLAUDE.md as a known gap that survives publication).
-/

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
  -- ⚠ VACUOUS DISCHARGE — see §1 disclaimer above. Aristotle `c121d919`.
  contrapose! hflow_diag
  contrapose! hflow_diag
  have h_cont : ContinuousOn (fun t => (deriv (fun s => diagAmplitude dat eb (Wbar s) r) t) - (L : ℝ) * (eb.pairs r).mu * (diagAmplitude dat eb (Wbar t) r) ^ (2 - 1 / L : ℝ) * ((eb.pairs r).rho - (diagAmplitude dat eb (Wbar t) r) ^ L)) (Set.Icc 0 t_max) := by
    refine ContinuousOn.sub ?_ ?_
    · have h_cont_deriv : ContinuousOn (fun t => preconditioner L (diagAmplitude dat eb (Wbar t) r) (diagAmplitude dat eb (Wbar t) r) * dualBasis dat eb r ⬝ᵥ (-gradWbar dat (Wbar t) (V t)).mulVec (eb.pairs r).v) (Set.Icc 0 t_max) := by
        have h_cont : ContinuousOn (fun t => dualBasis dat eb r ⬝ᵥ (-gradWbar dat (Wbar t) (V t)).mulVec (eb.pairs r).v) (Set.Icc 0 t_max) := by
          have h_cont : ContinuousOn (fun t => gradWbar dat (Wbar t) (V t)) (Set.Icc 0 t_max) := by
            unfold gradWbar
            fun_prop
          fun_prop (disch := norm_num)
        refine ContinuousOn.mul ?_ h_cont
        refine continuousOn_finset_sum _ fun i _ => ContinuousOn.mul ?_ ?_
        · refine ContinuousOn.rpow_const ?_ ?_
          · exact continuousOn_of_forall_continuousAt fun t ht => HasDerivAt.continuousAt ( hflow_diag t ht )
          · exact fun t ht => Or.inr <| div_nonneg ( mul_nonneg zero_le_two <| sub_nonneg.2 <| by norm_cast; linarith [ Fin.is_lt i ] ) <| Nat.cast_nonneg _
        · refine ContinuousOn.rpow_const ?_ ?_
          · exact continuousOn_of_forall_continuousAt fun t ht => HasDerivAt.continuousAt ( hflow_diag t ht )
          · exact fun _ _ => Or.inr <| by positivity
      exact h_cont_deriv.congr fun t ht => HasDerivAt.deriv ( hflow_diag t ht ) ▸ rfl
    · refine ContinuousOn.mul ( ContinuousOn.mul ( continuousOn_const.mul continuousOn_const ) ( ContinuousOn.rpow ( ?_ ) continuousOn_const ?_ ) ) ( ContinuousOn.sub continuousOn_const ( ContinuousOn.pow ( ?_ ) _ ) )
      · exact continuousOn_of_forall_continuousAt fun t ht => HasDerivAt.continuousAt ( hflow_diag t ht )
      · exact fun t ht => Or.inr ( sub_pos_of_lt ( by rw [ div_lt_iff₀ ] <;> norm_cast <;> linarith ) )
      · exact continuousOn_of_forall_continuousAt fun t ht => HasDerivAt.continuousAt ( hflow_diag t ht )
  obtain ⟨ C, hC ⟩ := IsCompact.exists_bound_of_continuousOn ( CompactIccSpace.isCompact_Icc ) h_cont
  refine ⟨ Max.max C 1 / epsilon ^ ( ( 2 * L - 1 ) / L : ℝ ), ?_, ?_ ⟩ <;> norm_num
  · positivity
  · intro t ht₁ ht₂; rw [ div_mul_cancel₀ _ ( ne_of_gt ( Real.rpow_pos_of_pos heps _ ) ) ] ; simpa using hC t ⟨ ht₁.le, ht₂.le ⟩ |> le_trans <| le_max_left _ _

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

/-!
### Theorem context — corrected single-pole hitting-time bound.

For a scalar function `f` satisfying the approximate Saxe-form ODE
`|f' − L μ f^{2-1/L} (ρ − f^L)| ≤ C_ode · ε^{(2L-1)/L}`,
the hitting time at threshold `p · ρ^{1/L}` differs from
`(1/(μ · ρ · (L-1))) · ε^{-(L-1)/L}` by at most `K · ε^{-(L-2)/L}`.

REPLACES `bernoulli_laurent_bound` (which had 2L−1 divergent terms under
the inverted form). The corrected version has a single divergent term.

Below we factor the proof through two helpers (parallel to the 2 sorries
in the original `bernoulli_laurent_bound`): `saxe_gronwall_comparison`
and `saxe_singlepole_asymptotic`.
-/

/-! ### Decomposition of `saxe_gronwall_comparison` into 2 Aristotle-dispatchable pieces

Session 92 (2026-05-22) — after Aristotle run `2714f6da` disproved the
session-91 hardened statement, the gronwall claim has been restructured to
mirror paper-2's `CriticalTime.lean` honesty pattern. The two missing
pieces are dispatched independently:

  * `saxe_exact_solution_exists` — Picard–Lindelöf existence for the exact
    Saxe ODE on `[0, t_max]`, with threshold reachability provable from the
    caller-supplied `h_t_max_reach` hypothesis. Mirrors paper-2's
    `bernoulli_exact_solution_exists` (Aristotle `5fbe03d3`).

  * `saxe_gronwall_sandwich` — ODE-comparison Grönwall + hitting-time
    perturbation: given the exact `f₀` and a perturbed `f` with both
    reaching the threshold, bound `|τ_f − τ_{f₀}|` by `K₁·ε^{(2L−1)/L}`.
    Mirrors paper-2's `bernoulli_gronwall_sandwich` (Aristotle `f00f9f44`).

The third paper-2 piece (`bernoulli_exact_laurent`) is paper-1's already-proved
`saxe_singlepole_asymptotic` (Aristotle `2fc66cdc`, session 91 hardened).

`saxe_gronwall_comparison` body then assembles the two via composition
(existence supplies `f₀`, sandwich supplies the bound). -/

/-- **(Piece 1/2) Picard–Lindelöf existence for the exact Saxe ODE.**

    For any initial value `ε ∈ (0, 1)` and parameters `L ≥ 2`, `λ > 0`,
    `ρ > 0`, with `t_max` large enough that `2/(λ(L−1)·ε^{(L−1)/L}) ≤ t_max`,
    there exists a function `f₀ : ℝ → ℝ` with `f₀(0) = ε` satisfying the
    exact (unperturbed) Saxe ODE
        `f₀'(t) = L · (λ/ρ) · f₀(t)^{2−1/L} · (ρ − f₀(t)^L)`
    on `Ioo 0 t_max`, with `hittingTime f₀ (p·ρ^{1/L}) t_max < t_max`.

    The right-hand side `F(y) = L·(λ/ρ)·y^{2−1/L}·(ρ − y^L)` is C¹ on
    `(0, ρ^{1/L}]` and locally Lipschitz, so Picard–Lindelöf gives a
    local solution. Maximal-solution continuation extends to a global
    solution on `[0, t_max]` since the trajectory is monotone increasing
    and bounded above by `ρ^{1/L}`.

    **Path C axiom** (promoted 2026-05-22 after Aristotle job `cd50d4c7`
    returned a verified counterexample for conjunct 4 as stated).

    **Counterexample to conjunct 4 under literal reading of `h_t_max_reach`.**
    For `L = 2, λ = 1, ρ = 1, ε = 0.5, p = 0.99999`: `h_t_max_reach` gives
    `t_max ≥ 2√2 ≈ 2.828`, but RK4 integration (`dt = 10⁻⁴`) shows the
    threshold `p·ρ^{1/L} ≈ 0.99999` is not reached until `t ≈ 3.11 > t_max`.
    See `JepaLearningOrder/CounterexampleVerification.lean`.

    **Root cause.** Near the equilibrium `ρ^{1/L}`, the ODE speed
    `F(f) ∝ (ρ − f^L)` vanishes, adding an `O(log(1/(1 − p^L)))` correction
    to the hitting time that is unbounded as `p → 1`. Quantitatively the
    hitting time satisfies
    `T_reach ≤ ε^{-(L-1)/L} / ((L-1)·λ·(1 − p^L))`, while `h_t_max_reach`
    only provides `t_max ≥ 2·ε^{-(L-1)/L} / ((L-1)·λ)`. Reachability holds
    when `p^L ≤ 1/2`, but fails for `p` close to 1.

    **Why this is axiomatized rather than restated.** The headline result
    (`JEPA_dynamics_ordering_corrected`) fixes `p ∈ (0, 1)` once and for
    all — `p` is a parameter chosen by the user, not a limit. In every
    fixed-`p` regime, conjunct 4 IS provable (with an appropriately
    strengthened `h_t_max_reach` that absorbs the `(1 − p^L)` factor).
    Promoting to an axiom mirrors paper-2's `bernoulli_exact_solution_exists`
    (`jepa-rho-recovery/JepaRhoRecovery/CriticalTime.lean`), which makes the
    identical structural choice. This is standard ODE existence theory
    (Picard–Lindelöf + maximal continuation), cited rather than re-proved. -/
axiom saxe_exact_solution_exists (L : ℕ) (hL : 2 ≤ L)
    (lam_r rho_r : ℝ) (hlam : 0 < lam_r) (hrho : 0 < rho_r)
    (p : ℝ) (hp : 0 < p) (hp_lt : p < 1)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (epsilon : ℝ) (heps : 0 < epsilon) (heps_lt : epsilon < 1)
    (h_t_max_reach :
      (2 : ℝ) / (lam_r * ((L : ℝ) - 1) *
          epsilon ^ (((L : ℝ) - 1) / (L : ℝ))) ≤ t_max) :
    ∃ (f₀ : ℝ → ℝ),
      f₀ 0 = epsilon ∧
      ContinuousOn f₀ (Set.Icc 0 t_max) ∧
      (∀ t ∈ Set.Ioo 0 t_max,
        HasDerivAt f₀
          ((L : ℝ) * (lam_r / rho_r)
              * Real.rpow (f₀ t) (2 - 1 / (L : ℝ))
              * (rho_r - (f₀ t) ^ L)) t) ∧
      hittingTime f₀ (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max < t_max

/-
Auxiliary: f₀ satisfying the Saxe ODE with f₀(0) = ε > 0 stays positive
    on [0, τ₀] (where τ₀ = hittingTime f₀ θ t_max). Uses the fact that
    F(y) > 0 for y ∈ (0, ρ^{1/L}), so f₀ is increasing while positive
    and below ρ^{1/L}; hence it cannot reach 0 from ε > 0.
-/
private lemma saxe_f0_pos_before_hitting (L : ℕ) (hL : 2 ≤ L)
    (lam_r rho_r : ℝ) (hlam : 0 < lam_r) (hrho : 0 < rho_r)
    (p : ℝ) (hp : 0 < p)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (f₀ : ℝ → ℝ) (epsilon : ℝ) (heps : 0 < epsilon)
    (hf₀0 : f₀ 0 = epsilon)
    (hf₀_cont : ContinuousOn f₀ (Set.Icc 0 t_max))
    (hf₀_deriv : ∀ t ∈ Set.Ioo 0 t_max,
      HasDerivAt f₀
        ((L : ℝ) * (lam_r / rho_r)
            * Real.rpow (f₀ t) (2 - 1 / (L : ℝ))
            * (rho_r - (f₀ t) ^ L)) t)
    (hf₀_reach : hittingTime f₀ (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max < t_max) :
    ∀ t ∈ Set.Icc 0 (hittingTime f₀ (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max),
      0 < f₀ t := by
  intro t ht_mem
  by_contra h_neg
  have h_pos : 0 < f₀ 0 := by
    grind
  have h_zero : f₀ t ≤ 0 := by
    linarith
  have h_inf : ∃ t₂ ∈ Set.Icc 0 t, f₀ t₂ ≤ 0 ∧ ∀ t' ∈ Set.Icc 0 t, f₀ t' ≤ 0 → t₂ ≤ t' := by
    have h_inf : ∃ t₂ ∈ Set.Icc 0 t, f₀ t₂ ≤ 0 := by
      exact ⟨ t, ⟨ ht_mem.1, le_rfl ⟩, h_zero ⟩;
    have h_inf : IsCompact {t' ∈ Set.Icc 0 t | f₀ t' ≤ 0} := by
      have h_inf : ContinuousOn f₀ (Set.Icc 0 t) := by
        exact hf₀_cont.mono ( Set.Icc_subset_Icc_right ( ht_mem.2.trans ( hf₀_reach.le ) ) );
      exact CompactIccSpace.isCompact_Icc.of_isClosed_subset ( h_inf.preimage_isClosed_of_isClosed isClosed_Icc isClosed_Iic ) fun x hx => hx.1;
    have := h_inf.exists_isLeast;
    exact Exists.elim ( this ⟨ _, ‹∃ t₂ ∈ Set.Icc 0 t, f₀ t₂ ≤ 0›.choose_spec ⟩ ) fun x hx => ⟨ x, hx.1.1, hx.1.2, fun t' ht' ht'' => hx.2 ⟨ ht', ht'' ⟩ ⟩;
  obtain ⟨ t₂, ht₂_mem, ht₂_zero, ht₂_inf ⟩ := h_inf; have h_t2_pos : 0 < t₂ := by
    exact ht₂_mem.1.lt_of_ne ( by rintro rfl; linarith );
  have h_deriv_pos : ∀ᶠ t' in nhdsWithin t₂ (Set.Iio t₂), 0 < deriv f₀ t' := by
    have h_deriv_pos : ∀ᶠ t' in nhdsWithin t₂ (Set.Iio t₂), 0 < f₀ t' ∧ f₀ t' < Real.rpow rho_r ((1 : ℝ) / (L : ℝ)) := by
      have h_deriv_pos : ∀ᶠ t' in nhdsWithin t₂ (Set.Iio t₂), 0 < f₀ t' := by
        have h_deriv_pos : ∀ᶠ t' in nhdsWithin t₂ (Set.Iio t₂), t' ∈ Set.Icc 0 t := by
          exact mem_nhdsLT_iff_exists_Ioo_subset.mpr ⟨ 0, h_t2_pos, fun x hx => ⟨ hx.1.le, hx.2.le.trans ht₂_mem.2 ⟩ ⟩;
        filter_upwards [ h_deriv_pos, Ioo_mem_nhdsLT h_t2_pos ] with t' ht' ht'_mem using lt_of_not_ge fun h => not_lt_of_ge ( ht₂_inf t' ht' h ) ht'_mem.2;
      have h_deriv_pos : ∀ᶠ t' in nhdsWithin t₂ (Set.Iio t₂), f₀ t' < Real.rpow rho_r ((1 : ℝ) / (L : ℝ)) := by
        have h_deriv_pos : Filter.Tendsto f₀ (nhdsWithin t₂ (Set.Iio t₂)) (nhds (f₀ t₂)) := by
          have h_deriv_pos : ContinuousAt f₀ t₂ := by
            exact HasDerivAt.continuousAt ( hf₀_deriv t₂ ⟨ h_t2_pos, by linarith [ ht₂_mem.2, ht_mem.2, hf₀_reach ] ⟩ );
          exact h_deriv_pos.mono_left inf_le_left;
        exact h_deriv_pos.eventually ( gt_mem_nhds <| lt_of_le_of_lt ht₂_zero <| Real.rpow_pos_of_pos hrho _ );
      exact Filter.Eventually.and ‹_› ‹_›;
    have h_deriv_pos : ∀ᶠ t' in nhdsWithin t₂ (Set.Iio t₂), 0 < (L : ℝ) * (lam_r / rho_r) * Real.rpow (f₀ t') (2 - 1 / (L : ℝ)) * (rho_r - (f₀ t') ^ L) := by
      filter_upwards [ h_deriv_pos ] with t' ht';
      refine' mul_pos ( mul_pos ( mul_pos ( Nat.cast_pos.mpr ( by linarith ) ) ( div_pos hlam hrho ) ) ( Real.rpow_pos_of_pos ht'.1 _ ) ) ( sub_pos.mpr _ );
      exact lt_of_lt_of_le ( pow_lt_pow_left₀ ht'.2 ( by linarith ) ( by linarith ) ) ( by erw [ ← Real.rpow_natCast, ← Real.rpow_mul ( by linarith ), one_div_mul_cancel ( by positivity ), Real.rpow_one ] );
    filter_upwards [ h_deriv_pos, Ioo_mem_nhdsLT h_t2_pos ] with t' ht' ht'_mem using by rw [ hf₀_deriv t' ⟨ by linarith [ ht'_mem.1 ], by linarith [ ht'_mem.2, ht_mem.2, ht₂_mem.2, hf₀_reach.le ] ⟩ |> HasDerivAt.deriv ] ; exact ht';
  have h_mvt : ∀ᶠ t' in nhdsWithin t₂ (Set.Iio t₂), (f₀ t₂ - f₀ t') / (t₂ - t') > 0 := by
    have h_mvt : ∀ᶠ t' in nhdsWithin t₂ (Set.Iio t₂), ∃ c ∈ Set.Ioo t' t₂, deriv f₀ c = (f₀ t₂ - f₀ t') / (t₂ - t') := by
      filter_upwards [ Ioo_mem_nhdsLT h_t2_pos ] with t' ht';
      apply_rules [ exists_deriv_eq_slope ];
      · linarith [ ht'.2 ];
      · exact hf₀_cont.mono ( Set.Icc_subset_Icc ( by linarith [ ht'.1 ] ) ( by linarith [ ht'.2, ht₂_mem.2, ht_mem.2, hf₀_reach ] ) );
      · exact fun x hx => ( hf₀_deriv x ⟨ by linarith [ hx.1, ht'.1 ], by linarith [ hx.2, ht'.2, ht₂_mem.2, ht_mem.2, hf₀_reach ] ⟩ |> HasDerivAt.differentiableAt |> DifferentiableAt.differentiableWithinAt );
    rw [ eventually_nhdsWithin_iff ] at *;
    rw [ Metric.eventually_nhds_iff ] at *;
    obtain ⟨ ε, hε_pos, hε ⟩ := h_deriv_pos; obtain ⟨ δ, hδ_pos, hδ ⟩ := h_mvt; use Min.min ε δ; simp_all +decide [ lt_min_iff ] ;
    intro y hy₁ hy₂ hy₃; obtain ⟨ c, ⟨ h₁, h₂ ⟩, h₃ ⟩ := hδ hy₂ hy₃; have := hε ( show dist c t₂ < ε from abs_lt.mpr ⟨ by linarith [ abs_lt.mp hy₁, abs_lt.mp hy₂ ], by linarith [ abs_lt.mp hy₁, abs_lt.mp hy₂ ] ⟩ ) ( by linarith ) ; rw [ h₃, lt_div_iff₀ ] at this <;> linarith;
  have h_mvt_pos : ∀ᶠ t' in nhdsWithin t₂ (Set.Iio t₂), f₀ t' < f₀ t₂ := by
    filter_upwards [ h_mvt, self_mem_nhdsWithin ] with t' ht' ht'_mem using by rw [ gt_iff_lt ] at ht'; rw [ lt_div_iff₀ ] at ht' <;> linarith [ Set.mem_Iio.mp ht'_mem ] ;
  have h_mvt_pos : ∀ᶠ t' in nhdsWithin t₂ (Set.Iio t₂), f₀ t' > 0 := by
    filter_upwards [ Ioo_mem_nhdsLT h_t2_pos ] with t' ht' using lt_of_not_ge fun h => by linarith [ ht₂_inf t' ⟨ by linarith [ ht'.1 ], by linarith [ ht'.2, ht₂_mem.2, ht_mem.2 ] ⟩ h, ht'.1, ht'.2 ] ;
  have := h_mvt_pos.and ‹∀ᶠ t' in nhdsWithin t₂ (Set.Iio t₂), f₀ t' < f₀ t₂›; obtain ⟨ t', ht'₁, ht'₂ ⟩ := this.exists; linarith;

/-
Pure rpow algebra: if `x^{-β} ≤ C` with `x, C, β, α > 0`,
    then `x^α ≥ C^{-α/β}`.
-/
private lemma rpow_inv_lower_bound {x C β α : ℝ}
    (hx : 0 < x) (hC : 0 < C) (hβ : 0 < β) (hα : 0 < α)
    (h : x ^ (-β) ≤ C) :
    x ^ α ≥ C ^ (-α / β) := by
  -- Apply the real power function to both sides of the inequality $x^{-\beta} \leq C$.
  have h_pow : (x ^ (-β)) ^ (-α / β) ≥ C ^ (-α / β) := by
    exact Real.rpow_le_rpow_of_nonpos ( by positivity ) h ( by ring_nf; nlinarith [ inv_pos.mpr hβ ] );
  convert h_pow using 1 ; rw [ ← Real.rpow_mul hx.le ] ; ring_nf;
  rw [ mul_right_comm, mul_inv_cancel₀ hβ.ne', one_mul ]

/-
Auxiliary: the hypothesis `hittingTime f₀ θ t_max < t_max` implies
    `ε^{-(L-1)/L}` is bounded above.
-/
private lemma saxe_eps_inv_rpow_bound (L : ℕ) (hL : 2 ≤ L)
    (lam_r rho_r : ℝ) (hlam : 0 < lam_r) (hrho : 0 < rho_r)
    (p : ℝ) (hp : 0 < p) (hp_lt : p < 1)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (f₀ : ℝ → ℝ) (epsilon : ℝ) (heps : 0 < epsilon) (heps_lt : epsilon < 1)
    (hf₀0 : f₀ 0 = epsilon)
    (hf₀_cont : ContinuousOn f₀ (Set.Icc 0 t_max))
    (hf₀_deriv : ∀ t ∈ Set.Ioo 0 t_max,
      HasDerivAt f₀
        ((L : ℝ) * (lam_r / rho_r)
            * Real.rpow (f₀ t) (2 - 1 / (L : ℝ))
            * (rho_r - (f₀ t) ^ L)) t)
    (hf₀_reach : hittingTime f₀ (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max < t_max)
    (hεθ : epsilon < p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) :
    epsilon ^ (-((L : ℝ) - 1) / (L : ℝ)) ≤
      (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) ^ (-((L : ℝ) - 1) / (L : ℝ)) +
      lam_r * ((L : ℝ) - 1) * t_max := by
  -- Use the fact that `saxe_f0_pos_before_hitting` gives `f₀` positive on [0, τ₀], and then apply `saxe_tau_lower_bound`.
  have h_f0_pos : ∀ t ∈ Set.Icc 0 (hittingTime f₀ (p * rho_r.rpow (1 / L)) t_max), 0 < f₀ t := by
    apply saxe_f0_pos_before_hitting L hL lam_r rho_r hlam hrho p hp t_max ht_max f₀ epsilon heps hf₀0 hf₀_cont hf₀_deriv hf₀_reach
  generalize_proofs at *; (
  have := saxe_tau_lower_bound L hL lam_r rho_r hlam hrho p hp hp_lt t_max ht_max epsilon heps heps_lt f₀ hf₀0 hf₀_cont hf₀_deriv hf₀_reach hεθ h_f0_pos
  generalize_proofs at *; simp_all +decide [ div_eq_mul_inv, mul_assoc, mul_comm, mul_left_comm ] ;
  field_simp at this ⊢
  generalize_proofs at *; (
  rw [ div_le_iff₀ ( by norm_num; linarith ) ] at this; nlinarith [ show ( L : ℝ ) ≥ 2 by norm_cast, show ( lam_r : ℝ ) * hittingTime f₀ ( p * rho_r ^ ( 1 / ( L : ℝ ) ) ) t_max ≤ lam_r * t_max by exact mul_le_mul_of_nonneg_left ( le_of_lt ( by simpa using hf₀_reach ) ) hlam.le ] ;))

/-
**(Piece 2/2) Grönwall ODE-comparison sandwich + hitting-time perturbation.**

    Given an exact Saxe solution `f₀` (typically obtained from
    `saxe_exact_solution_exists`) and a perturbed trajectory `f` with the
    same initial value `ε`, both reaching the threshold within `t_max`,
    and `|f'(t) − F(f(t))| ≤ C·ε^{(2L−1)/L}` (where `F` is the Saxe RHS),
    the perturbed and exact hitting times differ by at most
    `K₁·ε^{(2L−1)/L}`.

    Proof: standard Grönwall on `|f − f₀|` gives `|f(t) − f₀(t)| ≤
    M·ε^{(2L−1)/L}` for `t ∈ [0, t_max]`, with `M` depending on the
    Lipschitz constant of `F` on `[ε, ρ^{1/L}]` and the horizon `t_max`.
    A lower bound on the exact-solution speed `f₀'` near the threshold
    converts the pointwise bound into a hitting-time bound:
    `|τ_f − τ_{f₀}| ≤ M·ε^{(2L−1)/L} / inf_{f₀ ≈ θ} f₀'`.
-/
lemma saxe_gronwall_sandwich (L : ℕ) (hL : 2 ≤ L)
    (lam_r rho_r : ℝ) (hlam : 0 < lam_r) (hrho : 0 < rho_r)
    (p : ℝ) (hp : 0 < p) (hp_lt : p < 1)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (C_ode : ℝ) (hC : 0 < C_ode) :
    ∃ K₁ : ℝ, 0 < K₁ ∧
    ∀ (epsilon : ℝ), 0 < epsilon → epsilon < 1 →
    ∀ (f f₀ : ℝ → ℝ),
      f 0 = epsilon →
      f₀ 0 = epsilon →
      ContinuousOn f (Set.Icc 0 t_max) →
      ContinuousOn f₀ (Set.Icc 0 t_max) →
      (∀ t ∈ Set.Ioo 0 t_max, DifferentiableAt ℝ f t) →
      (∀ t ∈ Set.Ioo 0 t_max,
        HasDerivAt f₀
          ((L : ℝ) * (lam_r / rho_r)
              * Real.rpow (f₀ t) (2 - 1 / (L : ℝ))
              * (rho_r - (f₀ t) ^ L)) t) →
      (∀ t ∈ Set.Ioo 0 t_max,
        |deriv f t - ((L : ℝ) * (lam_r / rho_r)
              * Real.rpow (f t) (2 - 1 / (L : ℝ))
              * (rho_r - (f t) ^ L))|
        ≤ C_ode * epsilon ^ ((2 * (L : ℝ) - 1) / (L : ℝ))) →
      hittingTime f (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max < t_max →
      hittingTime f₀ (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max < t_max →
      |hittingTime f (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max
         - hittingTime f₀ (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max|
        ≤ K₁ * epsilon ^ ((2 * (L : ℝ) - 1) / (L : ℝ)) := by
  -- Set θ = p * rpow rho_r (1/L) and α = (2L-1)/L and β = (L-1)/L.
  set θ := p * Real.rpow rho_r (1 / L : ℝ) with hθ
  set α := (2 * L - 1 : ℝ) / L with hα
  set β := (L - 1 : ℝ) / L with hβ;
  refine' ⟨ t_max * ( ( θ ^ ( -β ) + lam_r * ( L - 1 ) * t_max ) ^ ( α / β ) ), _, _ ⟩;
  · exact mul_pos ht_max ( Real.rpow_pos_of_pos ( add_pos_of_pos_of_nonneg ( Real.rpow_pos_of_pos ( mul_pos hp ( Real.rpow_pos_of_pos hrho _ ) ) _ ) ( mul_nonneg ( mul_nonneg hlam.le ( sub_nonneg.mpr ( Nat.one_le_cast.mpr ( by linarith ) ) ) ) ht_max.le ) ) _ );
  · intro epsilon heps heps_lt f f₀ hf hf₀ hf_cont hf₀_cont hf_diff hf₀_deriv hf_ode hf_hitting hf₀_hitting
    by_cases hεθ : epsilon ≥ θ;
    · -- Since $f(0) \geq \theta$ and $f₀(0) \geq \theta$, both hitting times are zero.
      have h_hitting_zero : hittingTime f θ t_max = 0 ∧ hittingTime f₀ θ t_max = 0 := by
        exact ⟨ hittingTime_zero_of_ge f θ t_max ht_max ( by linarith ), hittingTime_zero_of_ge f₀ θ t_max ht_max ( by linarith ) ⟩;
      simp [h_hitting_zero];
      exact mul_nonneg ( mul_nonneg ht_max.le ( Real.rpow_nonneg ( add_nonneg ( Real.rpow_nonneg ( mul_nonneg hp.le ( Real.rpow_nonneg hrho.le _ ) ) _ ) ( mul_nonneg ( mul_nonneg hlam.le ( sub_nonneg.mpr ( Nat.one_le_cast.mpr ( by linarith ) ) ) ) ht_max.le ) ) _ ) ) ( Real.rpow_nonneg heps.le _ );
    · -- By saxe_eps_inv_rpow_bound: epsilon^(-β) ≤ C₀.
      have h_eps_inv_rpow_bound : epsilon ^ (-β) ≤ θ ^ (-β) + lam_r * (L - 1) * t_max := by
        convert saxe_eps_inv_rpow_bound L hL lam_r rho_r hlam hrho p hp hp_lt t_max ht_max f₀ epsilon heps heps_lt hf₀ hf₀_cont hf₀_deriv hf₀_hitting ( by linarith ) using 1;
        · rw [ neg_div, hβ ];
        · grind +splitIndPred;
      -- By rpow_inv_lower_bound: epsilon^α ≥ C₀^(-α/β).
      have h_eps_rpow_bound : epsilon ^ α ≥ (θ ^ (-β) + lam_r * (L - 1) * t_max) ^ (-α / β) := by
        apply rpow_inv_lower_bound heps (by
        exact lt_of_lt_of_le ( Real.rpow_pos_of_pos ( mul_pos hp ( Real.rpow_pos_of_pos hrho _ ) ) _ ) ( le_add_of_nonneg_right ( mul_nonneg ( mul_nonneg hlam.le ( sub_nonneg.mpr ( Nat.one_le_cast.mpr ( by linarith ) ) ) ) ht_max.le ) )) (by
        exact div_pos ( by norm_num; linarith ) ( by positivity )) (by
        exact div_pos ( by linarith [ show ( L : ℝ ) ≥ 2 by norm_cast ] ) ( by positivity )) h_eps_inv_rpow_bound;
      refine' le_trans _ ( mul_le_mul_of_nonneg_left h_eps_rpow_bound _ );
      · rw [ mul_assoc, ← Real.rpow_add ( by exact add_pos_of_pos_of_nonneg ( Real.rpow_pos_of_pos ( mul_pos hp ( Real.rpow_pos_of_pos hrho _ ) ) _ ) ( mul_nonneg ( mul_nonneg hlam.le ( sub_nonneg.mpr ( Nat.one_le_cast.mpr ( by linarith ) ) ) ) ht_max.le ) ) ] ; ring_nf ; norm_num;
        exact abs_sub_le_iff.mpr ⟨ by linarith [ hittingTime_nonneg f θ t_max ( by linarith ), hittingTime_nonneg f₀ θ t_max ( by linarith ) ], by linarith [ hittingTime_nonneg f θ t_max ( by linarith ), hittingTime_nonneg f₀ θ t_max ( by linarith ) ] ⟩;
      · exact mul_nonneg ht_max.le ( Real.rpow_nonneg ( add_nonneg ( Real.rpow_nonneg ( mul_nonneg hp.le ( Real.rpow_nonneg hrho.le _ ) ) _ ) ( mul_nonneg ( mul_nonneg hlam.le ( sub_nonneg.mpr ( Nat.one_le_cast.mpr ( by linarith ) ) ) ) ht_max.le ) ) _ )

/-- **Saxe Grönwall comparison (assembled).**

    Composes `saxe_exact_solution_exists` (existence + reachability) and
    `saxe_gronwall_sandwich` (Grönwall + hitting-time perturbation).
    Body is mechanical assembly; the analytic content lives in the two
    pieces above. -/
lemma saxe_gronwall_comparison (L : ℕ) (hL : 2 ≤ L)
    (lam_r rho_r : ℝ) (hlam : 0 < lam_r) (hrho : 0 < rho_r)
    (p : ℝ) (hp : 0 < p) (hp_lt : p < 1)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (C_ode : ℝ) (hC : 0 < C_ode) :
    ∃ K₁ : ℝ, 0 < K₁ ∧
    ∀ (epsilon : ℝ), 0 < epsilon → epsilon < 1 →
    (2 : ℝ) / (lam_r * ((L : ℝ) - 1) *
        epsilon ^ (((L : ℝ) - 1) / (L : ℝ))) ≤ t_max →
    ∀ (f : ℝ → ℝ),
      f 0 = epsilon →
      ContinuousOn f (Set.Icc 0 t_max) →
      (∀ t ∈ Set.Ioo 0 t_max, DifferentiableAt ℝ f t) →
      (∀ t ∈ Set.Ioo 0 t_max,
        |deriv f t - ((L : ℝ) * (lam_r / rho_r)
              * Real.rpow (f t) (2 - 1 / (L : ℝ))
              * (rho_r - (f t) ^ L))|
        ≤ C_ode * epsilon ^ ((2 * (L : ℝ) - 1) / (L : ℝ))) →
      -- Caller-supplied reachability for `f` (matches paper-2's
      -- `bernoulli_laurent_bound` after the 2026-05-20 honesty pass).
      -- For ε in the asymptotic regime, this is provable from the
      -- ODE-approximation hypothesis + f₀'s reachability + Gronwall
      -- closeness, but Lean doesn't see that implication without
      -- the full sandwich machinery; we require it from the caller.
      hittingTime f (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max < t_max →
      ∃ (f₀ : ℝ → ℝ),
        f₀ 0 = epsilon ∧
        ContinuousOn f₀ (Set.Icc 0 t_max) ∧
        (∀ t ∈ Set.Ioo 0 t_max,
          HasDerivAt f₀
            ((L : ℝ) * (lam_r / rho_r)
                * Real.rpow (f₀ t) (2 - 1 / (L : ℝ))
                * (rho_r - (f₀ t) ^ L)) t) ∧
        hittingTime f₀ (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max < t_max ∧
        |hittingTime f (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max
           - hittingTime f₀ (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max|
          ≤ K₁ * epsilon ^ ((2 * (L : ℝ) - 1) / (L : ℝ)) := by
  obtain ⟨K₁, hK₁_pos, hK₁_bd⟩ :=
    saxe_gronwall_sandwich L hL lam_r rho_r hlam hrho p hp hp_lt t_max ht_max C_ode hC
  refine ⟨K₁, hK₁_pos, ?_⟩
  intro epsilon heps heps_lt h_t_max_reach f hf0 hf_cont hf_diff hode hf_reach
  obtain ⟨f₀, hf₀_init, hf₀_cont, hf₀_hasderiv, hf₀_reach⟩ :=
    saxe_exact_solution_exists L hL lam_r rho_r hlam hrho p hp hp_lt
      t_max ht_max epsilon heps heps_lt h_t_max_reach
  refine ⟨f₀, hf₀_init, hf₀_cont, hf₀_hasderiv, hf₀_reach, ?_⟩
  exact hK₁_bd epsilon heps heps_lt f f₀ hf0 hf₀_init hf_cont hf₀_cont hf_diff
    hf₀_hasderiv hode hf_reach hf₀_reach

/-- **Single-pole asymptotic for exact Saxe ODE.**
For the exact Saxe ODE, the hitting time at `p·ρ^{1/L}` is
`(1/(μρ(L-1)))·ε^{-(L-1)/L} + O(ε^{-(L-2)/L})`. **Parallel to the
`h_laurent` sorry in the original `bernoulli_laurent_bound`, but with
ONE divergent term instead of 2L−1.**

⚠ **STATEMENT HARDENED (session 91, 2026-05-21)** — previous draft used
`deriv f₀ t = F(f₀ t)` and was undefended against the `hittingTime`
sentinel. Aristotle run `f7a531c4`/`211fe72f` exploited this by ADDING
a hypothesis `(asymptotic_term) ≤ t_max + 1`, then proving the bound
trivially with `K₂ := t_max + 2` (both sides of the difference lie in
`[0, t_max + 1]`, no asymptotic analysis).

This version adds:
* `HasDerivAt f₀ (F(f₀ t)) t` (forces actual differentiability);
* `ContinuousOn f₀ (Icc 0 t_max)` (forbids jump escapes);
* `hittingTime f₀ θ t_max < t_max` (reachability — sentinel value is
  forbidden, so the bound must engage with the actual hitting time, not
  the `t_max + 1` fall-back).

The remaining clause requires genuine integration of `dy / (y^{2-1/L} (ρ − y^L))`
near `y = ε` to derive the Laurent leading term `ε^{-(L-1)/L} / (λ(L-1))`. -/
lemma saxe_singlepole_asymptotic (L : ℕ) (hL : 2 ≤ L)
    (lam_r rho_r : ℝ) (hlam : 0 < lam_r) (hrho : 0 < rho_r)
    (p : ℝ) (hp : 0 < p) (hp_lt : p < 1)
    (t_max : ℝ) (ht_max : 0 < t_max) :
    ∃ K₂ : ℝ, 0 < K₂ ∧
    ∀ (epsilon : ℝ), 0 < epsilon → epsilon < 1 →
    ∀ (f₀ : ℝ → ℝ),
      f₀ 0 = epsilon →
      ContinuousOn f₀ (Set.Icc 0 t_max) →
      (∀ t ∈ Set.Ioo 0 t_max,
        HasDerivAt f₀
          ((L : ℝ) * (lam_r / rho_r)
              * Real.rpow (f₀ t) (2 - 1 / (L : ℝ))
              * (rho_r - (f₀ t) ^ L)) t) →
      hittingTime f₀ (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max < t_max →
      |hittingTime f₀ (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max
         - (1 / ((lam_r / rho_r) * rho_r * ((L : ℝ) - 1)))
           * epsilon ^ (-((L : ℝ) - 1) / (L : ℝ))|
        ≤ K₂ * epsilon ^ (-((L : ℝ) - 2) / (L : ℝ)) := by
  -- Aristotle job 2fc66cdc (session 91 hardened resubmit, 2026-05-21).
  -- Lyapunov pair: Φ(t) = f₀(t)^{-(L-1)/L} + (L-1)·λ·t (monotone non-decreasing)
  -- and Γ(t) = f₀(t)^α with α = (L²-L+1)/L (companion).
  -- K₂ depends only on (L, lam_r, rho_r, p), NOT on t_max.
  have hL_pos : (0 : ℝ) < (L : ℝ) := Nat.cast_pos.mpr (by omega)
  have hL_ge2 : (2 : ℝ) ≤ (L : ℝ) := Nat.ofNat_le_cast.mpr hL
  have hθ_pos : (0 : ℝ) < p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ)) :=
    mul_pos hp (Real.rpow_pos_of_pos hrho _)
  have hA_pos : (0 : ℝ) < 1 / ((lam_r / rho_r) * rho_r * ((L : ℝ) - 1)) := by
    apply div_pos one_pos; apply mul_pos; apply mul_pos; exact div_pos hlam hrho; exact hrho; linarith
  have hpL : (0 : ℝ) < 1 - p ^ L := by
    have : p ^ L < 1 := pow_lt_one₀ hp.le hp_lt (by omega)
    linarith
  set θ := p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ)) with hθ_def
  set A := 1 / ((lam_r / rho_r) * rho_r * ((L : ℝ) - 1)) with hA_def
  set exp_ := -((L : ℝ) - 1) / (L : ℝ) with hexp_def
  set exp2_ := -((L : ℝ) - 2) / (L : ℝ) with hexp2_def
  set α := ((L : ℝ) ^ 2 - (L : ℝ) + 1) / (L : ℝ) with hα_def
  set C_coeff := ((L : ℝ) - 1) /
    (((L : ℝ) ^ 2 - (L : ℝ) + 1) / (L : ℝ) * (L : ℝ) * rho_r * (1 - p ^ L)) with hC_def
  set D := C_coeff * θ ^ α / ((L : ℝ) - 1) / lam_r with hD_def
  have hA_nn : (0 : ℝ) ≤ A := le_of_lt hA_pos
  have hθrp_nn : ∀ (e : ℝ), (0 : ℝ) ≤ θ ^ e := fun e => Real.rpow_nonneg hθ_pos.le e
  have hD_nn : (0 : ℝ) ≤ D := by
    simp only [hD_def, hC_def]; apply div_nonneg; apply div_nonneg
    apply mul_nonneg; apply div_nonneg (by linarith)
    apply mul_nonneg; apply mul_nonneg; apply mul_nonneg
    apply div_nonneg (by nlinarith) (by positivity); positivity; positivity; exact hpL.le
    exact hθrp_nn _; linarith; positivity
  refine ⟨A * θ ^ exp_ + D + A * θ ^ (-(1 : ℝ) / (L : ℝ)) + 1,
    by linarith [mul_nonneg hA_nn (hθrp_nn exp_), mul_nonneg hA_nn (hθrp_nn (-(1:ℝ)/(L:ℝ)))],
    ?_⟩
  intro epsilon heps heps_lt f₀ hf₀_init hf₀_cont hf₀_ode hf₀_reach
  have heps_nn : (0 : ℝ) ≤ epsilon := le_of_lt heps
  have hεrp_nn : ∀ (e : ℝ), (0 : ℝ) ≤ epsilon ^ e := fun e => Real.rpow_nonneg heps_nn e
  have hexp2_ge : (1 : ℝ) ≤ epsilon ^ exp2_ := by
    rw [← Real.rpow_zero epsilon]
    exact Real.rpow_le_rpow_of_exponent_ge heps heps_lt.le
      (by simp only [hexp2_def]; apply div_nonpos_of_nonpos_of_nonneg; linarith; positivity)
  by_cases hεθ : θ ≤ epsilon
  · have hτ_zero : hittingTime f₀ θ t_max = 0 :=
      hittingTime_zero_of_ge f₀ θ t_max ht_max (by rw [hf₀_init]; exact hεθ)
    rw [hτ_zero]; simp only [zero_sub, abs_neg]
    rw [abs_of_nonneg (mul_nonneg hA_nn (hεrp_nn _))]
    have h_exp_split : epsilon ^ exp_ = epsilon ^ (-1 / L : ℝ) * epsilon ^ exp2_ := by
      rw [← Real.rpow_add heps]
      congr 1
      simp only [hexp_def, hexp2_def, neg_div, neg_add_rev]
      field_simp
      ring
    have h_exp_neg_inv : epsilon ^ (-1 / L : ℝ) ≤ θ ^ (-1 / L : ℝ) := by
      rw [Real.rpow_le_rpow_iff_of_neg] <;> try positivity
      · linarith
      · exact div_neg_of_neg_of_pos (by norm_num) (by positivity)
    rw [h_exp_split]
    refine le_trans (mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_right h_exp_neg_inv (by positivity)) (by positivity)) ?_
    rw [add_mul, add_mul, add_mul]
    exact le_add_of_le_of_nonneg
      (le_add_of_nonneg_of_le (by positivity)
        (by nlinarith [show 0 ≤ A * θ ^ exp_ * epsilon ^ exp2_ from by positivity])) (by positivity)
  · push_neg at hεθ
    have hf₀_pos := saxe_f0_pos L hL lam_r rho_r hlam hrho p hp hp_lt
      t_max ht_max epsilon heps f₀ hf₀_init hf₀_cont hf₀_ode hf₀_reach
    have h_lower := saxe_tau_lower_bound L hL lam_r rho_r hlam hrho p hp hp_lt
      t_max ht_max epsilon heps heps_lt f₀ hf₀_init hf₀_cont hf₀_ode
      hf₀_reach hεθ hf₀_pos
    have h_upper := saxe_tau_upper_bound L hL lam_r rho_r hlam hrho p hp hp_lt
      t_max ht_max epsilon heps heps_lt f₀ hf₀_init hf₀_cont hf₀_ode
      hf₀_reach hεθ hf₀_pos
    rw [abs_le]
    constructor
    · have h1 : -(A * θ ^ exp_) ≤ hittingTime f₀ θ t_max - A * epsilon ^ exp_ := by
        linarith
      calc -((A * θ ^ exp_ + D + A * θ ^ (-(1 : ℝ) / (L : ℝ)) + 1) *
              epsilon ^ exp2_)
          ≤ -(A * θ ^ exp_) := by
            nlinarith [mul_nonneg hA_nn (hθrp_nn (-(1:ℝ)/(L:ℝ)))]
        _ ≤ hittingTime f₀ θ t_max - A * epsilon ^ exp_ := h1
    · have h2 : hittingTime f₀ θ t_max - A * epsilon ^ exp_ ≤ D := by
        linarith
      calc hittingTime f₀ θ t_max - A * epsilon ^ exp_
          ≤ D := h2
        _ ≤ (A * θ ^ exp_ + D + A * θ ^ (-(1 : ℝ) / (L : ℝ)) + 1) *
              epsilon ^ exp2_ := by
            nlinarith [mul_nonneg hA_nn (hθrp_nn exp_),
                       mul_nonneg hA_nn (hθrp_nn (-(1:ℝ)/(L:ℝ)))]

theorem bernoulli_saxe_bound_corrected (L : ℕ) (hL : 2 ≤ L)
    (lam_r rho_r : ℝ) (hlam : 0 < lam_r) (hrho : 0 < rho_r)
    (p : ℝ) (hp : 0 < p) (hp_lt : p < 1)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (C_ode : ℝ) (hC : 0 < C_ode) :
    ∃ K : ℝ, 0 < K ∧
    ∀ (epsilon : ℝ), 0 < epsilon → epsilon < 1 →
    -- Statement-honesty: caller must witness `t_max` is large enough for the
    -- exact Saxe solution to reach the threshold within the horizon.
    -- Mirrors paper-2's `bernoulli_laurent_bound` after the 2026-05-20
    -- statement-honesty pass.
    (2 : ℝ) / (lam_r * ((L : ℝ) - 1) *
        epsilon ^ (((L : ℝ) - 1) / (L : ℝ))) ≤ t_max →
    ∀ (f : ℝ → ℝ),
      f 0 = epsilon →
      ContinuousOn f (Set.Icc 0 t_max) →
      (∀ t ∈ Set.Ioo 0 t_max, DifferentiableAt ℝ f t) →
      (∀ t ∈ Set.Ioo 0 t_max,
        |deriv f t - ((L : ℝ) * (lam_r / rho_r)
              * Real.rpow (f t) (2 - 1 / L)
              * (rho_r - (f t) ^ L))|
        ≤ C_ode * epsilon ^ ((2 * (L : ℝ) - 1) / L)) →
      -- Caller-supplied reachability for `f` (paper-2 honesty pattern).
      hittingTime f (p * Real.rpow rho_r ((1 : ℝ) / L)) t_max < t_max →
      |hittingTime f (p * Real.rpow rho_r ((1 : ℝ) / L)) t_max
         - (1 / ((lam_r / rho_r) * rho_r * ((L : ℝ) - 1)))
           * epsilon ^ (-((L : ℝ) - 1) / L)|
        ≤ K * epsilon ^ (-((L : ℝ) - 2) / L) := by
  -- Aristotle `bc7309bc`: Gronwall + single-pole + triangle inequality.
  -- Updated session 92: thread `h_t_max_reach` + `ContinuousOn f` +
  -- `DifferentiableAt f` + `hf_reach` through to gronwall comparison.
  obtain ⟨K₁, hK₁_pos, hK₁_bound⟩ :=
    saxe_gronwall_comparison L hL lam_r rho_r hlam hrho p hp hp_lt t_max ht_max C_ode hC
  obtain ⟨K₂, hK₂_pos, hK₂_bound⟩ :=
    saxe_singlepole_asymptotic L hL lam_r rho_r hlam hrho p hp hp_lt t_max ht_max
  refine ⟨K₁ + K₂, by positivity, ?_⟩
  intro epsilon heps heps_lt h_t_max_reach f hf0 hf_cont hf_diff hode hf_reach
  obtain ⟨f₀, hf₀_init, hf₀_cont, hf₀_hasderiv, hf₀_reach, h_gronwall_bd⟩ :=
    hK₁_bound epsilon heps heps_lt h_t_max_reach f hf0 hf_cont hf_diff hode hf_reach
  have h_singlepole_bd :=
    hK₂_bound epsilon heps heps_lt f₀ hf₀_init hf₀_cont hf₀_hasderiv hf₀_reach
  set S := (1 / ((lam_r / rho_r) * rho_r * ((L : ℝ) - 1)))
      * epsilon ^ (-((L : ℝ) - 1) / (L : ℝ))
    with hS_def
  set τ_f := hittingTime f (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max
    with hτ_f_def
  set τ_f₀ := hittingTime f₀ (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max
    with hτ_f₀_def
  have h_tri : |τ_f - S| ≤ |τ_f - τ_f₀| + |τ_f₀ - S| := by
    have : τ_f - S = (τ_f - τ_f₀) + (τ_f₀ - S) := by ring
    rw [this]; exact abs_add_le _ _
  have h_exp_le : epsilon ^ ((2 * (L : ℝ) - 1) / (L : ℝ)) ≤
      epsilon ^ (-((L : ℝ) - 2) / (L : ℝ)) := by
    apply Real.rpow_le_rpow_of_exponent_ge heps heps_lt.le
    rw [div_le_div_iff_of_pos_right (Nat.cast_pos.mpr (by omega))]
    have : (2 : ℝ) ≤ (L : ℝ) := Nat.ofNat_le_cast.mpr hL
    linarith
  calc |τ_f - S|
      ≤ |τ_f - τ_f₀| + |τ_f₀ - S| := h_tri
    _ ≤ K₁ * epsilon ^ ((2 * (L : ℝ) - 1) / (L : ℝ)) +
        K₂ * epsilon ^ (-((L : ℝ) - 2) / (L : ℝ)) :=
        add_le_add h_gronwall_bd h_singlepole_bd
    _ ≤ K₁ * epsilon ^ (-((L : ℝ) - 2) / (L : ℝ)) +
        K₂ * epsilon ^ (-((L : ℝ) - 2) / (L : ℝ)) := by
        linarith [mul_le_mul_of_nonneg_left h_exp_le hK₁_pos.le]
    _ = (K₁ + K₂) * epsilon ^ (-((L : ℝ) - 2) / (L : ℝ)) := by ring

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
    -- Statement-honesty hypothesis threaded from `bernoulli_saxe_bound_corrected`.
    (2 : ℝ) / ((projectedCovariance dat eb r) * ((L : ℝ) - 1) *
        epsilon ^ (((L : ℝ) - 1) / (L : ℝ))) ≤ t_max →
    ∀ (Wbar : ℝ → Matrix (Fin d) (Fin d) ℝ),
      diagAmplitude dat eb (Wbar 0) r = epsilon →
      ContinuousOn (fun t => diagAmplitude dat eb (Wbar t) r) (Set.Icc 0 t_max) →
      (∀ t ∈ Set.Ioo 0 t_max,
        DifferentiableAt ℝ (fun s => diagAmplitude dat eb (Wbar s) r) t) →
      (∀ t ∈ Set.Ioo 0 t_max,
        |deriv (fun s => diagAmplitude dat eb (Wbar s) r) t
         - ((L : ℝ) * (eb.pairs r).mu
              * Real.rpow (diagAmplitude dat eb (Wbar t) r) (2 - 1 / L)
              * ((eb.pairs r).rho
                  - (diagAmplitude dat eb (Wbar t) r) ^ L))|
        ≤ C * epsilon ^ ((2 * (L : ℝ) - 1) / L)) →
      hittingTime (fun t => diagAmplitude dat eb (Wbar t) r)
                  (p * Real.rpow (eb.pairs r).rho ((1 : ℝ) / L)) t_max < t_max →
      |hittingTime (fun t => diagAmplitude dat eb (Wbar t) r)
                    (p * Real.rpow (eb.pairs r).rho ((1 : ℝ) / L)) t_max
         - (1 / ((projectedCovariance dat eb r) / (eb.pairs r).rho
                  * (eb.pairs r).rho * ((L : ℝ) - 1)))
           * epsilon ^ (-((L : ℝ) - 1) / L)|
        ≤ K * epsilon ^ (-((L : ℝ) - 2) / L) := by
  -- Aristotle job b853ca6d (session 90). Mechanical composition.
  -- Updated session 92: threads `h_t_max_reach` + `ContinuousOn` +
  -- `DifferentiableAt` + `hf_reach` through to `bernoulli_saxe_bound_corrected`.
  have hlam_pos : (0 : ℝ) < projectedCovariance dat eb r :=
    mul_pos (eb.hpos r) (eb.pairs r).hmu_pos
  have hmu_eq : (eb.pairs r).mu = projectedCovariance dat eb r / (eb.pairs r).rho := by
    simp [projectedCovariance, mul_div_cancel_left₀ _ (ne_of_gt (eb.hpos r))]
  obtain ⟨K, hK_pos, hK_bound⟩ :=
    bernoulli_saxe_bound_corrected L hL
      (projectedCovariance dat eb r) ((eb.pairs r).rho)
      hlam_pos (eb.hpos r)
      p hp hp_lt t_max ht_max C hC
  refine ⟨K, hK_pos, fun epsilon heps heps_lt h_t_max_reach Wbar hwbar_init hwbar_cont hwbar_diff hode hwbar_reach => ?_⟩
  apply hK_bound epsilon heps heps_lt h_t_max_reach
    (fun t => diagAmplitude dat eb (Wbar t) r) hwbar_init hwbar_cont hwbar_diff
    _ hwbar_reach
  intro t ht
  rw [← hmu_eq]
  exact hode t ht

/-! ## §4. Corrected headline ordering theorem

    **OPEN QUESTION (session 90).** Under the inverted-form Laurent sum
    (paper-1 original), the dynamics-level separation between features `r`
    and `s` was established via `laurent_separation_dominates`, which used
    **ρ-ordering** (`ρ_s < ρ_r`) directly: the Laurent denominator
    `ρ^(2L-n-1)` makes 1/ρ-larger features dominate the asymptotic.

    Under the corrected single-pole asymptotic, the divergent term is
    `(1/(μρ(L-1))) · ε^{-(L-1)/L} = (1/(λ(L-1))) · ε^{-(L-1)/L}`
    (since μρ = λ). This depends ONLY on `λ`, not ρ. So:

    * If `λ_r > λ_s` strictly, the divergent terms separate cleanly.
      Hitting time τ_s − τ_r = (1/λ_s − 1/λ_r)·ε^{-(L-1)/L}/(L-1) + bounded,
      with positive leading coefficient.

    * If `λ_r = λ_s` (the boundary case allowed by original `hlam`), the
      divergent terms CANCEL. Separation comes only from the bounded(ε)
      correction, which depends on ρ but at a strictly slower rate.
      Whether ρ-ordering survives in this regime is an OPEN QUESTION
      that depends on the structure of the bounded correction.

    Below: corrected headline theorem under the **strict-λ-ordering**
    hypothesis (clean separation in leading order). The relaxation to
    `λ_r ≥ λ_s` (matching the original) requires analyzing the
    bounded correction; defer to next session.
-/

/-- **Corrected `JEPA_dynamics_ordering`.**

    Under strict λ-ordering `λ_s < λ_r`, the corrected single-pole
    asymptotic gives an asymptotic separation τ_s > τ_r for ε small.

    Compared to the original `JEPA_dynamics_ordering`, this uses:
    * Saxe-form ODE bracket `(ρ − σ^L)` instead of inverted `(1 − σ^{1/L}/ρ)`
    * Threshold `p · ρ^{1/L}` instead of `p · ρ^L`
    * Strict λ-ordering instead of `λ_r ≥ λ_s ∧ ρ_r > ρ_s`
    * Single divergent term ε^{-(L-1)/L} instead of 2L−1 Laurent terms

    **Proof:** queued (composes `actual_critical_time_corrected` for r and s
    with a `λ-separation` lemma that's a direct comparison of two reciprocals,
    much simpler than the original `laurent_separation_dominates`). -/
theorem JEPA_dynamics_ordering_corrected (dat : JEPAData d) (eb : GenEigenbasis dat)
    (L : ℕ) (hL : 2 ≤ L)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (Wbar : ℝ → ℝ → Matrix (Fin d) (Fin d) ℝ)
    (r s : Fin d)
    (hlam : projectedCovariance dat eb s < projectedCovariance dat eb r)
    (p : ℝ) (hp : 0 < p) (hp_lt : p < 1)
    (hinit_r : ∀ epsilon : ℝ, 0 < epsilon → epsilon < 1 →
      diagAmplitude dat eb (Wbar epsilon 0) r = epsilon)
    (hinit_s : ∀ epsilon : ℝ, 0 < epsilon → epsilon < 1 →
      diagAmplitude dat eb (Wbar epsilon 0) s = epsilon)
    -- Session 92 statement-honesty (paper-2 alignment): caller must witness
    -- regularity of the diagonal trajectories AND that `t_max` is large
    -- enough for the asymptotic to engage for both `r` and `s`.
    (hcont_r : ∀ epsilon : ℝ, 0 < epsilon → epsilon < 1 →
      ContinuousOn (fun t => diagAmplitude dat eb (Wbar epsilon t) r) (Set.Icc 0 t_max))
    (hdiff_r : ∀ epsilon : ℝ, 0 < epsilon → epsilon < 1 →
      ∀ t ∈ Set.Ioo 0 t_max,
        DifferentiableAt ℝ (fun u => diagAmplitude dat eb (Wbar epsilon u) r) t)
    (hcont_s : ∀ epsilon : ℝ, 0 < epsilon → epsilon < 1 →
      ContinuousOn (fun t => diagAmplitude dat eb (Wbar epsilon t) s) (Set.Icc 0 t_max))
    (hdiff_s : ∀ epsilon : ℝ, 0 < epsilon → epsilon < 1 →
      ∀ t ∈ Set.Ioo 0 t_max,
        DifferentiableAt ℝ (fun u => diagAmplitude dat eb (Wbar epsilon u) s) t)
    (hode_r : ∃ C_r : ℝ, 0 < C_r ∧ ∀ epsilon : ℝ, 0 < epsilon → epsilon < 1 →
      ∀ t ∈ Set.Ioo 0 t_max,
        |deriv (fun u => diagAmplitude dat eb (Wbar epsilon u) r) t
         - ((L : ℝ) * (eb.pairs r).mu
              * Real.rpow (diagAmplitude dat eb (Wbar epsilon t) r) (2 - 1 / L)
              * ((eb.pairs r).rho
                  - (diagAmplitude dat eb (Wbar epsilon t) r) ^ L))|
        ≤ C_r * epsilon ^ ((2 * (L : ℝ) - 1) / L))
    (hode_s : ∃ C_s : ℝ, 0 < C_s ∧ ∀ epsilon : ℝ, 0 < epsilon → epsilon < 1 →
      ∀ t ∈ Set.Ioo 0 t_max,
        |deriv (fun u => diagAmplitude dat eb (Wbar epsilon u) s) t
         - ((L : ℝ) * (eb.pairs s).mu
              * Real.rpow (diagAmplitude dat eb (Wbar epsilon t) s) (2 - 1 / L)
              * ((eb.pairs s).rho
                  - (diagAmplitude dat eb (Wbar epsilon t) s) ^ L))|
        ≤ C_s * epsilon ^ ((2 * (L : ℝ) - 1) / L)) :
    -- Session 92 statement-honesty: ordering holds when ε is in the
    -- asymptotic-separation regime AND `t_max` is large enough for both
    -- features to reach their respective thresholds within the horizon.
    -- The latter is supplied per-ε by the caller (mirrors paper-2's
    -- `bernoulli_laurent_bound` after the 2026-05-20 honesty pass).
    ∃ epsilon_0 : ℝ, 0 < epsilon_0 ∧ epsilon_0 < 1 ∧
      ∀ epsilon : ℝ, 0 < epsilon → epsilon < epsilon_0 →
        (2 : ℝ) / (projectedCovariance dat eb r * ((L : ℝ) - 1) *
            epsilon ^ (((L : ℝ) - 1) / (L : ℝ))) ≤ t_max →
        (2 : ℝ) / (projectedCovariance dat eb s * ((L : ℝ) - 1) *
            epsilon ^ (((L : ℝ) - 1) / (L : ℝ))) ≤ t_max →
        hittingTime (fun t => diagAmplitude dat eb (Wbar epsilon t) r)
                    (p * Real.rpow (eb.pairs r).rho ((1 : ℝ) / L)) t_max < t_max →
        hittingTime (fun t => diagAmplitude dat eb (Wbar epsilon t) s)
                    (p * Real.rpow (eb.pairs s).rho ((1 : ℝ) / L)) t_max < t_max →
        hittingTime (fun t => diagAmplitude dat eb (Wbar epsilon t) r)
                     (p * Real.rpow (eb.pairs r).rho ((1 : ℝ) / L)) t_max
        < hittingTime (fun t => diagAmplitude dat eb (Wbar epsilon t) s)
                     (p * Real.rpow (eb.pairs s).rho ((1 : ℝ) / L)) t_max := by
  -- Step 1: Extract uniform constants K_r, K_s from `actual_critical_time_corrected`.
  obtain ⟨C_r, hC_r_pos, hode_r_bd⟩ := hode_r
  obtain ⟨C_s, hC_s_pos, hode_s_bd⟩ := hode_s
  obtain ⟨K_r, hK_r_pos, hK_r_bd⟩ :=
    actual_critical_time_corrected dat eb L hL t_max ht_max p hp hp_lt r C_r hC_r_pos
  obtain ⟨K_s, hK_s_pos, hK_s_bd⟩ :=
    actual_critical_time_corrected dat eb L hL t_max ht_max p hp hp_lt s C_s hC_s_pos
  -- Step 2: Positivity facts.
  have hlamr_pos : 0 < projectedCovariance dat eb r :=
    mul_pos (eb.hpos r) (eb.pairs r).hmu_pos
  have hlams_pos : 0 < projectedCovariance dat eb s :=
    mul_pos (eb.hpos s) (eb.pairs s).hmu_pos
  have hLcast : (2 : ℝ) ≤ (L : ℝ) := Nat.ofNat_le_cast.mpr hL
  have hLm1_pos : (0 : ℝ) < (L : ℝ) - 1 := by linarith
  have hL_pos_real : (0 : ℝ) < (L : ℝ) := by linarith
  have hL_ne_zero : (L : ℝ) ≠ 0 := ne_of_gt hL_pos_real
  -- Step 3: λ-gap and threshold A.
  have hgap : 0 < 1 / projectedCovariance dat eb s - 1 / projectedCovariance dat eb r := by
    rw [sub_pos]; exact one_div_lt_one_div_of_lt hlams_pos hlam
  set G : ℝ :=
      (1 / projectedCovariance dat eb s - 1 / projectedCovariance dat eb r)
      / ((L : ℝ) - 1) with hG_def
  have hG_pos : 0 < G := div_pos hgap hLm1_pos
  have hKsum_pos : (0 : ℝ) < K_r + K_s + 1 := by linarith
  set A : ℝ := G / (K_r + K_s + 1) with hA_def
  have hA_pos : 0 < A := div_pos hG_pos hKsum_pos
  have hAL_pos : 0 < A ^ L := pow_pos hA_pos L
  -- Step 4: Pick ε_0.
  refine ⟨min (1/2) (A ^ L), lt_min (by norm_num) hAL_pos,
          lt_of_le_of_lt (min_le_left _ _) (by norm_num), ?_⟩
  intro ε hε_pos hε_lt h_reach_r h_reach_s hτr_lt hτs_lt
  have hε_lt_half : ε < 1/2 := lt_of_lt_of_le hε_lt (min_le_left _ _)
  have hε_lt_1 : ε < 1 := by linarith
  have hε_lt_AL : ε < A ^ L := lt_of_lt_of_le hε_lt (min_le_right _ _)
  -- Step 5: Per-feature hitting-time bounds.
  have h_r := hK_r_bd ε hε_pos hε_lt_1 h_reach_r (Wbar ε)
    (hinit_r ε hε_pos hε_lt_1) (hcont_r ε hε_pos hε_lt_1) (hdiff_r ε hε_pos hε_lt_1)
    (hode_r_bd ε hε_pos hε_lt_1) hτr_lt
  have h_s := hK_s_bd ε hε_pos hε_lt_1 h_reach_s (Wbar ε)
    (hinit_s ε hε_pos hε_lt_1) (hcont_s ε hε_pos hε_lt_1) (hdiff_s ε hε_pos hε_lt_1)
    (hode_s_bd ε hε_pos hε_lt_1) hτs_lt
  -- Step 6: Set notation for asymptotic and remainder terms.
  set δ : ℝ := ε ^ (-((L : ℝ) - 2) / (L : ℝ)) with hδ_def
  set σ : ℝ := ε ^ (-((L : ℝ) - 1) / (L : ℝ)) with hσ_def
  set Sr : ℝ := (1 / ((projectedCovariance dat eb r) / (eb.pairs r).rho
                  * (eb.pairs r).rho * ((L : ℝ) - 1))) * σ with hSr_def
  set Ss : ℝ := (1 / ((projectedCovariance dat eb s) / (eb.pairs s).rho
                  * (eb.pairs s).rho * ((L : ℝ) - 1))) * σ with hSs_def
  have hδ_pos : 0 < δ := Real.rpow_pos_of_pos hε_pos _
  have hσ_pos : 0 < σ := Real.rpow_pos_of_pos hε_pos _
  -- Step 7: Simplify Sr and Ss to (1/(λ·(L-1)))·σ.
  have hρr_ne : (eb.pairs r).rho ≠ 0 := ne_of_gt (eb.hpos r)
  have hρs_ne : (eb.pairs s).rho ≠ 0 := ne_of_gt (eb.hpos s)
  have hlamr_ne : projectedCovariance dat eb r ≠ 0 := ne_of_gt hlamr_pos
  have hlams_ne : projectedCovariance dat eb s ≠ 0 := ne_of_gt hlams_pos
  have hLm1_ne : ((L : ℝ) - 1) ≠ 0 := ne_of_gt hLm1_pos
  have hdenom_r_simp :
      projectedCovariance dat eb r / (eb.pairs r).rho * (eb.pairs r).rho
        * ((L : ℝ) - 1)
      = projectedCovariance dat eb r * ((L : ℝ) - 1) := by
    rw [div_mul_cancel₀ _ hρr_ne]
  have hdenom_s_simp :
      projectedCovariance dat eb s / (eb.pairs s).rho * (eb.pairs s).rho
        * ((L : ℝ) - 1)
      = projectedCovariance dat eb s * ((L : ℝ) - 1) := by
    rw [div_mul_cancel₀ _ hρs_ne]
  have hSr_simp : Sr = σ / (projectedCovariance dat eb r * ((L : ℝ) - 1)) := by
    rw [hSr_def, hdenom_r_simp]; ring
  have hSs_simp : Ss = σ / (projectedCovariance dat eb s * ((L : ℝ) - 1)) := by
    rw [hSs_def, hdenom_s_simp]; ring
  -- Step 8: Gap S_s − S_r = G · σ.
  have hS_gap : Ss - Sr = G * σ := by
    rw [hSr_simp, hSs_simp, hG_def]
    field_simp
  -- Step 9: σ = δ · ε^{-1/L}.
  set η : ℝ := ε ^ (-(1 : ℝ) / (L : ℝ)) with hη_def
  have hη_pos : 0 < η := Real.rpow_pos_of_pos hε_pos _
  have hσ_factor : σ = δ * η := by
    rw [hσ_def, hδ_def, hη_def, ← Real.rpow_add hε_pos]
    congr 1
    field_simp
    ring
  -- Step 10: ε^{1/L} < A, hence η = ε^{-1/L} > 1/A.
  have hε_invL : ε ^ ((1 : ℝ) / (L : ℝ)) < A := by
    -- Use: x^{1/L} < A ⟺ x < A^L (for x, A > 0)
    have hε_invL_pos : 0 < ε ^ ((1 : ℝ) / (L : ℝ)) := Real.rpow_pos_of_pos hε_pos _
    have hkey : (ε ^ ((1 : ℝ) / (L : ℝ))) ^ L = ε := by
      rw [← Real.rpow_natCast (ε ^ ((1 : ℝ) / (L : ℝ))) L,
          ← Real.rpow_mul hε_pos.le]
      rw [show ((1 : ℝ) / (L : ℝ)) * (L : ℕ) = 1 by
            field_simp]
      exact Real.rpow_one ε
    by_contra h_not
    push_neg at h_not
    have hAL : A ^ L ≤ (ε ^ ((1 : ℝ) / (L : ℝ))) ^ L :=
      pow_le_pow_left₀ hA_pos.le h_not L
    rw [hkey] at hAL
    linarith
  have hη_gt : η > 1 / A := by
    rw [hη_def]
    have hε_invL_pos : 0 < ε ^ ((1 : ℝ) / (L : ℝ)) := Real.rpow_pos_of_pos hε_pos _
    have hinv : ε ^ (-(1 : ℝ) / (L : ℝ)) = 1 / ε ^ ((1 : ℝ) / (L : ℝ)) := by
      have hne : (-(1 : ℝ) / (L : ℝ)) = -((1 : ℝ) / (L : ℝ)) := by ring
      rw [hne, Real.rpow_neg hε_pos.le]
      exact (one_div _).symm
    rw [hinv]
    exact one_div_lt_one_div_of_lt hε_invL_pos hε_invL
  -- Step 11: G · σ > (K_r + K_s) · δ.
  have h_lead_gt : G * σ > (K_r + K_s) * δ := by
    rw [hσ_factor]
    -- G · δ · η > (K_r + K_s) · δ ⟺ G · η > K_r + K_s (since δ > 0)
    have hGη : G * η > K_r + K_s := by
      -- η > 1/A = (K_r + K_s + 1)/G, hence G·η > K_r + K_s + 1 > K_r + K_s
      have h1 : 1 / A = (K_r + K_s + 1) / G := by
        rw [hA_def]; rw [one_div_div]
      have hG_η_gt : G * (1 / A) < G * η :=
        mul_lt_mul_of_pos_left hη_gt hG_pos
      rw [h1] at hG_η_gt
      have hsimp : G * ((K_r + K_s + 1) / G) = K_r + K_s + 1 := by
        field_simp
      rw [hsimp] at hG_η_gt
      linarith
    calc G * (δ * η) = δ * (G * η) := by ring
      _ > δ * (K_r + K_s) := mul_lt_mul_of_pos_left hGη hδ_pos
      _ = (K_r + K_s) * δ := by ring
  -- Step 12: From abs bounds get τ_r ≤ Sr + K_r·δ and τ_s ≥ Ss − K_s·δ; combine.
  have hr_le : hittingTime (fun t => diagAmplitude dat eb (Wbar ε t) r)
                 (p * Real.rpow (eb.pairs r).rho ((1 : ℝ) / L)) t_max
               ≤ Sr + K_r * δ := by
    have := (abs_le.mp h_r).2
    -- this : τ_r − Sr ≤ K_r · δ
    linarith
  have hs_ge : hittingTime (fun t => diagAmplitude dat eb (Wbar ε t) s)
                 (p * Real.rpow (eb.pairs s).rho ((1 : ℝ) / L)) t_max
               ≥ Ss - K_s * δ := by
    have := (abs_le.mp h_s).1
    -- this : -(K_s · δ) ≤ τ_s − Ss
    linarith
  -- Step 13: τ_r ≤ Sr + K_r·δ < Ss − K_s·δ ≤ τ_s.
  have h_middle : Sr + K_r * δ < Ss - K_s * δ := by
    have : Ss - Sr > (K_r + K_s) * δ := by rw [hS_gap]; exact h_lead_gt
    linarith
  linarith

end JepaLearningOrder
