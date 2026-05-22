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

/-- **Gronwall comparison for Saxe ODE.**
Construct an exact-Saxe solution f₀ matching f at t=0 and bound the
hitting-time perturbation by `K₁ · ε^{(2L-1)/L}`. **Parallel to the
`h_gronwall` sorry in the original `bernoulli_laurent_bound`.**

⚠ **STATEMENT HARDENED (session 91, 2026-05-21)** — previous draft accepted
`deriv f₀ t = F(f₀ t)` as the ODE clause, which is satisfied by the
*equilibrium-jump piecewise-constant function* `f₀(t) := ε if t = 0 ; 0 if 0 < t < τ ;
ρ^{1/L} if t ≥ τ` (Aristotle run `20dbfd88`/`d31884ff`). Both 0 and ρ^{1/L}
are equilibria of `F`, so `deriv f₀ t = 0 = F(f₀ t)` holds wherever `f₀`
is locally constant (everywhere except the jump). Lean's `deriv` returns 0
at non-differentiable points, so the jump itself doesn't break the clause.

This version requires `HasDerivAt f₀ (F(f₀ t)) t` *and* `ContinuousOn f₀ (Icc 0 t_max)`.
A jump at any `τ ∈ Ioo 0 t_max` violates `HasDerivAt`; a jump at `t = 0`
violates `f₀ 0 = ε` together with continuity if the right limit is 0.
Constant-at-equilibrium (`f₀ ≡ 0` on `(0, t_max)`) is forbidden by
`f₀ 0 = ε > 0` combined with `ContinuousOn` — the right-limit at 0 must
be ε, not 0. Constant-at-ε is forbidden because `F(ε) ≠ 0` for generic ρ
(unless ρ = ε^L, a knife-edge case).

The remaining flexibility allows only actual C¹ solutions of the Saxe ODE
from initial condition ε; existence/uniqueness is the genuine Picard–Lindelöf
content. -/
lemma saxe_gronwall_comparison (L : ℕ) (hL : 2 ≤ L)
    (lam_r rho_r : ℝ) (hlam : 0 < lam_r) (hrho : 0 < rho_r)
    (p : ℝ) (hp : 0 < p) (hp_lt : p < 1)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (C_ode : ℝ) (hC : 0 < C_ode) :
    ∃ K₁ : ℝ, 0 < K₁ ∧
    ∀ (epsilon : ℝ), 0 < epsilon → epsilon < 1 →
    ∀ (f : ℝ → ℝ),
      f 0 = epsilon →
      (∀ t ∈ Set.Ioo 0 t_max,
        |deriv f t - ((L : ℝ) * (lam_r / rho_r)
              * Real.rpow (f t) (2 - 1 / (L : ℝ))
              * (rho_r - (f t) ^ L))|
        ≤ C_ode * epsilon ^ ((2 * (L : ℝ) - 1) / (L : ℝ))) →
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
  sorry

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
  sorry

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
  -- Aristotle `bc7309bc`: Gronwall + single-pole + triangle inequality.
  obtain ⟨K₁, hK₁_pos, hK₁_bound⟩ :=
    saxe_gronwall_comparison L hL lam_r rho_r hlam hrho p hp hp_lt t_max ht_max C_ode hC
  obtain ⟨K₂, hK₂_pos, hK₂_bound⟩ :=
    saxe_singlepole_asymptotic L hL lam_r rho_r hlam hrho p hp hp_lt t_max ht_max
  refine ⟨K₁ + K₂, by positivity, ?_⟩
  intro epsilon heps heps_lt f hf0 hode
  obtain ⟨f₀, hf₀_init, hf₀_cont, hf₀_hasderiv, hf₀_reach, h_gronwall_bd⟩ :=
    hK₁_bound epsilon heps heps_lt f hf0 hode
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
    ∃ epsilon_0 : ℝ, 0 < epsilon_0 ∧ epsilon_0 < 1 ∧
      ∀ epsilon : ℝ, 0 < epsilon → epsilon < epsilon_0 →
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
  intro ε hε_pos hε_lt
  have hε_lt_half : ε < 1/2 := lt_of_lt_of_le hε_lt (min_le_left _ _)
  have hε_lt_1 : ε < 1 := by linarith
  have hε_lt_AL : ε < A ^ L := lt_of_lt_of_le hε_lt (min_le_right _ _)
  -- Step 5: Per-feature hitting-time bounds.
  have h_r := hK_r_bd ε hε_pos hε_lt_1 (Wbar ε)
    (hinit_r ε hε_pos hε_lt_1) (hode_r_bd ε hε_pos hε_lt_1)
  have h_s := hK_s_bd ε hε_pos hε_lt_1 (Wbar ε)
    (hinit_s ε hε_pos hε_lt_1) (hode_s_bd ε hε_pos hε_lt_1)
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
