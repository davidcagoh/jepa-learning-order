import Mathlib
import JepaLearningOrder.Lemmas
import JepaLearningOrder.OffDiagHelpers
import JepaLearningOrder.JEPA

/-!
# Bootstrap Sub-Lemmas for JEPA

This file contains three sub-lemmas that together prove `bootstrap_consistency`
(Proposition 6.5 in the paper) without a maximal-interval argument:

1. `offDiag_ftc`         — off-diagonal amplitude bound via FTC + slow Wbar
2. `pd_lower_from_offDiag` — Frobenius PD lower bound from diagonal/off-diagonal structure
3. `tracking_bound_from_gronwall` — tracking error bound assembling contraction_ode_structure
                                    + contractive_gronwall_decay

The key insight: the off-diagonal bound follows directly from FTC (no bootstrap), because
`offDiagAmplitude` is linear in `Wbar`, so its derivative is bounded by `hWbar_slow`.
With the off-diagonal bound in hand, the PD lower bound on `Wbar * SigmaXX * Wbar^T` follows
from the eigenbasis structure. The tracking bound then follows from the already-proved
`contraction_ode_structure` and `contractive_gronwall_decay`.

**New hypothesis added to `bootstrap_consistency`**:
  `hWbar_init : ∃ K₀ : ℝ, 0 < K₀ ∧ matFrobNorm (Wbar 0) ≤ K₀ * epsilon ^ ((1 : ℝ) / L)`
This is already a hypothesis of `JEPA_rho_ordering` (line 1387), so adding it to
`bootstrap_consistency` does not increase the overall assumption count.
-/

set_option linter.style.longLine false
set_option linter.style.whitespace false

open scoped Matrix

/-! ## Sub-Lemma 1: Off-Diagonal Bound via FTC -/

/-- **Lemma B.1 (Off-diagonal bound from slow encoder).**
    If `Wbar(0)` has Frobenius norm O(ε^{1/L}) and `Wbar` moves at speed O(ε²),
    then the off-diagonal amplitudes `c_{rs}(t) = u_r^T Wbar(t) v_s` remain O(ε^{1/L})
    on any compact interval [0, t_max].

    **Proof strategy (FTC + Cauchy-Schwarz):**

    Step 1: The function `g : ℝ → ℝ := fun t => offDiagAmplitude dat eb (Wbar t) r s`
            equals `dotProduct (dualBasis dat eb r) ((Wbar t).mulVec (eb.pairs s).v)`.
            This is a composition of a fixed linear functional with the differentiable
            path `t ↦ Wbar t`.

    Step 2: `g` has derivative at each `t ∈ (0, t_max)`:
              `g'(t) = dotProduct (dualBasis dat eb r) ((deriv Wbar t).mulVec (eb.pairs s).v)`
            Proof: unfold as `∑ i, (dualBasis dat eb r) i * ∑ j, (Wbar t) i j * (eb.pairs s).v j`,
            then apply `HasDerivAt.sum` + `HasDerivAt.const_mul` + `HasDerivAt.sum` + entry-wise
            derivative from `hWbar_diff`.

    Step 3: Bound `|g'(t)|` using Cauchy-Schwarz:
              `|g'(t)| ≤ ‖dualBasis dat eb r‖₂ · ‖(deriv Wbar t).mulVec (eb.pairs s).v‖₂`
              `        ≤ ‖dualBasis dat eb r‖₂ · matFrobNorm (deriv Wbar t) · ‖(eb.pairs s).v‖₂`
              `        ≤ C_uv · K₁ · ε²`
            where `C_uv := ‖dualBasis dat eb r‖₂ · ‖(eb.pairs s).v‖₂` is a finite constant
            (norms of fixed vectors from the eigenbasis).

    Step 4: Apply the mean value theorem / FTC:
              `|g(t) - g(0)| ≤ C_uv · K₁ · ε² · t ≤ C_uv · K₁ · ε² · t_max`

    Step 5: Bound `|g(0)|`:
              `|g(0)| ≤ ‖dualBasis dat eb r‖₂ · matFrobNorm (Wbar 0) · ‖(eb.pairs s).v‖₂`
                      `≤ C_uv · K₀ · ε^{1/L}`

    Step 6: Combine:
              `|g(t)| ≤ C_uv · K₀ · ε^{1/L} + C_uv · K₁ · ε² · t_max`
              `       = C_uv · (K₀ + K₁ · t_max · ε^{2 - 1/L}) · ε^{1/L}`
              `       ≤ C_uv · (K₀ + K₁ · t_max) · ε^{1/L}`   [since ε^{2-1/L} ≤ 1 for ε<1, L≥2]

    Take `K := C_uv · (K₀ + K₁ · t_max)`.

    Key Mathlib API:
    - `norm_le_of_forall_norm_deriv_right_le` or MVT via `exists_deriv_eq_slope`
    - `Real.inner_le_iff`, `inner_mul_le_norm_mul_iff` for Cauchy-Schwarz on dot products
    - `Matrix.norm_mulVec_le` or `matFrobNorm_mulVec_le` for ‖A v‖ ≤ ‖A‖_F ‖v‖
    - `HasDerivAt.sum`, `HasDerivAt.const_mul`, entry-wise derivative from `DifferentiableAt`
-/
lemma offDiag_ftc {d : ℕ} (dat : JEPAData d) (eb : GenEigenbasis dat)
    (Wbar : ℝ → Matrix (Fin d) (Fin d) ℝ)
    (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    (L : ℕ) (hL : 2 ≤ L)
    (t_max : ℝ) (ht_max : 0 < t_max)
    -- Frobenius norm bound at t = 0 (already a hypothesis of JEPA_rho_ordering)
    (hWbar_init : ∃ K₀ : ℝ, 0 < K₀ ∧ matFrobNorm (Wbar 0) ≤ K₀ * epsilon ^ ((1 : ℝ) / L))
    -- Slow encoder dynamics (hypothesis of bootstrap_consistency)
    (hWbar_slow : ∃ K₁ : ℝ, 0 < K₁ ∧ ∀ t ∈ Set.Icc 0 t_max,
        matFrobNorm (deriv Wbar t) ≤ K₁ * epsilon ^ 2)
    -- Differentiability of Wbar (follows from HasDerivAt; needed for FTC)
    (hWbar_diff : ∀ t ∈ Set.Ioo 0 t_max, DifferentiableAt ℝ Wbar t)
    -- Continuity (for MVT / FTC interval)
    (hWbar_cont : ContinuousOn Wbar (Set.Icc 0 t_max)) :
    ∃ K : ℝ, 0 < K ∧ ∀ r s : Fin d, r ≠ s → ∀ t ∈ Set.Icc 0 t_max,
        |offDiagAmplitude dat eb (Wbar t) r s| ≤ K * epsilon ^ ((1 : ℝ) / L) := by
  obtain ⟨K₀, hK₀_pos, hK₀_bound⟩ := hWbar_init
  obtain ⟨K₁, hK₁_pos, hK₁_bound⟩ := hWbar_slow
  -- The constant K absorbs: C_uv from eigenbasis norms + t_max growth
  -- Use K = (K₀ + K₁ * t_max) * C where C depends on eigenbasis norms
  -- For the bound to be uniform in r, s, we use:
  -- C_uv(r,s) ≤ C_max = max_{r,s} ‖dualBasis r‖₂ · ‖(eb.pairs s).v‖₂ (finite, Fin d)
  sorry


/-! ## Sub-Lemma 2: Frobenius PD Lower Bound from Eigenbasis Structure -/

/-- **Lemma B.2 (PD lower bound from diagonal amplitudes).**
    If the diagonal amplitudes `σ_r(t) = diagAmplitude dat eb (Wbar t) r` are bounded below
    by `c_w · ε^{1/L}` and the off-diagonal amplitudes are bounded by `δ · ε^{1/L}` with
    `δ < c_w / (2 * √d)`, then `Wbar t * dat.SigmaXX * (Wbar t)^T` satisfies the Frobenius
    PD lower bound `‖M * A‖_F ≥ c₀ · ε^{2/L} · ‖M‖_F` for all matrices M.

    **Proof strategy (eigenbasis expansion + perturbation):**

    Step 1: For any vector `v : Fin d → ℝ`, expand in the eigenbasis `{v_s}`:
              `(Wbar t)^T v = ∑_s [(u_s^T v / μ_s)] · (Wbar t)^T v_s`   (if {u_s, v_s} biorthogonal)
            Actually: use the decomposition
              `v = ∑_s (dotProduct (dualBasis dat eb s) v / μ_s) • (eb.pairs s).v`
            (reconstruction formula from biorthogonality `hbiorthog`).

    Step 2: `(Wbar t)^T v = ∑_s coeff_s · (Wbar t)^T (eb.pairs s).v`
            The s-th component in the dual basis: `u_r^T (Wbar t)^T v_s = offDiagAmplitude r s`.
            Diagonal term r = s: `σ_s`.

    Step 3: `‖(Wbar t)^T v‖² ≥ ∑_r (u_r^T (Wbar t)^T v)² - cross_terms`
            The diagonal contribution: `≥ (c_w · ε^{1/L})² · ∑_r coeff_r²`
            The off-diagonal perturbation: `≤ d · (δ · ε^{1/L})² · ∑_s coeff_s²`
            For δ < c_w / (2√d): net `≥ (c_w/2)² · ε^{2/L} · ‖v‖²/C_basis²`.

    Step 4: `v^T (Wbar * SigmaXX * Wbar^T) v = (Wbar^T v)^T SigmaXX (Wbar^T v)`
            `≥ λ_min(SigmaXX) · ‖Wbar^T v‖² ≥ λ_min(SigmaXX) · (c_w/2)² · ε^{2/L} / C_basis²`.

    Step 5: Apply `frobenius_pd_lower_bound` with the resulting λ and the PD matrix A = Wbar * SigmaXX * Wbar^T.

    Key Mathlib API:
    - `Matrix.PosDef` for `dat.SigmaXX`
    - `pd_quadratic_lower_bound` from Lemmas.lean
    - `frobenius_pd_lower_bound` from Lemmas.lean
    - Biorthogonality: `eb.hbiorthog`
    - `Real.sqrt_le_sqrt`, `Finset.sum_le_sum`
-/
lemma pd_lower_from_offDiag {d : ℕ} (hd : 0 < d) (dat : JEPAData d) (eb : GenEigenbasis dat)
    (Wbar : Matrix (Fin d) (Fin d) ℝ)
    (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    (L : ℕ) (hL : 2 ≤ L)
    -- Diagonal amplitudes bounded below: σ_r ≥ c_w · ε^{1/L}
    (c_w : ℝ) (hc_w : 0 < c_w)
    (hdiag : ∀ r : Fin d, diagAmplitude dat eb Wbar r ≥ c_w * epsilon ^ ((1 : ℝ) / L))
    -- Off-diagonal amplitudes bounded: |c_{rs}| ≤ δ · ε^{1/L}
    (δ : ℝ) (hδ_nn : 0 ≤ δ)
    (hoff : ∀ r s : Fin d, r ≠ s →
        |offDiagAmplitude dat eb Wbar r s| ≤ δ * epsilon ^ ((1 : ℝ) / L))
    -- Off-diagonal perturbation is small relative to diagonal (ensures net PD)
    (hδ_small : δ * Real.sqrt d < c_w / 2) :
    ∃ c₀ : ℝ, 0 < c₀ ∧ ∀ M : Matrix (Fin d) (Fin d) ℝ,
        matFrobNorm (M * (Wbar * dat.SigmaXX * Wbarᵀ)) ≥
            c₀ * epsilon ^ ((2 : ℝ) / L) * matFrobNorm M := by
  sorry


/-! ## Sub-Lemma 3: Tracking Bound via Contractive Gronwall -/

/-- **Lemma B.3 (Tracking bound from contractive Gronwall).**
    Given:
    - The Phase A bound `‖V(0) - V_qs(Wbar(0))‖_F ≤ C_A · ε^{2(L-1)/L}` (hPhaseA),
    - The Frobenius PD lower bound on `Wbar(t) * SigmaXX * Wbar(t)^T` (hPD_lower),
    - The drift bound `‖d/dt V_qs(Wbar(t))‖_F ≤ D₀ · ε²` (hDrift_bound),
    - The V gradient-flow ODE (hV_flow_ode),
    the tracking error satisfies `‖V(t) - V_qs(Wbar(t))‖_F ≤ C · ε^{2(L-1)/L}` on all of [0, t_max].

    **Proof strategy (assemble existing proved lemmas):**

    Step 1: Apply `contraction_ode_structure` (proved, Aristotle 020b76be) to obtain
            constants c₀, D₀ > 0 such that for a.e. t:
              `f'(t) ≤ -(c₀ · ε^{2/L}) · f(t) + D₀ · ε²`
            where `f(t) = matFrobNorm (V t - quasiStaticDecoder dat (Wbar t))`.

    Step 2: Apply `contractive_gronwall_decay` (proved, Aristotle 1afe6f24):
              `f(t) ≤ f(0) · exp(-c₀ · ε^{2/L} · t) + (D₀ · ε² / (c₀ · ε^{2/L})) · (1 - exp(...))`
              `     ≤ f(0) + (D₀/c₀) · ε^{2(L-1)/L}`
            (since `ε² / ε^{2/L} = ε^{2 - 2/L} = ε^{2(L-1)/L}`).

    Step 3: From `hPhaseA`: `f(0) ≤ C_A · ε^{2(L-1)/L}`.

    Step 4: Combine: `f(t) ≤ (C_A + D₀/c₀) · ε^{2(L-1)/L}`. Take `C := C_A + D₀/c₀`.

    Step 5: For continuity (required by contractive_gronwall_decay):
            `f` is continuous on [0, t_max] by `frozen_tracking_continuousOn`-style argument:
            `V` is continuous (from HasDerivAt), `quasiStaticDecoder ∘ Wbar` is continuous
            (hypothesis hVqs_cont), and matFrobNorm is continuous.

    Step 6: `hf_nn`: `f(t) ≥ 0` since `matFrobNorm = sqrt(...)`.

    Key Mathlib API:
    - `contraction_ode_structure` : proved in JEPA.lean §5.4
    - `contractive_gronwall_decay` : proved in Lemmas.lean §4
    - `Real.exp_neg`, `Real.rpow_add`, `Real.rpow_natCast`
    - `div_add_div_same`, `le_add_of_nonneg_right`
-/
lemma tracking_bound_from_gronwall {d : ℕ} (hd : 0 < d) (dat : JEPAData d) (eb : GenEigenbasis dat)
    (L : ℕ) (hL : 2 ≤ L) (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (V Wbar : ℝ → Matrix (Fin d) (Fin d) ℝ)
    -- Gradient-flow ODE for V
    (hV_flow_ode : ∀ t ∈ Set.Icc 0 t_max,
        HasDerivAt V (-(gradV dat (Wbar t) (V t))) t)
    -- Phase A: initial tracking error is already O(ε^{2(L-1)/L})
    (hPhaseA : ∃ C_A : ℝ, 0 < C_A ∧
        matFrobNorm (V 0 - quasiStaticDecoder dat (Wbar 0)) ≤
          C_A * epsilon ^ (2 * ((L : ℝ) - 1) / L))
    -- Frobenius PD lower bound on Wbar(t) SigmaXX Wbar(t)^T (from sub-lemma 2)
    (hPD_lower : ∃ c₀ : ℝ, 0 < c₀ ∧ ∀ t ∈ Set.Icc 0 t_max,
        ∀ M : Matrix (Fin d) (Fin d) ℝ,
          matFrobNorm (M * (Wbar t * dat.SigmaXX * (Wbar t)ᵀ)) ≥
            c₀ * epsilon ^ ((2 : ℝ) / L) * matFrobNorm M)
    -- V_qs ∘ Wbar is differentiable on (0, t_max)
    (hVqs_deriv_exists : ∀ t ∈ Set.Ico 0 t_max,
        ∃ Vqs_d : Matrix (Fin d) (Fin d) ℝ,
          HasDerivAt (fun s => quasiStaticDecoder dat (Wbar s)) Vqs_d t)
    -- Drift bound: ‖d/dt V_qs(Wbar(t))‖_F ≤ D₀ · ε²
    (hDrift_bound : ∃ D₀ : ℝ, 0 < D₀ ∧ ∀ t ∈ Set.Ico 0 t_max,
        matFrobNorm (deriv (fun s => quasiStaticDecoder dat (Wbar s)) t) ≤ D₀ * epsilon ^ 2)
    -- Tracking error is nonzero on (0, t_max) (needed for matFrobNorm differentiability)
    (hDelta_nz : ∀ t ∈ Set.Ico 0 t_max,
        V t - quasiStaticDecoder dat (Wbar t) ≠ 0)
    -- Continuity hypotheses (follow from HasDerivAt but stated explicitly)
    (hVqs_cont : ContinuousOn (fun t => quasiStaticDecoder dat (Wbar t)) (Set.Icc 0 t_max)) :
    ∃ C : ℝ, 0 < C ∧ ∀ t ∈ Set.Icc 0 t_max,
        matFrobNorm (V t - quasiStaticDecoder dat (Wbar t)) ≤
          C * epsilon ^ (2 * ((L : ℝ) - 1) / L) := by
  obtain ⟨C_A, hCA_pos, hCA_bound⟩ := hPhaseA
  -- Step 1: Get contractive ODE from contraction_ode_structure (already proved)
  have hContraction := contraction_ode_structure hd dat L hL epsilon heps t_max ht_max V Wbar
      hV_flow_ode hVqs_deriv_exists hDrift_bound hPD_lower hDelta_nz
  obtain ⟨c₀, D₀, hc₀_pos, hD₀_pos, hODE⟩ := hContraction
  -- Step 2: Continuity of the tracking error norm
  have hf_cont : ContinuousOn (fun t => matFrobNorm (V t - quasiStaticDecoder dat (Wbar t)))
      (Set.Icc 0 t_max) := by
    have hV_cont : ContinuousOn V (Set.Icc 0 t_max) := by
      intro t ht
      have := hV_flow_ode t ht
      rw [hasDerivAt_pi] at this
      exact tendsto_pi_nhds.mpr fun i =>
        (this i |>.continuousAt |>.continuousWithinAt)
    unfold matFrobNorm
    refine ContinuousOn.sqrt ?_
    exact continuousOn_finset_sum _ fun i _ =>
      continuousOn_finset_sum _ fun j _ =>
        ContinuousOn.pow (ContinuousOn.sub
          (continuousOn_pi.mp (continuousOn_pi.mp hV_cont i) j)
          (continuousOn_pi.mp (continuousOn_pi.mp hVqs_cont i) j)) _
  -- Step 3: Apply contractive_gronwall_decay
  -- lam := c₀ * ε^{2/L}, D := D₀ * ε²
  set lam := c₀ * epsilon ^ ((2 : ℝ) / L) with hlam_def
  set D := D₀ * epsilon ^ 2 with hD_def
  have hlam_pos : 0 < lam := mul_pos hc₀_pos (Real.rpow_pos_of_pos heps _)
  have hD_nn : 0 ≤ D := mul_nonneg hD₀_pos.le (by positivity)
  have hf_nn : ∀ t ∈ Set.Icc 0 t_max,
      0 ≤ matFrobNorm (V t - quasiStaticDecoder dat (Wbar t)) :=
    fun t _ => Real.sqrt_nonneg _
  have hf_deriv : ∀ t ∈ Set.Ico 0 t_max,
      ∃ f' : ℝ, HasDerivAt (fun s => matFrobNorm (V s - quasiStaticDecoder dat (Wbar s))) f' t
              ∧ f' ≤ -lam * matFrobNorm (V t - quasiStaticDecoder dat (Wbar t)) + D := by
    intro t ht
    obtain ⟨f', hf'_deriv, hf'_bound⟩ := hODE t ht
    exact ⟨f', hf'_deriv, by linarith⟩
  have hGW := contractive_gronwall_decay ht_max hlam_pos hD_nn hf_cont hf_nn hf_deriv
  -- Step 4: Translate Gronwall output to ε^{2(L-1)/L} bound
  -- GW gives: f(t) ≤ f(0) * exp(-lam * t) + (D/lam) * (1 - exp(-lam * t))
  --         ≤ f(0) + D/lam  (both terms bounded by dropping the exp factor)
  -- D/lam = D₀ * ε² / (c₀ * ε^{2/L}) = (D₀/c₀) * ε^{2(L-1)/L}
  refine ⟨C_A + D₀ / c₀, by positivity, fun t ht => ?_⟩
  have hGW_t := hGW t ht
  have hexp_nn : 0 ≤ Real.exp (-lam * t) := Real.exp_nonneg _
  have hexp_le1 : Real.exp (-lam * t) ≤ 1 := by
    rw [Real.exp_le_one_iff]
    have : 0 ≤ t := ht.1
    nlinarith [hlam_pos.le]
  have hDlam_nn : 0 ≤ D / lam := div_nonneg hD_nn hlam_pos.le
  have h_D_over_lam : D / lam = (D₀ / c₀) * epsilon ^ (2 * ((L : ℝ) - 1) / L) := by
    -- D/lam = (D₀ * ε²) / (c₀ * ε^{2/L}) = (D₀/c₀) * ε^{2-2/L} = (D₀/c₀) * ε^{2(L-1)/L}
    -- Proof: simp [hlam_def, hD_def], convert ε^2 (ℕ-pow) to ε^(2:ℝ) (rpow)
    -- via Real.rpow_natCast, then use Real.rpow_sub heps, field_simp, ring.
    sorry
  calc matFrobNorm (V t - quasiStaticDecoder dat (Wbar t))
      ≤ matFrobNorm (V 0 - quasiStaticDecoder dat (Wbar 0)) * Real.exp (-lam * t)
        + (D / lam) * (1 - Real.exp (-lam * t)) := hGW_t
    _ ≤ matFrobNorm (V 0 - quasiStaticDecoder dat (Wbar 0)) * 1 + (D / lam) * 1 := by
          have hf0_nn : 0 ≤ matFrobNorm (V 0 - quasiStaticDecoder dat (Wbar 0)) :=
            Real.sqrt_nonneg _
          have h1 : matFrobNorm (V 0 - quasiStaticDecoder dat (Wbar 0)) * Real.exp (-lam * t)
              ≤ matFrobNorm (V 0 - quasiStaticDecoder dat (Wbar 0)) * 1 :=
            mul_le_mul_of_nonneg_left hexp_le1 hf0_nn
          have h2 : (D / lam) * (1 - Real.exp (-lam * t)) ≤ (D / lam) * 1 := by
            apply mul_le_mul_of_nonneg_left _ hDlam_nn
            linarith
          linarith
    _ = matFrobNorm (V 0 - quasiStaticDecoder dat (Wbar 0)) + D / lam := by ring
    _ ≤ C_A * epsilon ^ (2 * ((L : ℝ) - 1) / L)
        + (D₀ / c₀) * epsilon ^ (2 * ((L : ℝ) - 1) / L) := by
          rw [h_D_over_lam]; linarith
    _ = (C_A + D₀ / c₀) * epsilon ^ (2 * ((L : ℝ) - 1) / L) := by ring
