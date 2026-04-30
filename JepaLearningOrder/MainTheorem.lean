import Mathlib
import JepaLearningOrder.Lemmas
import JepaLearningOrder.OffDiagHelpers
import JepaLearningOrder.JEPA
import JepaLearningOrder.PDLowerHelpers
import JepaLearningOrder.BootstrapLemmas

/-!
# Main Theorem: JEPA ρ*-Ordering (Updated)

This file contains the updated `JEPA_rho_ordering` theorem with `hPD_lower` removed
as a named hypothesis. Instead, the diagonal amplitude lower bound and Gershgorin
constants are taken as hypotheses, and `hPD_lower` is derived internally via
`uniform_pd_lower_from_compactness`.

**Change from the version in JEPA.lean**: The hypothesis
```
hPD_lower : ∃ c₀ : ℝ, 0 < c₀ ∧ ∀ t ∈ Set.Icc 0 t_max,
    ∀ M : Matrix (Fin d) (Fin d) ℝ,
      matFrobNorm (M * (Wbar t * dat.SigmaXX * (Wbar t)ᵀ)) ≥
        c₀ * epsilon ^ ((2 : ℝ) / L) * matFrobNorm M
```
has been replaced by:
```
c_w : ℝ, hc_w : 0 < c_w
hdiag_t : ∀ t ∈ Set.Icc 0 t_max, ∀ r : Fin d,
    diagAmplitude dat eb (Wbar t) r ≥ c_w * epsilon ^ ((1 : ℝ) / L)
δ_off : ℝ, hδ_nn : 0 ≤ δ_off, hδ_small : δ_off * ((d : ℝ) - 1) < c_w
hoff_unif : ∀ t ∈ Set.Icc 0 t_max, ∀ r s : Fin d, r ≠ s →
    |offDiagAmplitude dat eb (Wbar t) r s| ≤ δ_off * epsilon ^ ((1 : ℝ) / L)
```
The uniform PD lower bound is then derived from `uniform_pd_lower_from_compactness`.
-/

set_option linter.style.longLine false
set_option linter.style.whitespace false

open scoped Matrix

variable {d : ℕ}

/-- **Theorem 8.1 (JEPA ρ*-ordering — updated).**

    Same as `JEPA_rho_ordering` in JEPA.lean, but with `hPD_lower` derived internally
    from diagonal/off-diagonal amplitude conditions via `uniform_pd_lower_from_compactness`
    (compactness argument). This reduces the hypothesis count by 1. -/
theorem JEPA_rho_ordering' (dat : JEPAData d) (eb : GenEigenbasis dat)
    (L : ℕ) (hL : 2 ≤ L) (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (V Wbar : ℝ → Matrix (Fin d) (Fin d) ℝ)
    -- Gradient flow from balanced initialisation
    (h_init : BalancedInit d L epsilon)
    -- (H1) Encoder moves slowly: ‖Ẇ̄(t)‖_F ≤ K ε² (from preconditioned gradient flow)
    (hWbar_slow : ∃ K : ℝ, 0 < K ∧ ∀ t ∈ Set.Icc 0 t_max,
        matFrobNorm (deriv Wbar t) ≤ K * epsilon ^ 2)
    (hWbar_init : ∃ K₀ : ℝ, 0 < K₀ ∧
        matFrobNorm (Wbar 0) ≤ K₀ * epsilon ^ ((1 : ℝ) / L))
    -- (H2) Decoder satisfies gradient-flow ODE V̇ = -∇_V ℒ(W̄(t), V(t))
    (hV_flow_ode : ∀ t ∈ Set.Icc 0 t_max,
        HasDerivAt V (-(gradV dat (Wbar t) (V t))) t)
    (hV_init : ∃ K₀ : ℝ, 0 < K₀ ∧
        matFrobNorm (V 0) ≤ K₀ * epsilon ^ ((1 : ℝ) / L))
    -- (H3) Off-diagonal amplitudes are O(ε^{1/L}) on [0, t_max].
    (hoff_small : ∃ K : ℝ, 0 < K ∧ ∀ r s : Fin d, r ≠ s → ∀ t ∈ Set.Icc 0 t_max,
        |offDiagAmplitude dat eb (Wbar t) r s| ≤ K * epsilon ^ ((1 : ℝ) / L))
    -- Regularity
    (hWbar_cont : ContinuousOn Wbar (Set.Icc 0 t_max))
    (hV_cont : ContinuousOn V (Set.Icc 0 t_max))
    (hVqs_cont : ContinuousOn (fun t => quasiStaticDecoder dat (Wbar t)) (Set.Icc 0 t_max))
    -- Phase A
    (hPhaseA : ∃ C_A : ℝ, 0 < C_A ∧
        matFrobNorm (V 0 - quasiStaticDecoder dat (Wbar 0)) ≤
          C_A * epsilon ^ (2 * ((↑L : ℝ) - 1) / ↑L))
    -- Phase B ODE inputs
    (hVqs_deriv_exists : ∀ t ∈ Set.Ico 0 t_max,
        ∃ Vqs_d : Matrix (Fin d) (Fin d) ℝ,
          HasDerivAt (fun s => quasiStaticDecoder dat (Wbar s)) Vqs_d t)
    (hDrift_bound : ∃ D₀ : ℝ, 0 < D₀ ∧ ∀ t ∈ Set.Ico 0 t_max,
        matFrobNorm (deriv (fun s => quasiStaticDecoder dat (Wbar s)) t) ≤ D₀ * epsilon ^ 2)
    -- (H6) Diagonal amplitude lower bound + Gershgorin constants
    --      (replaces the former hPD_lower hypothesis)
    (c_w : ℝ) (hc_w : 0 < c_w)
    (hdiag_t : ∀ t ∈ Set.Icc 0 t_max, ∀ r : Fin d,
        diagAmplitude dat eb (Wbar t) r ≥ c_w * epsilon ^ ((1 : ℝ) / L))
    (δ_off : ℝ) (hδ_nn : 0 ≤ δ_off) (hδ_small : δ_off * ((d : ℝ) - 1) < c_w)
    (hoff_unif : ∀ t ∈ Set.Icc 0 t_max, ∀ r s : Fin d, r ≠ s →
        |offDiagAmplitude dat eb (Wbar t) r s| ≤ δ_off * epsilon ^ ((1 : ℝ) / L))
    -- Tracking error is nonzero on (0, t_max)
    (hDelta_nz : ∀ t ∈ Set.Ico 0 t_max,
        V t - quasiStaticDecoder dat (Wbar t) ≠ 0)
    :
    -- (A) Quasi-static decoder
    (∃ C : ℝ, 0 < C ∧ ∀ t ∈ Set.Icc 0 t_max,
      matFrobNorm (V t - quasiStaticDecoder dat (Wbar t)) ≤ C * epsilon ^ (2 * ((L : ℝ) - 1) / L))
    ∧
    -- (B) Off-diagonal alignment
    (∃ C : ℝ, 0 < C ∧ ∀ r s : Fin d, r ≠ s → ∀ t ∈ Set.Icc 0 t_max,
      |offDiagAmplitude dat eb (Wbar t) r s| ≤ C * epsilon ^ ((1 : ℝ) / L))
    ∧
    (∃ C : ℝ, 0 < C ∧ ∀ r : Fin d, ∀ t ∈ Set.Icc 0 t_max,
      sinAngle dat eb (Wbar t) r ≤ C * epsilon ^ ((1 : ℝ) / L))
    ∧
    -- (C) Feature ordering
    (∃ epsilon_0 : ℝ, 0 < epsilon_0 ∧ epsilon < epsilon_0 →
      ∀ r s : Fin d, (eb.pairs s).rho < (eb.pairs r).rho →
      projectedCovariance dat eb s < projectedCovariance dat eb r →
      (L : ℝ) / (projectedCovariance dat eb r * (eb.pairs r).rho ^ (2 * L - 2) * epsilon ^ ((1 : ℝ) / L))
      < (L : ℝ) / (projectedCovariance dat eb s * (eb.pairs s).rho ^ (2 * L - 2) * epsilon ^ ((1 : ℝ) / L)))
    ∧
    -- (D) Depth is a sharp threshold
    (L = 1 → ∀ r s : Fin d, r ≠ s →
      ∀ C : ℝ, 0 < C →
      ∃ sigma_r sigma_s : ℝ → ℝ,
        ∫ u in Set.Ioo 0 (C / epsilon), preconditioner 1 (sigma_r u) (sigma_s u) ≥ C / epsilon)
    ∧
    -- (E) JEPA vs. MAE
    (∀ r s : Fin d, r ≠ s →
      projectedCovariance dat eb r = projectedCovariance dat eb s →
      (eb.pairs s).rho < (eb.pairs r).rho →
      (eb.pairs r).rho ^ (2 * L - 2 : ℕ) / (eb.pairs s).rho ^ (2 * L - 2 : ℕ) > 1) := by
  -- Derive hPD_lower from the diagonal/off-diagonal conditions via compactness
  obtain hd | hd := Nat.eq_zero_or_pos d
  case inl =>
    subst hd
    exact ⟨⟨1, one_pos, fun t _ => by
            simp [matFrobNorm, quasiStaticDecoder, Finset.univ_eq_empty]
            exact Real.rpow_nonneg heps.le _⟩,
           ⟨1, one_pos, fun r => Fin.elim0 r⟩,
           ⟨1, one_pos, fun r => Fin.elim0 r⟩,
           ⟨1, fun _ r => Fin.elim0 r⟩,
           fun h => absurd h (by omega),
           fun r => Fin.elim0 r⟩
  case inr =>
  -- Derive the uniform PD lower bound
  have hPD_lower := uniform_pd_lower_from_compactness hd dat eb epsilon heps heps_small L hL
    t_max ht_max c_w hc_w δ_off hδ_nn hδ_small Wbar hWbar_cont hdiag_t hoff_unif
  -- Now delegate to the original JEPA_rho_ordering
  exact JEPA_rho_ordering dat eb L hL epsilon heps heps_small t_max ht_max V Wbar
    h_init hWbar_slow hWbar_init hV_flow_ode hV_init hoff_small hWbar_cont hV_cont hVqs_cont
    hPhaseA hVqs_deriv_exists hDrift_bound hPD_lower hDelta_nz

/-! ## Section 7: Dynamics-level ρ*-ordering (strongest result)

    The previous `JEPA_rho_ordering'` gives the *formula-level* ordering of
    critical times. This section assembles Jobs E (`diagAmp_ODE`),
    F (`jepa_critical_time_diag`), and G (`actual_critical_time`) into the
    dynamics-level ordering theorem: under the same hypotheses as
    `JEPA_rho_ordering'`, if `ρ_r* > ρ_s*` and `λ_r* ≥ λ_s*`, then for all
    sufficiently small ε the *actual* JEPA hitting times satisfy
    `T̂_r < T̂_s`.

    See `my_theorems/strongest_result_roadmap.md` for the proof plan and
    `my_theorems/paper.tex` Section 6 (Theorem~\ref{thm:dynamics-ordering})
    for the math statement.

    Status: stub with `sorry` — to be discharged after Jobs E, F, G land.
-/

/-- **Theorem (Dynamics-level ρ*-ordering of feature learning).**
    Under (A1)-(A5), (H1)-(H4), and bootstrap, if `ρ_r* > ρ_s*` and
    `λ_r* ≥ λ_s*`, then the actual JEPA training dynamics satisfy
    `T̂_r < T̂_s` for all ε sufficiently small.

    This subsumes `critical_time_ordering` (which gave only the formula-level
    n=1 Laurent term ordering). -/
theorem JEPA_dynamics_ordering (dat : JEPAData d) (eb : GenEigenbasis dat)
    (L : ℕ) (hL : 2 ≤ L)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (V Wbar : ℝ → ℝ → Matrix (Fin d) (Fin d) ℝ)
      -- Wbar and V parameterised by (epsilon, t)
    (hWbar_slow : ∀ epsilon : ℝ, 0 < epsilon → epsilon < 1 →
      ∃ K : ℝ, 0 < K ∧ ∀ t ∈ Set.Icc 0 t_max,
        matFrobNorm (deriv (Wbar epsilon) t) ≤ K * epsilon ^ 2)
    (hV_flow_ode : ∀ epsilon : ℝ, 0 < epsilon → epsilon < 1 →
      ∀ t ∈ Set.Icc 0 t_max,
        HasDerivAt (V epsilon) (-(gradV dat (Wbar epsilon t) (V epsilon t))) t)
    (htrack : ∀ epsilon : ℝ, 0 < epsilon → epsilon < 1 →
      ∃ K : ℝ, 0 < K ∧ ∀ t ∈ Set.Icc 0 t_max,
        matFrobNorm (V epsilon t - quasiStaticDecoder dat (Wbar epsilon t)) ≤
          K * epsilon ^ (2 * ((L : ℝ) - 1) / L))
    (hoff : ∀ epsilon : ℝ, 0 < epsilon → epsilon < 1 →
      ∃ K : ℝ, 0 < K ∧ ∀ r s : Fin d, r ≠ s → ∀ t ∈ Set.Icc 0 t_max,
        |offDiagAmplitude dat eb (Wbar epsilon t) r s|
          ≤ K * epsilon ^ ((1 : ℝ) / L))
    (r s : Fin d)
    (hrho : (eb.pairs s).rho < (eb.pairs r).rho)
    (hlam : projectedCovariance dat eb s ≤ projectedCovariance dat eb r)
    (p : ℝ) (hp : 0 < p) (hp_lt : p < 1) :
    ∃ epsilon_0 : ℝ, 0 < epsilon_0 ∧ epsilon_0 < 1 ∧
      ∀ epsilon : ℝ, 0 < epsilon → epsilon < epsilon_0 →
        hittingTime (fun t => diagAmplitude dat eb (Wbar epsilon t) r)
                     (p * (eb.pairs r).rho ^ L) t_max
        < hittingTime (fun t => diagAmplitude dat eb (Wbar epsilon t) s)
                     (p * (eb.pairs s).rho ^ L) t_max := by
  -- Proof skeleton (see roadmap):
  -- 1. Apply diagAmp_ODE for both r and s to get Bernoulli ODEs with
  --    error ε^{(2L-1)/L}.
  -- 2. Apply jepa_critical_time_diag to extract t*_r and t*_s as the
  --    Laurent expansions in ε^{1/L}.
  -- 3. Apply actual_critical_time twice to bound |T̂ - t*| = O(ε^{-(L-2)/L})
  --    for both r and s.
  -- 4. Compute t*_s - t*_r: leading λ-only terms cancel when λ_r = λ_s
  --    (for the equal-λ case), or differ by a positive amount when λ_r > λ_s.
  --    The first ρ-distinguishing term is the n=1 summand
  --    L (ρ_r^{-(2L-2)} - ρ_s^{-(2L-2)})/(λ ε^{1/L}) > 0.
  -- 5. The total separation is Θ(ε^{-1/L}); the perturbation error is
  --    o(ε^{-1/L}), so for ε small enough, T̂_r < T̂_s.
  sorry
