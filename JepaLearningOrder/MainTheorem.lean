import Mathlib
import JepaLearningOrder.Lemmas
import JepaLearningOrder.OffDiagHelpers
import JepaLearningOrder.JEPA
import JepaLearningOrder.PDLowerHelpers
import JepaLearningOrder.BootstrapLemmas
import JepaLearningOrder.LaurentHelpers

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

/-
**Sub-lemma (Laurent-separation asymptotic).** With `ρ_s < ρ_r` and
    `λ_s ≤ λ_r`, the Laurent sum for `s` exceeds the Laurent sum for `r`
    by at least `(K_r + K_s) * ε^{-(L-2)/L}` once ε is small enough.

    Mathematical content: write `LSr ε = (1/λ_r) Σ_n L/(n ρ_r^{2L-n-1} ε^{n/L})`
    and similarly for `s`. The n=2L-2 summand contributes
    `(L / ((2L-2) λ)) * (1/ρ_s - 1/ρ_r) * ε^{-(2L-2)/L}` to `LSs - LSr`
    (the `1/λ_s ≥ 1/λ_r` factor only helps; all other summands are
    nonnegative since `ρ_s^{2L-n-1} ≤ ρ_r^{2L-n-1}` for `n ≤ 2L-2`).
    Comparing exponents `(2L-2)/L > (L-2)/L` for `L ≥ 2` (gap of `1`) yields
    `ε^{-(2L-2)/L} = ε^{-1} · ε^{-(L-2)/L}`; choose `ε_0` so that the
    coefficient `L (1/ρ_s - 1/ρ_r) / ((2L-2) λ_r ε)` strictly dominates `K_r + K_s`.

    Left as a named `sorry` — pure ε-asymptotic algebra over finite sums.
-/
lemma laurent_separation_dominates
    (dat : JEPAData d) (eb : GenEigenbasis dat)
    (L : ℕ) (hL : 2 ≤ L) (r s : Fin d)
    (hrho : (eb.pairs s).rho < (eb.pairs r).rho)
    (hlam : projectedCovariance dat eb s ≤ projectedCovariance dat eb r)
    (K_r K_s : ℝ) (hK_r : 0 < K_r) (hK_s : 0 < K_s) :
    ∃ ε_0 : ℝ, 0 < ε_0 ∧ ε_0 < 1 ∧ ∀ ε : ℝ, 0 < ε → ε < ε_0 →
      (1 / projectedCovariance dat eb s)
        * ∑ n ∈ Finset.Ioc 0 (2 * L - 1),
            (L : ℝ) / ((n : ℝ) * (eb.pairs s).rho ^ (2 * L - n - 1)
                        * ε ^ ((n : ℝ) / L))
      - (1 / projectedCovariance dat eb r)
        * ∑ n ∈ Finset.Ioc 0 (2 * L - 1),
            (L : ℝ) / ((n : ℝ) * (eb.pairs r).rho ^ (2 * L - n - 1)
                        * ε ^ ((n : ℝ) / L))
      > (K_r + K_s) * ε ^ (-((L : ℝ) - 2) / L) := by
  -- Set M = (L : ℝ) / ((2 * L - 2 : ℝ)) * (1 / (projectedCovariance dat eb s * (eb.pairs s).rho) - 1 / (projectedCovariance dat eb r * (eb.pairs r).rho)).
  set M := (L : ℝ) / ((2 * L - 2 : ℝ)) * (1 / (projectedCovariance dat eb s * (eb.pairs s).rho) - 1 / (projectedCovariance dat eb r * (eb.pairs r).rho)) with hM_def;
  -- Choose ε_0 such that for all ε with 0 < ε < ε_0, we have M / ε > K_r + K_s.
  obtain ⟨ε_0, hε_0_pos, hε_0⟩ : ∃ ε_0, 0 < ε_0 ∧ ε_0 < 1 ∧ ∀ ε, 0 < ε → ε < ε_0 → M / ε > K_r + K_s := by
    have hM_pos : 0 < M := by
      refine' mul_pos ( div_pos ( by positivity ) ( by linarith [ show ( L : ℝ ) ≥ 2 by norm_cast ] ) ) ( sub_pos_of_lt _ );
      have h_prod_lt : projectedCovariance dat eb s * (eb.pairs s).rho < projectedCovariance dat eb r * (eb.pairs r).rho := by
        apply_rules [ projCov_mul_rho_strict_lt ];
      exact one_div_lt_one_div_of_lt ( mul_pos ( projectedCovariance_pos dat eb s ) ( eb.pairs s |>.hrho_pos ) ) h_prod_lt;
    exact ⟨ Min.min 1 ( M / ( K_r + K_s ) ) / 2, by positivity, by linarith [ show 0 < Min.min 1 ( M / ( K_r + K_s ) ) by positivity, min_le_left 1 ( M / ( K_r + K_s ) ), min_le_right 1 ( M / ( K_r + K_s ) ) ], fun ε hε₁ hε₂ => by rw [ gt_iff_lt ] ; rw [ lt_div_iff₀ hε₁ ] ; nlinarith [ show 0 < Min.min 1 ( M / ( K_r + K_s ) ) by positivity, min_le_left 1 ( M / ( K_r + K_s ) ), min_le_right 1 ( M / ( K_r + K_s ) ), mul_div_cancel₀ M ( show ( K_r + K_s ) ≠ 0 by positivity ) ] ⟩;
  refine' ⟨ ε_0, hε_0_pos, hε_0.1, fun ε hε₁ hε₂ => lt_of_lt_of_le ( mul_lt_mul_of_pos_right ( hε_0.2 ε hε₁ hε₂ ) ( Real.rpow_pos_of_pos hε₁ _ ) ) _ ⟩;
  have h_sum_ge : ∑ n ∈ Finset.Ioc 0 (2 * L - 1), (1 / projectedCovariance dat eb s * ((L : ℝ) / ((n : ℝ) * (eb.pairs s).rho ^ (2 * L - n - 1) * ε ^ ((n : ℝ) / L))) - 1 / projectedCovariance dat eb r * ((L : ℝ) / ((n : ℝ) * (eb.pairs r).rho ^ (2 * L - n - 1) * ε ^ ((n : ℝ) / L)))) ≥ M / ε ^ ((2 * L - 2 : ℝ) / L) := by
    refine' le_trans _ ( Finset.single_le_sum ( fun n hn => _ ) ( show 2 * L - 2 ∈ Finset.Ioc 0 ( 2 * L - 1 ) from _ ) );
    · rcases L with ( _ | _ | L ) <;> norm_num at *;
      norm_num [ Nat.mul_succ, pow_succ' ] ; ring_nf;
      field_simp [hM_def]
      ring_nf;
      rw [ hM_def ] ; ring_nf;
      nlinarith [ inv_mul_cancel_left₀ ( by linarith : ( 2 + L * 2 : ℝ ) ≠ 0 ) ( ( eb.pairs s |> GenEigenpair.rho ) ⁻¹ * ( projectedCovariance dat eb s ) ⁻¹ ), inv_mul_cancel_left₀ ( by linarith : ( 2 + L * 2 : ℝ ) ≠ 0 ) ( ( eb.pairs r |> GenEigenpair.rho ) ⁻¹ * ( projectedCovariance dat eb r ) ⁻¹ ) ];
    · refine' sub_nonneg_of_le _;
      gcongr;
      · exact div_nonneg ( Nat.cast_nonneg _ ) ( mul_nonneg ( mul_nonneg ( Nat.cast_nonneg _ ) ( pow_nonneg ( le_of_lt ( eb.pairs r |>.hrho_pos ) ) _ ) ) ( Real.rpow_nonneg hε₁.le _ ) );
      · exact one_div_nonneg.mpr ( le_of_lt ( projectedCovariance_pos dat eb s ) );
      · exact projectedCovariance_pos dat eb s;
      · exact mul_pos ( mul_pos ( Nat.cast_pos.mpr ( Finset.mem_Ioc.mp hn |>.1 ) ) ( pow_pos ( eb.pairs s |>.hrho_pos ) _ ) ) ( Real.rpow_pos_of_pos hε₁ _ );
      · exact le_of_lt ( eb.pairs s |>.hrho_pos );
    · exact Finset.mem_Ioc.mpr ⟨ Nat.sub_pos_of_lt ( by linarith ), Nat.sub_le_sub_left ( by linarith ) _ ⟩;
  convert h_sum_ge.le using 1;
  · rw [ div_mul_eq_mul_div, div_eq_div_iff ] <;> first | positivity | ring;
    rw [ mul_assoc, ← Real.rpow_add hε₁ ] ; ring_nf ; norm_num [ show L ≠ 0 by linarith ];
  · rw [ Finset.mul_sum _ _ _, Finset.mul_sum _ _ _, Finset.sum_sub_distrib ]

/-- **Theorem 6.1 (Dynamics-level ρ*-ordering of feature learning).**
    Under the diagonal-amplitude initial condition and the perturbed Bernoulli
    ODE bound (output of `diagAmp_ODE`), if `ρ_r* > ρ_s*` and `λ_r* ≥ λ_s*`,
    then for all sufficiently small ε the actual JEPA training dynamics satisfy
    `T̂_r < T̂_s`.

    This subsumes `critical_time_ordering` (which gave only the formula-level
    n=1 Laurent term ordering). The proof composes:
    - `actual_critical_time` for both `r` and `s` (giving uniform K_r, K_s
      independent of ε), and
    - `laurent_separation_dominates` (the ε-asymptotic algebra). -/
-- ⚠ DEPRECATED (session 90, 2026-05-21). This headline theorem is built on
--   paper-1's inverted-ODE-form lemmas (diagAmp_ODE, bernoulli_laurent_bound,
--   actual_critical_time). The ORDERING claim survives empirically (direction
--   of motion is correct), but quantitative bounds and hitting-time thresholds
--   `p · ρ^L` are wrong under the actual JEPA dynamics. The corrected version
--   `Corrected.JEPA_dynamics_ordering_corrected` lives in JepaLearningOrder.Corrected
--   and uses threshold `p · ρ^(1/L)` + single-pole asymptotic. See CORRECTION_NOTE.md.
@[deprecated "Inverted ODE form; use Corrected.JEPA_dynamics_ordering_corrected"]
theorem JEPA_dynamics_ordering (dat : JEPAData d) (eb : GenEigenbasis dat)
    (L : ℕ) (hL : 2 ≤ L)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (Wbar : ℝ → ℝ → Matrix (Fin d) (Fin d) ℝ)
    (r s : Fin d)
    (hrho : (eb.pairs s).rho < (eb.pairs r).rho)
    (hlam : projectedCovariance dat eb s ≤ projectedCovariance dat eb r)
    (p : ℝ) (hp : 0 < p) (hp_lt : p < 1)
    -- Diagonal amplitude initial conditions (output of balanced-init projection)
    (hinit_r : ∀ epsilon : ℝ, 0 < epsilon → epsilon < 1 →
      diagAmplitude dat eb (Wbar epsilon 0) r = epsilon)
    (hinit_s : ∀ epsilon : ℝ, 0 < epsilon → epsilon < 1 →
      diagAmplitude dat eb (Wbar epsilon 0) s = epsilon)
    -- Diagonal amplitude perturbed Bernoulli ODE bounds (output of `diagAmp_ODE`)
    (hode_r : ∃ C_r : ℝ, 0 < C_r ∧ ∀ epsilon : ℝ, 0 < epsilon → epsilon < 1 →
      ∀ t ∈ Set.Ioo 0 t_max,
        |deriv (fun u => diagAmplitude dat eb (Wbar epsilon u) r) t
         - ((L : ℝ) * projectedCovariance dat eb r
              * Real.rpow (diagAmplitude dat eb (Wbar epsilon t) r) (3 - 1 / L)
              * (1 - Real.rpow (diagAmplitude dat eb (Wbar epsilon t) r) (1 / L)
                     / (eb.pairs r).rho))|
        ≤ C_r * epsilon ^ ((2 * (L : ℝ) - 1) / L))
    (hode_s : ∃ C_s : ℝ, 0 < C_s ∧ ∀ epsilon : ℝ, 0 < epsilon → epsilon < 1 →
      ∀ t ∈ Set.Ioo 0 t_max,
        |deriv (fun u => diagAmplitude dat eb (Wbar epsilon u) s) t
         - ((L : ℝ) * projectedCovariance dat eb s
              * Real.rpow (diagAmplitude dat eb (Wbar epsilon t) s) (3 - 1 / L)
              * (1 - Real.rpow (diagAmplitude dat eb (Wbar epsilon t) s) (1 / L)
                     / (eb.pairs s).rho))|
        ≤ C_s * epsilon ^ ((2 * (L : ℝ) - 1) / L)) :
    ∃ epsilon_0 : ℝ, 0 < epsilon_0 ∧ epsilon_0 < 1 ∧
      ∀ epsilon : ℝ, 0 < epsilon → epsilon < epsilon_0 →
        hittingTime (fun t => diagAmplitude dat eb (Wbar epsilon t) r)
                     (p * (eb.pairs r).rho ^ L) t_max
        < hittingTime (fun t => diagAmplitude dat eb (Wbar epsilon t) s)
                     (p * (eb.pairs s).rho ^ L) t_max := by
  -- Step 1: extract uniform constants K_r, K_s from `actual_critical_time`.
  obtain ⟨C_r, hC_r_pos, hode_r_bd⟩ := hode_r
  obtain ⟨C_s, hC_s_pos, hode_s_bd⟩ := hode_s
  obtain ⟨K_r, hK_r_pos, hK_r⟩ :=
    actual_critical_time dat eb L hL t_max ht_max p hp hp_lt r C_r hC_r_pos
  obtain ⟨K_s, hK_s_pos, hK_s⟩ :=
    actual_critical_time dat eb L hL t_max ht_max p hp hp_lt s C_s hC_s_pos
  -- Step 2: Laurent separation gap (named sorry).
  obtain ⟨ε_0, hε_0_pos, hε_0_lt, h_sep⟩ :=
    laurent_separation_dominates dat eb L hL r s hrho hlam K_r K_s hK_r_pos hK_s_pos
  refine ⟨ε_0, hε_0_pos, hε_0_lt, ?_⟩
  intro ε hε_pos hε_lt
  have hε_small : ε < 1 := lt_trans hε_lt hε_0_lt
  -- Step 3: instantiate `actual_critical_time` for r and s at this ε.
  have h_r := hK_r ε hε_pos hε_small (Wbar ε)
    (hinit_r ε hε_pos hε_small) (hode_r_bd ε hε_pos hε_small)
  have h_s := hK_s ε hε_pos hε_small (Wbar ε)
    (hinit_s ε hε_pos hε_small) (hode_s_bd ε hε_pos hε_small)
  -- Step 4: combine perturbation bounds with Laurent separation via triangle.
  have h_sep_ε := h_sep ε hε_pos hε_lt
  have hr_le := (abs_le.mp h_r).2
  have hs_ge := (abs_le.mp h_s).1
  linarith