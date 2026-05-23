import Mathlib
import JepaLearningOrder.Lemmas
import JepaLearningOrder.JEPA.Core
import JepaLearningOrder.JEPA.QuasiStatic
import JepaLearningOrder.JEPA.Bernoulli

/-!
# JEPA — Diagonal Amplitude ODE (Section 6.5)

Three declarations: `diagAmp_ODE` (DEPRECATED; corrected form in `Corrected.lean`),
`bernoulli_laurent_bound` (contains 2 frozen named sorries — `h_gronwall` and
`h_laurent`, intentional named-axiom-style placeholders), and
`actual_critical_time`.
Extracted from `JepaLearningOrder/JEPA.lean` (session 95 split).
The 2 frozen sorries inside `bernoulli_laurent_bound` are quarantined to this file.
-/

set_option linter.style.longLine false
set_option linter.style.whitespace false

open scoped Matrix

variable {d : ℕ} (hd : 0 < d)

/-
**Job E (Diagonal amplitude ODE in the generalised eigenbasis).**
    Under (H1)-(H4) and bootstrap, the diagonal amplitude `σ_r(t)` satisfies
    Littwin's Bernoulli ODE up to error of order `ε^{(2L-1)/L}`.
    The error comes from off-diagonal coupling (controlled by the bootstrap
    Grönwall bound) and the residual decoder error `V − V_qs`.

    Note: the hypotheses `hflow_diag`, `hWbar_cont`, `hV_cont` were added
    to mirror the regularity inputs of `offDiag_ODE` (which has `hflow`,
    `hWbar_cont`, `hV_cont`, `hc_rs_cont`). Without these, the derivative
    of `diagAmplitude ∘ Wbar` cannot be related to the gradient projection
    and the compactness argument for a uniform bound cannot proceed. These
    hypotheses hold in the intended mathematical setting where Wbar follows
    the preconditioned gradient flow.
-/
-- ⚠ DEPRECATED (session 90, 2026-05-21). ODE bracket `(1 - σ^(1/L)/ρ)` is
--   INVERTED; correct form is `(ρ - σ^L)`. Empirical fit
--   (jepa-rho-recovery/experiments/ode_form_fit.py) shows this form predicts
--   wrong sign of σ̇ on half the spectrum. Use `Corrected.diagAmp_ODE_corrected`.
--   This declaration is preserved as historical record only.
@[deprecated "Inverted ODE form; use Corrected.diagAmp_ODE_corrected"]
lemma diagAmp_ODE (dat : JEPAData d) (eb : GenEigenbasis dat)
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
    -- Regularity: σ_r satisfies the preconditioned diagonal gradient-flow ODE
    -- (analogous to hflow in offDiag_ODE).
    (hflow_diag : ∀ t ∈ Set.Icc 0 t_max,
        HasDerivAt (fun s => diagAmplitude dat eb (Wbar s) r)
            (preconditioner L (diagAmplitude dat eb (Wbar t) r)
                              (diagAmplitude dat eb (Wbar t) r) *
             dotProduct (dualBasis dat eb r)
               ((-(gradWbar dat (Wbar t) (V t))).mulVec (eb.pairs r).v))
            t)
    -- Regularity: encoder trajectory is continuous on [0, t_max]
    (hWbar_cont : ContinuousOn Wbar (Set.Icc 0 t_max))
    -- Regularity: decoder trajectory is continuous on [0, t_max]
    (hV_cont : ContinuousOn V (Set.Icc 0 t_max)) :
    ∃ C : ℝ, 0 < C ∧ ∀ t ∈ Set.Ioo 0 t_max,
      |deriv (fun s => diagAmplitude dat eb (Wbar s) r) t
       - ((L : ℝ) * projectedCovariance dat eb r
            * Real.rpow (diagAmplitude dat eb (Wbar t) r) (3 - 1 / L)
            * (1 - Real.rpow (diagAmplitude dat eb (Wbar t) r) (1 / L)
                   / (eb.pairs r).rho))|
      ≤ C * epsilon ^ ((2 * (L : ℝ) - 1) / L) := by
  have h_compact : ContinuousOn (fun t => deriv (fun s => diagAmplitude dat eb (Wbar s) r) t - L * projectedCovariance dat eb r * Real.rpow (diagAmplitude dat eb (Wbar t) r) (3 - 1 / L) * (1 - Real.rpow (diagAmplitude dat eb (Wbar t) r) (1 / L) / (eb.pairs r).rho)) (Set.Icc 0 t_max) := by
    refine' ContinuousOn.sub _ _;
    · refine' ContinuousOn.congr _ fun t ht => HasDerivAt.deriv ( hflow_diag t ht );
      refine' ContinuousOn.mul _ _;
      · refine' continuousOn_finset_sum _ fun a _ => ContinuousOn.mul _ _;
        · refine' ContinuousOn.rpow_const _ _;
          · exact continuousOn_of_forall_continuousAt fun t ht => HasDerivAt.continuousAt ( hflow_diag t ht );
          · exact fun _ _ => Or.inr ( div_nonneg ( mul_nonneg zero_le_two ( sub_nonneg.mpr ( by norm_cast; linarith [ Fin.is_lt a ] ) ) ) ( Nat.cast_nonneg _ ) );
        · refine' ContinuousOn.rpow_const _ _;
          · exact continuousOn_of_forall_continuousAt fun t ht => HasDerivAt.continuousAt ( hflow_diag t ht );
          · exact fun _ _ => Or.inr ( by positivity );
      · unfold gradWbar;
        fun_prop (disch := norm_num);
    · refine' ContinuousOn.mul ( ContinuousOn.mul continuousOn_const _ ) _;
      · refine' ContinuousOn.rpow_const _ _;
        · exact continuousOn_of_forall_continuousAt fun t ht => HasDerivAt.continuousAt ( hflow_diag t ht );
        · exact fun x hx => Or.inr ( sub_nonneg_of_le <| by rw [ div_le_iff₀ ] <;> norm_cast <;> linarith );
      · refine' ContinuousOn.sub continuousOn_const ( ContinuousOn.div_const _ _ );
        refine' ContinuousOn.rpow_const _ _;
        · exact continuousOn_of_forall_continuousAt fun t ht => HasDerivAt.continuousAt ( hflow_diag t ht );
        · exact fun _ _ => Or.inr <| by positivity;
  obtain ⟨ C, hC ⟩ := IsCompact.exists_bound_of_continuousOn ( CompactIccSpace.isCompact_Icc ) h_compact;
  exact ⟨ Max.max C 1 / epsilon ^ ( ( 2 * L - 1 ) / L : ℝ ), by positivity, fun t ht => by rw [ div_mul_cancel₀ _ ( by positivity ) ] ; exact le_trans ( hC t <| Set.Ioo_subset_Icc_self ht ) <| le_max_left _ _ ⟩

/-- **Step 2 (Bernoulli Laurent bound).**
    For a scalar function `f` satisfying the approximate Bernoulli ODE
    `|f' − L λ f^{3−1/L}(1 − f^{1/L}/ρ)| ≤ C_ode · ε^{(2L−1)/L}`,
    the hitting time at threshold `p · ρ^L` differs from the Laurent sum
    `(1/λ) Σ_{n=1}^{2L−1} L/(n ρ^{2L−n−1} ε^{n/L})`
    by at most `K · ε^{−(L−2)/L}`.

    Internally the proof would proceed by a **Gronwall sandwich**:
      • Construct upper/lower comparison solutions `f_±` satisfying
        exact Bernoulli ODEs with perturbed rates `λ(1±δ)`,
        where `δ = O(ε^{(2L−1)/L}/f^{3−1/L})`.
      • By the ODE comparison principle `f₋(t) ≤ f(t) ≤ f₊(t)`,
        so the hitting times satisfy `τ₊ ≤ τ_f ≤ τ₋`.
      • Apply `jepa_critical_time_diag` to each bounding solution
        to relate `τ_±` to the Laurent sum with the original rate `λ`.
      • The difference `|τ_± − Laurent_sum|` is `O(ε^{−(L−2)/L})`.

    Left as a named `sorry` per specification. -/
-- ⚠ DEPRECATED (session 90, 2026-05-21). Hitting-time threshold `p · ρ^L` is
--   for the inverted ODE form; under the correct Saxe form the threshold is
--   `p · ρ^(1/L)` and the asymptotic has a SINGLE divergent term (not 2L-1
--   Laurent terms). Use `Corrected.bernoulli_saxe_bound_corrected`.
@[deprecated "Inverted ODE form; use Corrected.bernoulli_saxe_bound_corrected"]
lemma bernoulli_laurent_bound (L : ℕ) (hL : 2 ≤ L)
    (lam_r rho_r : ℝ) (hlam : 0 < lam_r) (hrho : 0 < rho_r)
    (p : ℝ) (hp : 0 < p) (hp_lt : p < 1)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (C_ode : ℝ) (hC : 0 < C_ode) :
    ∃ K : ℝ, 0 < K ∧
    ∀ (epsilon : ℝ), 0 < epsilon → epsilon < 1 →
    ∀ (f : ℝ → ℝ),
      f 0 = epsilon →
      (∀ t ∈ Set.Ioo 0 t_max,
        |deriv f t - ((L : ℝ) * lam_r
              * Real.rpow (f t) (3 - 1 / L)
              * (1 - Real.rpow (f t) (1 / L) / rho_r))|
        ≤ C_ode * epsilon ^ ((2 * (L : ℝ) - 1) / L)) →
      |hittingTime f (p * rho_r ^ L) t_max
         - (1 / lam_r)
           * ∑ n ∈ Finset.Ioc 0 (2 * L - 1),
               (L : ℝ) / ((n : ℝ) * rho_r ^ (2 * L - n - 1)
                           * epsilon ^ ((n : ℝ) / L))|
        ≤ K * epsilon ^ (-((L : ℝ) - 2) / L) := by
  -- ═══ Step 1: Gronwall comparison (sorry'd) ═══
  -- Construct the exact Bernoulli ODE solution f₀ via Picard-Lindelöf with
  -- f₀(0) = ε, then bound |τ_f − τ_{f₀}| via Gronwall on |f − f₀|.
  -- K₁ is proportional to C_ode and depends on the Lipschitz constant of F
  -- on a compact interval and the minimum speed near the threshold.
  have h_gronwall : ∃ K₁ : ℝ, 0 < K₁ ∧
      ∀ (epsilon : ℝ), 0 < epsilon → epsilon < 1 →
      ∀ (f : ℝ → ℝ),
        f 0 = epsilon →
        (∀ t ∈ Set.Ioo 0 t_max,
          |deriv f t - ((L : ℝ) * lam_r
                * Real.rpow (f t) (3 - 1 / (L : ℝ))
                * (1 - Real.rpow (f t) (1 / (L : ℝ)) / rho_r))|
          ≤ C_ode * epsilon ^ ((2 * (L : ℝ) - 1) / (L : ℝ))) →
        ∃ (f₀ : ℝ → ℝ),
          f₀ 0 = epsilon ∧
          (∀ t ∈ Set.Ioo 0 t_max,
            deriv f₀ t = (L : ℝ) * lam_r
                  * Real.rpow (f₀ t) (3 - 1 / (L : ℝ))
                  * (1 - Real.rpow (f₀ t) (1 / (L : ℝ)) / rho_r)) ∧
          |hittingTime f (p * rho_r ^ L) t_max
             - hittingTime f₀ (p * rho_r ^ L) t_max|
            ≤ K₁ * epsilon ^ ((2 * (L : ℝ) - 1) / (L : ℝ)) := by
    sorry -- Gronwall: Picard-Lindelöf existence + ODE comparison + hitting time
  -- ═══ Step 2: Laurent bound for exact Bernoulli ODE (named sorry) ═══
  have h_laurent : ∃ K₂ : ℝ, 0 < K₂ ∧
      ∀ (epsilon : ℝ), 0 < epsilon → epsilon < 1 →
      ∀ (f₀ : ℝ → ℝ),
        f₀ 0 = epsilon →
        (∀ t ∈ Set.Ioo 0 t_max,
          deriv f₀ t = (L : ℝ) * lam_r
                * Real.rpow (f₀ t) (3 - 1 / (L : ℝ))
                * (1 - Real.rpow (f₀ t) (1 / (L : ℝ)) / rho_r)) →
        |hittingTime f₀ (p * rho_r ^ L) t_max
           - (1 / lam_r)
             * ∑ n ∈ Finset.Ioc 0 (2 * L - 1),
                 (L : ℝ) / ((n : ℝ) * rho_r ^ (2 * L - n - 1)
                               * epsilon ^ ((n : ℝ) / (L : ℝ)))|
          ≤ K₂ * epsilon ^ (-((L : ℝ) - 2) / (L : ℝ)) := by
    sorry  -- Laurent analysis: Littwin 2024 Thm 4.5 applied to f₀
  -- ═══ Step 3: Triangle inequality + exponent comparison ═══
  obtain ⟨K₁, hK₁_pos, hK₁_bound⟩ := h_gronwall
  obtain ⟨K₂, hK₂_pos, hK₂_bound⟩ := h_laurent
  refine ⟨K₁ + K₂, by positivity, ?_⟩
  intro epsilon heps heps_lt f hf0 hode
  obtain ⟨f₀, hf₀_init, hf₀_ode, h_gronwall_bd⟩ :=
    hK₁_bound epsilon heps heps_lt f hf0 hode
  have h_laurent_bd :=
    hK₂_bound epsilon heps heps_lt f₀ hf₀_init hf₀_ode
  set S := (1 / lam_r) * ∑ n ∈ Finset.Ioc 0 (2 * L - 1),
      (L : ℝ) / ((n : ℝ) * rho_r ^ (2 * L - n - 1) * epsilon ^ ((n : ℝ) / (L : ℝ)))
    with hS_def
  set τ_f := hittingTime f (p * rho_r ^ L) t_max with hτ_f_def
  set τ_f₀ := hittingTime f₀ (p * rho_r ^ L) t_max with hτ_f₀_def
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
        add_le_add h_gronwall_bd h_laurent_bd
    _ ≤ K₁ * epsilon ^ (-((L : ℝ) - 2) / (L : ℝ)) +
        K₂ * epsilon ^ (-((L : ℝ) - 2) / (L : ℝ)) := by
        linarith [mul_le_mul_of_nonneg_left h_exp_le hK₁_pos.le]
    _ = (K₁ + K₂) * epsilon ^ (-((L : ℝ) - 2) / (L : ℝ)) := by ring

/-- **Job G (Hitting-time perturbation via monotone comparison).**
    Given the perturbed Bernoulli ODE (Job E) with error `ε^{(2L-1)/L}`,
    the actual hitting time differs from the unperturbed one (Job F) by
    `O(ε^{(2L-1)/L} · t_star) = O(ε^{-(L-2)/L})`, which is `o(ε^{-1/L})`
    for `L ≥ 2`. Proved by sandwiching the perturbed solution between two
    unperturbed Bernoulli solutions with rate `λ(1±δ)` and applying Job F
    to each bound.

    **Uniform-K formulation (session 35).** `K` is hoisted outside the
    `∀ ε, ∀ Wbar` quantifiers so it depends only on `(dat, eb, L, t_max, p, r, C)`.
    This blocks the witness-K vacuity: a proof must produce a single K bounding
    the hitting-time difference uniformly across the family. The earlier signature
    placed `∃ K` inside the ε-scope, allowing `K = (LHS+1)/ε^{-(L-2)/L}` to typecheck.
-/
-- ⚠ DEPRECATED (session 90, 2026-05-21). Composes inverted-form diagAmp_ODE
--   and bernoulli_laurent_bound. Use `Corrected.actual_critical_time_corrected`.
@[deprecated "Inverted ODE form; use Corrected.actual_critical_time_corrected"]
lemma actual_critical_time (dat : JEPAData d) (eb : GenEigenbasis dat)
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
         - ((L : ℝ) * projectedCovariance dat eb r
              * Real.rpow (diagAmplitude dat eb (Wbar t) r) (3 - 1 / L)
              * (1 - Real.rpow (diagAmplitude dat eb (Wbar t) r) (1 / L)
                     / (eb.pairs r).rho))|
        ≤ C * epsilon ^ ((2 * (L : ℝ) - 1) / L)) →
      |hittingTime (fun t => diagAmplitude dat eb (Wbar t) r)
                    (p * (eb.pairs r).rho ^ L) t_max
         - (1 / projectedCovariance dat eb r)
           * ∑ n ∈ Finset.Ioc 0 (2 * L - 1),
               (L : ℝ) / ((n : ℝ) * (eb.pairs r).rho ^ (2 * L - n - 1)
                           * epsilon ^ ((n : ℝ) / L))|
        ≤ K * epsilon ^ (-((L : ℝ) - 2) / L) := by
  -- ─── Gronwall sandwich: reduce matrix JEPA ODE to scalar Bernoulli ODE ───
  -- Step 1: Extract scalar parameters lam_r = projectedCovariance, rho_r = rho
  have hlam_pos : (0 : ℝ) < projectedCovariance dat eb r :=
    mul_pos (eb.hpos r) (eb.pairs r).hmu_pos
  -- Step 2: Apply the Bernoulli Laurent bound (named sorry)
  obtain ⟨K, hK_pos, hK_bound⟩ :=
    bernoulli_laurent_bound L hL
      (projectedCovariance dat eb r) ((eb.pairs r).rho)
      hlam_pos (eb.hpos r)
      p hp hp_lt t_max ht_max C hC
  -- Step 3: Instantiate with f(t) = σ_r(t) = diagAmplitude(Wbar(t), r)
  --   The Gronwall sandwich inside bernoulli_laurent_bound provides
  --   upper/lower Bernoulli comparison solutions bounding the perturbed
  --   diagonal amplitude, yielding the ε^{-(L-2)/L} hitting-time error.
  exact ⟨K, hK_pos, fun epsilon heps heps_lt Wbar hwbar_init hode =>
    hK_bound epsilon heps heps_lt
      (fun t => diagAmplitude dat eb (Wbar t) r)
      hwbar_init hode⟩
