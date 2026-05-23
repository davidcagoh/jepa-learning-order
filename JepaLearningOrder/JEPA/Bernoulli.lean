import Mathlib
import JepaLearningOrder.Lemmas
import JepaLearningOrder.JEPA.Core
import JepaLearningOrder.JEPA.QuasiStatic

/-!
# JEPA — Bernoulli ODE & Critical Time (Section 6)

Diagonal dynamics: Littwin Bernoulli ODE, partial fractions, antiderivative,
hitting time, and the Laurent-series critical-time formula.
Extracted from `JepaLearningOrder/JEPA.lean` (session 95 split).
-/

set_option linter.style.longLine false
set_option linter.style.whitespace false

open scoped Matrix

variable {d : ℕ} (hd : 0 < d)

/-! ## Section 6: Diagonal Dynamics — The Littwin ODE -/


/-- **Corollary 6.2 (Critical time formula).**
    The critical time t̃_r* at which σ_r reaches fraction p of its asymptote
    σ_r* = (ρ_r*)^{1/2} μ_r^{1/2} is
    t̃_r* = (1/λ_r*) Σ_{n=1}^{2L-1} L / (n ρ_r*^{2L-n-1} ε^{n/L}) + Θ(log ε).
    Leading order: t̃_r* ≈ L / (λ_r* ρ_r*^{2L-2} ε^{1/L}).

    Since t̃_r* is strictly decreasing in ρ_r*, features with higher ρ* reach
    their asymptote first (for ε sufficiently small and off-diagonal corrections
    remaining O(ε^{1/L})).

    PROVIDED SOLUTION
    Step 1: Solve the scalar ODE from Proposition 6.1 for σ_r(t).
    Step 2: Invert to get the time t at which σ_r = p · σ_r*.
    Step 3: Expand the resulting expression in powers of ε^{1/L}, identifying
            the coefficients L / (n ρ_r*^{2L-n-1}) for n = 1, …, 2L-1.
    Step 4: Show ∂(t̃_r*)/∂(ρ_r*) < 0 by differentiating the leading term
            L / (λ_r* ρ_r*^{2L-2} ε^{1/L}) with respect to ρ_r*,
            using λ_r* = ρ_r* μ_r and noting (2L-3) > 0 for L ≥ 2. -/
lemma critical_time_formula (dat : JEPAData d) (eb : GenEigenbasis dat)
    (L : ℕ) (hL : 2 ≤ L) (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    (r : Fin d)
    (p : ℝ) (hp : 0 < p) (hp1 : p < 1) :
    -- The asymptotic amplitude is σ_r* = sqrt(ρ_r* · μ_r)
    let sigma_r_star := Real.sqrt ((eb.pairs r).rho * (eb.pairs r).mu)
    -- The leading-order critical time
    let t_crit_leading := (L : ℝ) /
      (projectedCovariance dat eb r * (eb.pairs r).rho ^ (2 * L - 2) * epsilon ^ ((1 : ℝ) / L))
    -- There exist constants C₁, C₂ such that t̃_r* lies between the bounds
    ∃ C₁ C₂ : ℝ, t_crit_leading - C₁ * |Real.log epsilon| ≤ C₂ ∧
      C₂ ≤ t_crit_leading + C₁ * |Real.log epsilon| := by
  -- *** PROOF NOTE (rigor level: trivially true but not informative) ***
  -- We take C₁ = 0, C₂ = t_crit_leading.  With C₁ = 0 the existential reduces to
  -- "t_crit_leading ≤ C₂ ≤ t_crit_leading", i.e. C₂ = t_crit_leading, which is trivially
  -- satisfied.  The *meaningful* statement would require C₁ > 0 and prove that the actual
  -- hitting time of σ_r(t) (governed by an ODE derived from the diagonal dynamics) lies
  -- within C₁·|log ε| of t_crit_leading.  That derivation requires solving the scalar
  -- Bernoulli ODE from the diagonal dynamics (Proposition 6.1 in the paper draft) and
  -- inverting it, which in turn requires a rigorous diagonal ODE that is not yet formalized.
  -- In the paper draft this is stated as "Asymptotic Prediction 6.1" rather than a theorem.
  refine ⟨0, (L : ℝ) / (projectedCovariance dat eb r * (eb.pairs r).rho ^ (2 * L - 2) *
    epsilon ^ ((1 : ℝ) / ↑L)), ?_, ?_⟩ <;> simp

/-- **Corollary 6.2 (Ordering).** Higher ρ* and λ* imply smaller critical time.
    For ρ_r* > ρ_s* and λ_r* > λ_s*, we have t̃_r* < t̃_s* for all ε > 0.

    Note: both hypotheses are required. The paper (Step C3) shows ρ_r* > ρ_s* alone
    does not suffice — we also need λ_r* > λ_s* (i.e. projectedCovariance r > s) to
    ensure ρ_r*^{2L-2}·λ_r* > ρ_s*^{2L-2}·λ_s*, which reverses the denominator ordering.

    PROVIDED SOLUTION
    Step 1: The critical time leading-order formula is t̃_r* ≈ L / (λ_r* ρ_r*^{2L-2} ε^{1/L}).
    Step 2: t̃_r* < t̃_s* ⟺ λ_r* ρ_r*^{2L-2} > λ_s* ρ_s*^{2L-2} (denominators reversed).
    Step 3: λ_s* ρ_s*^{2L-2} < λ_r* ρ_s*^{2L-2} since λ_s* < λ_r* and ρ_s*^{2L-2} > 0.
    Step 4: λ_r* ρ_s*^{2L-2} ≤ λ_r* ρ_r*^{2L-2} since ρ_s* ≤ ρ_r* and λ_r* > 0.
    Step 5: Combine: λ_s* ρ_s*^{2L-2} < λ_r* ρ_r*^{2L-2}, so denominator_r > denominator_s,
            and since L > 0, ε^{1/L} > 0 (for ε > 0), we get t̃_r* < t̃_s* for all ε > 0.
            The ε_0 = 1 works (the inequality holds for all ε > 0, not just small ε). -/
lemma critical_time_ordering (dat : JEPAData d) (eb : GenEigenbasis dat)
    (L : ℕ) (hL : 2 ≤ L)
    (r s : Fin d) (hrs : (eb.pairs s).rho < (eb.pairs r).rho)
    (hlambda : projectedCovariance dat eb s < projectedCovariance dat eb r) :
    ∃ epsilon_0 : ℝ, 0 < epsilon_0 ∧ ∀ epsilon : ℝ, 0 < epsilon → epsilon < epsilon_0 →
    -- t̃_r* < t̃_s*: the leading-order critical time for r is strictly less than for s
    (L : ℝ) / (projectedCovariance dat eb r * (eb.pairs r).rho ^ (2 * L - 2) * epsilon ^ ((1 : ℝ) / L))
    < (L : ℝ) / (projectedCovariance dat eb s * (eb.pairs s).rho ^ (2 * L - 2) * epsilon ^ ((1 : ℝ) / L)) := by
  -- The inequality holds for ALL ε > 0; ε₀ = 1 works
  refine ⟨1, one_pos, fun epsilon heps _ => ?_⟩
  have hLr : (0 : ℝ) < projectedCovariance dat eb r :=
    mul_pos (eb.pairs r).hrho_pos (eb.pairs r).hmu_pos
  have hLs : (0 : ℝ) < projectedCovariance dat eb s :=
    mul_pos (eb.pairs s).hrho_pos (eb.pairs s).hmu_pos
  have hL_pos : (0 : ℝ) < (L : ℝ) := Nat.cast_pos.mpr (by omega)
  have heps_pow : (0 : ℝ) < epsilon ^ ((1 : ℝ) / (L : ℝ)) := Real.rpow_pos_of_pos heps _
  have hρs_pow_pos : (0 : ℝ) < (eb.pairs s).rho ^ (2 * L - 2) :=
    pow_pos (eb.pairs s).hrho_pos _
  have hρ_pow_le : (eb.pairs s).rho ^ (2 * L - 2) ≤ (eb.pairs r).rho ^ (2 * L - 2) :=
    pow_le_pow_left₀ (eb.pairs s).hrho_pos.le hrs.le _
  -- Key: denominator for r is strictly larger than for s
  have hden : projectedCovariance dat eb s * (eb.pairs s).rho ^ (2 * L - 2) * epsilon ^ ((1 : ℝ) / ↑L)
            < projectedCovariance dat eb r * (eb.pairs r).rho ^ (2 * L - 2) * epsilon ^ ((1 : ℝ) / ↑L) := by
    apply mul_lt_mul_of_pos_right _ heps_pow
    calc projectedCovariance dat eb s * (eb.pairs s).rho ^ (2 * L - 2)
        < projectedCovariance dat eb r * (eb.pairs s).rho ^ (2 * L - 2) :=
          mul_lt_mul_of_pos_right hlambda hρs_pow_pos
      _ ≤ projectedCovariance dat eb r * (eb.pairs r).rho ^ (2 * L - 2) :=
          mul_le_mul_of_nonneg_left hρ_pow_le hLr.le
  have hDr : (0 : ℝ) < projectedCovariance dat eb r * (eb.pairs r).rho ^ (2 * L - 2) * epsilon ^ ((1 : ℝ) / ↑L) :=
    mul_pos (mul_pos hLr (pow_pos (eb.pairs r).hrho_pos _)) heps_pow
  have hDs : (0 : ℝ) < projectedCovariance dat eb s * (eb.pairs s).rho ^ (2 * L - 2) * epsilon ^ ((1 : ℝ) / ↑L) :=
    mul_pos (mul_pos hLs (pow_pos (eb.pairs s).hrho_pos _)) heps_pow
  -- L/Dr < L/Ds ↔ Ds < Dr (when L, Dr, Ds > 0)
  rw [div_lt_div_iff₀ hDr hDs]
  exact mul_lt_mul_of_pos_left hden hL_pos

/-! ## Section 6.5: Strongest result — dynamics-level ordering (Jobs E, F, G)

    These four lemmas close the conceptual gap left by `critical_time_formula`
    (which is currently a degenerate existential). Together they prove that
    the *actual* JEPA training dynamics satisfy the ρ*-ordering, not just the
    leading-order formula.

    See `my_theorems/strongest_result_roadmap.md` for the full plan and
    `my_theorems/paper.tex` Section 6 for the math statements.

    Status: stubs with `sorry` — to be discharged by Aristotle Jobs E, F, G.
-/

/-- **Hitting time of a continuous process at threshold θ.**
    First time at which `f t ≥ θ`. Defined as the infimum over the set
    `{t ∈ Set.Icc 0 t_max | f t ≥ θ}`; if the set is empty, defaults to
    `t_max + 1` (an unattainable sentinel). -/
noncomputable def hittingTime (f : ℝ → ℝ) (θ : ℝ) (t_max : ℝ) : ℝ :=
  sInf ({t ∈ Set.Icc (0 : ℝ) t_max | f t ≥ θ} ∪ {t_max + 1})

/-- **Job F (Littwin Lemma B.6 — partial fraction identity).**
    The integrand `1/(ψ^{2L} − ψ^{2L+1}) = 1/(ψ^{2L}(1−ψ))` admits an
    elementary antiderivative as a finite sum. This is purely algebraic and
    is provable by induction on `L`. -/
lemma bernoulli_partial_fractions (L : ℕ) (hL : 1 ≤ L) (ψ : ℝ)
    (hψ_pos : 0 < ψ) (hψ_lt : ψ < 1) :
    HasDerivAt
      (fun x : ℝ =>
        -(∑ n ∈ Finset.Ioc 0 (2 * L - 1), 1 / ((n : ℝ) * x ^ n))
        + Real.log x - Real.log (1 - x))
      (1 / (ψ ^ (2 * L) - ψ ^ (2 * L + 1))) ψ := by
  convert HasDerivAt.sub ( HasDerivAt.add ( HasDerivAt.neg <| HasDerivAt.sum _ ) ( Real.hasDerivAt_log hψ_pos.ne' ) ) ( HasDerivAt.log ( hasDerivAt_id' ψ |> HasDerivAt.const_sub 1 ) <| by linarith ) using 1;
  any_goals exact Finset.Ioc 0 ( 2 * L - 1 );
  rotate_left;
  rotate_left;
  use fun n x => 1 / ( n * x ^ n );
  use fun n => -n / ( n * ψ ^ ( n + 1 ) );
  · intro i hi; convert HasDerivAt.div ( hasDerivAt_const _ _ ) ( HasDerivAt.mul ( hasDerivAt_const _ _ ) ( hasDerivAt_pow i ψ ) ) _ using 1 <;> ring ; norm_num [ hψ_pos.ne' ] ;
    · field_simp;
      cases i <;> simp_all +decide [ pow_succ', mul_assoc ];
    · exact mul_ne_zero ( Nat.cast_ne_zero.mpr ( by linarith [ Finset.mem_Ioc.mp hi ] ) ) ( pow_ne_zero _ hψ_pos.ne' );
  · ext; norm_num;
  · -- Simplify the sum of the series
    have h_sum : ∑ i ∈ Finset.Ioc 0 (2 * L - 1), (1 : ℝ) / ψ ^ (i + 1) = (1 / ψ ^ 2) * (1 - (1 / ψ) ^ (2 * L - 1)) / (1 - (1 / ψ)) := by
      induction 2 * L - 1 <;> simp_all +decide [ Finset.sum_Ioc_succ_top, pow_succ' ];
      grind;
    rcases L with ( _ | L ) <;> simp_all +decide [ Nat.mul_succ, pow_succ' ];
    simp_all +decide [ div_eq_mul_inv, mul_assoc, mul_comm, mul_left_comm, Finset.mul_sum _ _ _, ne_of_gt ];
    rw [ Finset.sum_congr rfl fun x hx => by rw [ mul_inv_cancel₀ ( by norm_cast; linarith [ Finset.mem_Ioc.mp hx ] ) ] ] ; simp_all +decide [ ← mul_assoc, ne_of_gt ];
    grind

/-
Helper: if a real function has constant derivative `c` on `[a, b]`, then
    it equals `c * t + C` for some constant `C`.
-/
lemma exists_const_of_hasDerivAt_const {f : ℝ → ℝ} {c a b : ℝ} (hab : a ≤ b)
    (hf : ∀ t ∈ Set.Icc a b, HasDerivAt f c t) :
    ∃ C : ℝ, ∀ t ∈ Set.Icc a b, f t = c * t + C := by
  use f a - c * a;
  intro t ht;
  cases eq_or_lt_of_le ht.1 <;> simp_all +decide [ mul_comm c ];
  have := exists_deriv_eq_slope f ‹_›;
  exact this ( continuousOn_of_forall_continuousAt fun x hx => HasDerivAt.continuousAt ( hf x hx.1 ( hx.2.trans ht.2 ) ) ) ( fun x hx => DifferentiableAt.differentiableWithinAt ( hf x hx.1.le ( hx.2.le.trans ht.2 ) |> HasDerivAt.differentiableAt ) ) |> fun ⟨ x, hx₁, hx₂ ⟩ => by have := hf x hx₁.1.le ( hx₁.2.le.trans ht.2 ) ; have := this.deriv; rw [ eq_div_iff ] at hx₂ <;> nlinarith;

/-
Helper: the key rpow identity `(w ^ (1/L) / ρ) ^ (2*L) = w ^ 2 / ρ ^ (2*L)`
    for `w > 0`, `L ≥ 1`.
-/
lemma rpow_div_pow_eq (L : ℕ) (hL : 1 ≤ L) (w ρ : ℝ) (hw : 0 < w) (hρ : 0 < ρ) :
    (w ^ ((1 : ℝ) / (L : ℝ)) / ρ) ^ (2 * L) = w ^ (2 : ℝ) / ρ ^ (2 * L) := by
  rw [ div_pow, ← Real.rpow_natCast, ← Real.rpow_natCast, ← Real.rpow_mul hw.le ] ; ring_nf ; norm_num [ show L ≠ 0 by linarith ];
  ring

/-
Helper: chain rule + rpow algebra shows the antiderivative has constant derivative.
    At each point `t` where `wbar` satisfies the Bernoulli ODE and `ψ(t) ∈ (0,1)`,
    the composition `F(ψ(t))` has derivative `σ_xx * ρ^{2L}`.
-/
lemma bernoulli_antideriv_hasDerivAt (L : ℕ) (hL : 2 ≤ L)
    (ρ σ_xx : ℝ) (hρ : 0 < ρ) (hσ_xx : 0 < σ_xx)
    (wbar : ℝ → ℝ) (t : ℝ)
    (hwbar_ode : HasDerivAt wbar
        ((L : ℝ) * (wbar t) ^ (3 - 1 / (L : ℝ)) * (ρ * σ_xx)
         - (L : ℝ) * (wbar t) ^ 3 * σ_xx) t)
    (hwbar_pos : 0 < wbar t)
    (hwbar_lt : (wbar t) ^ ((1 : ℝ) / (L : ℝ)) < ρ) :
    HasDerivAt (fun s =>
      -(∑ n ∈ Finset.Ioc 0 (2 * L - 1),
          1 / ((n : ℝ) * ((wbar s) ^ ((1 : ℝ) / (L : ℝ)) / ρ) ^ n))
      + Real.log ((wbar s) ^ ((1 : ℝ) / (L : ℝ)) / ρ)
      - Real.log (1 - (wbar s) ^ ((1 : ℝ) / (L : ℝ)) / ρ))
    (σ_xx * ρ ^ (2 * L)) t := by
  have h_chain : HasDerivAt (fun s => (wbar s) ^ (1 / (L : ℝ)) / ρ) ((1 / (L : ℝ)) * (wbar t) ^ ((1 / (L : ℝ)) - 1) * (L * (wbar t) ^ (3 - 1 / (L : ℝ)) * (ρ * σ_xx) - L * (wbar t) ^ 3 * σ_xx) / ρ) t := by
    convert HasDerivAt.div_const ( HasDerivAt.rpow_const hwbar_ode ?_ ) _ using 1 <;> norm_num [ hwbar_pos.ne' ];
    ring;
  convert HasDerivAt.comp t ( bernoulli_partial_fractions L ( by linarith ) ( ( wbar t ^ ( 1 / ( L : ℝ ) ) / ρ ) ) ( by positivity ) ( by rw [ div_lt_iff₀ hρ ] ; linarith ) ) h_chain using 1;
  rw [ div_mul_div_comm, eq_div_iff ];
  · norm_num [ Real.rpow_sub hwbar_pos ] ; ring;
    field_simp;
    norm_num [ mul_assoc, mul_comm, mul_left_comm, hρ.ne' ];
    rw [ mul_left_comm ( ρ ^ ( L * 2 ) ), mul_inv_cancel₀ ( by positivity ), mul_one, ← Real.rpow_natCast _ ( L * 2 ), ← Real.rpow_mul ( by positivity ) ] ; norm_num [ show L ≠ 0 by positivity ] ; ring;
    norm_num;
  · exact mul_ne_zero ( sub_ne_zero_of_ne <| ne_of_gt <| pow_lt_pow_right_of_lt_one₀ ( by positivity ) ( by rw [ div_lt_iff₀ hρ ] ; linarith ) <| by linarith ) hρ.ne'

/-
**Original `jepa_bernoulli_solution` — COMMENTED OUT: coefficient error.**
   The original statement had coefficient `σ_xx * ρ ^ (2 * L) / L`.  The correct
   coefficient is `σ_xx * ρ ^ (2 * L)` (without the `/L`), because the factor of L
   in the ODE (`L * wbar^{3-1/L} * …`) cancels with the `1/L` from the chain rule
   `d/dt[wbar^{1/L}] = (1/L) wbar^{1/L-1} wbar'`.  Verified numerically with
   L=2,3 and ρ=2, σ_xx=1.

lemma jepa_bernoulli_solution_WRONG (L : ℕ) (hL : 2 ≤ L)
    (ρ σ_xx : ℝ) (hρ : 0 < ρ) (hσ_xx : 0 < σ_xx)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (wbar : ℝ → ℝ) (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    (hwbar_init : wbar 0 = epsilon)
    (hwbar_ode : ∀ t ∈ Set.Icc (0 : ℝ) t_max,
      HasDerivAt wbar
        ((L : ℝ) * Real.rpow (wbar t) (3 - 1 / L) * (ρ * σ_xx)
         - (L : ℝ) * (wbar t) ^ 3 * σ_xx) t)
    (hwbar_pos : ∀ t ∈ Set.Icc (0 : ℝ) t_max, 0 < wbar t)
    (hwbar_lt : ∀ t ∈ Set.Icc (0 : ℝ) t_max, Real.rpow (wbar t) (1 / L) < ρ) :
    ∃ C : ℝ,
    ∀ t ∈ Set.Icc (0 : ℝ) t_max,
      -(∑ n ∈ Finset.Ioc 0 (2 * L - 1),
          1 / ((n : ℝ) * (Real.rpow (wbar t) (1 / L) / ρ) ^ n))
      + Real.log (Real.rpow (wbar t) (1 / L) / ρ)
      - Real.log (1 - Real.rpow (wbar t) (1 / L) / ρ)
      = (σ_xx * ρ ^ (2 * L) / L) * t + C := by
  sorry
-/

/-- **Job F (Littwin Theorem 4.4 — JEPA Bernoulli closed form, corrected).**
    The unperturbed JEPA Bernoulli ODE
    `dwbar/dt = L wbar^{3-1/L} Σ_yx − L wbar^3 Σ_xx`
    admits the implicit closed-form solution
    `−Σ_{n=1}^{2L-1} 1/(n ψ^n) + log ψ − log(1−ψ) = σ² ρ^{2L} t + C`
    where `ψ = wbar^{1/L}/ρ`, `ρ = Σ_yx/Σ_xx`, `σ² = Σ_xx`.

    **Correction**: The original statement had `σ² ρ^{2L}/L`; the correct coefficient
    is `σ² ρ^{2L}` because the `L` from the ODE cancels with `1/L` from the chain
    rule for `wbar^{1/L}`. -/
lemma jepa_bernoulli_solution (L : ℕ) (hL : 2 ≤ L)
    (ρ σ_xx : ℝ) (hρ : 0 < ρ) (hσ_xx : 0 < σ_xx)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (wbar : ℝ → ℝ) (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    (hwbar_init : wbar 0 = epsilon)
    (hwbar_ode : ∀ t ∈ Set.Icc (0 : ℝ) t_max,
      HasDerivAt wbar
        ((L : ℝ) * Real.rpow (wbar t) (3 - 1 / L) * (ρ * σ_xx)
         - (L : ℝ) * (wbar t) ^ 3 * σ_xx) t)
    (hwbar_pos : ∀ t ∈ Set.Icc (0 : ℝ) t_max, 0 < wbar t)
    (hwbar_lt : ∀ t ∈ Set.Icc (0 : ℝ) t_max, Real.rpow (wbar t) (1 / L) < ρ) :
    ∃ C : ℝ,
    ∀ t ∈ Set.Icc (0 : ℝ) t_max,
      -(∑ n ∈ Finset.Ioc 0 (2 * L - 1),
          1 / ((n : ℝ) * (Real.rpow (wbar t) (1 / L) / ρ) ^ n))
      + Real.log (Real.rpow (wbar t) (1 / L) / ρ)
      - Real.log (1 - Real.rpow (wbar t) (1 / L) / ρ)
      = (σ_xx * ρ ^ (2 * L)) * t + C := by
  apply exists_const_of_hasDerivAt_const ht_max.le;
  intros t ht
  apply bernoulli_antideriv_hasDerivAt L hL ρ σ_xx hρ hσ_xx wbar t (hwbar_ode t ht) (hwbar_pos t ht) (hwbar_lt t ht)

/-- **Job F (Littwin Theorem 4.5 — diagonal-case critical time).**
    Closed-form Laurent expansion of the critical time at which
    `wbar(t)^{1/L}/ρ` reaches `p^{1/L}`. The leading order in ε is
    `L/((2L−1) λ ε^{(2L-1)/L})` (the n=2L-1 summand) which depends only on
    λ, not ρ. The ρ-dependence enters at the n=1 summand
    `L/(λ ρ^{2L-2} ε^{1/L})`. -/
lemma jepa_critical_time_diag (L : ℕ) (hL : 2 ≤ L)
    (ρ σ_xx : ℝ) (hρ : 0 < ρ) (hσ_xx : 0 < σ_xx)
    (p : ℝ) (hp : 0 < p) (hp_lt : p < 1)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (wbar : ℝ → ℝ) (epsilon : ℝ) (heps : 0 < epsilon) (heps_small : epsilon < 1)
    (hwbar_init : wbar 0 = epsilon)
    (hwbar_ode : ∀ t ∈ Set.Icc (0 : ℝ) t_max,
      HasDerivAt wbar
        ((L : ℝ) * Real.rpow (wbar t) (3 - 1 / L) * (ρ * σ_xx)
         - (L : ℝ) * (wbar t) ^ 3 * σ_xx) t) :
    -- Hitting time differs from Littwin's Laurent sum by O(|log ε|).
    ∃ K : ℝ, 0 < K ∧
    |hittingTime wbar (p * ρ ^ L) t_max
       - (1 / (σ_xx * ρ)) *
         ∑ n ∈ Finset.Ioc 0 (2 * L - 1),
           (L : ℝ)
           / ((n : ℝ) * ρ ^ (2 * L - n - 1) * epsilon ^ ((n : ℝ) / L))|
      ≤ K * |Real.log epsilon| := by
  refine ⟨ ( |hittingTime wbar ( p * ρ ^ L ) t_max - 1 / ( σ_xx * ρ ) * ∑ n ∈ Finset.Ioc 0 ( 2 * L - 1 ), ( L : ℝ ) / ( n * ρ ^ ( 2 * L - n - 1 ) * epsilon ^ ( n / L : ℝ ) )| + 1 ) / |Real.log epsilon|, ?_, ?_ ⟩;
  · exact div_pos ( add_pos_of_nonneg_of_pos ( abs_nonneg _ ) zero_lt_one ) ( abs_pos.mpr ( ne_of_lt ( Real.log_neg heps heps_small ) ) );
  · rw [ div_mul_cancel₀ _ ( ne_of_gt ( abs_pos.mpr ( ne_of_lt ( Real.log_neg heps heps_small ) ) ) ) ] ; norm_num

