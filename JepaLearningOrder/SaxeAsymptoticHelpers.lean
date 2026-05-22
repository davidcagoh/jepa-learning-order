/-
Helper lemmas for the proof of `saxe_singlepole_asymptotic`.
These establish hittingTime properties and energy/Lyapunov bounds
for the Saxe ODE.
-/
import JepaLearningOrder.JEPA
import Mathlib

namespace JepaLearningOrder

open Real

/-! ### hittingTime properties -/

/-
The hittingTime is non-negative when t_max ≥ 0.
-/
lemma hittingTime_nonneg (f : ℝ → ℝ) (θ t_max : ℝ) (ht : 0 ≤ t_max) :
    0 ≤ hittingTime f θ t_max := by
  -- Since the set is non-empty and bounded below by 0, its infimum is non-negative.
  apply Real.sInf_nonneg;
  grind +splitImp

/-
When hittingTime < t_max, the threshold set is non-empty.
-/
lemma hittingTime_set_nonempty (f : ℝ → ℝ) (θ t_max : ℝ)
    (hlt : hittingTime f θ t_max < t_max) :
    ({t ∈ Set.Icc (0 : ℝ) t_max | f t ≥ θ}).Nonempty := by
  -- By definition of hittingTime, if it is less than t_max, then the set {t ∈ Set.Icc 0 t_max | f t ≥ θ} is non-empty.
  unfold hittingTime at hlt;
  contrapose! hlt; aesop;

/-
When hittingTime < t_max, it equals the sInf of the threshold set.
-/
lemma hittingTime_eq_csInf (f : ℝ → ℝ) (θ t_max : ℝ) (ht : 0 ≤ t_max)
    (hlt : hittingTime f θ t_max < t_max) :
    hittingTime f θ t_max = sInf {t ∈ Set.Icc (0 : ℝ) t_max | f t ≥ θ} := by
  unfold hittingTime at *;
  norm_num [ Set.union_comm, Set.union_left_comm, Set.union_assoc, Real.sInf_def ] at *;
  rw [ @csSup_insert ];
  · contrapose! hlt;
    rw [ csSup_insert ];
    · grind;
    · exact ⟨ 0, fun x hx => hx.1.1 ⟩;
    · exact Set.nonempty_iff_ne_empty.mpr ( by rintro h; rw [ h ] at hlt; norm_num at hlt; linarith );
  · exact ⟨ 0, fun x hx => hx.1.1 ⟩;
  · contrapose! hlt; aesop;

/-
hittingTime is in [0, t_max] when < t_max.
-/
lemma hittingTime_mem_Icc (f : ℝ → ℝ) (θ t_max : ℝ) (ht : 0 ≤ t_max)
    (hlt : hittingTime f θ t_max < t_max) :
    hittingTime f θ t_max ∈ Set.Icc 0 t_max := by
  exact ⟨ hittingTime_nonneg f θ t_max ht, le_of_lt hlt ⟩

/-
f achieves ≥ θ at the hittingTime, when continuous and hittingTime < t_max.
-/
lemma f_ge_at_hittingTime (f : ℝ → ℝ) (θ t_max : ℝ) (ht : 0 < t_max)
    (hcont : ContinuousOn f (Set.Icc 0 t_max))
    (hlt : hittingTime f θ t_max < t_max) :
    f (hittingTime f θ t_max) ≥ θ := by
  have := hittingTime_eq_csInf f θ t_max ht.le hlt;
  have h_inf : IsClosed {t ∈ Set.Icc 0 t_max | f t ≥ θ} := by
    exact hcont.preimage_isClosed_of_isClosed isClosed_Icc isClosed_Ici;
  have := h_inf.csInf_mem ( hittingTime_set_nonempty f θ t_max hlt );
  exact ‹hittingTime f θ t_max = sInf { t | t ∈ Set.Icc 0 t_max ∧ f t ≥ θ } › ▸ this ( ⟨ 0, fun x hx => hx.1.1 ⟩ ) |>.2

/-
When f 0 ≥ θ and t_max > 0, hittingTime = 0.
-/
lemma hittingTime_zero_of_ge (f : ℝ → ℝ) (θ t_max : ℝ) (ht : 0 < t_max)
    (hge : f 0 ≥ θ) :
    hittingTime f θ t_max = 0 := by
  refine' le_antisymm ( csInf_le _ _ ) ( le_csInf _ _ );
  · exact ⟨ 0, fun x hx => hx.elim ( fun hx => hx.1.1 ) fun hx => hx.symm ▸ by linarith ⟩;
  · exact Or.inl ⟨ ⟨ le_rfl, ht.le ⟩, hge ⟩;
  · exact ⟨ 0, Or.inl ⟨ ⟨ by norm_num, ht.le ⟩, hge ⟩ ⟩;
  · grind

/-
f is strictly below θ before the hitting time.
-/
lemma f_lt_before_hittingTime (f : ℝ → ℝ) (θ t_max : ℝ) (ht : 0 ≤ t_max)
    (hlt : hittingTime f θ t_max < t_max)
    {t : ℝ} (ht_mem : t ∈ Set.Ico 0 (hittingTime f θ t_max)) :
    f t < θ := by
  contrapose! ht_mem; simp_all +decide [ hittingTime ] ;
  intro ht_nonneg; exact (by
  by_cases ht_le_tmax : t ≤ t_max;
  · exact csInf_le ⟨ 0, by rintro x ( rfl | ⟨ ⟨ hx₁, hx₂ ⟩, hx₃ ⟩ ) <;> linarith ⟩ ( Set.mem_insert_of_mem _ ⟨ ⟨ ht_nonneg, ht_le_tmax ⟩, ht_mem ⟩ );
  · exact le_trans hlt.le ( by linarith ));

/-
When f 0 < θ, ContinuousOn, and hittingTime < t_max, f equals θ at hittingTime.
-/
lemma f_eq_at_hittingTime (f : ℝ → ℝ) (θ t_max : ℝ) (ht : 0 < t_max)
    (hcont : ContinuousOn f (Set.Icc 0 t_max))
    (hlt : hittingTime f θ t_max < t_max)
    (hinit : f 0 < θ) :
    f (hittingTime f θ t_max) = θ := by
  have h_ge : f (hittingTime f θ t_max) ≥ θ := by
    exact?
  have h_le : f (hittingTime f θ t_max) ≤ θ := by
    -- By continuity of $f$ at $hittingTime$, we have $\lim_{t \to hittingTime^-} f(t) = f(hittingTime)$.
    have h_cont_lim_left : Filter.Tendsto f (nhdsWithin (hittingTime f θ t_max) (Set.Iio (hittingTime f θ t_max))) (nhds (f (hittingTime f θ t_max))) := by
      apply_rules [ Filter.Tendsto.mono_left, hcont.continuousAt ];
      · apply hittingTime_mem_Icc; exact ht.le; exact hlt;
      · rw [ nhdsWithin_le_iff ];
        refine' mem_nhdsLT_iff_exists_Ioo_subset.mpr _;
        refine' ⟨ 0, _, _ ⟩ <;> norm_num;
        · exact lt_of_le_of_ne ( hittingTime_nonneg f θ t_max ht.le ) ( Ne.symm <| by rintro h; norm_num [ h ] at *; linarith );
        · exact fun x hx => ⟨ hx.1.le, hx.2.le.trans hlt.le ⟩;
    have h_le : ∀ᶠ t in nhdsWithin (hittingTime f θ t_max) (Set.Iio (hittingTime f θ t_max)), f t < θ := by
      have h_le : ∀ t ∈ Set.Ico 0 (hittingTime f θ t_max), f t < θ := by
        intros t ht_mem
        apply f_lt_before_hittingTime f θ t_max (by linarith) hlt ht_mem;
      rw [ eventually_nhdsWithin_iff ];
      filter_upwards [ lt_mem_nhds ( show 0 < hittingTime f θ t_max from lt_of_le_of_ne ( hittingTime_nonneg f θ t_max ht.le ) ( Ne.symm <| by rintro h; exact absurd hinit <| by simpa [ h ] using h_ge ) ) ] with x hx₁ hx₂ using h_le x ⟨ hx₁.le, hx₂ ⟩;
    exact le_of_tendsto h_cont_lim_left ( h_le.mono fun x hx => le_of_lt hx )
  exact le_antisymm h_le h_ge

/-! ### Saxe ODE Lyapunov bounds -/

set_option maxHeartbeats 800000 in

/-
**Lower bound on hitting time via Lyapunov function Φ.**
For f₀ satisfying the exact Saxe ODE with f₀(0) = ε < θ and hitting time τ < t_max,
the Lyapunov function Φ(t) = f₀(t)^{-(L-1)/L} + (L-1)·λ·t is non-decreasing.
This gives τ ≥ A·(ε^{-(L-1)/L} - θ^{-(L-1)/L}).
-/
lemma saxe_tau_lower_bound (L : ℕ) (hL : 2 ≤ L)
    (lam_r rho_r : ℝ) (hlam : 0 < lam_r) (hrho : 0 < rho_r)
    (p : ℝ) (hp : 0 < p) (hp_lt : p < 1)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (epsilon : ℝ) (heps : 0 < epsilon) (heps_lt : epsilon < 1)
    (f₀ : ℝ → ℝ)
    (hf₀_init : f₀ 0 = epsilon)
    (hf₀_cont : ContinuousOn f₀ (Set.Icc 0 t_max))
    (hf₀_ode : ∀ t ∈ Set.Ioo 0 t_max,
      HasDerivAt f₀
        ((L : ℝ) * (lam_r / rho_r)
            * Real.rpow (f₀ t) (2 - 1 / (L : ℝ))
            * (rho_r - (f₀ t) ^ L)) t)
    (hf₀_reach : hittingTime f₀ (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max < t_max)
    (hεθ : epsilon < p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ)))
    (hf₀_pos : ∀ t ∈ Set.Icc 0 (hittingTime f₀ (p * rpow rho_r (1 / (L : ℝ))) t_max),
      0 < f₀ t) :
    hittingTime f₀ (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max ≥
      1 / ((lam_r / rho_r) * rho_r * ((L : ℝ) - 1)) *
      (epsilon ^ (-((L : ℝ) - 1) / (L : ℝ)) -
       (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) ^ (-((L : ℝ) - 1) / (L : ℝ))) := by
  -- Define the Lyapunov function Φ(t) = f₀(t) ^ (-(L-1)/L) + (L-1)*lam_r*t.
  set phi : ℝ → ℝ := fun t => (f₀ t) ^ (-(L - 1) / L : ℝ) + (L - 1) * lam_r * t;
  -- By definition of $phi$, we know that its derivative is non-negative.
  have h_phi_deriv_nonneg : ∀ t ∈ Set.Ioo 0 (hittingTime f₀ (p * rho_r.rpow (1 / L)) t_max), 0 ≤ deriv phi t := by
    intro t ht
    have h_deriv : deriv phi t = (-(L - 1) / L) * (f₀ t) ^ (-(L - 1) / L - 1 : ℝ) * (deriv f₀ t) + (L - 1) * lam_r := by
      convert HasDerivAt.deriv ( HasDerivAt.add ( HasDerivAt.rpow_const ( hf₀_ode t ⟨ ht.1, ht.2.trans_le <| le_of_lt hf₀_reach ⟩ |> HasDerivAt.differentiableAt |> DifferentiableAt.hasDerivAt ) _ ) ( HasDerivAt.const_mul ( ( L - 1 : ℝ ) * lam_r ) ( hasDerivAt_id t ) ) ) using 1 <;> norm_num;
      · ring;
      · exact Or.inl <| ne_of_gt <| hf₀_pos t <| Set.Ioo_subset_Icc_self ht
    generalize_proofs at *; (
    rw [ h_deriv, hf₀_ode t ⟨ ht.1, ht.2.trans_le <| by linarith ⟩ |> HasDerivAt.deriv ] ; ring_nf ; norm_num [ show L ≠ 0 by linarith ] ;generalize_proofs at *; (
                                                                                                                  norm_num [ sq, mul_assoc, mul_comm, mul_left_comm, ne_of_gt ( zero_lt_two.trans_le hL ), ne_of_gt hrho ] ; ring_nf ;generalize_proofs at *; (
                                                                                                                  norm_num [ mul_assoc, ← Real.rpow_add ( hf₀_pos t ⟨ ht.1.le, ht.2.le ⟩ ) ] ; ring_nf ; norm_num [ ne_of_gt ( zero_lt_two.trans_le hL ), ne_of_gt hrho ] ;generalize_proofs at *; (
                                                                                                                  exact mul_le_mul_of_nonneg_right ( mul_le_mul_of_nonneg_right ( le_mul_of_one_le_right ( by positivity ) ( by norm_cast; linarith ) ) ( by exact pow_nonneg ( le_of_lt ( hf₀_pos t ⟨ ht.1.le, ht.2.le ⟩ ) ) _ ) ) ( by positivity ) ;))));
  -- Since $\phi$ is non-decreasing on $[0, \tau]$, we have $\phi(\tau) \geq \phi(0)$.
  have h_phi_ge_phi0 : phi (hittingTime f₀ (p * rho_r.rpow (1 / L)) t_max) ≥ phi 0 := by
    by_contra h_contra;
    -- Apply the mean value theorem to $\phi$ on the interval $[0, \tau]$.
    obtain ⟨c, hc⟩ : ∃ c ∈ Set.Ioo 0 (hittingTime f₀ (p * rho_r.rpow (1 / L)) t_max), deriv phi c = (phi (hittingTime f₀ (p * rho_r.rpow (1 / L)) t_max) - phi 0) / (hittingTime f₀ (p * rho_r.rpow (1 / L)) t_max - 0) := by
      apply_rules [ exists_deriv_eq_slope ];
      · exact lt_of_le_of_ne ( hittingTime_nonneg _ _ _ ht_max.le ) ( Ne.symm <| by aesop_cat );
      · exact ContinuousOn.add ( ContinuousOn.rpow ( hf₀_cont.mono ( Set.Icc_subset_Icc le_rfl hf₀_reach.le ) ) continuousOn_const <| by intro t ht; exact Or.inl <| ne_of_gt <| hf₀_pos t ht ) <| ContinuousOn.mul continuousOn_const continuousOn_id;
      · refine' fun t ht => DifferentiableAt.differentiableWithinAt _;
        exact DifferentiableAt.add ( DifferentiableAt.rpow ( hf₀_ode t ⟨ ht.1, ht.2.trans_le ( by linarith [ Set.mem_Ioo.mp ht ] ) ⟩ |> HasDerivAt.differentiableAt ) ( by norm_num ) ( ne_of_gt ( hf₀_pos t ⟨ ht.1.le, ht.2.le ⟩ ) ) ) ( DifferentiableAt.mul ( differentiableAt_const _ ) ( differentiableAt_id ) );
    rw [ eq_div_iff ] at hc <;> nlinarith [ h_phi_deriv_nonneg c hc.1, hc.1.1, hc.1.2 ];
  -- Since $f₀(\tau) = \theta$, we have $\phi(\tau) = \theta^{-(L-1)/L} + (L-1)\lambda\tau$.
  have h_phi_tau : phi (hittingTime f₀ (p * rho_r.rpow (1 / L)) t_max) = (p * rho_r.rpow (1 / L)) ^ (-(L - 1) / L : ℝ) + (L - 1) * lam_r * hittingTime f₀ (p * rho_r.rpow (1 / L)) t_max := by
    grind +suggestions;
  simp +zetaDelta at *;
  field_simp at *;
  rw [ div_le_iff₀ ] <;> nlinarith [ show ( L : ℝ ) ≥ 2 by norm_cast, hf₀_init ▸ h_phi_ge_phi0 ]

/-
**Upper bound on hitting time via Lyapunov comparison Φ − C·Γ.**
For f₀ satisfying the exact Saxe ODE with f₀(0) = ε < θ and hitting time τ < t_max,
the function Φ(t) − C·Γ(t) is non-increasing (where Γ(t) = f₀(t)^α,
α = (L²−L+1)/L, C chosen so the derivative is ≤ 0).
This gives τ ≤ A·ε^{-(L-1)/L} + D for a constant D independent of ε.
-/
set_option maxHeartbeats 800000 in
lemma saxe_tau_upper_bound (L : ℕ) (hL : 2 ≤ L)
    (lam_r rho_r : ℝ) (hlam : 0 < lam_r) (hrho : 0 < rho_r)
    (p : ℝ) (hp : 0 < p) (hp_lt : p < 1)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (epsilon : ℝ) (heps : 0 < epsilon) (heps_lt : epsilon < 1)
    (f₀ : ℝ → ℝ)
    (hf₀_init : f₀ 0 = epsilon)
    (hf₀_cont : ContinuousOn f₀ (Set.Icc 0 t_max))
    (hf₀_ode : ∀ t ∈ Set.Ioo 0 t_max,
      HasDerivAt f₀
        ((L : ℝ) * (lam_r / rho_r)
            * Real.rpow (f₀ t) (2 - 1 / (L : ℝ))
            * (rho_r - (f₀ t) ^ L)) t)
    (hf₀_reach : hittingTime f₀ (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max < t_max)
    (hεθ : epsilon < p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ)))
    (hf₀_pos : ∀ t ∈ Set.Icc 0 (hittingTime f₀ (p * rpow rho_r (1 / (L : ℝ))) t_max),
      0 < f₀ t) :
    hittingTime f₀ (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max ≤
      1 / ((lam_r / rho_r) * rho_r * ((L : ℝ) - 1)) *
        epsilon ^ (-((L : ℝ) - 1) / (L : ℝ)) +
      ((L : ℝ) - 1) / (((L : ℝ) ^ 2 - (L : ℝ) + 1) / (L : ℝ) * (L : ℝ) * rho_r * (1 - p ^ L)) *
        (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) ^ (((L : ℝ) ^ 2 - (L : ℝ) + 1) / (L : ℝ)) /
        ((L : ℝ) - 1) / lam_r := by
  revert hf₀_ode hf₀_reach hεθ hf₀_pos;
  intro hf₀_ode hf₀_reach hεθ hf₀_pos
  generalize hτ : hittingTime f₀ (p * rho_r.rpow (1 / L)) t_max = τ
  have hτ_le : τ ≤ t_max := by
    linarith;
  have hΦ_deriv : ∀ t ∈ Set.Ioo 0 τ, deriv (fun t => (f₀ t) ^ (-(L - 1 : ℝ) / L) + (L - 1) * lam_r * t) t = (L - 1) * (lam_r / rho_r) * (f₀ t) ^ L := by
    intro t ht
    have h_deriv : deriv (fun t => (f₀ t) ^ (-(L - 1 : ℝ) / L)) t = (-(L - 1 : ℝ) / L) * (f₀ t) ^ (-(L - 1 : ℝ) / L - 1) * deriv f₀ t := by
      convert HasDerivAt.deriv ( HasDerivAt.rpow_const ( hf₀_ode t ⟨ ht.1, ht.2.trans_le hτ_le ⟩ ) _ ) using 1 <;> norm_num;
      · rw [ hf₀_ode t ⟨ ht.1, ht.2.trans_le hτ_le ⟩ |> HasDerivAt.deriv ] ; norm_num ; ring;
      · exact Or.inl <| ne_of_gt <| hf₀_pos t ⟨ ht.1.le, ht.2.le.trans <| by linarith ⟩;
    convert congr_arg ( fun x => x + ( L - 1 ) * lam_r ) h_deriv using 1;
    · convert HasDerivAt.deriv ( HasDerivAt.add ( hasDerivAt_deriv_iff.mpr _ ) ( HasDerivAt.const_mul _ ( hasDerivAt_id t ) ) ) using 1;
      · ring;
      · exact DifferentiableAt.rpow ( hf₀_ode t ⟨ ht.1, ht.2.trans_le hτ_le ⟩ |> HasDerivAt.differentiableAt ) ( by norm_num ) ( ne_of_gt ( hf₀_pos t ⟨ ht.1.le, ht.2.le.trans ( by linarith ) ⟩ ) );
    · rw [ hf₀_ode t ⟨ ht.1, ht.2.trans_le hτ_le ⟩ |> HasDerivAt.deriv ] ; ring;
      norm_num [ sq, mul_assoc, mul_comm, mul_left_comm, ne_of_gt ( zero_lt_two.trans_le hL ), ne_of_gt hrho ] ; ring;
      norm_num [ mul_assoc, ← Real.rpow_add ( hf₀_pos t ⟨ ht.1.le, ht.2.le.trans ( by linarith ) ⟩ ) ] ; ring;
  have hΓ_deriv : ∀ t ∈ Set.Ioo 0 τ, deriv (fun t => (f₀ t) ^ ((L ^ 2 - L + 1 : ℝ) / L)) t = ((L ^ 2 - L + 1 : ℝ) / L) * L * (lam_r / rho_r) * (f₀ t) ^ L * (rho_r - (f₀ t) ^ L) := by
    intro t ht
    have hΓ_deriv_step : deriv (fun t => (f₀ t) ^ ((L ^ 2 - L + 1 : ℝ) / L)) t = ((L ^ 2 - L + 1 : ℝ) / L) * (f₀ t) ^ ((L ^ 2 - L + 1 : ℝ) / L - 1) * deriv f₀ t := by
      convert HasDerivAt.deriv ( HasDerivAt.rpow_const ( hf₀_ode t ⟨ ht.1, ht.2.trans_le hτ_le ⟩ |> HasDerivAt.differentiableAt |> DifferentiableAt.hasDerivAt ) _ ) using 1 <;> norm_num;
      · ring;
      · exact Or.inl <| ne_of_gt <| hf₀_pos t ⟨ ht.1.le, ht.2.le.trans <| by linarith ⟩;
    rw [ hΓ_deriv_step, hf₀_ode t ⟨ ht.1, ht.2.trans_le hτ_le ⟩ |> HasDerivAt.deriv ] ; ring;
    norm_num [ sq, pow_three, mul_assoc, ne_of_gt ( zero_lt_two.trans_le hL ) ] ; ring;
    norm_num [ mul_assoc, mul_comm, mul_left_comm, ← Real.rpow_add ( hf₀_pos t ⟨ ht.1.le, ht.2.le.trans ( by linarith ) ⟩ ) ] ; ring;
  have hΦΓ_deriv : ∀ t ∈ Set.Ioo 0 τ, deriv (fun t => (f₀ t) ^ (-(L - 1 : ℝ) / L) + (L - 1) * lam_r * t - ((L - 1) / ((L ^ 2 - L + 1 : ℝ) / L * L * rho_r * (1 - p ^ L))) * (f₀ t) ^ ((L ^ 2 - L + 1 : ℝ) / L)) t ≤ 0 := by
    intro t ht
    have hΦΓ_deriv : deriv (fun t => (f₀ t) ^ (-(L - 1 : ℝ) / L) + (L - 1) * lam_r * t - ((L - 1) / ((L ^ 2 - L + 1 : ℝ) / L * L * rho_r * (1 - p ^ L))) * (f₀ t) ^ ((L ^ 2 - L + 1 : ℝ) / L)) t = (L - 1) * (lam_r / rho_r) * (f₀ t) ^ L * (1 - (rho_r - (f₀ t) ^ L) / (rho_r * (1 - p ^ L))) := by
      convert HasDerivAt.deriv ( HasDerivAt.sub ( hΦ_deriv t ht ▸ hasDerivAt_deriv_iff.mpr _ ) ( HasDerivAt.const_mul _ ( hΓ_deriv t ht ▸ hasDerivAt_deriv_iff.mpr _ ) ) ) using 1 <;> norm_num [ mul_assoc, mul_comm, mul_left_comm ];
      · field_simp;
        rw [ mul_div_mul_right _ _ ( by nlinarith [ show ( L : ℝ ) ≥ 2 by norm_cast ] ) ];
      · exact DifferentiableAt.rpow ( hf₀_ode t ⟨ ht.1, ht.2.trans_le hτ_le ⟩ |> HasDerivAt.differentiableAt ) ( by norm_num ) ( ne_of_gt ( hf₀_pos t ⟨ ht.1.le, ht.2.le.trans ( by linarith ) ⟩ ) );
      · exact DifferentiableAt.rpow ( hf₀_ode t ⟨ ht.1, ht.2.trans_le hτ_le ⟩ |> HasDerivAt.differentiableAt ) ( by norm_num ) ( ne_of_gt ( hf₀_pos t ⟨ ht.1.le, ht.2.le.trans ( by linarith ) ⟩ ) );
    have hΦΓ_deriv_nonpos : (rho_r - (f₀ t) ^ L) / (rho_r * (1 - p ^ L)) ≥ 1 := by
      have hΦΓ_deriv_nonpos : f₀ t ^ L < p ^ L * rho_r := by
        have hΦΓ_deriv_nonpos : f₀ t < p * rho_r.rpow (1 / L) := by
          apply f_lt_before_hittingTime f₀ (p * rho_r.rpow (1 / L)) t_max (by linarith) (by
          exact hf₀_reach) (by
          exact ⟨ ht.1.le, by linarith [ ht.2 ] ⟩);
        convert pow_lt_pow_left₀ hΦΓ_deriv_nonpos ( le_of_lt ( hf₀_pos t ⟨ by linarith [ ht.1 ], by linarith [ ht.2 ] ⟩ ) ) ( by linarith : L ≠ 0 ) using 1 ; ring;
        norm_num [ ← Real.rpow_natCast, ← Real.rpow_mul hrho.le, show L ≠ 0 by linarith ];
      rw [ ge_iff_le, le_div_iff₀ ] <;> nlinarith [ pow_pos hp L, pow_lt_one₀ hp.le hp_lt ( by linarith : L ≠ 0 ) ];
    exact hΦΓ_deriv.symm ▸ mul_nonpos_of_nonneg_of_nonpos ( mul_nonneg ( mul_nonneg ( sub_nonneg.mpr ( Nat.one_le_cast.mpr ( by linarith ) ) ) ( div_nonneg hlam.le hrho.le ) ) ( pow_nonneg ( le_of_lt ( hf₀_pos t ⟨ by linarith [ ht.1 ], by linarith [ ht.2 ] ⟩ ) ) _ ) ) ( sub_nonpos.mpr hΦΓ_deriv_nonpos );
  have hΦΓ_noninc : (f₀ τ) ^ (-(L - 1 : ℝ) / L) + (L - 1) * lam_r * τ - ((L - 1) / ((L ^ 2 - L + 1 : ℝ) / L * L * rho_r * (1 - p ^ L))) * (f₀ τ) ^ ((L ^ 2 - L + 1 : ℝ) / L) ≤ (f₀ 0) ^ (-(L - 1 : ℝ) / L) + (L - 1) * lam_r * 0 - ((L - 1) / ((L ^ 2 - L + 1 : ℝ) / L * L * rho_r * (1 - p ^ L))) * (f₀ 0) ^ ((L ^ 2 - L + 1 : ℝ) / L) := by
    by_cases hτ_pos : 0 < τ;
    · have := exists_deriv_eq_slope ( f := fun t => f₀ t ^ ( - ( L - 1 : ℝ ) / L ) + ( L - 1 ) * lam_r * t - ( L - 1 ) / ( ( L ^ 2 - L + 1 : ℝ ) / L * L * rho_r * ( 1 - p ^ L ) ) * f₀ t ^ ( ( L ^ 2 - L + 1 : ℝ ) / L ) ) hτ_pos;
      contrapose! this;
      refine' ⟨ _, _, _ ⟩;
      · refine' ContinuousOn.sub _ _;
        · refine' ContinuousOn.add _ _;
          · refine' ContinuousOn.rpow _ _ _;
            · exact hf₀_cont.mono ( Set.Icc_subset_Icc_right hτ_le );
            · exact continuousOn_const;
            · exact fun x hx => Or.inl <| ne_of_gt <| hf₀_pos x ⟨ hx.1, hx.2.trans <| by linarith ⟩;
          · exact continuousOn_const.mul continuousOn_id;
        · exact ContinuousOn.mul continuousOn_const ( ContinuousOn.rpow ( hf₀_cont.mono ( Set.Icc_subset_Icc le_rfl hτ_le ) ) continuousOn_const <| by intro t ht; exact Or.inl <| ne_of_gt <| hf₀_pos t <| by aesop );
      · refine' fun t ht => DifferentiableAt.differentiableWithinAt _;
        refine' DifferentiableAt.sub _ _;
        · exact DifferentiableAt.add ( DifferentiableAt.rpow ( hf₀_ode t ⟨ ht.1, ht.2.trans_le hτ_le ⟩ |> HasDerivAt.differentiableAt ) ( by norm_num ) ( ne_of_gt ( hf₀_pos t ⟨ ht.1.le, ht.2.le.trans ( by linarith ) ⟩ ) ) ) ( DifferentiableAt.mul ( differentiableAt_const _ ) ( differentiableAt_id ) );
        · exact DifferentiableAt.mul ( differentiableAt_const _ ) ( DifferentiableAt.rpow ( hf₀_ode t ⟨ ht.1, ht.2.trans_le hτ_le ⟩ |> HasDerivAt.differentiableAt ) ( by norm_num ) ( ne_of_gt ( hf₀_pos t ⟨ ht.1.le, ht.2.le.trans ( by linarith ) ⟩ ) ) );
      · intro c hc; rw [ ne_eq, eq_div_iff ] <;> nlinarith [ hΦΓ_deriv c hc ] ;
    · norm_num [ show τ = 0 by linarith [ hittingTime_nonneg f₀ ( p * rho_r.rpow ( 1 / L ) ) t_max ( by linarith ) ] ] at *;
  have hΦΓ_noninc : (f₀ τ) ^ (-(L - 1 : ℝ) / L) + (L - 1) * lam_r * τ - ((L - 1) / ((L ^ 2 - L + 1 : ℝ) / L * L * rho_r * (1 - p ^ L))) * (f₀ τ) ^ ((L ^ 2 - L + 1 : ℝ) / L) ≤ (f₀ 0) ^ (-(L - 1 : ℝ) / L) := by
    refine le_trans hΦΓ_noninc ?_;
    norm_num [ hf₀_init ];
    exact mul_nonneg ( div_nonneg ( sub_nonneg.mpr ( Nat.one_le_cast.mpr ( by linarith ) ) ) ( mul_nonneg ( mul_nonneg ( mul_nonneg ( div_nonneg ( by nlinarith [ show ( L : ℝ ) ≥ 2 by norm_cast ] ) ( by positivity ) ) ( by positivity ) ) ( by positivity ) ) ( sub_nonneg.mpr ( pow_le_one₀ ( by positivity ) hp_lt.le ) ) ) ) ( Real.rpow_nonneg heps.le _ );
  have hΦΓ_noninc : (L - 1) * lam_r * τ ≤ (f₀ 0) ^ (-(L - 1 : ℝ) / L) + ((L - 1) / ((L ^ 2 - L + 1 : ℝ) / L * L * rho_r * (1 - p ^ L))) * (f₀ τ) ^ ((L ^ 2 - L + 1 : ℝ) / L) := by
    linarith [ Real.rpow_pos_of_pos ( hf₀_pos τ ⟨ by linarith [ hittingTime_nonneg f₀ ( p * rho_r.rpow ( 1 / L ) ) t_max ( by linarith ) ], by linarith [ hittingTime_nonneg f₀ ( p * rho_r.rpow ( 1 / L ) ) t_max ( by linarith ) ] ⟩ ) ( - ( L - 1 : ℝ ) / L ) ];
  have hΦΓ_noninc : (L - 1) * lam_r * τ ≤ (f₀ 0) ^ (-(L - 1 : ℝ) / L) + ((L - 1) / ((L ^ 2 - L + 1 : ℝ) / L * L * rho_r * (1 - p ^ L))) * (p * rho_r.rpow (1 / L)) ^ ((L ^ 2 - L + 1 : ℝ) / L) := by
    have := f_eq_at_hittingTime f₀ ( p * rho_r.rpow ( 1 / L ) ) t_max ( by linarith ) hf₀_cont ( by linarith ) ( by linarith ) ; aesop;
  convert div_le_div_of_nonneg_right hΦΓ_noninc ( show 0 ≤ ( L - 1 : ℝ ) * lam_r by exact mul_nonneg ( sub_nonneg.mpr <| Nat.one_le_cast.mpr <| by linarith ) hlam.le ) using 1 ; ring;
  · nontriviality;
    nlinarith [ inv_mul_cancel_left₀ ( show ( L : ℝ ) * lam_r - lam_r ≠ 0 by nlinarith [ show ( L : ℝ ) ≥ 2 by norm_cast ] ) τ ];
  · grind

/-
**f₀ remains positive before the hitting time under the Saxe ODE.**
If f₀ satisfies the Saxe ODE with f₀(0) = ε > 0, then f₀(t) > 0
for all t ∈ [0, hittingTime].
-/
set_option maxHeartbeats 800000 in
lemma saxe_f0_pos (L : ℕ) (hL : 2 ≤ L)
    (lam_r rho_r : ℝ) (hlam : 0 < lam_r) (hrho : 0 < rho_r)
    (p : ℝ) (hp : 0 < p) (hp_lt : p < 1)
    (t_max : ℝ) (ht_max : 0 < t_max)
    (epsilon : ℝ) (heps : 0 < epsilon)
    (f₀ : ℝ → ℝ)
    (hf₀_init : f₀ 0 = epsilon)
    (hf₀_cont : ContinuousOn f₀ (Set.Icc 0 t_max))
    (hf₀_ode : ∀ t ∈ Set.Ioo 0 t_max,
      HasDerivAt f₀
        ((L : ℝ) * (lam_r / rho_r)
            * Real.rpow (f₀ t) (2 - 1 / (L : ℝ))
            * (rho_r - (f₀ t) ^ L)) t)
    (hf₀_reach : hittingTime f₀ (p * Real.rpow rho_r ((1 : ℝ) / (L : ℝ))) t_max < t_max) :
    ∀ t ∈ Set.Icc 0 (hittingTime f₀ (p * rpow rho_r (1 / (L : ℝ))) t_max),
      0 < f₀ t := by
  intro t ht
  by_contra h_neg;
  -- Let $t₀$ be the first time $f₀$ becomes non-positive.
  obtain ⟨t₀, ht₀⟩ : ∃ t₀ ∈ Set.Icc 0 t, f₀ t₀ ≤ 0 ∧ ∀ t' ∈ Set.Ico 0 t₀, f₀ t' > 0 := by
    use sInf {t' ∈ Set.Icc 0 t | f₀ t' ≤ 0};
    have h_inf : IsClosed {t' ∈ Set.Icc 0 t | f₀ t' ≤ 0} := by
      have h_closed : ContinuousOn f₀ (Set.Icc 0 t) := by
        exact hf₀_cont.mono ( Set.Icc_subset_Icc_right ( ht.2.trans ( by linarith [ ht.1, ht.2, hittingTime_mem_Icc f₀ ( p * rho_r.rpow ( 1 / L ) ) t_max ( by linarith ) hf₀_reach ] ) ) );
      exact h_closed.preimage_isClosed_of_isClosed isClosed_Icc isClosed_Iic;
    have := h_inf.csInf_mem;
    simp +zetaDelta at *;
    exact ⟨ this ⟨ t, ⟨ ⟨ by linarith, by linarith ⟩, h_neg ⟩ ⟩ ⟨ 0, fun x hx => hx.1.1 ⟩ |>.1, this ⟨ t, ⟨ ⟨ by linarith, by linarith ⟩, h_neg ⟩ ⟩ ⟨ 0, fun x hx => hx.1.1 ⟩ |>.2, fun t' ht' ht'' => not_le.1 fun h => ht''.not_ge <| csInf_le ⟨ 0, fun x hx => hx.1.1 ⟩ ⟨ ⟨ ht', by linarith [ this ⟨ t, ⟨ ⟨ by linarith, by linarith ⟩, h_neg ⟩ ⟩ ⟨ 0, fun x hx => hx.1.1 ⟩ |>.1.2 ] ⟩, h ⟩ ⟩;
  -- Since $f₀$ is continuous on $[0, t₀]$ and $f₀(t₀) \leq 0$, we have $f₀(t₀) = 0$.
  have h_f₀_t₀_zero : f₀ t₀ = 0 := by
    by_cases ht₀_zero : t₀ = 0;
    · grind;
    · have h_f₀_t₀_zero : Filter.Tendsto f₀ (nhdsWithin t₀ (Set.Iio t₀)) (nhds (f₀ t₀)) := by
        have := hf₀_cont.continuousWithinAt ( show t₀ ∈ Set.Icc 0 t_max from ⟨ ht₀.1.1, ht₀.1.2.trans ( ht.2.trans hf₀_reach.le ) ⟩ );
        refine' this.mono_left _;
        rw [ nhdsWithin_le_iff ];
        exact mem_nhdsLT_iff_exists_Ioo_subset.mpr ⟨ 0, lt_of_le_of_ne ht₀.1.1 ( Ne.symm ht₀_zero ), fun x hx => ⟨ hx.1.le, hx.2.le.trans ( ht₀.1.2.trans ( ht.2.trans hf₀_reach.le ) ) ⟩ ⟩;
      exact le_antisymm ht₀.2.1 ( le_of_tendsto_of_tendsto tendsto_const_nhds h_f₀_t₀_zero <| Filter.eventually_of_mem ( Ioo_mem_nhdsLT <| show 0 < t₀ from lt_of_le_of_ne ht₀.1.1 <| Ne.symm ht₀_zero ) fun x hx => le_of_lt <| ht₀.2.2 x ⟨ hx.1.le, hx.2 ⟩ );
  -- By the mean value theorem, there exists some $c \in (t₀/2, t₀)$ such that $f₀'(c) = (f₀(t₀) - f₀(t₀/2)) / (t₀ - t₀/2)$.
  obtain ⟨c, hc⟩ : ∃ c ∈ Set.Ioo (t₀ / 2) t₀, deriv f₀ c = (f₀ t₀ - f₀ (t₀ / 2)) / (t₀ - t₀ / 2) := by
    apply_rules [ exists_deriv_eq_slope ];
    · linarith [ ht₀.1.1, show t₀ > 0 from lt_of_le_of_ne ht₀.1.1 ( Ne.symm <| by rintro rfl; linarith ) ];
    · exact hf₀_cont.mono ( Set.Icc_subset_Icc ( by linarith [ ht₀.1.1 ] ) ( by linarith [ ht₀.1.2, ht.2, hf₀_reach ] ) );
    · exact fun x hx => ( hf₀_ode x ⟨ by linarith [ hx.1, ht₀.1.1 ], by linarith [ hx.2, ht₀.1.2, ht.2, hf₀_reach ] ⟩ |> HasDerivAt.differentiableAt |> DifferentiableAt.differentiableWithinAt );
  -- Since $f₀(c) > 0$ and $f₀(c)$ is close to $0$, we have $f₀'(c) > 0$.
  have h_deriv_pos : deriv f₀ c > 0 := by
    rw [ hf₀_ode c ⟨ by linarith [ hc.1.1, ht₀.1.1 ], by linarith [ hc.1.2, ht₀.1.2, ht.2, show hittingTime f₀ ( p * rho_r.rpow ( 1 / ( L : ℝ ) ) ) t_max ≤ t_max from le_of_lt hf₀_reach ] ⟩ |> HasDerivAt.deriv ];
    refine' mul_pos ( mul_pos ( mul_pos ( Nat.cast_pos.mpr ( by linarith ) ) ( div_pos hlam hrho ) ) ( Real.rpow_pos_of_pos ( ht₀.2.2 c ⟨ by linarith [ hc.1.1, ht₀.1.1 ], by linarith [ hc.1.2, ht₀.1.2 ] ⟩ ) _ ) ) ( sub_pos.mpr _ );
    refine' lt_of_le_of_lt ( pow_le_pow_left₀ ( le_of_lt ( ht₀.2.2 c ⟨ by linarith [ hc.1.1, ht₀.1.1 ], by linarith [ hc.1.2, ht₀.1.2 ] ⟩ ) ) ( show f₀ c ≤ p * rho_r.rpow ( 1 / ( L : ℝ ) ) from _ ) _ ) _;
    · refine' le_of_not_gt fun h => _;
      have h_contra : hittingTime f₀ (p * rho_r.rpow (1 / (L : ℝ))) t_max ≤ c := by
        unfold hittingTime;
        refine' csInf_le _ _ <;> norm_num;
        · exact ⟨ 0, fun x hx => hx.1.1 ⟩;
        · exact Or.inr ⟨ ⟨ by linarith [ hc.1.1, ht₀.1.1 ], by linarith [ hc.1.2, ht₀.1.2, ht.2, show hittingTime f₀ ( p * rho_r.rpow ( 1 / ( L : ℝ ) ) ) t_max ≤ t_max from le_of_lt hf₀_reach ] ⟩, by simpa using h.le ⟩;
      linarith [ hc.1.1, hc.1.2, ht.1, ht.2, ht₀.1.1, ht₀.1.2 ];
    · norm_num [ mul_pow, ← Real.rpow_natCast, ← Real.rpow_mul hrho.le ];
      norm_num [ show L ≠ 0 by linarith ];
      exact mul_lt_of_lt_one_left hrho ( pow_lt_one₀ hp.le hp_lt ( by linarith ) );
  rw [ hc.2, gt_iff_lt, lt_div_iff₀ ] at h_deriv_pos <;> nlinarith [ ht₀.1.1, ht₀.1.2, hc.1.1, hc.1.2, ht₀.2.2 ( t₀ / 2 ) ⟨ by linarith [ ht₀.1.1, ht₀.1.2, hc.1.1, hc.1.2 ], by linarith [ ht₀.1.1, ht₀.1.2, hc.1.1, hc.1.2 ] ⟩ ]

end JepaLearningOrder