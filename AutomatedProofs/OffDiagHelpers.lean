import Mathlib
import AutomatedProofs.Lemmas

set_option linter.style.longLine false
set_option linter.style.whitespace false

/-- Helper: ε^{(2L-1)/L} ≤ ε^{1/L} for 0 < ε < 1 and L ≥ 2, since (2L-1)/L ≥ 1/L. -/
lemma offDiag_eps_rpow_le {epsilon : ℝ} {L : ℕ}
    (heps : 0 < epsilon) (heps1 : epsilon < 1) (hL : 2 ≤ L) :
    epsilon ^ ((2 * (L : ℝ) - 1) / (L : ℝ)) ≤ epsilon ^ ((1 : ℝ) / (L : ℝ)) := by
  apply Real.rpow_le_rpow_of_exponent_ge heps heps1.le
  have hL_pos : (0 : ℝ) < (↑L : ℝ) := Nat.cast_pos.mpr (by omega)
  have hL2 : (2 : ℝ) ≤ (↑L : ℝ) := by exact_mod_cast hL
  rw [div_eq_mul_inv, div_eq_mul_inv]
  apply mul_le_mul_of_nonneg_right _ (inv_nonneg.mpr hL_pos.le)
  nlinarith

/-- Helper: ∫₀ᵗ κ·P(s) ds ≤ κ·C_int for all t ∈ [0, t_max], given ∫_{Ioo} P ≤ C_int.
    Used in `offDiag_bound` to supply the `hα_int_bound` argument to
    `gronwall_approx_ode_bound`. -/
lemma offDiag_integral_bound {t_max : ℝ} (ht_max : 0 < t_max)
    {P : ℝ → ℝ} {κ C_int : ℝ}
    (hκ : 0 ≤ κ) (hC_int : 0 < C_int)
    (hP_nn : ∀ t ∈ Set.Icc 0 t_max, 0 ≤ P t)
    (hP_cont : ContinuousOn P (Set.Icc 0 t_max))
    (h_int : ∫ u in Set.Ioo 0 t_max, P u ≤ C_int) :
    ∀ t ∈ Set.Icc 0 t_max,
      ∫ s in (0 : ℝ)..t, (κ * P s) ≤ κ * C_int := by
  norm_num [intervalIntegral.integral_of_le ht_max.le] at *
  intro t ht ht'
  rw [intervalIntegral.integral_of_le ht, MeasureTheory.integral_Ioc_eq_integral_Ioo]
  refine' mul_le_mul_of_nonneg_left _ hκ
  refine' le_trans (MeasureTheory.setIntegral_mono_set _ _ _) h_int
  · exact hP_cont.integrableOn_Icc.mono_set Set.Ioo_subset_Icc_self
  · filter_upwards [MeasureTheory.ae_restrict_mem measurableSet_Ioo] with u hu
      using hP_nn u hu.1.le hu.2.le
  · exact MeasureTheory.ae_of_all _ fun x hx => ⟨hx.1, hx.2.trans_le ht'⟩
