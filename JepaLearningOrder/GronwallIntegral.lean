import Mathlib

/-!
# Variable-coefficient Grönwall inequality, integral form

This file proves the integral form of Grönwall's inequality with a variable coefficient β.

## Main result

`gronwall_le_of_le_integral`: If `u : ℝ → ℝ` is continuous on `[a, b]`, `β` is non-negative
and integrable on `[a, b]`, and
  `u t ≤ c + ∫ s in a..t, β s * u s`
for all `t ∈ [a, b]`, then
  `u t ≤ c * exp (∫ s in a..t, β s)`.

This completes the TODO in `Mathlib.Analysis.ODE.Gronwall`, which noted (as of v4.28.0):
> Once we have FTC, prove an inequality for a function satisfying `‖f' x‖ ≤ K x * ‖f x‖ + ε`.

## Proof outline

Define the integrating-factor function
  `G s = (c + ∫ r in a..s, β r * u r) * exp(-∫ r in a..s, β r)`.

By FTC, `G' s = β s * (u s - F s) * exp(-∫β) ≤ 0` (since `u s ≤ F s` and `β s ≥ 0`).
So `G` is antitone, giving `G t ≤ G a = c`. Multiplying through by `exp(∫β) > 0` yields the claim.

## References

* Teschl, *Ordinary Differential Equations and Dynamical Systems* (2012), proof of Theorem 2.10
  (Grönwall's lemma, p. 47).
-/

open MeasureTheory intervalIntegral Real Set

variable {a b : ℝ}

/-- **Grönwall's inequality, integral form (variable coefficient).**

If `u` is continuous on `[a, b]`, `β` is non-negative and continuous on `[a, b]`, and
  `u t ≤ c + ∫ s in a..t, β s * u s`
for all `t ∈ [a, b]`, then `u t ≤ c * exp (∫ s in a..t, β s)`.

This is the variable-coefficient generalisation of the constant-coefficient bound in
`gronwallBound_ε0`. -/
theorem gronwall_le_of_le_integral
    (hab : a ≤ b)
    {u β : ℝ → ℝ} {c : ℝ}
    (hu_cont : ContinuousOn u (Icc a b))
    (hβ_cont : ContinuousOn β (Icc a b))
    (hβ_nn : ∀ s ∈ Icc a b, 0 ≤ β s)
    (hβ_int : IntervalIntegrable β volume a b)
    (hbound : ∀ t ∈ Icc a b, u t ≤ c + ∫ s in a..t, β s * u s) :
    ∀ t ∈ Icc a b, u t ≤ c * exp (∫ s in a..t, β s) := by
  intro t ht
  have hβu_cont : ContinuousOn (fun s => β s * u s) (Icc a b) := hβ_cont.mul hu_cont
  -- It suffices to bound F(t) := c + ∫ₐᵗ βu by c * exp(∫ₐᵗ β)
  suffices h_F : c + ∫ s in a..t, β s * u s ≤ c * exp (∫ s in a..t, β s) from
    le_trans (hbound t ht) h_F
  -- Define auxiliary functions
  -- B(s) = ∫ₐˢ β,  F(s) = c + ∫ₐˢ βu,  G(s) = F(s) * exp(-B(s))
  set B : ℝ → ℝ := fun s => ∫ r in a..s, β r with hB_def
  set F : ℝ → ℝ := fun s => c + ∫ r in a..s, β r * u r with hF_def
  set G : ℝ → ℝ := fun s => F s * exp (-B s) with hG_def
  -- G(a) = c
  have hGa : G a = c := by simp [hG_def, hF_def, hB_def, integral_same]
  -- B and F are continuous on [a, b]
  have hB_cont : ContinuousOn B (Icc a b) := by
    apply (continuousOn_primitive_interval' hβ_int left_mem_uIcc).mono
    rw [uIcc_of_le hab]
  have hF_cont : ContinuousOn F (Icc a b) :=
    continuousOn_const.add
      ((continuousOn_primitive_interval'
        (hβu_cont.intervalIntegrable_of_Icc hab) left_mem_uIcc).mono
        (by rw [uIcc_of_le hab]))
  -- G is continuous on [a, b]
  have hG_cont : ContinuousOn G (Icc a b) := hF_cont.mul hB_cont.neg.rexp
  -- StronglyMeasurableAtFilter instances (for FTC applications in the interior)
  have smaf_β : ∀ s ∈ Ioo a b, StronglyMeasurableAtFilter β (nhds s) volume :=
    ContinuousOn.stronglyMeasurableAtFilter isOpen_Ioo (hβ_cont.mono Ioo_subset_Icc_self)
  have smaf_βu : ∀ s ∈ Ioo a b, StronglyMeasurableAtFilter (fun r => β r * u r) (nhds s) volume :=
    ContinuousOn.stronglyMeasurableAtFilter isOpen_Ioo (hβu_cont.mono Ioo_subset_Icc_self)
  -- G is antitone: G' ≤ 0 on (a, b)
  have hG_anti : AntitoneOn G (Icc a b) := by
    apply antitoneOn_of_deriv_nonpos (convex_Icc a b) hG_cont
    · -- DifferentiableOn G on the interior
      intro s hs
      rw [interior_Icc] at hs
      have hβs_int : IntervalIntegrable β volume a s :=
        (hβ_cont.mono (Icc_subset_Icc_right hs.2.le)).intervalIntegrable_of_Icc hs.1.le
      have hβus_int : IntervalIntegrable (fun r => β r * u r) volume a s :=
        (hβu_cont.mono (Icc_subset_Icc_right hs.2.le)).intervalIntegrable_of_Icc hs.1.le
      have hB_da : HasDerivAt B (β s) s :=
        integral_hasDerivAt_right hβs_int (smaf_β s hs)
          (hβ_cont.continuousAt (Icc_mem_nhds hs.1 hs.2))
      have hI_da : HasDerivAt (fun x => ∫ r in a..x, β r * u r) (β s * u s) s :=
        integral_hasDerivAt_right hβus_int (smaf_βu s hs)
          (hβu_cont.continuousAt (Icc_mem_nhds hs.1 hs.2))
      have hF_da : HasDerivAt F (β s * u s) s := by
        simpa using (hasDerivAt_const s c).add hI_da
      exact (hF_da.mul hB_da.neg.exp).differentiableAt.differentiableWithinAt
    · -- deriv G ≤ 0 on the interior
      intro s hs
      rw [interior_Icc] at hs
      have hs_mem : s ∈ Icc a b := ⟨hs.1.le, hs.2.le⟩
      have hβs_int : IntervalIntegrable β volume a s :=
        (hβ_cont.mono (Icc_subset_Icc_right hs.2.le)).intervalIntegrable_of_Icc hs.1.le
      have hβus_int : IntervalIntegrable (fun r => β r * u r) volume a s :=
        (hβu_cont.mono (Icc_subset_Icc_right hs.2.le)).intervalIntegrable_of_Icc hs.1.le
      have hB_da : HasDerivAt B (β s) s :=
        integral_hasDerivAt_right hβs_int (smaf_β s hs)
          (hβ_cont.continuousAt (Icc_mem_nhds hs.1 hs.2))
      have hI_da : HasDerivAt (fun x => ∫ r in a..x, β r * u r) (β s * u s) s :=
        integral_hasDerivAt_right hβus_int (smaf_βu s hs)
          (hβu_cont.continuousAt (Icc_mem_nhds hs.1 hs.2))
      have hF_da : HasDerivAt F (β s * u s) s := by
        simpa using (hasDerivAt_const s c).add hI_da
      have hG_da : HasDerivAt G
          (β s * u s * exp (-B s) + F s * (exp (-B s) * (-β s))) s :=
        hF_da.mul hB_da.neg.exp
      rw [hG_da.deriv]
      have hus_le_Fs : u s ≤ F s := hbound s hs_mem
      have hrw : β s * u s * exp (-B s) + F s * (exp (-B s) * (-β s)) =
          β s * (u s - F s) * exp (-B s) := by ring
      rw [hrw]
      exact mul_nonpos_of_nonpos_of_nonneg
        (mul_nonpos_of_nonneg_of_nonpos (hβ_nn s hs_mem) (by linarith))
        (exp_pos _).le
  -- G(t) ≤ G(a) = c
  have hGt : G t ≤ c := hGa ▸ hG_anti (left_mem_Icc.mpr hab) ht ht.1
  -- F(t) * exp(-∫β) ≤ c, so F(t) ≤ c * exp(∫β)
  have hEB_pos : 0 < exp (-B t) := exp_pos _
  calc c + ∫ s in a..t, β s * u s
      = F t := rfl
    _ ≤ c / exp (-B t) := (le_div_iff₀ hEB_pos).mpr (by linarith [hGt])
    _ = c * exp (∫ s in a..t, β s) := by
        rw [hB_def, exp_neg, div_inv_eq_mul]
